"""Нагляд за воркерами: процеси в постачанні, задача asyncio в розробці.

Дві реалізації одного інтерфейсу `Supervisor`:

* `ProcessSupervisor` — постачання. Тримає N підпроцесів `python -m app.worker`,
  перезапускає їх при виході. Код 75 означає планову переробку процесу
  (пам'ять torch), тож перезапуск миттєвий; будь-який інший ненульовий код —
  аварія, і пауза перед перезапуском росте, щоб не крутити цикл падінь
  сотнею перезапусків на секунду.

* `InlineSupervisor` — розробка, CI, stub-режим. Той самий `JobRunner`, але в
  `asyncio.to_thread`. Дозволено рівно тому, що в цих режимах у конвеєрі
  немає ні torch, ні docling; у постачанні inline заборонено — усі чотири
  причини з `app/worker.py` лишаються в силі.

ЗАВЕРШЕННЯ БЕЗ СИРІТ
Осиротілий `python.exe` після закриття вікна — класична скарга на десктопні
застосунки з sidecar'ами, і димовий тест інсталятора перевіряє саме її. Тому
`stop()` спершу просить чемно (SIGTERM), а через `GRACE_SECONDS` вбиває.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from app.settings import Settings
from app.jobs.exit_codes import RECYCLE_EXIT_CODE

__all__ = ["Supervisor", "ProcessSupervisor", "InlineSupervisor", "create_supervisor"]

log = logging.getLogger("asistent.supervisor")

GRACE_SECONDS = 8.0
MIN_RESTART_DELAY = 0.5
MAX_RESTART_DELAY = 30.0

EmitFn = Callable[[str, dict[str, Any]], None]


class Supervisor(Protocol):
    mode: str

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    def status(self) -> dict[str, Any]: ...


class NullSupervisor:
    """`worker_mode="off"`: індексації немає, API працює лише на читання."""

    mode = "off"

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    def status(self) -> dict[str, Any]:
        return {"mode": self.mode, "running": 0, "detail": "Індексацію вимкнено."}


class ProcessSupervisor:
    """N підпроцесів `python -m app.worker` з автоперезапуском."""

    mode = "process"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._procs: list[subprocess.Popen[bytes]] = []
        self._task: asyncio.Task[None] | None = None
        self._stopping = False
        self._restarts = 0
        self._delay = MIN_RESTART_DELAY

    # ------------------------------------------------------------ запуск
    def _spawn(self) -> subprocess.Popen[bytes]:
        env = dict(os.environ)
        env.setdefault("ASISTENT_DATA_DIR", str(self.settings.paths().data_dir))
        if self.settings.stub:
            env["ASISTENT_STUB"] = "1"
        # -X utf8 обов'язковий на Windows: кириличні імена файлів у межі
        # subprocess інакше падають на cp1251.
        cmd = [sys.executable, "-X", "utf8", "-m", "app.worker",
               "--max-jobs", str(self.settings.worker_recycle_after)]

        # `python -m app.worker` шукає пакет у sys.path, а той для НОВОГО
        # процесу починається з його робочого каталогу. Успадкований cwd тут
        # не годиться: запуск API через `uvicorn --app-dir backend` (або з
        # оболонки Tauri, де cwd — каталог застосунку) лишав воркер без
        # пакета `app`, і він падав із ModuleNotFoundError у нескінченному
        # циклі перезапуску, тоді як API виглядав здоровим. Тому корінь
        # виводимо з розташування цього файлу, а не з середовища.
        package_root = str(Path(__file__).resolve().parents[2])
        env["PYTHONPATH"] = os.pathsep.join(
            [package_root, *(p for p in (env.get("PYTHONPATH"),) if p)]
        )
        log.info("Запускаю воркер: %s (cwd=%s)", " ".join(cmd), package_root)
        return subprocess.Popen(
            cmd, env=env, cwd=package_root, stdin=subprocess.DEVNULL
        )

    async def start(self) -> None:
        self._stopping = False
        for _ in range(max(1, self.settings.worker_count)):
            self._procs.append(self._spawn())
        self._task = asyncio.create_task(self._watch(), name="worker-supervisor")

    async def _watch(self) -> None:
        try:
            while not self._stopping:
                await asyncio.sleep(1.0)
                for i, proc in enumerate(list(self._procs)):
                    code = proc.poll()
                    if code is None:
                        continue
                    if code == RECYCLE_EXIT_CODE:
                        # Планова переробка: пам'ять повернуто ОС, стартуємо одразу.
                        self._delay = MIN_RESTART_DELAY
                    else:
                        self._restarts += 1
                        log.warning("Воркер завершився з кодом %s — перезапуск через %.1f с",
                                    code, self._delay)
                        await asyncio.sleep(self._delay)
                        # Експоненційна пауза: цикл падінь не має з'їдати CPU.
                        self._delay = min(MAX_RESTART_DELAY, self._delay * 2)
                    if self._stopping:
                        return
                    self._procs[i] = self._spawn()
        except asyncio.CancelledError:  # pragma: no cover — нормальне завершення
            raise

    # ------------------------------------------------------------ зупинка
    async def stop(self) -> None:
        self._stopping = True
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._task = None
        for proc in self._procs:
            if proc.poll() is None:
                proc.terminate()
        deadline = time.monotonic() + GRACE_SECONDS
        for proc in self._procs:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                await asyncio.to_thread(proc.wait, remaining or 0.1)
            except Exception:  # noqa: BLE001 — таймаут очікування
                log.warning("Воркер pid=%s не завершився чемно — вбиваю.", proc.pid)
                proc.kill()
        self._procs.clear()

    def status(self) -> dict[str, Any]:
        alive = [p.pid for p in self._procs if p.poll() is None]
        return {
            "mode": self.mode,
            "running": len(alive),
            "pids": alive,
            "restarts": self._restarts,
        }


class InlineSupervisor:
    """Конвеєр у пулі потоків API-процесу. Лише розробка, CI і stub-режим."""

    mode = "inline"

    def __init__(self, settings: Settings, *, db: Any, emit: EmitFn) -> None:
        self.settings = settings
        self.db = db
        self.emit = emit
        self._task: asyncio.Task[None] | None = None
        self._stopping = False
        self._busy = False
        self._done = 0
        self._runner: Any | None = None

    def _get_runner(self) -> Any:
        if self._runner is None:
            from app.jobs.runner import JobRunner

            self._runner = JobRunner(self.db, self.settings, emit=self.emit)
        return self._runner

    async def start(self) -> None:
        self._stopping = False
        self._task = asyncio.create_task(self._loop(), name="worker-inline")

    async def _loop(self) -> None:
        runner = self._get_runner()
        try:
            while not self._stopping:
                self._busy = True
                try:
                    worked = await asyncio.to_thread(runner.run_once)
                finally:
                    self._busy = False
                if worked:
                    self._done += 1
                    continue
                await asyncio.sleep(self.settings.worker_poll_interval_s)
        except asyncio.CancelledError:  # pragma: no cover
            raise
        except Exception:  # noqa: BLE001 — inline-воркер не має валити API
            log.exception("Inline-воркер зупинився через помилку")

    async def stop(self) -> None:
        self._stopping = True
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._task = None

    def status(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "running": 1 if self._task and not self._task.done() else 0,
            "busy": self._busy,
            "completed": self._done,
        }


def create_supervisor(settings: Settings, *, db: Any, emit: EmitFn) -> Supervisor:
    if settings.worker_mode == "off":
        return NullSupervisor()
    if settings.worker_mode == "inline":
        return InlineSupervisor(settings, db=db, emit=emit)
    return ProcessSupervisor(settings)
