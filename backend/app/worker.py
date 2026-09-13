"""Процес-воркер індексації. Запуск: `python -m app.worker`.

ЧОМУ ПРОЦЕС, А НЕ ПОТІК — ЧОТИРИ ПРИЧИНИ, І ЖОДНА З НИХ НЕ ЕСТЕТИЧНА

1. **GIL.** `onnxruntime` і `torch` тримають інтерпретатор довгими блоками
   всередині C. Індексація на 1000 сторінок у потоці заморозила б SSE-стрім
   чату: викладач бачив би, як відповідь зупиняється посеред речення на
   кілька секунд і продовжується. Процес такого не робить за побудовою.

2. **Ізоляція збоїв.** Некоректний PDF валить `pdfium` не винятком, а
   сегфолтом. У потоці це вбиває застосунок разом із чатом, історією і
   незбереженою роботою; у процесі це рівно один провалений документ.

3. **Скасовуваність.** Потік, застряглий у C-виклику, перервати НЕМОЖЛИВО —
   Python просто не отримає керування, щоб перевірити прапорець. Процес
   можна вбити. Тому «скасовуване індексування» чесно реалізується лише
   процесами; усе інше було б обіцянкою, яку код не виконує.

4. **Пам'ять.** Кешувальний алокатор `torch` не повертає звільнені блоки ОС.
   Воркер після сорока документів тримає ~3 ГБ, і на 8-гігабайтній машині це
   починає конкурувати з LM Studio. Переробка процесу після N завдань
   повертає пам'ять миттєво; для потоків такого механізму не існує.

КОД ВИХОДУ 75 = «ПЕРЕРОБИ МЕНЕ»
Досягнувши `worker_recycle_after` завдань, воркер завершується з кодом 75
(`EX_TEMPFAIL`). Наглядач бачить саме цей код і запускає новий процес без
паузи й без запису в журнал помилок. Будь-який інший ненульовий код — це
справжня аварія, і наглядач сповільнює перезапуск.
"""

from __future__ import annotations

from app import net_guard

# Гард ставиться ПЕРШИМ ділом у процесі, до будь-якого імпорту, що створює
# HTTP-клієнти. Docling вимагає `enable_remote_services=True` навіть для
# 127.0.0.1 — саме цей гард робить той прапорець безпечним.
net_guard.install()

import argparse  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import signal  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from typing import Any  # noqa: E402

from app.config import export_model_env  # noqa: E402
from app.jobs.exit_codes import RECYCLE_EXIT_CODE  # noqa: E402
from app.jobs.runner import JobRunner, open_database  # noqa: E402
from app.settings import Settings, get_settings  # noqa: E402

# RECYCLE_EXIT_CODE лишається в `__all__` для сумісності, але ЖИВЕ у
# `app.jobs.exit_codes`: коли він визначався тут, супервізор імпортував його
# з `app.worker`, а `app.worker` через `app.jobs.runner` тягнув пакет
# `app.jobs`, який імпортував супервізор — цикл.
__all__ = ["RECYCLE_EXIT_CODE", "WorkerStop", "run_worker", "main"]

log = logging.getLogger("asistent.worker")


class WorkerStop(Exception):
    """SIGTERM/SIGINT: доробити поточне завдання й вийти."""


def parent_is_gone(original_ppid: int) -> bool:
    """Чи помер процес, який нас запустив.

    SIGKILL батька не каскадує на дітей: на Unix воркер усиновлює init і
    його PPID стає 1, після чого він живе ВІЧНО — тримає стару копію коду,
    стару базу й далі краде завдання з черги. Спостережено на практиці:
    два таких воркери пережили кілька перезапусків API й мовчки конкурували
    з новими за ті самі завдання.

    Плагін оболонки Tauri цього теж не вирішує (tauri#11686), а Job Object
    прикриває лише Windows. Тому єдина справді переносна гарантія — щоб
    воркер сам стежив за батьком.

    `original_ppid` фіксується на старті: порівняння зі збереженим значенням
    ловить і випадок, коли PID батька перевикористали під інший процес.
    """
    if os.name == "nt":  # pragma: no cover — на Windows працює Job Object
        return False
    current = os.getppid()
    return current != original_ppid or current == 1


def _install_signal_handlers(state: dict[str, bool]) -> None:
    def handler(signum: int, _frame: Any) -> None:
        # НЕ вбиваємо процес одразу: посеред сторінки це втрата вже
        # виконаної роботи. Ставимо прапорець — цикл вийде на найближчій
        # межі завдання, а лізинг однаково поверне завдання в чергу, якщо
        # нас усе-таки вб'ють жорстко.
        log.info("Воркер отримав сигнал %s — завершуюсь після поточного завдання.", signum)
        state["stop"] = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, handler)
        except (ValueError, OSError):
            # Не головний потік або платформа без сигналу — не критично.
            pass


def run_worker(
    settings: Settings | None = None,
    *,
    max_jobs: int | None = None,
    once: bool = False,
    idle_timeout: float | None = None,
) -> int:
    """Цикл воркера. Повертає код виходу процесу."""
    settings = settings or get_settings()
    os.environ.update(export_model_env(settings.paths()))

    state = {"stop": False}
    _install_signal_handlers(state)

    db = open_database(settings)
    runner = JobRunner(db, settings)
    limit = max_jobs if max_jobs is not None else settings.worker_recycle_after
    idle_since: float | None = None

    log.info(
        "Воркер стартував: pid=%s, режим=%s, ліміт завдань=%s, stub=%s",
        os.getpid(), settings.default_ingest_mode.value, limit, settings.stub,
    )
    original_ppid = os.getppid()
    try:
        while not state["stop"]:
            # Перевірка перед кожним завданням, а не в окремому потоці:
            # осиротілий воркер має піти тихо, але ЛИШЕ на межі завдань,
            # щоб не кинути документ у стані «Обробляється» з живим лізингом.
            if parent_is_gone(original_ppid):
                log.info("Батьківський процес зник — воркер %s завершується.", os.getpid())
                return 0
            # Прострочені лізинги повертаємо в чергу першими: інакше після
            # аварійного перезапуску документ висів би «Обробляється» до
            # закінчення лізингу навіть тоді, коли воркер уже вільний.
            runner.queue.reclaim_expired()
            worked = runner.run_once()
            if worked:
                idle_since = None
                if limit and runner.jobs_done >= limit:
                    log.info("Виконано %s завдань — переробляю процес.", runner.jobs_done)
                    return RECYCLE_EXIT_CODE
                continue
            if once:
                return 0
            now = time.monotonic()
            idle_since = idle_since if idle_since is not None else now
            if idle_timeout is not None and (now - idle_since) >= idle_timeout:
                return 0
            time.sleep(settings.worker_poll_interval_s)
    finally:
        try:
            db.close()
        except Exception:  # noqa: BLE001 — вихід не має падати на закритті
            log.debug("Помилка закриття БД воркера", exc_info=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="app.worker", description="Воркер індексації Асістента"
    )
    parser.add_argument("--data-dir", default=None, help="Каталог даних застосунку")
    parser.add_argument("--max-jobs", type=int, default=None,
                        help="Скільки завдань виконати до переробки процесу")
    parser.add_argument("--once", action="store_true",
                        help="Виконати доступні завдання й вийти")
    parser.add_argument("--idle-timeout", type=float, default=None,
                        help="Вийти після N секунд простою")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    if args.data_dir:
        os.environ["ASISTENT_DATA_DIR"] = str(args.data_dir)

    settings = get_settings()
    return run_worker(
        settings, max_jobs=args.max_jobs, once=args.once, idle_timeout=args.idle_timeout
    )


if __name__ == "__main__":  # pragma: no cover — точка входу процесу
    raise SystemExit(main())
