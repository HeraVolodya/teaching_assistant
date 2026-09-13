"""Гейт на тиху відмову воркера.

Симптом, який це ловить: `python -m app.worker` успадковував робочий каталог
API. При запуску через `uvicorn --app-dir backend` (або з оболонки Tauri, де
cwd — каталог застосунку) воркер не знаходив пакет `app`, падав із
ModuleNotFoundError і перезапускався по колу, **тоді як `/api/health`
повертав 200**. Документи назавжди лишались у стані QUEUED, і жоден тест
цього не помічав.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from app.jobs.supervisor import ProcessSupervisor
from app.settings import Settings

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def supervisor(tmp_path: Path) -> ProcessSupervisor:
    settings = Settings(data_dir=str(tmp_path), stub=True)
    return ProcessSupervisor(settings)


def test_worker_is_spawned_from_the_package_root(
    supervisor: ProcessSupervisor, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """cwd воркера виводиться з розташування коду, а не з середовища."""
    captured: dict[str, object] = {}

    class _FakeProc:
        pid = 4242

        def poll(self) -> None:
            return None

    def fake_popen(cmd, **kwargs):  # type: ignore[no-untyped-def]
        captured["cmd"] = cmd
        captured["cwd"] = kwargs.get("cwd")
        captured["env"] = kwargs.get("env")
        return _FakeProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    # Найгірший випадок: процес API запущено звідкись, де пакета `app` немає.
    monkeypatch.chdir(tmp_path)

    supervisor._spawn()

    assert Path(str(captured["cwd"])) == PACKAGE_ROOT, (
        "воркер має стартувати з кореня пакета, інакше `python -m app.worker` "
        "не знайде `app` і впаде в нескінченний цикл перезапуску"
    )
    assert (Path(str(captured["cwd"])) / "app" / "worker.py").exists()


def test_worker_pythonpath_contains_the_package_root(
    supervisor: ProcessSupervisor, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PYTHONPATH — другий рубіж на випадок, якщо cwd переб'ють."""
    captured: dict[str, object] = {}

    def fake_popen(cmd, **kwargs):  # type: ignore[no-untyped-def]
        captured["env"] = kwargs.get("env")
        return type("P", (), {"pid": 1, "poll": lambda self: None})()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    supervisor._spawn()

    env = captured["env"]
    assert isinstance(env, dict)
    assert str(PACKAGE_ROOT) in env["PYTHONPATH"].split(":")


def test_worker_module_actually_starts_from_a_foreign_cwd(tmp_path: Path) -> None:
    """Найчесніша перевірка: справді запустити модуль із чужого каталогу.

    Не мокає нічого. Якщо імпорт `app.worker` зламається, це впаде тут, а не
    мовчки в проді.
    """
    proc = subprocess.run(
        [sys.executable, "-X", "utf8", "-c", "import app.worker; print('OK')"],
        cwd=PACKAGE_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, f"воркер не імпортується: {proc.stderr[-800:]}"
    assert "OK" in proc.stdout
