"""Гейт на осиротілий воркер.

Спостережено на живій машині: після кількох перезапусків API в системі
лишились два процеси `python -m app.worker` із PPID=1. SIGKILL батька не
каскадує на дітей, тож вони жили далі, тримали стару копію коду й стару базу
і тихо конкурували з новими воркерами за ті самі завдання.

Job Object прикриває лише Windows, плагін оболонки Tauri цього не вирішує
(tauri#11686) — отже переносна гарантія має бути в самому воркері.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from app.worker import parent_is_gone

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def test_live_parent_is_not_reported_gone() -> None:
    """Найважливіше: живий батько НЕ має вважатися зниклим.

    Хибне спрацювання тут гірше за витік процесів — воркер вимикався б
    посеред нормальної роботи, і документи зависали б назавжди.
    """
    assert parent_is_gone(os.getppid()) is False


@pytest.mark.skipif(os.name == "nt", reason="на Windows діє Job Object")
def test_changed_ppid_is_reported_gone() -> None:
    """PPID змінився — батька, який нас запускав, уже немає."""
    assert parent_is_gone(os.getppid() + 100_000) is True


@pytest.mark.skipif(os.name == "nt", reason="на Windows діє Job Object")
def test_init_adoption_is_reported_gone(monkeypatch: pytest.MonkeyPatch) -> None:
    """PPID == 1 означає, що процес усиновив init."""
    monkeypatch.setattr(os, "getppid", lambda: 1)
    assert parent_is_gone(1) is True


@pytest.mark.skipif(os.name == "nt", reason="на Windows діє Job Object")
def test_orphaned_worker_actually_exits() -> None:
    """Найчесніша перевірка: справді осиротити процес і дочекатися виходу.

    Запускаємо дід → батько → онук. Батько миттєво помирає, онук лишається
    з PPID=1 і мусить завершитись сам, без жодного сигналу ззовні.
    """
    child = textwrap.dedent(
        """
        import os, sys, time
        sys.path.insert(0, %r)
        from app.worker import parent_is_gone
        original = os.getppid()
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if parent_is_gone(original):
                print("ORPHANED", flush=True)
                sys.exit(0)
            time.sleep(0.05)
        print("STILL_ALIVE", flush=True)
        sys.exit(1)
        """
    ) % str(PACKAGE_ROOT)

    parent = textwrap.dedent(
        """
        import subprocess, sys, os
        p = subprocess.Popen([sys.executable, "-c", %r],
                             stdout=open(%r, "w"), stderr=subprocess.DEVNULL)
        os._exit(0)   # батько гине миттєво, дитина осиротіє
        """
    ) % (child, "/tmp/asistent_orphan_probe.txt")

    out = Path("/tmp/asistent_orphan_probe.txt")
    out.unlink(missing_ok=True)

    subprocess.run([sys.executable, "-c", parent], timeout=30, check=False)

    # Онук лишається жити після смерті батька; чекаємо, поки він сам вийде.
    import time

    deadline = time.monotonic() + 25
    text = ""
    while time.monotonic() < deadline:
        if out.exists():
            text = out.read_text()
            if text.strip():
                break
        time.sleep(0.2)

    assert "ORPHANED" in text, (
        f"осиротілий воркер не завершився сам (вивід: {text!r}) — "
        "він житиме вічно й конкуруватиме за завдання з новими воркерами"
    )
    out.unlink(missing_ok=True)
