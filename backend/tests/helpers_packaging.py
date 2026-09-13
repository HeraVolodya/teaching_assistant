"""Спільні помічники для тестів пакування.

Скрипти постачання живуть у `<корінь>/scripts`, а не в пакеті `app`: вони не
частина застосунку і не мають до нього імпортуватися. Щоб тести могли їх
перевіряти, каталог додається в `sys.path` один раз саме тут.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
TAURI_DIR = REPO_ROOT / "src-tauri"
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def script(name: str) -> ModuleType:
    """Імпортувати скрипт постачання як модуль."""
    return importlib.import_module(name)


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def rust_code_lines(path: Path) -> int:
    """Рядки коду Rust без коментарів і порожніх рядків.

    Коментарі в цьому проєкті — не шум, а носій обґрунтувань, тому бюджет
    «оболонка ~300 рядків» міряється саме кодом.
    """
    count = 0
    for raw in read(path).splitlines():
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        count += 1
    return count
