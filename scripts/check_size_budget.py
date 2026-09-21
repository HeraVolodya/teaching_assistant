#!/usr/bin/env python3
"""Перевірка розміру артефактів збірки проти бюджету.

Існує заради одного конкретного сценарію: хтось додає залежність, `uv` тягне
колесо torch із `download.pytorch.org` замість PyPI, і в інсталятор непомітно
заїжджає 2.5 ГБ CUDA/cuDNN. На машині розробника з гігабітом це «збірка стала
повільніша», у закритому контурі — «інсталятор не влазить на носій, який ми
привезли в академію».

Бюджети свідомо близькі до фактичних розмірів: запас має бути малим, інакше
перевірка нічого не ловить.

Приклад:
    python scripts/check_size_budget.py --max-mb 900 dist/Asistent_0.1.0_x64-setup.exe
    python scripts/check_size_budget.py --max-mb 1400 src-tauri/resources/runtime
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# UTF-8 НЕЗАЛЕЖНО ВІД КОНСОЛІ.
# На Windows `sys.stdout` має кодування cp1252 з обробником `strict`, тож будь-яка
# кирилиця у виводі валить скрипт з UnicodeEncodeError — саме так падав крок
# перевірки бюджету в CI. У workflow це закрито змінною PYTHONUTF8, але скрипт
# мусить лишатися самодостатнім: docs/DEPLOY.md пропонує запускати його вручну
# з `cmd` на машині викладача, де жодних змінних середовища не виставлено.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    except (AttributeError, ValueError, OSError):  # pragma: no cover — перенаправлений потік
        pass


def size_of(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file() and not p.is_symlink())


def human(n: int) -> str:
    value = float(n)
    for unit in ("Б", "КБ", "МБ", "ГБ"):
        if value < 1024 or unit == "ГБ":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} ГБ"


def largest(path: Path, count: int = 10) -> list[tuple[int, Path]]:
    """Найбільші файли — перше, що треба побачити при перевищенні бюджету."""
    if path.is_file():
        return [(path.stat().st_size, path)]
    files = [(p.stat().st_size, p) for p in path.rglob("*") if p.is_file() and not p.is_symlink()]
    return sorted(files, reverse=True)[:count]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Порівняти розмір артефактів із бюджетом.")
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--max-mb", type=int, required=True)
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args(argv)

    total = 0
    for path in args.paths:
        if not path.exists():
            print(f"Немає такого шляху: {path}", file=sys.stderr)
            return 2
        size = size_of(path)
        total += size
        print(f"{human(size):>10}  {path}")

    budget = args.max_mb * 1024 * 1024
    print(f"{human(total):>10}  РАЗОМ (бюджет {args.max_mb} МБ)")

    if total > budget:
        print(f"\nПЕРЕВИЩЕНО на {human(total - budget)}. Найбільші файли:", file=sys.stderr)
        for path in args.paths:
            for size, item in largest(path, args.top):
                print(f"  {human(size):>10}  {item}", file=sys.stderr)
        print(
            "\nНайімовірніша причина — колесо torch із CUDA-залежностями "
            "(вони мають маркер platform_system == 'Linux' і на Windows/macOS "
            "з'являються лише з download.pytorch.org).",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
