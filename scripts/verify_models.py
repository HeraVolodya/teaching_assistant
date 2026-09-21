#!/usr/bin/env python3
"""Перевірка цілісності моделей за `manifest.json` (SHA-256 кожного файлу).

Викликається:
  * майстром першого запуску після встановлення з USB — тоді потрібен `--json`,
    щоб UI показав прогрес і назвав конкретний зіпсований файл;
  * у CI перед пакуванням — щоб у інсталятор не потрапив огризок.

Свідомо НЕ імпортує ані `fetch_models`, ані нічого з `app`: це той код, який
працює на машині викладача в закритому контурі. Він мусить бути читабельним
самодостатнім файлом на стандартній бібліотеці, у якому немає жодного рядка,
здатного відкрити сокет. Дублювання `sha256_file` — свідома ціна цієї
властивості.

Приклад:
    python scripts/verify_models.py --models-dir "%LOCALAPPDATA%\\Asistent\\models"
    python scripts/verify_models.py --models-dir ./models --json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
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

MANIFEST_NAME = "manifest.json"

OK = "ok"
MISSING = "missing"
SIZE_MISMATCH = "size_mismatch"
HASH_MISMATCH = "hash_mismatch"

_UK = {
    OK: "гаразд",
    MISSING: "файл відсутній",
    SIZE_MISMATCH: "невірний розмір",
    HASH_MISMATCH: "невірна контрольна сума",
}


@dataclass(frozen=True, slots=True)
class FileResult:
    model_id: str
    relpath: str
    status: str
    expected_bytes: int
    actual_bytes: int

    @property
    def ok(self) -> bool:
        return self.status == OK

    def describe(self) -> str:
        return f"[{self.model_id}] {self.relpath}: {_UK[self.status]}"


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def human(n: int) -> str:
    value = float(n)
    for unit in ("Б", "КБ", "МБ", "ГБ"):
        if value < 1024 or unit == "ГБ":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} ГБ"


def load_manifest(models_dir: Path) -> dict:
    path = models_dir / MANIFEST_NAME
    if not path.exists():
        raise SystemExit(
            f"Не знайдено {path}. Каталог моделей неповний або вказано не ту теку. "
            f"Моделі постачаються на носії разом з інсталятором."
        )
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Пошкоджений {path}: {exc}") from exc


def verify_file(models_dir: Path, model_id: str, entry: dict, *, deep: bool = True) -> FileResult:
    """Перевірити один файл. `deep=False` — лише наявність і розмір (швидка проба)."""
    rel = entry["relpath"]
    expected_bytes = int(entry["bytes"])
    path = models_dir / rel

    if not path.exists():
        return FileResult(model_id, rel, MISSING, expected_bytes, 0)

    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        # Розмір — безкоштовний фільтр: обірване копіювання з USB ловиться тут,
        # без читання 1.8 ГБ.
        return FileResult(model_id, rel, SIZE_MISMATCH, expected_bytes, actual_bytes)

    if deep and sha256_file(path) != entry["sha256"]:
        return FileResult(model_id, rel, HASH_MISMATCH, expected_bytes, actual_bytes)

    return FileResult(model_id, rel, OK, expected_bytes, actual_bytes)


def verify(
    models_dir: Path,
    manifest: dict,
    *,
    deep: bool = True,
    on_progress=None,
) -> list[FileResult]:
    """Перевірити всі файли маніфесту.

    `on_progress(done, total, result)` викликається після кожного файлу — саме
    через нього майстер першого запуску малює прогрес-бар: перевірка 1.8 ГБ
    триває десятки секунд, і мовчазний застосунок у цей час виглядає зависшим.
    """
    entries = [(m["id"], f) for m in manifest.get("models", []) for f in m.get("files", [])]
    total = len(entries)
    results: list[FileResult] = []
    for index, (model_id, entry) in enumerate(entries, start=1):
        result = verify_file(models_dir, model_id, entry, deep=deep)
        results.append(result)
        if on_progress is not None:
            on_progress(index, total, result)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Перевірити моделі за manifest.json.")
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--quick", action="store_true", help="Лише наявність і розмір, без SHA-256.")
    parser.add_argument("--json", action="store_true", help="Машинний вивід для UI майстра.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    models_dir = args.models_dir.expanduser().resolve()
    manifest = load_manifest(models_dir)

    def progress(done: int, total: int, result: FileResult) -> None:
        if args.json or args.quiet:
            return
        mark = "✓" if result.ok else "✗"
        print(f"  {mark} [{done}/{total}] {result.relpath}", file=sys.stderr)

    results = verify(models_dir, manifest, deep=not args.quick, on_progress=progress)
    bad = [r for r in results if not r.ok]

    if args.json:
        print(
            json.dumps(
                {
                    "models_dir": str(models_dir),
                    "profile": manifest.get("profile"),
                    "fingerprint": manifest.get("fingerprint"),
                    "checked": len(results),
                    "failed": len(bad),
                    "problems": [
                        {
                            "model": r.model_id,
                            "relpath": r.relpath,
                            "status": r.status,
                            "message": _UK[r.status],
                            "expected_bytes": r.expected_bytes,
                            "actual_bytes": r.actual_bytes,
                        }
                        for r in bad
                    ],
                },
                ensure_ascii=False,
            )
        )
    elif bad:
        print(f"\nПОМИЛКА: пошкоджено або відсутньо {len(bad)} з {len(results)} файлів:", file=sys.stderr)
        for r in bad:
            print(f"  {r.describe()}", file=sys.stderr)
        print(
            "\nСкопіюйте теку моделей з носія повторно. Якщо помилка повторюється — "
            "носій пошкоджено; звірте контрольні суми з паперовим актом постачання.",
            file=sys.stderr,
        )
    elif not args.quiet:
        print(f"Перевірено {len(results)} файлів ({human(manifest.get('total_bytes', 0))}) — усе гаразд.")

    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
