#!/usr/bin/env python3
"""Гейт закритого контуру: доводить, що застосунок не може вийти в мережу.

Довіра до продукту тримається на твердженні «жоден навчальний матеріал не
залишає цей комп'ютер». Це твердження мусить бути виконуваним кодом і кроком
CI, а не обіцянкою в README. Скрипт валить збірку, якщо:

  * будь-яка спроба вихідного з'єднання (httpx / urllib / сирий сокет) НЕ
    заблокована;
  * офлайн-змінні HuggingFace не виставлені;
  * імпорт API-процесу тягне `torch`, `docling` або `rapidocr` — це не питання
    мережі, а питання 6 секунд до чутливості вікна на Windows, і ловиться воно
    тим самим прогоном (див. CONTRACT.md, правило 1).

Виконується в найшвидшому рівні CI, на кожен пуш.
"""

from __future__ import annotations

import importlib
import os
import socket
import sys
import urllib.request
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND))

# Шлях доповнюється вище — інакше `app` не знайдеться при запуску з кореня.
from app import net_guard

# Заборонені в API-процесі важкі імпорти.
FORBIDDEN_IN_API = ("torch", "docling", "rapidocr")

# Модулі, які має тягнути API-процес. Якщо `app.main` ще не існує (модуль
# пишеться паралельно) — перевірка чесно пропускається з попередженням.
API_MODULES = ("app.config", "app.domain", "app.db.repositories", "app.main")

failures: list[str] = []


def expect_blocked(what: str, fn) -> None:
    try:
        fn()
    except net_guard.OutboundNetworkBlocked:
        print(f"  OK   {what}: заблоковано")
        return
    except Exception as exc:  # noqa: BLE001
        # Будь-який інший виняток означає, що ми не дійшли до гарду — це не
        # доказ. У CI без мережі так виглядає, наприклад, збій DNS.
        failures.append(f"{what}: очікували OutboundNetworkBlocked, отримали {type(exc).__name__}: {exc}")
        return
    failures.append(f"{what}: З'ЄДНАННЯ ПРОЙШЛО. Гард не працює.")


def main() -> int:
    net_guard.install()

    print("Мережевий гард:")
    expect_blocked("сирий сокет 1.1.1.1:443", lambda: socket.create_connection(("1.1.1.1", 443), 3))
    expect_blocked(
        "urllib → huggingface.co",
        lambda: urllib.request.urlopen("https://huggingface.co", timeout=5),
    )
    try:
        import httpx

        expect_blocked("httpx → example.com", lambda: httpx.get("https://example.com", timeout=5))
    except ImportError:
        failures.append("httpx не встановлено — гард httpx неперевірений")

    print("Петля назад дозволена:")
    if not net_guard.is_allowed("127.0.0.1"):
        failures.append("127.0.0.1 заблоковано — LM Studio стане недосяжним")
    else:
        print("  OK   127.0.0.1 дозволено")

    print("Офлайн-змінні:")
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_SYMLINKS", "PYTHONUTF8"):
        value = os.environ.get(key)
        if value != "1":
            failures.append(f"{key}={value!r}, очікували '1'")
        else:
            print(f"  OK   {key}=1")

    print("Важкі імпорти в API-процесі:")
    for name in API_MODULES:
        try:
            importlib.import_module(name)
        except ImportError as exc:
            print(f"  ПРОПУЩЕНО {name}: {exc}")
            continue
        print(f"  імпортовано {name}")
    for heavy in FORBIDDEN_IN_API:
        if heavy in sys.modules:
            failures.append(
                f"API-процес імпортує {heavy!r}. Це +1.5–4 с холодного старту на Windows "
                f"і пряме порушення CONTRACT.md, правило 1."
            )
        else:
            print(f"  OK   {heavy} не імпортовано")

    if failures:
        print("\nГЕЙТ ОФЛАЙНУ ПРОВАЛЕНО:", file=sys.stderr)
        for item in failures:
            print(f"  - {item}", file=sys.stderr)
        return 1

    print("\nГейт офлайну пройдено.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
