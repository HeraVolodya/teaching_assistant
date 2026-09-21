"""Димовий тест інсталятора: фікстура й допоміжні функції.

Сам сценарій виконується лише на запакованому інсталяторі в CI (рівень 3).
Тут перевіряється те, що можна перевірити локально й що ламається тихо:
згенерований PDF мусить бути ВАЛІДНИМ — інакше димовий тест падатиме на
завантаженні документа й звинувачуватиме конвеєр приймання замість фікстури.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from helpers_packaging import script

smoke = script("smoke_installer")


@pytest.fixture
def pdf(tmp_path: Path) -> Path:
    return smoke.make_test_pdf(tmp_path / "fixture.pdf")


def test_фікстура_є_валідним_pdf(pdf: Path) -> None:
    data = pdf.read_bytes()
    assert data.startswith(b"%PDF-1.4")
    assert data.rstrip().endswith(b"%%EOF")
    assert b"xref" in data and b"startxref" in data and b"trailer" in data


def test_фікстура_має_рівно_три_сторінки(pdf: Path) -> None:
    pdfium = pytest.importorskip("pypdfium2")
    doc = pdfium.PdfDocument(str(pdf))
    try:
        assert len(doc) == 3
    finally:
        doc.close()


def test_текст_витягується_і_несе_маркер(pdf: Path) -> None:
    """Без витяжного текстового шару димовий тест не мав би що знайти."""
    pdfium = pytest.importorskip("pypdfium2")
    doc = pdfium.PdfDocument(str(pdf))
    try:
        pages = [doc[i].get_textpage().get_text_range() for i in range(len(doc))]
    finally:
        doc.close()
    assert all(smoke.MARKER in text for text in pages)
    # Маркер сторінки унікальний — так тест бачить, ЯКА сторінка процитована.
    assert f"{smoke.MARKER}-02" in pages[1]


def test_зміщення_xref_вказують_на_обʼєкти(pdf: Path) -> None:
    """Найпоширеніша помилка саморобного PDF — з'їхала таблиця xref."""
    data = pdf.read_bytes()
    start = int(data.rsplit(b"startxref", 1)[1].split(b"%%EOF")[0].strip())
    assert data[start:start + 4] == b"xref"
    body = data[start:].split(b"trailer")[0].splitlines()
    for line in body[2:]:                       # пропускаємо "xref" і "0 N"
        if line.endswith(b" n "):
            offset = int(line.split()[0])
            assert data[offset:offset + 1].isdigit()
            assert b" obj" in data[offset:offset + 20]


def test_multipart_коректно_обгортає_файл() -> None:
    body, content_type = smoke.multipart("file", "підручник.pdf", b"%PDF-1.4\n", {"mode": "fast"})
    boundary = content_type.split("boundary=")[1]
    assert body.startswith(f"--{boundary}".encode())
    assert body.endswith(f"--{boundary}--\r\n".encode())
    assert 'filename="підручник.pdf"'.encode() in body
    assert b'name="mode"' in body and b"fast" in body


def test_ендпоїнти_узгоджені_з_реальним_api() -> None:
    """Маршрути димового тесту звіряються з FastAPI-застосунком.

    Розбіжність тут коштує провалу релізного прогону через 40 хвилин збірки —
    тому вона ловиться на кожному пуші.
    """
    from app.main import create_app

    # Через openapi(), а не через `.routes`: у свіжих версіях FastAPI
    # підключені роутери загорнуті й прямого `.path` не мають.
    routes = set(create_app().openapi()["paths"])
    for path in (
        smoke.HEALTH_PATH,
        smoke.ASSISTANTS_PATH,
        smoke.SESSIONS_PATH,
        smoke.CHAT_PATH,
        smoke.UPLOAD_PATH,
        smoke.DOCUMENT_PATH,
    ):
        assert path in routes, f"{path} немає серед маршрутів API"

    assert smoke.STARTUP_TIMEOUT_S == 60      # вимога плану: падіння через 60 с


def test_стани_документа_збігаються_з_доменом() -> None:
    from app.domain import DocStatus

    known = {s.value for s in DocStatus}
    assert known >= smoke.READY_STATES
    assert known >= smoke.FAILED_STATES
    assert DocStatus.READY.value in smoke.READY_STATES


def test_порт_передається_тією_ж_змінною_що_читає_бекенд() -> None:
    """Оболонка задає порт через ASISTENT_PORT — у `app.main` немає argparse."""
    from helpers_packaging import TAURI_DIR, read

    from app.settings import Settings

    assert Settings.model_config["env_prefix"] == "ASISTENT_"
    assert "port" in Settings.model_fields
    assert '"ASISTENT_PORT"' in read(TAURI_DIR / "src" / "sidecar.rs")


def test_пошук_осиротілих_процесів_не_падає(tmp_path: Path) -> None:
    # Головна перевірка kill-on-exit; тут доводимо лише, що вона виконувана
    # на цій ОС і не помилково знаходить чужі процеси.
    assert smoke.orphan_processes(tmp_path) == []


def test_режим_створення_фікстури(tmp_path: Path) -> None:
    target = tmp_path / "out.pdf"
    assert smoke.main(["--make-pdf", str(target)]) == 0
    assert target.stat().st_size > 500
