"""Копія для читання: вибір програми, запасні шляхи, помилки, перевикористання.

Справжній Word тут не запускається ніколи: програми підміняються, і
перевіряється саме логіка вибору — що за чим пробується, що кешується і
яку помилку з якою підказкою отримає викладач.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from app.ingestion import convert
from app.ingestion.convert import (
    ConversionCancelled,
    ConversionError,
    Rendition,
    office_script,
    prepare_rendition,
    remove_renditions,
    view_file,
)
from app.ingestion.formats import ACCEPTED_SUFFIXES, source_kind

NO_TOOLS = {"word": False, "excel": False, "powerpoint": False,
            "libreoffice": False, "djvulibre": False}


def _doc(tmp_path: Path, name: str, data: bytes = b"PK\x03\x04 test") -> Path:
    docs = tmp_path / "docs"
    docs.mkdir(exist_ok=True)
    path = docs / name
    path.write_bytes(data)
    return path


def _tools(monkeypatch, **present: bool) -> None:
    monkeypatch.setattr(convert, "available_tools", lambda: {**NO_TOOLS, **present})


# -------------------------------------------------------------- формати
@pytest.mark.parametrize(
    ("name", "kind"),
    [("a.pdf", "pdf"), ("a.MD", "text"), ("a.docx", "word"), ("a.doc", "word"),
     ("a.rtf", "word"), ("a.odt", "word"), ("a.pptx", "slides"), ("a.ppt", "slides"),
     ("a.djvu", "djvu"), ("a.djv", "djvu"), ("a.xlsx", "sheet"), ("a.xls", "sheet"),
     ("a.csv", "sheet"), ("a.ods", "sheet"), ("a.exe", None)],
)
def test_source_kind_by_suffix(name: str, kind: str | None) -> None:
    assert source_kind(name) == kind
    assert (Path(name).suffix.lower() in ACCEPTED_SUFFIXES) is (kind is not None)


# ------------------------------------------------------- PDF і текст як є
def test_pdf_and_text_are_read_from_the_original(tmp_path) -> None:
    for name in ("книга.pdf", "конспект.md"):
        path = _doc(tmp_path, name)
        rendition = prepare_rendition(path, documents_dir=path.parent)
        assert rendition.path == path and not rendition.is_copy


# ----------------------------------------------------------------- Office
def test_word_converts_to_a_pdf_copy_next_to_the_original(tmp_path, monkeypatch) -> None:
    path = _doc(tmp_path, "abc.docx")
    _tools(monkeypatch, word=True)
    calls: list[str] = []

    def fake_office(app, src, dst, **_kw):
        calls.append(app)
        dst.write_bytes(b"%PDF-1.7")
        return "ASISTENT-OK"

    monkeypatch.setattr(convert, "_office", fake_office)
    rendition = prepare_rendition(path, documents_dir=path.parent)
    assert calls == ["word"]
    assert rendition.path == path.parent / "abc.view.pdf"
    assert rendition.tool == "word" and rendition.is_copy
    assert (path.parent / "abc.view.json").is_file()
    assert view_file(path.parent, path) == (rendition.path, "pdf")


def test_finished_copy_is_reused_on_reingest(tmp_path, monkeypatch) -> None:
    """Word на 300 сторінках думає десятки секунд — переіндексація не має платити знову."""
    path = _doc(tmp_path, "abc.docx")
    _tools(monkeypatch, word=True)
    calls: list[str] = []

    def fake_office(app, src, dst, **_kw):
        calls.append(app)
        dst.write_bytes(b"%PDF-1.7")
        return "ASISTENT-OK"

    monkeypatch.setattr(convert, "_office", fake_office)
    prepare_rendition(path, documents_dir=path.parent)
    again = prepare_rendition(path, documents_dir=path.parent)
    assert calls == ["word"]
    assert again.reused and again.tool == "word"


def test_powerpoint_reports_visible_slide_numbers(tmp_path, monkeypatch) -> None:
    path = _doc(tmp_path, "abc.pptx")
    _tools(monkeypatch, powerpoint=True)

    def fake_office(app, src, dst, **_kw):
        dst.write_bytes(b"%PDF-1.7")
        return "ASISTENT-OK slides=1,2,4"          # третій слайд прихований

    monkeypatch.setattr(convert, "_office", fake_office)
    rendition = prepare_rendition(path, documents_dir=path.parent)
    assert rendition.page_labels == {1: "слайд 1", 2: "слайд 2", 3: "слайд 4"}


def test_libreoffice_is_the_fallback_when_word_fails(tmp_path, monkeypatch) -> None:
    path = _doc(tmp_path, "abc.doc")
    _tools(monkeypatch, word=True, libreoffice=True)

    def broken_word(*_a, **_kw):
        raise convert._ToolFailed("Word завис")

    def fake_libre(src, dst, target, **_kw):
        dst.write_bytes(b"%PDF-1.7")

    monkeypatch.setattr(convert, "_office", broken_word)
    monkeypatch.setattr(convert, "_libreoffice", fake_libre)
    rendition = prepare_rendition(path, documents_dir=path.parent)
    assert rendition.tool == "libreoffice"


def test_password_protected_file_fails_fast_with_its_own_code(tmp_path, monkeypatch) -> None:
    path = _doc(tmp_path, "abc.docx")
    _tools(monkeypatch, word=True, libreoffice=True)

    def locked(*_a, **_kw):
        raise convert._ToolFailed("The password is incorrect", encrypted=True)

    monkeypatch.setattr(convert, "_office", locked)
    monkeypatch.setattr(convert, "_libreoffice", lambda *a, **k: pytest.fail("не пробувати далі"))
    with pytest.raises(ConversionError) as info:
        prepare_rendition(path, documents_dir=path.parent)
    assert info.value.code == "OFFICE_ENCRYPTED"
    assert "пароль" in info.value.hint


@pytest.mark.parametrize(("name", "needle"), [("abc.doc", "LibreOffice"),
                                              ("abc.djvu", "DjVuLibre")])
def test_missing_converter_explains_what_to_install(tmp_path, monkeypatch, name, needle) -> None:
    path = _doc(tmp_path, name)
    _tools(monkeypatch)
    with pytest.raises(ConversionError) as info:
        prepare_rendition(path, documents_dir=path.parent)
    assert info.value.code == "CONVERTER_MISSING"
    assert needle in info.value.hint


def test_docx_without_any_converter_is_read_natively(tmp_path, monkeypatch) -> None:
    """Без Office і LibreOffice DOCX усе одно читається — але сторінки чесно «≈»."""
    path = _doc(tmp_path, "abc.docx")
    _tools(monkeypatch)
    monkeypatch.setattr(convert, "_docling_markdown",
                        lambda src, kind: ("# Розділ 1\n\nТекст.\n", {}, True))
    rendition = prepare_rendition(path, documents_dir=path.parent)
    assert rendition.path == path.parent / "abc.view.md"
    assert rendition.approximate_pages and rendition.tool == "docling"
    assert any("приблизні" in w for w in rendition.warnings)
    assert view_file(path.parent, path) == (rendition.path, "text")


def test_converters_none_runs_no_program(tmp_path, monkeypatch) -> None:
    path = _doc(tmp_path, "abc.rtf")
    _tools(monkeypatch, word=True, libreoffice=True)
    monkeypatch.setattr(convert, "_office", lambda *a, **k: pytest.fail("Word заборонено"))
    with pytest.raises(ConversionError) as info:
        prepare_rendition(path, documents_dir=path.parent, converters="none")
    assert info.value.code == "CONVERTER_MISSING"


def test_stub_mode_never_starts_office(tmp_path, monkeypatch) -> None:
    path = _doc(tmp_path, "abc.docx")
    monkeypatch.setattr(convert.subprocess, "Popen", lambda *a, **k: pytest.fail("процес"))
    rendition = prepare_rendition(path, documents_dir=path.parent, stub=True)
    assert rendition.tool == "stub" and rendition.path == path


# --------------------------------------------------------------- таблиці
def test_spreadsheet_becomes_markdown_with_sheet_labels(tmp_path) -> None:
    openpyxl = pytest.importorskip("openpyxl")
    path = tmp_path / "docs" / "abc.xlsx"
    path.parent.mkdir()
    book = openpyxl.Workbook()
    book.active.title = "Норми"
    book.active.append(["Норматив", "Час"])
    book.active.append(["Розгортання", "90 с"])
    book.save(path)
    rendition = prepare_rendition(path, documents_dir=path.parent, stub=True)
    assert rendition.tool == "sheet"
    assert rendition.page_labels == {1: "арк. Норми"}
    assert "| Розгортання | 90 с |" in rendition.path.read_text(encoding="utf-8")


def test_ods_needs_excel_and_says_so(tmp_path, monkeypatch) -> None:
    path = _doc(tmp_path, "abc.ods")
    _tools(monkeypatch)
    with pytest.raises(ConversionError) as info:
        prepare_rendition(path, documents_dir=path.parent)
    assert info.value.code == "CONVERTER_MISSING"
    assert "XLSX" in info.value.hint


def test_empty_workbook_is_no_text(tmp_path) -> None:
    openpyxl = pytest.importorskip("openpyxl")
    path = tmp_path / "docs" / "abc.xlsx"
    path.parent.mkdir()
    openpyxl.Workbook().save(path)
    with pytest.raises(ConversionError) as info:
        prepare_rendition(path, documents_dir=path.parent)
    assert info.value.code == "NO_TEXT"


# ------------------------------------------------------------ прибирання
def test_remove_renditions_deletes_every_copy(tmp_path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    for suffix in (".view.pdf", ".view.md", ".view.json"):
        (docs / f"abc{suffix}").write_text("x", encoding="utf-8")
    remove_renditions(docs, "abc")
    assert not list(docs.iterdir())


def test_rendition_profile_names_the_program() -> None:
    rendition = Rendition(path=Path("a.view.pdf"), kind="word", tool="word")
    assert rendition.profile() == {"kind": "word", "tool": "word",
                                   "version": convert.RENDITION_VERSION}


# --------------------------------------------------------- сценарій Office
def test_office_script_quotes_paths_and_quits_word_by_ref(tmp_path) -> None:
    src = tmp_path / "прив'язка.docx"
    script = office_script("word", src, tmp_path / "out.pdf", tmp_path / "pid")
    # Одинарна лапка в шляху подвоюється — інакше рядок PowerShell обірвався б.
    assert "прив''язка.docx" in script
    # Без [ref] Word.Quit не викликається, і WINWORD.EXE лишається жити у фоні.
    assert "$app.Quit([ref]0)" in script
    assert "AutomationSecurity = 3" in script
    # Хибний пароль замість діалогу; не довше 15 символів — межа Excel.
    assert len(convert._PASSWORD) <= 15


def test_powerpoint_script_never_quits_a_running_powerpoint(tmp_path) -> None:
    script = office_script("powerpoint", tmp_path / "a.pptx", tmp_path / "a.pdf", tmp_path / "p")
    assert "if (-not $running) { try { $app.Quit() } catch {} }" in script


# ------------------------------------------------------ тайм-аут і скасування
def test_hung_program_is_killed_after_timeout() -> None:
    cmd = [sys.executable, "-c", "import time; time.sleep(30)"]
    with pytest.raises(convert._ToolFailed, match="не відповіла"):
        convert._run(cmd, timeout=0.6, cancel=None)


def test_cancel_stops_the_program() -> None:
    cmd = [sys.executable, "-c", "import time; time.sleep(30)"]
    with pytest.raises(ConversionCancelled):
        convert._run(cmd, timeout=30, cancel=lambda: True)


def test_finished_program_returns_its_output() -> None:
    result = convert._run([sys.executable, "-c", "print('ok')"], timeout=30, cancel=None)
    assert result.returncode == 0 and b"ok" in result.stdout
    assert isinstance(result, subprocess.CompletedProcess)


def test_frontend_accepts_exactly_what_the_backend_accepts() -> None:
    """Розширення, якого немає у фронтенді, екран мовчки відкидав би ще до
    відправлення — саме так DOCX «не завантажувався» до появи копій."""
    import re

    source = Path(__file__).resolve().parents[2] / "frontend" / "src" / "lib" / "formats.ts"
    if not source.is_file():
        pytest.skip("фронтенд не в цьому checkout")
    block = source.read_text(encoding="utf-8").split("ACCEPTED_EXTENSIONS = [", 1)[1]
    block = block.split("] as const", 1)[0]
    assert set(re.findall(r'"(\.[a-z0-9]+)"', block)) == set(ACCEPTED_SUFFIXES)
