"""Гейти тріажу рівня 0.

Головна відмова, заради якої цей модуль існує: PDF, чия `ToUnicode` CMap
мапить кириличні гліфи на ВАЛІДНІ, але хибні кодпойнти. `parse_score` Docling
там дорівнює 1.0 при цілковито нечитабельному тексті — ловить це лише
словникова перевірка.
"""

from __future__ import annotations

import pytest

from app.domain import OcrModeName, PageClass
from app.ingestion.probe import (
    ProbeReport,
    classify_page,
    cost_weight,
    looks_like_formula,
    looks_like_table,
    measure_page,
    mojibake_ratio,
    probe_text,
    window_boundaries,
)
from app.ingestion.uk_lexicon import is_known_word, lexicon_hit_rate

GOOD = (
    "Гармата призначена для ураження живої сили та вогневих засобів противника. "
    "Розрахунок визначає установки для стрільби за таблицями стрільби, "
    "враховуючи метеорологічні та балістичні умови. "
)
# Кириличні літери, законні кодпойнти — але слів такого вигляду не існує.
BROKEN_CMAP = "Ерсдп ыиуцн жщхїб фєґкв ъэяю шчьтм пролджэ ыуцкен гшщзхї фывапр олджэя чсмить бю "


def test_broken_tounicode_cmap_is_detected_by_lexicon() -> None:
    """Саме та відмова, якої Docling НЕ ловить."""
    page = probe_text(BROKEN_CMAP * 4).pages[0]
    assert page.cyrillic_ratio > 0.9
    assert page.mojibake_ratio == 0.0, "mojibake тут немає — текст «валідний»"
    assert page.lexicon_hit_rate is not None and page.lexicon_hit_rate < 0.45
    assert page.page_class is PageClass.DIGITAL_BROKEN
    assert page.ocr_mode is OcrModeName.FULL_PAGE


def test_healthy_ukrainian_page_is_left_alone() -> None:
    page = probe_text(GOOD * 4).pages[0]
    assert page.page_class is PageClass.DIGITAL_CLEAN
    assert page.lexicon_hit_rate is not None and page.lexicon_hit_rate >= 0.45


def test_english_page_is_not_flagged_as_broken() -> None:
    """cyrillic_ratio <= 0.3 знімає словникову перевірку — англійські джерела в корпусі є."""
    text = "The projectile trajectory depends on the elevation angle and muzzle velocity. " * 6
    page = probe_text(text).pages[0]
    assert page.page_class is PageClass.DIGITAL_CLEAN


def test_lexicon_hit_rate_is_none_on_thin_evidence() -> None:
    """На майже порожній сторінці статистика недостовірна — краще None, ніж хибний OCR."""
    assert lexicon_hit_rate(["гармата", "снаряд"]) is None


def test_lexicon_matches_by_stem() -> None:
    assert is_known_word("гарматний")
    assert is_known_word("траєкторії")
    assert not is_known_word("ыиуцн")


# ------------------------------------------------------------------- mojibake
@pytest.mark.parametrize(
    "text",
    ["текст ���", "/glyph<C0>/glyph<C1>", "/g12/g13/g14"],
)
def test_mojibake_markers_are_counted(text: str) -> None:
    assert mojibake_ratio(text) > 0.02


def test_clean_text_has_zero_mojibake() -> None:
    assert mojibake_ratio(GOOD) == 0.0


def test_mojibake_page_goes_to_full_page_ocr() -> None:
    page = measure_page("/glyph<C0>" * 40 + GOOD, page_number=1)
    assert page.page_class is PageClass.DIGITAL_BROKEN
    assert page.ocr_mode is OcrModeName.FULL_PAGE


# --------------------------------------------------------------- класифікація
def test_scanned_page_has_no_text_but_has_raster() -> None:
    page = measure_page("", page_number=1, bitmap_coverage=0.85, picture_count=1)
    assert page.page_class is PageClass.SCANNED
    assert page.ocr_mode is OcrModeName.FULL_PAGE


def test_blank_page_is_never_sent_to_ocr() -> None:
    """OCR порожньої сторінки — чиста витрата хвилин на тисячосторінковому підручнику."""
    page = measure_page("   \n\n", page_number=1, bitmap_coverage=0.0)
    assert page.ocr_mode is OcrModeName.NONE


def test_digital_page_with_scanned_inserts_uses_region_level_ocr() -> None:
    page = measure_page(GOOD * 3, page_number=1, bitmap_coverage=0.4)
    assert page.page_class is PageClass.MIXED
    assert page.ocr_mode is OcrModeName.DEFAULT


# ------------------------------------------------------------------- вартість
def test_cost_weight_formula_matches_the_plan() -> None:
    """w = 1 + 3·скан + 2·таблиця + 1·формула + 0.5·рисунки."""
    page = measure_page("", page_number=1, bitmap_coverage=0.9)
    page.has_table = True
    page.has_formula = True
    page.picture_count = 2
    assert cost_weight(page) == pytest.approx(1 + 3 + 2 + 1 + 1.0)


def test_picture_count_is_capped() -> None:
    """Сторінка з 200 декоративними гліфами не коштує у сто разів більше."""
    page = measure_page(GOOD, page_number=1)
    page.picture_count = 200
    assert cost_weight(page) <= 1 + 2 + 1 + 3.0


def test_progress_denominator_is_weighted_not_linear() -> None:
    report = probe_text(GOOD * 3 + "\f" + BROKEN_CMAP * 4)
    assert report.page_count == 2
    assert report.total_weight > report.page_count


def test_page_info_conversion_carries_triage_to_the_database() -> None:
    page = measure_page(BROKEN_CMAP * 4, page_number=7)
    info = page.to_page_info(page_label="2-7")
    assert info.page_number == 7
    assert info.page_label == "2-7"
    assert info.page_class is PageClass.DIGITAL_BROKEN
    assert info.lexicon_hit_rate is not None


# ----------------------------------------------------------------- евристики
def test_table_and_formula_hints() -> None:
    assert looks_like_table("Таблиця 3.1 — установки для стрільби")
    assert looks_like_table("а  1  2\nб  3  4\nв  5  6\nг  7  8")
    assert looks_like_formula("V = V₀ · cos α ± Δ")
    assert not looks_like_formula("Звичайний абзац без математики.")


# ---------------------------------------------------------------------- вікна
def test_windows_avoid_ending_inside_a_table() -> None:
    """Таблиця, розірвана межею вікна, губить структуру при зчепленні."""
    report = ProbeReport(page_count=50)
    for number in range(1, 51):
        page = measure_page(GOOD, page_number=number)
        page.has_table = number in (24, 25, 26)
        report.pages.append(page)
    windows = window_boundaries(report, window=25)
    assert windows[0][1] == 23
    assert windows[0] == (1, 23) and windows[1][0] == 24


def test_windows_cover_every_page_exactly_once() -> None:
    report = probe_text("сторінка\f" * 60)
    covered = [p for lo, hi in window_boundaries(report, window=25) for p in range(lo, hi + 1)]
    assert covered == list(range(1, report.page_count + 1))


def test_probe_report_is_empty_but_valid_for_zero_pages() -> None:
    assert window_boundaries(ProbeReport(page_count=0)) == []
    assert ProbeReport().dominant_class is PageClass.DIGITAL_CLEAN


def test_classify_is_a_pure_function_of_metrics() -> None:
    page = measure_page(GOOD * 3, page_number=1)
    assert classify_page(page) == (PageClass.DIGITAL_CLEAN, OcrModeName.DEFAULT)


# --------------------------------------------------------------- реальний PDF
def _scanned_pdf(pdfium, path) -> str:
    """PDF з одним великим растром — імітація сканованої сторінки підручника."""
    pdf = pdfium.PdfDocument.new()
    page = pdf.new_page(595, 842)
    bitmap = pdfium.PdfBitmap.new_native(400, 600, pdfium.raw.FPDFBitmap_BGRA)
    bitmap.fill_rect((200, 200, 200, 255), 0, 0, 400, 600)
    image = pdfium.PdfImage.new(pdf)
    image.set_bitmap(bitmap)
    image.set_matrix(pdfium.PdfMatrix().scale(500, 700).translate(40, 60))
    page.insert_obj(image)
    page.gen_content()
    pdf.save(str(path))
    return str(path)


def test_bitmap_coverage_is_actually_measured_on_a_real_pdf(tmp_path) -> None:
    """Регресія на тиху відмову.

    pypdfium2 перейменував `PdfObject.get_pos()` на `get_bounds()` у 5.x. Якщо
    проба це проковтне, `bitmap_coverage` буде завжди 0, КОЖНА сканована
    сторінка стане DIGITAL_CLEAN і жоден сканований підручник не отримає OCR —
    без жодного повідомлення про помилку.
    """
    # pypdfium2 живе лише у групі залежностей `worker` — на «голому» CI його немає.
    pdfium = pytest.importorskip("pypdfium2")
    from app.ingestion.probe import probe_pdf

    report = probe_pdf(_scanned_pdf(pdfium, tmp_path / "scan.pdf"))
    assert report.warnings == []
    page = report.pages[0]
    assert page.picture_count == 1
    assert page.bitmap_coverage > 0.5, "покриття растром не виміряно — див. _object_bounds"
    assert page.page_class is PageClass.SCANNED
    assert page.ocr_mode is OcrModeName.FULL_PAGE
