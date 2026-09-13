"""Бенчмарк приймання: хв/стор., ConfidenceReport, пам'ять, guard TableFormer.

Тести працюють без Docling: `parse_document` має робочий текстовий шлях для
`.txt`/`.md`, тож увесь звіт будується наскрізь і в CI без моделей.
"""

from __future__ import annotations

import math

import pytest

from app.domain import IngestMode, QualityGrade
from scripts import bench_ingest as BI


class _Page:
    """Сторінка з оцінками — рівно ті поля, на які дивиться звіт."""

    def __init__(self, **scores):
        self.page_number = scores.pop("page_number", 1)
        self.parse_score = scores.get("parse_score")
        self.layout_score = scores.get("layout_score")
        self.table_score = scores.get("table_score")
        self.ocr_score = scores.get("ocr_score")


# --------------------------------------------------------- ConfidenceReport
def test_grade_is_taken_from_the_worst_score_not_the_mean():
    """Сторінка з поламаною таблицею — це погана сторінка, скільки б не було
    решти. Усереднення сховало б рівно те, заради чого робиться вимір."""
    pages = [_Page(parse_score=0.95, layout_score=0.95, table_score=0.30)]
    distribution = BI.grade_distribution(pages)
    assert distribution == {QualityGrade.POOR.value: 1}


def test_grade_distribution_counts_all_grades_and_unknown():
    pages = [
        _Page(parse_score=0.99, layout_score=0.99),
        _Page(parse_score=0.85),
        _Page(parse_score=0.60),
        _Page(),                       # оцінок немає зовсім
    ]
    distribution = BI.grade_distribution(pages)
    assert distribution[QualityGrade.EXCELLENT.value] == 1
    assert distribution[QualityGrade.GOOD.value] == 1
    assert distribution[QualityGrade.FAIR.value] == 1
    assert distribution["НЕВІДОМО"] == 1


# ---------------------------------------------------------------- пам'ять
def test_peak_memory_is_a_positive_number_of_megabytes():
    """Одиниці `ru_maxrss` різні на macOS (байти) і Linux (кілобайти) —
    помилка тут дає у звіті «12 ГБ» замість «12 МБ»."""
    value = BI.peak_memory_mb()
    assert math.isnan(value) or 1.0 < value < 100_000.0


# ------------------------------------------------------- guard TableFormer
def test_tableformer_guard_detection_is_honest_about_missing_docling():
    guard = BI.detect_tableformer_guard()
    assert isinstance(guard.docling_installed, bool)
    if not guard.docling_installed:
        assert guard.guard_present is None
        assert "docling" in guard.detail
    else:
        assert guard.version


# ------------------------------------------------------------------ прогін
def test_bench_document_measures_a_text_file(tmp_path):
    document = tmp_path / "методичка.txt"
    document.write_text(
        "Розділ 1. Балістика\n\n" + ("Деривація снаряда — це відхилення. " * 60),
        encoding="utf-8",
    )
    bench = BI.bench_document(document, IngestMode.FAST)

    assert not bench.error
    assert bench.pages >= 1
    assert bench.seconds >= 0.0
    assert bench.seconds_per_page >= 0.0
    assert bench.minutes_per_1000_pages == pytest.approx(bench.seconds_per_page * 1000 / 60)
    assert bench.mode == "FAST"


def test_bench_document_records_the_error_instead_of_crashing(tmp_path):
    from app.backends.stub_backend import is_stub_enabled

    broken = tmp_path / "битий.pdf"
    broken.write_bytes(b"not a pdf at all")
    bench = BI.bench_document(broken, IngestMode.FAST)

    if is_stub_enabled():
        # ASISTENT_STUB=1 підмінює парсер детермінованою заглушкою, і вона
        # ЗОБОВ'ЯЗАНА позначити свій вихід синтетичним — інакше бенчмарк у CI
        # рапортував би швидкість, якої не існує.
        assert not bench.error
        assert any("заглушк" in w.casefold() for w in bench.warnings)
    else:
        # Або парсер упорався (тоді сторінок нуль), або записав помилку — але не впав.
        assert bench.error or bench.pages == 0


def test_run_bench_builds_the_full_report(tmp_path):
    for name in ("а.txt", "б.txt"):
        (tmp_path / name).write_text(
            "Тема 1. Стрільба\n\n" + ("Поправка на деривацію береться з таблиць. " * 40),
            encoding="utf-8",
        )
    documents = sorted(tmp_path.glob("*.txt"))
    result = BI.run_bench(documents, [IngestMode.FAST], verbose=False)

    assert len(result.benches) == 2
    assert result.environment["platform"]
    rows = BI._mode_rows(result)
    assert rows[0]["режим"] == "FAST"
    assert rows[0]["документів"] == 2

    markdown = BI.render_markdown(result)
    assert "Швидкість за режимами" in markdown
    assert "TableFormer і Apple Silicon" in markdown
    assert "#3202" in markdown            # число з issue має бути у звіті

    csv_path, md_path = BI.write_outputs(result, tmp_path / "out")
    assert csv_path.exists() and md_path.exists()
    assert "с/стор." in csv_path.read_text(encoding="utf-8-sig")


def test_repeat_marks_the_warm_runs(tmp_path):
    (tmp_path / "а.txt").write_text("Текст сторінки. " * 50, encoding="utf-8")
    result = BI.run_bench(
        sorted(tmp_path.glob("*.txt")), [IngestMode.FAST], repeat=2, verbose=False
    )
    names = [b.document for b in result.benches]
    assert names[0] == "а.txt"
    assert "прогін 2" in names[1]


def test_both_modes_are_reported_separately(tmp_path):
    (tmp_path / "а.txt").write_text("Текст сторінки. " * 50, encoding="utf-8")
    result = BI.run_bench(
        sorted(tmp_path.glob("*.txt")), [IngestMode.FAST, IngestMode.DEEP], verbose=False
    )
    assert set(result.by_mode()) == {"FAST", "DEEP"}
    assert {row["режим"] for row in BI._mode_rows(result)} == {"FAST", "DEEP"}
