"""OCR-бейк-оф: CER, WER і українські плутанини (Віха 1).

Головне, що тут перевіряється, — детекція плутанини «і/i». Це та помилка, яку
CER майже не бачить, а пошук не переживає взагалі.
"""

from __future__ import annotations

import math

import pytest

from app.eval import ocr_metrics as O
from scripts import ocr_bakeoff as B


# ------------------------------------------------------- відстань і CER/WER
def test_edit_distance_basics():
    assert O.edit_distance("гаубиця", "гаубиця") == 0
    assert O.edit_distance("гаубиця", "гаубица") == 1
    assert O.edit_distance("", "три") == 3
    assert O.edit_distance(["а", "б"], ["а", "б", "в"]) == 1


def test_cer_and_wer_on_a_known_pair():
    reference = "Гаубиця Д-30 калібру 122 мм"
    hypothesis = "Гаубица Д-3O калібру 122 мм"     # 2 заміни, 5 слів
    assert O.cer(reference, hypothesis) == pytest.approx(2 / len(reference), abs=1e-9)
    assert O.wer(reference, hypothesis) == pytest.approx(2 / 5)


def test_cer_can_exceed_one_when_the_engine_invents_text():
    """Рушій, що видав удвічі більше сміття, має отримати CER > 1, а не «100%»."""
    assert O.cer("абв", "абв абв абв абв") > 1.0


def test_cer_of_empty_reference_is_nan_or_inf():
    assert math.isnan(O.cer("", ""))
    assert math.isinf(O.cer("", "щось"))


def test_normalization_keeps_case_but_drops_soft_hyphen():
    assert O.normalize_for_scoring("Гау­биця   Д-30\n") == "Гаубиця Д-30"
    assert O.normalize_for_scoring("ГАУБИЦЯ") == "ГАУБИЦЯ"


# ------------------------------------------------------------- плутанини
def test_latin_i_instead_of_cyrillic_is_detected():
    """і (U+0456) → i (U+0069): сторінка виглядає бездоганно, пошук мертвий."""
    reference = "поділ і межі"
    hypothesis = "подiл i межi"
    classes, raw = O.confusion_counts(reference, hypothesis)

    assert classes["і/i"] == 3
    assert raw[("і", "i")] == 3
    # CER при цьому мізерний — саме тому потрібен окремий лічильник.
    assert O.cer(reference, hypothesis) < 0.3


def test_uppercase_i_confusion_is_detected():
    classes, _ = O.confusion_counts("Інструкція", "Iнструкція")
    assert classes["і/i"] == 1


def test_yi_ye_ghe_and_y_confusions():
    assert O.confusion_counts("їжа", "iжа")[0]["ї/i"] == 1
    assert O.confusion_counts("є", "e")[0]["є/e"] == 1
    assert O.confusion_counts("ґанок", "ганок")[0]["ґ/г"] == 1
    assert O.confusion_counts("бий", "бии")[0]["и/й"] == 1


def test_apostrophe_loss_is_counted_separately():
    """FTS5 тримає апостроф усередині слова: «пять» і «п'ять» — різні токени."""
    classes, _ = O.confusion_counts("п'ять снарядів", "пять снарядів")
    assert classes["апостроф"] == 1


def test_apostrophe_substitution_is_counted_too():
    classes, _ = O.confusion_counts("п'ять", "п’ять")
    assert classes["апостроф"] == 1


def test_homoglyph_class_catches_cyrillic_to_latin():
    classes, _ = O.confusion_counts("сорок", "copoк")
    assert classes["гомогліфи"] == 4      # с→c, о→o, р→p, о→o


def test_soft_cer_shows_the_price_of_the_apostrophe():
    score = O.score_page("p1", "рушій", "п'ять об'єктів", "пять обєктів")
    assert score.cer > score.cer_soft
    assert score.apostrophe_cost > 0


def test_align_pairs_reports_only_the_differences():
    pairs = O.align_pairs("абвгд", "абвгд")
    assert pairs == []
    pairs = O.align_pairs("абв", "абг")
    assert pairs == [("в", "г")]


def test_alignment_handles_insertions_and_deletions():
    assert ("", "х") in O.align_pairs("аб", "ахб")
    assert ("х", "") in O.align_pairs("ахб", "аб")


# ------------------------------------------------------------------ звіти
def test_aggregate_reports_mean_and_median_and_sorts_by_cer():
    scores = [
        O.score_page("p1", "добрий", "текст сторінки", "текст сторінки"),
        O.score_page("p2", "добрий", "текст сторінки", "текст сторiнки"),
        O.score_page("p3", "поганий", "текст сторінки", "щось зовсім інше тут"),
    ]
    reports = O.aggregate(scores)
    assert [r.engine for r in reports] == ["добрий", "поганий"]
    good = reports[0]
    assert good.pages == 2
    assert good.median_cer <= good.mean_cer or good.mean_cer == pytest.approx(good.median_cer)
    assert good.confusions["і/i"] == 1


def test_aggregate_excludes_failed_pages_but_counts_them():
    scores = [
        O.score_page("p1", "рушій", "текст", "текст"),
        O.score_page("p2", "рушій", "текст", "", error="RuntimeError: впав"),
    ]
    report = O.aggregate(scores)[0]
    assert report.pages == 1
    assert report.failures == 1


def test_report_rows_have_identical_columns():
    scores = [
        O.score_page("p1", "a", "поділ і межі", "подiл i межi"),
        O.score_page("p1", "b", "поділ і межі", "поділ і межі"),
    ]
    rows = O.report_rows(O.aggregate(scores))
    assert {frozenset(r) for r in rows} == {frozenset(rows[0])}


# ------------------------------------------------------------ скрипт бейк-офу
def test_collect_pairs_matches_by_stem_and_warns_about_orphans(tmp_path):
    pages = tmp_path / "pages"
    gold = tmp_path / "gold"
    pages.mkdir()
    gold.mkdir()
    (pages / "p001.png").write_bytes(b"fake")
    (pages / "p002.png").write_bytes(b"fake")
    (gold / "p001.txt").write_text("Текст сторінки", encoding="utf-8")
    (gold / "p003.txt").write_text("Осиротіла транскрипція", encoding="utf-8")

    pairs, warnings = B.collect_pairs(pages, gold)
    assert [p.name for p in pairs] == ["p001"]
    assert any("p002" in w for w in warnings)
    assert any("p003" in w for w in warnings)


def test_collect_pairs_skips_empty_transcripts(tmp_path):
    pages = tmp_path / "pages"
    gold = tmp_path / "gold"
    pages.mkdir()
    gold.mkdir()
    (pages / "p1.png").write_bytes(b"fake")
    (gold / "p1.txt").write_text("   \n", encoding="utf-8")
    pairs, warnings = B.collect_pairs(pages, gold)
    assert pairs == []
    assert any("порожня" in w for w in warnings)


class _FakeEngine:
    """Рушій, що псує кириличні «і» — рівно та поломка, яку шукає Віха 1."""

    key = "фейк-латинська-і"
    title = "Фейковий рушій"

    def status(self) -> B.EngineStatus:
        return B.EngineStatus(True, "тестовий")

    def recognize(self, image) -> str:
        return image.read_text(encoding="utf-8").replace("і", "i")


class _BrokenEngine:
    key = "фейк-битий"
    title = "Битий рушій"

    def status(self) -> B.EngineStatus:
        return B.EngineStatus(True, "тестовий")

    def recognize(self, image) -> str:
        raise RuntimeError("рушій упав на сторінці")


def test_run_bakeoff_end_to_end(tmp_path, capsys):
    pages = tmp_path / "pages"
    gold = tmp_path / "gold"
    pages.mkdir()
    gold.mkdir()
    text = "Поділ і межі відповідальності артилерійського підрозділу"
    (pages / "p1.png").write_text(text, encoding="utf-8")
    (gold / "p1.txt").write_text(text, encoding="utf-8")

    pairs, _ = B.collect_pairs(pages, gold)
    engines = [
        (_FakeEngine(), B.EngineStatus(True, "тестовий")),
        (_BrokenEngine(), B.EngineStatus(True, "тестовий")),
        (B.TesseractEngine(lang="неіснуюча"), B.EngineStatus(False, "мовний пакет відсутній")),
    ]
    result = B.run_bakeoff(pairs, engines, verbose=False)

    by_engine = {r.engine: r for r in result.reports}
    assert by_engine["фейк-латинська-і"].confusions["і/i"] >= 3
    assert by_engine["фейк-битий"].failures == 1
    assert result.statuses["tesseract-ukr"].available is False

    markdown = B.render_markdown(result)
    assert "Українські плутанини" in markdown
    # Пропущений рушій лишається у звіті з причиною.
    assert "Пропущені рушії" in markdown and "tesseract-ukr" in markdown

    per_page, summary = B.write_csv(result, tmp_path / "out")
    assert per_page.exists() and summary.exists()
    assert "conf:і/i" in per_page.read_text(encoding="utf-8-sig")


def test_discover_engines_reports_availability_honestly():
    engines = B.discover_engines(["tesseract-ukr", "rapidocr-eslav", "apple-vision"])
    assert [e.key for e, _ in engines] == ["tesseract-ukr", "rapidocr-eslav", "apple-vision"]
    for engine, status in engines:
        assert isinstance(status.available, bool)
        if not status.available:
            assert status.detail, f"{engine.key} мусить пояснити, чому недоступний"


def test_unknown_engine_fails_loudly():
    with pytest.raises(SystemExit, match="Невідомий рушій"):
        B.discover_engines(["чогось-такого-немає"])


def test_rapidocr_engine_keys_differ_by_language():
    eslav = B.RapidOcrEngine("eslav")
    cyrillic = B.RapidOcrEngine("cyrillic")
    assert eslav.key != cyrillic.key
    assert "eslav" in eslav.title
