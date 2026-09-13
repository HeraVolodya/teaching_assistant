"""Правдивість таблиць (Віха 2): поклітинний F1 і діагностика структури."""

from __future__ import annotations

import math

import pytest

from app.eval import table_metrics as T
from scripts import table_fidelity as F

GOLD = [
    ["Дальність, м", "Кут підвищення", "Поправка на деривацію"],
    ["1000", "0-12", "0,002"],
    ["2000", "0-25", "0,005"],
]


# --------------------------------------------------------------- нормалізація
def test_normalize_cell_unifies_apostrophes_and_dashes_but_not_decimals():
    assert T.normalize_cell("п’ять") == T.normalize_cell("п'ять")
    assert T.normalize_cell("0–12") == T.normalize_cell("0-12")
    # Кома як десятковий роздільник — значуща: «15,3» ≠ «15.3».
    assert T.normalize_cell("15,3") != T.normalize_cell("15.3")


# ---------------------------------------------------------------------- F1
def test_perfect_extraction_scores_one():
    p, r, f1 = T.cell_f1(GOLD, GOLD)
    assert (p, r, f1) == (1.0, 1.0, 1.0)


def test_single_wrong_cell_lowers_both_precision_and_recall():
    predicted = [row[:] for row in GOLD]
    predicted[1][2] = "0,020"
    p, r, f1 = T.cell_f1(GOLD, predicted)
    assert p == pytest.approx(8 / 9)
    assert r == pytest.approx(8 / 9)
    assert f1 == pytest.approx(8 / 9)


def test_scrambled_structure_keeps_content_f1_but_kills_positional():
    """Це і є діагноз: значення прочитані, структура поламана."""
    transposed = [list(column) for column in zip(*GOLD, strict=True)]
    score = T.score_table("таблиця стрільби", "v2", GOLD, transposed)
    assert score.positional_f1 < 0.4
    assert score.content_f1 == pytest.approx(1.0)
    assert score.structure_loss > 0.5


def test_empty_cells_do_not_inflate_the_score():
    """Порожня сітка правильного розміру не має отримати жодного бала."""
    empty = [["" for _ in row] for row in GOLD]
    p, r, f1 = T.cell_f1(GOLD, empty)
    assert f1 == 0.0
    assert math.isnan(p) or p == 0.0
    assert r == 0.0


def test_empty_ratio_triggers_the_same_threshold_as_the_runtime():
    half_empty = [GOLD[0], ["1000", "", ""], ["", "", ""]]
    score = T.score_table("t", "v1", GOLD, half_empty)
    assert score.empty_ratio > T.EMPTY_CELL_REPAIR_THRESHOLD
    assert score.needs_vlm_repair


def test_shape_mismatch_is_reported():
    score = T.score_table("t", "v1", GOLD, GOLD[:2])
    assert not score.shape_ok
    assert score.pred_shape == (2, 3)


# ------------------------------------------------------------------ розбір
def test_markdown_separator_row_is_not_data():
    """Рядок `|---|` як дані зсунув би всі координати на одиницю."""
    markdown = (
        "| Дальність, м | Кут |\n"
        "|---|---|\n"
        "| 1000 | 0-12 |\n"
    )
    grid = T.grid_from_markdown(markdown)
    assert grid == [["Дальність, м", "Кут"], ["1000", "0-12"]]


def test_html_colspan_is_expanded_so_columns_do_not_shift():
    """Об'єднаний заголовок — типова балістична таблиця."""
    html = (
        "<table><tr><th colspan='2'>Дальність, м</th><th>Кут</th></tr>"
        "<tr><td>1000</td><td>1200</td><td>0-12</td></tr></table>"
    )
    grid = T.grid_from_html(html)
    assert grid[0] == ["Дальність, м", "Дальність, м", "Кут"]
    assert grid[1] == ["1000", "1200", "0-12"]


def test_html_rowspan_is_carried_to_the_next_row():
    html = (
        "<table><tr><td rowspan='2'>Заряд повний</td><td>1000</td></tr>"
        "<tr><td>2000</td></tr></table>"
    )
    grid = T.grid_from_html(html)
    assert grid[0][0] == "Заряд повний"
    assert "2000" in grid[1]


def test_csv_gold_is_loaded_with_bom(tmp_path):
    path = tmp_path / "t.csv"
    path.write_text("﻿Дальність,Кут\n1000,0-12\n", encoding="utf-8")
    assert T.load_grid_csv(path) == [["Дальність", "Кут"], ["1000", "0-12"]]


# ------------------------------------------------------------------ скрипт
def _write_gold(tmp_path):
    """Еталон пишеться `csv.writer`, а не join'ом.

    Це не педантизм: у балістичній таблиці кома є і в заголовку («Дальність, м»),
    і в кожному числі («0,002»). Наївний `",".join` перетворив би таблицю 3×3 на
    4×4 з розсипаними значеннями — і саме такий еталон найлегше не помітити.
    """
    import csv

    gold = tmp_path / "gold"
    gold.mkdir()
    with (gold / "tab1.csv").open("w", encoding="utf-8", newline="") as fh:
        csv.writer(fh).writerows(GOLD)
    return gold


def test_collect_cases_reads_gold_and_warns_about_missing_pages(tmp_path):
    gold = _write_gold(tmp_path)
    tables = tmp_path / "pages"
    tables.mkdir()
    cases, warnings = F.collect_cases(gold, tables)
    assert [c.name for c in cases] == ["tab1"]
    assert cases[0].cells == 9
    assert any("tab1" in w for w in warnings)


def test_score_predictions_treats_a_missing_table_as_a_miss(tmp_path):
    gold = _write_gold(tmp_path)
    cases, _ = F.collect_cases(gold)
    scores = F.score_predictions(cases, {"v2": {}})
    assert scores[0].error == "таблицю не знайдено"
    assert scores[0].positional_f1 == 0.0


def test_predictions_mode_compares_two_engines_end_to_end(tmp_path):
    gold = _write_gold(tmp_path)
    good = tmp_path / "v1-accurate"
    bad = tmp_path / "v2-accurate"
    good.mkdir()
    bad.mkdir()
    (good / "tab1.md").write_text(
        "| Дальність, м | Кут підвищення | Поправка на деривацію |\n"
        "|---|---|---|\n"
        "| 1000 | 0-12 | 0,002 |\n"
        "| 2000 | 0-25 | 0,005 |\n",
        encoding="utf-8",
    )
    (bad / "tab1.html").write_text(
        "<table><tr><td>Дальність, м</td><td>Кут підвищення</td></tr>"
        "<tr><td>1000</td><td>0-12</td></tr></table>",
        encoding="utf-8",
    )

    cases, _ = F.collect_cases(gold)
    predictions = {name: F.load_predictions(tmp_path / name) for name in ("v1-accurate", "v2-accurate")}
    scores = F.score_predictions(cases, predictions)
    by_mode = {s.engine: s for s in scores}

    assert by_mode["v1-accurate"].positional_f1 == pytest.approx(1.0)
    assert by_mode["v2-accurate"].positional_f1 < 1.0
    assert by_mode["v2-accurate"].positional_precision == pytest.approx(1.0)

    result = F.FidelityResult(scores=scores)
    markdown = F.render_markdown(result)
    assert "Зведення по режимах" in markdown
    assert "v1-accurate" in markdown

    path = F.write_csv(result, tmp_path / "out")
    assert path.exists() and "F1 позиц." in path.read_text(encoding="utf-8-sig")


def test_docling_availability_is_reported_honestly():
    ok, detail = F.docling_available()
    assert isinstance(ok, bool)
    assert detail
    if not ok:
        assert "docling" in detail


def test_load_grid_rejects_unknown_format(tmp_path):
    path = tmp_path / "t.xlsx"
    path.write_bytes(b"PK")
    with pytest.raises(ValueError, match="Невідомий формат"):
        F.load_grid(path)
