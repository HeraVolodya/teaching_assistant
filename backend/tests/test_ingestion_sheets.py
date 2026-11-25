"""Таблиці Excel / CSV → Markdown-копія: аркуш = «сторінка», блоки з заголовком."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from app.ingestion.docling_pipeline import ElementKind, ParseOptions, parse_document
from app.ingestion.probe import probe_text
from app.ingestion.sheets import (
    BLOCK_CHARS,
    Sheet,
    cell_text,
    read_workbook,
    render_markdown,
    spreadsheet_to_markdown,
)

openpyxl = pytest.importorskip("openpyxl")


def _workbook(path: Path) -> Path:
    book = openpyxl.Workbook()
    ws = book.active
    ws.title = "Норми"
    ws["A1"] = "Таблиця 1. Нормативи розгортання"
    ws.merge_cells("A1:D1")
    ws.append(["Гармата", "Норматив", "Відмінно", "Дата"])
    ws.append(["Д-30", "Розгортання", 90, dt.date(2026, 11, 3)])
    ws.append(["Д-30", "Згортання", 1.5, dt.datetime(2026, 11, 4, 9, 30)])
    ws["B5"] = "Частка влучань"
    ws["C5"] = 0.25
    ws["C5"].number_format = "0%"
    ws["A5"] = "2С1"
    ws.merge_cells("A5:A6")
    ws["B6"] = "Вогонь | прямою наводкою"
    ws["C6"] = True
    book.create_sheet("Порожній")
    extra = book.create_sheet("Список")
    for name in ("Бусоль", "Далекомір", "Компас"):
        extra.append([name])
    out = path / "норми.xlsx"
    book.save(out)
    return out


def test_cell_text_matches_what_excel_shows() -> None:
    assert cell_text(None) == ""
    assert cell_text(15300.0) == "15300"
    assert cell_text(1.5) == "1,5"            # десяткова кома, як в українському Excel
    assert cell_text(0.25, "0%") == "25 %"
    assert cell_text(dt.date(2026, 1, 9)) == "09.01.2026"
    assert cell_text(dt.datetime(2026, 1, 9, 7, 5)) == "09.01.2026 07:05"
    assert cell_text(True) == "так"
    assert cell_text("  два\nрядки ") == "два рядки"


def test_workbook_sheets_become_pages_with_sheet_labels(tmp_path) -> None:
    result = spreadsheet_to_markdown(_workbook(tmp_path))
    # Порожній аркуш — не сторінка: цитувати там нічого.
    assert result.page_labels == {1: "арк. Норми", 2: "арк. Список"}
    assert result.markdown.count("\f") == 2
    assert result.markdown.endswith("\n\f\n")


def test_merged_cells_are_spread_and_title_row_is_not_a_header(tmp_path) -> None:
    markdown = spreadsheet_to_markdown(_workbook(tmp_path)).markdown
    lines = markdown.splitlines()
    # Злитий рядок-назва — абзац над таблицею, а не рядок даних і не заголовок.
    assert "Таблиця 1. Нормативи розгортання" in lines
    assert "| Гармата | Норматив | Відмінно | Дата |" in lines
    # Вертикально злита «2С1» дістається обом рядкам, «|» у клітинці екрановано.
    assert "| 2С1 | Частка влучань | 25 % |  |" in lines
    assert "| 2С1 | Вогонь ¦ прямою наводкою | так |  |" in lines
    assert "| Д-30 | Розгортання | 90 | 03.11.2026 |" in lines
    assert "Аркуш «Норми», рядки 3–6." in lines


def test_single_column_sheet_is_text_not_a_table(tmp_path) -> None:
    markdown = spreadsheet_to_markdown(_workbook(tmp_path)).markdown
    page = markdown.split("\f")[1]
    assert "Бусоль\nДалекомір\nКомпас" in page
    assert "|" not in page


def test_long_sheet_is_cut_into_blocks_with_repeated_header() -> None:
    rows = [(1, ["№", "Ціль", "Дальність, м", "Примітка"])]
    rows += [(n, [str(n - 1), f"Ціль {n - 1}", str(1000 + n), "спостерігається з НП"])
             for n in range(2, 202)]
    markdown = render_markdown([Sheet(name="Цілі", rows=rows)]).markdown
    headers = markdown.count("| № | Ціль | Дальність, м | Примітка |")
    assert headers >= 5                                   # блоків багато
    assert "Аркуш «Цілі», рядки 2–" in markdown
    assert markdown.count("Аркуш «Цілі», ряд") == headers  # кожен блок підписано


def test_blocks_fit_the_leaf_budget_after_parsing(tmp_path) -> None:
    """Таблиця атомарна для чанкера; блок понад бюджет дав би гігантський фрагмент."""
    rows = [(1, ["Ціль", "Опис"])]
    rows += [(n, [f"Ціль {n}", "довгий опис цілі " * 6]) for n in range(2, 120)]
    path = tmp_path / "цілі.view.md"
    path.write_text(render_markdown([Sheet(name="Цілі", rows=rows)]).markdown, encoding="utf-8")
    parsed = parse_document(path, ParseOptions())
    tables = [e for e in parsed.elements if e.kind is ElementKind.TABLE]
    assert len(tables) > 3
    assert all(len(t.text) <= BLOCK_CHARS + 200 for t in tables)
    # Підпис блоку — частина самої таблиці, тож рядки не губляться при чанкуванні.
    assert all(t.text.startswith("Аркуш «Цілі», ряд") for t in tables)
    assert all("Опис = довгий опис" in t.text for t in tables)


def test_sheet_pages_line_up_with_the_text(tmp_path) -> None:
    """Розрив U+000C теж займає позицію: без цього третій аркуш «з'їжджав» на четвертий."""
    sheets = [Sheet(name=f"А{i}", rows=[(1, ["Поле", "Значення"]), (2, [f"x{i}", str(i)])])
              for i in range(1, 4)]
    markdown = render_markdown(sheets).markdown
    report = probe_text(markdown)
    assert report.page_count == 3
    path = tmp_path / "книга.view.md"
    path.write_text(markdown, encoding="utf-8")
    parsed = parse_document(path, ParseOptions())
    headings = [(e.text, e.page_no) for e in parsed.elements if e.kind is ElementKind.HEADING]
    assert headings == [("Аркуш «А1»", 1), ("Аркуш «А2»", 2), ("Аркуш «А3»", 3)]
    for page in report.pages:
        assert markdown[page.char_from:page.char_to].lstrip().startswith("# Аркуш")


def test_csv_in_cp1251_with_semicolons(tmp_path) -> None:
    path = tmp_path / "журнал.csv"
    path.write_bytes("Прізвище;Оцінка\nПетренко;5\nКоваль;4\n".encode("cp1251"))
    sheets, _ = read_workbook(path, csv_name="Журнал")
    assert sheets[0].name == "Журнал"
    assert sheets[0].rows == [(1, ["Прізвище", "Оцінка"]), (2, ["Петренко", "5"]),
                              (3, ["Коваль", "4"])]


def test_unsupported_suffix_is_refused(tmp_path) -> None:
    path = tmp_path / "таблиця.ods"
    path.write_bytes(b"PK")
    with pytest.raises(ValueError, match="XLSX"):
        read_workbook(path)
