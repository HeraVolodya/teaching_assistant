"""Таблиця Excel.

Аркуш «Таблиця» — таблиці з відповіді як справжні «Таблиці Excel»: стиль зі
смугами, фільтр і сортування в шапці, шапка закріплена при прокручуванні й
повторюється при друці, перенесення тексту. Аркуш «Джерела» — що означає кожен [n].

Комірка, у якій ЛИШЕ маркер («[3]» у стовпці «Джерело»), розгортається в
«[3] Бойовий статут…, с. 15»: у таблиці, яку відкрили окремо від чату, голий
номер нічого не каже. Числа стають числами («122», «15,3»), щоб працювали
сортування й формули; рядок з одиницями («122 мм») лишається текстом.

Відповідь без таблиць (скажімо, конспект, який попросили «в Excel») стає
аркушем «Зміст» — по рядку на абзац чи пункт.
"""

from __future__ import annotations

import io
import re
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from app.export.outline import Block, Outline, markers, plain, strip_markers

if TYPE_CHECKING:
    from app.export.render import SourceRef

__all__ = ["render_xlsx"]

_MARKERS_ONLY = re.compile(r"(?:\[\d{1,3}\][\s,;]*)+")
_NUMBER = re.compile(r"[-−]?\d{1,3}(?:[   ]?\d{3})*(?:[.,]\d+)?|[-−]?\d+(?:[.,]\d+)?")
_EXCEL_TEXT_LIMIT = 32_000


def render_xlsx(
    outline: Outline,
    sources: Mapping[int, SourceRef],
    cited: Sequence[SourceRef],
    *,
    note: str,
) -> tuple[bytes, dict[str, int]]:
    from openpyxl import Workbook

    workbook = Workbook()
    sheet = workbook.active
    tables = [b for b in outline.blocks if b.kind == "table" and b.header]
    if tables:
        sheet.title = "Таблиця"
        rows = _tables_sheet(sheet, outline, tables, sources)
    else:
        sheet.title = "Зміст"
        rows = _text_sheet(sheet, outline)
    _sources_sheet(workbook.create_sheet("Джерела"), cited, note)

    workbook.properties.title = outline.title
    workbook.properties.creator = workbook.properties.lastModifiedBy = "Асистент викладача"
    workbook.properties.language = "uk-UA"

    buffer = io.BytesIO()
    workbook.save(buffer)
    return buffer.getvalue(), {"rows": rows, "tables": len(tables)} if tables else {"rows": rows}


def _styles() -> dict[str, Any]:
    from openpyxl.styles import Alignment, Font

    return {
        "title": Font(bold=True, size=14),
        "subtitle": Font(bold=True, size=12),
        "head_align": Alignment(wrap_text=True, vertical="center", horizontal="center"),
        "cell_align": Alignment(wrap_text=True, vertical="top"),
    }


def _tables_sheet(sheet: Any, outline: Outline, tables: list[Block],
                  sources: Mapping[int, SourceRef]) -> int:
    """Кожна таблиця — справжня «Таблиця Excel» (Вставлення → Таблиця): стиль
    зі смугами, кнопки фільтра й сортування в шапці, автоматичне розширення
    при дописуванні рядків. Колір шапки й смуг дає стиль, а не ручна заливка,
    тож викладач змінює оформлення одним кліком у «Конструкторі таблиць»."""
    from openpyxl.utils import get_column_letter

    style = _styles()
    sheet.cell(1, 1, outline.title).font = style["title"]
    captions = _captions(outline)
    widths: dict[int, int] = {}
    row = 3
    total = 0
    max_cols = 0
    first_header: int | None = None
    for index, block in enumerate(tables, start=1):
        if len(tables) > 1:
            sheet.cell(row, 1, captions.get(id(block)) or f"Таблиця {index}").font = style["subtitle"]
            row += 1
        header_row = row
        first_header = first_header or header_row
        for col, text in enumerate(_unique(block.header), start=1):
            cell = sheet.cell(row, col, text)
            cell.alignment = style["head_align"]
            widths[col] = max(widths.get(col, 0), len(text))
        row += 1
        for values in block.rows or [[]]:
            for col in range(1, len(block.header) + 1):
                raw = values[col - 1] if col - 1 < len(values) else ""
                cell = sheet.cell(row, col, _value(raw, sources))
                cell.alignment = style["cell_align"]
                widths[col] = max(widths.get(col, 0), len(str(cell.value or "")))
            row += 1
        total += len(block.rows)
        max_cols = max(max_cols, len(block.header))
        _excel_table(sheet, f"Table{index}", header_row, len(block.header), row - 1, "TableStyleMedium4")
        row += 1
    for col, width in widths.items():
        # Довгий текст — переносом у стовпці до 60 знаків, а не рядком на пів екрана.
        sheet.column_dimensions[get_column_letter(col)].width = max(10, min(60, int(width * 1.1) + 2))
    if first_header is not None:
        if len(tables) == 1:
            sheet.freeze_panes = sheet.cell(first_header + 1, 1)
        _print_setup(sheet, max_cols, first_header)
    return total


def _excel_table(sheet: Any, name: str, header_row: int, cols: int, last_row: int, style: str) -> None:
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.table import Table, TableStyleInfo

    table = Table(displayName=name, ref=f"A{header_row}:{get_column_letter(cols)}{last_row}")
    table.tableStyleInfo = TableStyleInfo(name=style, showFirstColumn=False, showLastColumn=False,
                                          showRowStripes=True, showColumnStripes=False)
    sheet.add_table(table)


def _unique(header: list[str]) -> list[str]:
    """Назви стовпців «Таблиці Excel» мусять бути непорожні й різні —
    інакше Excel відкриває файл через «відновлення вмісту»."""
    seen: dict[str, int] = {}
    out = []
    for col, raw in enumerate(header, start=1):
        name = plain(strip_markers(raw)) or f"Стовпець {col}"
        count = seen.get(name.lower(), 0)
        seen[name.lower()] = count + 1
        out.append(name if not count else f"{name} ({count + 1})")
    return out


def _print_setup(sheet: Any, cols: int, header_row: int) -> None:
    """Друк: широка таблиця — альбомно, усе в ширину сторінки, шапка на кожному аркуші."""
    sheet.page_setup.orientation = "landscape" if cols > 3 else "portrait"
    sheet.page_setup.paperSize = sheet.PAPERSIZE_A4
    sheet.page_setup.fitToWidth = 1
    sheet.page_setup.fitToHeight = 0
    sheet.sheet_properties.pageSetUpPr.fitToPage = True
    sheet.print_title_rows = f"{header_row}:{header_row}"


def _captions(outline: Outline) -> dict[int, str]:
    """Заголовок, що стоїть перед таблицею, — її підпис на аркуші."""
    out: dict[int, str] = {}
    last = ""
    for block in outline.blocks:
        if block.kind == "heading":
            last = plain(strip_markers(block.text))
        elif block.kind == "table" and last:
            out[id(block)] = last
    return out


def _text_sheet(sheet: Any, outline: Outline) -> int:
    from openpyxl.styles import Font

    style = _styles()
    sheet.cell(1, 1, outline.title).font = style["title"]
    sheet.column_dimensions["A"].width = 110
    row = 3
    count = 0
    for block in outline.blocks:
        lines: list[tuple[str, bool]] = []
        if block.kind == "heading":
            lines = [(plain(strip_markers(block.text)), True)]
        elif block.kind == "paragraph":
            lines = [(plain(block.text), False)]
        elif block.kind == "bullets":
            lines = [(f"{'    – ' if block.depth(i) else '• '}{plain(item)}", False)
                     for i, item in enumerate(block.items)]
        elif block.kind == "numbered":
            lines = [(f"{i + 1}. {plain(item)}", False) for i, item in enumerate(block.items)]
        for text, bold in lines:
            cell = sheet.cell(row, 1, text[:_EXCEL_TEXT_LIMIT])
            cell.alignment = style["cell_align"]
            if bold:
                cell.font = Font(bold=True)
            row += 1
            count += not bold
    return count


def _sources_sheet(sheet: Any, cited: Sequence[SourceRef], note: str) -> None:
    from openpyxl.styles import Font

    style = _styles()
    for col, text in enumerate(("№", "Документ", "Сторінки"), start=1):
        sheet.cell(1, col, text).alignment = style["head_align"]
    for row, source in enumerate(cited, start=2):
        for col, value in enumerate((source.ordinal, source.nice_title, source.label), start=1):
            sheet.cell(row, col, value).alignment = style["cell_align"]
    _excel_table(sheet, "Sources", 1, 3, max(2, len(cited) + 1), "TableStyleLight11")
    sheet.cell(max(2, len(cited) + 1) + 2, 1, note).font = Font(italic=True, color="595959")
    sheet.column_dimensions["A"].width = 6
    sheet.column_dimensions["B"].width = 70
    sheet.column_dimensions["C"].width = 16


def _value(raw: str, sources: Mapping[int, SourceRef]) -> str | int | float:
    text = plain(raw)
    stripped = text.strip()
    if stripped and _MARKERS_ONLY.fullmatch(stripped):
        return "; ".join(
            f"[{n}] {sources[n].short()}" if n in sources else f"[{n}]" for n in markers(stripped)
        )
    if _NUMBER.fullmatch(stripped) and not re.match(r"^[-−]?0\d", stripped):
        number = stripped.replace("−", "-").replace(" ", "").replace(" ", "").replace(" ", "")
        number = number.replace(",", ".")
        try:
            value = float(number)
        except ValueError:
            return stripped
        return int(value) if value.is_integer() and "." not in number else value
    return text[:_EXCEL_TEXT_LIMIT]
