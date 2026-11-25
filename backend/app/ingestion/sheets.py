"""Таблиці Excel / CSV → Markdown-копія для текстового конвеєра.

ЧОМУ НЕ PDF. Аркуш Excel не має сторінок: друкований PDF ріже широку таблицю
на смуги, і TableFormer потім збирає назад те, що й так лежить у клітинках
готовим. Тому клітинки читаються напряму, а «сторінкою» для цитати стає
аркуш — викладач шукатиме «арк. Норми», а не 7-му друковану смугу.

ЧОМУ ТАБЛИЦЯ РІЖЕТЬСЯ НА БЛОКИ. Таблиця для чанкера атомарна: її не можна
розрізати посередині. Аркуш на 2000 рядків як один атомарний блок став би
одним гігантським фрагментом, і пошук знаходив би «увесь аркуш» на будь-яке
питання. Блоки по ~30 рядків із ПОВТОРЕНИМ заголовком — кожен рядок блоку
серіалізується трійками «рядок, стовпець = значення» (`serialize_table_triplets`),
тож переживає будь-який розріз і не втрачає назв стовпців.

ЗЛИТІ КЛІТИНКИ розгортаються значенням верхньої лівої: «Значення» над двома
стовпцями дає заголовок обом, а «Гармати» на чотири рядки — кожному з них.
Інакше половина трійок мала б порожній заголовок або порожню категорію.
"""

from __future__ import annotations

import csv
import datetime as dt
import io
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.ingestion.docling_pipeline import serialize_table_triplets

__all__ = [
    "Sheet",
    "SheetRendition",
    "read_workbook",
    "render_markdown",
    "spreadsheet_to_markdown",
]

MAX_ROWS_PER_SHEET = 20_000
MAX_COLUMNS = 60
# Бюджет блоку — у символах СЕРІАЛІЗОВАНИХ трійок, бо саме вони йдуть у чанк.
# 1300 + підпис блоку лягає в ціль листка (1400) з запасом до м'якого
# максимуму (2200).
BLOCK_CHARS = 1300
BLOCK_ROWS = 40
# Повний режим openpyxl (зі злитими клітинками) тримає весь аркуш у пам'яті.
# Для великих файлів — потоковий read_only без злиттів.
FULL_MODE_MAX_BYTES = 25 * 1024 * 1024

PAGE_BREAK = "\n\f\n"


@dataclass(slots=True)
class Sheet:
    """Аркуш як список рядків. Номер рядка — той, що бачить викладач в Excel."""

    name: str
    rows: list[tuple[int, list[str]]] = field(default_factory=list)
    truncated: bool = False


@dataclass(slots=True)
class SheetRendition:
    markdown: str
    page_labels: dict[int, str]
    warnings: list[str] = field(default_factory=list)
    sheets: int = 0
    rows: int = 0


# ----------------------------------------------------------------- значення
_WS_RE = re.compile(r"\s+")


def _number(value: float, number_format: str = "") -> str:
    if math.isnan(value) or math.isinf(value):
        return ""
    if "%" in (number_format or ""):
        return f"{value * 100:.6g}".replace(".", ",") + " %"
    if value.is_integer() and abs(value) < 1e15:
        return str(int(value))
    text = f"{value:.10g}"
    # Десяткова кома — так число стоїть і в українському Excel, і в
    # підручнику, з яким його звірятимуть.
    return text.replace(".", ",") if "e" not in text else text


def cell_text(value: Any, number_format: str = "") -> str:
    """Значення клітинки → рядок, яким його показав би Excel."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "так" if value else "ні"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return _number(value, number_format)
    if isinstance(value, dt.datetime):
        if (value.hour, value.minute, value.second) == (0, 0, 0):
            return value.strftime("%d.%m.%Y")
        return value.strftime("%d.%m.%Y %H:%M")
    if isinstance(value, dt.date):
        return value.strftime("%d.%m.%Y")
    if isinstance(value, dt.time):
        return value.strftime("%H:%M:%S" if value.second else "%H:%M")
    return _WS_RE.sub(" ", str(value)).strip()


# ------------------------------------------------------------------ читання
def read_workbook(path: str | Path, *, csv_name: str = "") -> tuple[list[Sheet], list[str]]:
    """Прочитати книгу. Повертає аркуші й попередження.

    У CSV аркуша немає, а ім'я файлу на диску — UUID; тому назву «аркуша»
    передає той, хто знає справжню назву документа.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return [_read_csv(path, csv_name or "Таблиця")], []
    if suffix == ".xls":
        return _read_xls(path)
    if suffix in (".xlsx", ".xlsm"):
        return _read_xlsx(path)
    raise ValueError(f"Формат {suffix} не читається напряму — спершу перетворіть на XLSX.")


def _read_xlsx(path: Path) -> tuple[list[Sheet], list[str]]:
    import openpyxl

    warnings: list[str] = []
    full = path.stat().st_size <= FULL_MODE_MAX_BYTES
    # data_only=True: формула → збережене Excel значення. Без цього в базу
    # потрапили б «=B2*C2» замість чисел, які викладач бачить на екрані.
    book = openpyxl.load_workbook(path, data_only=True, read_only=not full)
    if not full:
        warnings.append("Файл великий: злиті клітинки не розгорнуто (потоковий режим читання).")
    sheets: list[Sheet] = []
    try:
        for ws in book.worksheets:
            if getattr(ws, "sheet_state", "visible") == "veryHidden":
                continue
            merged: dict[tuple[int, int], Any] = {}
            if full:
                for rng in ws.merged_cells.ranges:
                    anchor = ws.cell(rng.min_row, rng.min_col)
                    for r in range(rng.min_row, rng.max_row + 1):
                        for c in range(rng.min_col, min(rng.max_col, MAX_COLUMNS) + 1):
                            merged[(r, c)] = (anchor.value, anchor.number_format)
            sheet = Sheet(name=str(ws.title))
            for r_index, row in enumerate(ws.iter_rows(max_col=MAX_COLUMNS), start=1):
                if len(sheet.rows) >= MAX_ROWS_PER_SHEET:
                    sheet.truncated = True
                    break
                cells: list[str] = []
                for c_index, cell in enumerate(row, start=1):
                    value = getattr(cell, "value", None)
                    fmt = getattr(cell, "number_format", "") or ""
                    if value is None and (r_index, c_index) in merged:
                        value, fmt = merged[(r_index, c_index)]
                    cells.append(cell_text(value, fmt))
                if any(cells):
                    sheet.rows.append((r_index, cells))
            sheets.append(sheet)
    finally:
        book.close()
    return sheets, warnings


def _read_xls(path: Path) -> tuple[list[Sheet], list[str]]:
    import xlrd

    try:
        book = xlrd.open_workbook(str(path), formatting_info=True)
    except NotImplementedError:
        book = xlrd.open_workbook(str(path))
    sheets: list[Sheet] = []
    for sh in book.sheets():
        if getattr(sh, "visibility", 0) == 2:          # «дуже прихований»
            continue
        merged: dict[tuple[int, int], tuple[int, int]] = {}
        for rlo, rhi, clo, chi in getattr(sh, "merged_cells", []) or []:
            for r in range(rlo, rhi):
                for c in range(clo, min(chi, MAX_COLUMNS)):
                    merged[(r, c)] = (rlo, clo)
        sheet = Sheet(name=str(sh.name))
        for r in range(sh.nrows):
            if len(sheet.rows) >= MAX_ROWS_PER_SHEET:
                sheet.truncated = True
                break
            cells: list[str] = []
            for c in range(min(sh.ncols, MAX_COLUMNS)):
                src = merged.get((r, c), (r, c))
                cells.append(_xls_cell(book, sh.cell(*src)))
            if any(cells):
                sheet.rows.append((r + 1, cells))
        sheets.append(sheet)
    return sheets, []


def _xls_cell(book: Any, cell: Any) -> str:
    import xlrd

    if cell.ctype == xlrd.XL_CELL_DATE:
        try:
            return cell_text(xlrd.xldate.xldate_as_datetime(cell.value, book.datemode))
        except (ValueError, OverflowError):
            return cell_text(cell.value)
    if cell.ctype == xlrd.XL_CELL_BOOLEAN:
        return cell_text(bool(cell.value))
    if cell.ctype in (xlrd.XL_CELL_ERROR, xlrd.XL_CELL_EMPTY, xlrd.XL_CELL_BLANK):
        return ""
    fmt = ""
    if cell.ctype == xlrd.XL_CELL_NUMBER:
        try:
            xf = book.xf_list[cell.xf_index]
            fmt = book.format_map[xf.format_key].format_str
        except (AttributeError, IndexError, KeyError, TypeError):
            fmt = ""
    return cell_text(cell.value, fmt)


def _read_csv(path: Path, name: str) -> Sheet:
    raw = path.read_bytes()
    text = ""
    for encoding in ("utf-8-sig", "cp1251"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:  # pragma: no cover — cp1251 декодує будь-які байти
        text = raw.decode("utf-8", errors="replace")
    sample = text[:20_000]
    try:
        delimiter = csv.Sniffer().sniff(sample, delimiters=";,\t|").delimiter
    except csv.Error:
        # Український Excel зберігає CSV через «;» — кома там десяткова.
        delimiter = ";" if sample.count(";") >= sample.count(",") else ","
    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    sheet = Sheet(name=name)
    for index, row in enumerate(reader, start=1):
        if len(sheet.rows) >= MAX_ROWS_PER_SHEET:
            sheet.truncated = True
            break
        cells = [cell_text(c) for c in row[:MAX_COLUMNS]]
        if any(cells):
            sheet.rows.append((index, cells))
    return sheet


# ------------------------------------------------------------------ розмітка
def _column_letter(index: int) -> str:
    """1 → A, 27 → AA — так стовпці підписані в самому Excel."""
    letters = ""
    while index > 0:
        index, rest = divmod(index - 1, 26)
        letters = chr(ord("A") + rest) + letters
    return letters


def _md_cell(text: str) -> str:
    # «|» усередині клітинки розірвав би рядок Markdown-таблиці.
    return text.replace("|", "¦").strip()


def _normalized(sheet: Sheet) -> tuple[list[tuple[int, list[str]]], int]:
    """Обрізати порожні краї й зібрати рядки однакової ширини.

    Повертає рядки та номер першого використаного стовпця (1-based).
    """
    rows = [(n, list(cells)) for n, cells in sheet.rows if any(cells)]
    if not rows:
        return [], 1
    first = min(next(i for i, c in enumerate(cells) if c) for _, cells in rows)
    last = max(max(i for i, c in enumerate(cells) if c) for _, cells in rows)
    out: list[tuple[int, list[str]]] = []
    for number, cells in rows:
        cells = (cells + [""] * (last + 1 - len(cells)))[first:last + 1]
        filled = [c for c in cells if c]
        # Рядок, де всі клітинки — копія ОДНОГО злитого значення, — це назва
        # таблиці над нею («Таблиця 3. Норми»), а не рядок даних.
        if len(filled) > 1 and len(set(filled)) == 1:
            cells = [filled[0]] + [""] * (len(cells) - 1)
        out.append((number, cells))
    return out, first + 1


def _header_index(rows: list[tuple[int, list[str]]], width: int) -> int:
    """Рядок заголовків: перший, де заповнено хоча б половину стовпців."""
    need = max(2, math.ceil(width * 0.5))
    head = rows[:10]
    for index, (_n, cells) in enumerate(head):
        if sum(1 for c in cells if c) >= need:
            return index
    for index, (_n, cells) in enumerate(head):
        if sum(1 for c in cells if c) >= 2:
            return index
    return 0


def _render_sheet(sheet: Sheet, lines: list[str]) -> int:
    """Дописати аркуш у `lines`. Повертає кількість рядків даних."""
    rows, first_col = _normalized(sheet)
    if not rows:
        return 0
    width = len(rows[0][1])
    lines.append(f"# Аркуш «{sheet.name}»")
    lines.append("")

    if width == 1:
        # Аркуш-список (один стовпець) — це текст, а не таблиця.
        for start in range(0, len(rows), 30):
            lines.extend(_md_cell(cells[0]) for _n, cells in rows[start:start + 30])
            lines.append("")
        return len(rows)

    head = _header_index(rows, width)
    for _n, cells in rows[:head]:
        lines.append(" — ".join(c for c in cells if c))
        lines.append("")
    header = [
        cell or f"стовпець {_column_letter(first_col + i)}"
        for i, cell in enumerate(rows[head][1])
    ]
    body = rows[head + 1:]
    if not body:
        lines.append(" | ".join(c for c in rows[head][1] if c))
        lines.append("")
        return 1

    for block in _blocks(header, body):
        first, last = block[0][0], block[-1][0]
        span = f"рядок {first}" if first == last else f"рядки {first}–{last}"
        lines.append(f"Аркуш «{sheet.name}», {span}.")
        lines.append("| " + " | ".join(_md_cell(h) for h in header) + " |")
        lines.append("|" + "|".join(" --- " for _ in header) + "|")
        for _n, cells in block:
            lines.append("| " + " | ".join(_md_cell(c) for c in cells) + " |")
        lines.append("")
    return len(body)


def _blocks(
    header: list[str], body: list[tuple[int, list[str]]]
) -> Iterable[list[tuple[int, list[str]]]]:
    """Нарізати тіло на блоки за бюджетом серіалізованих трійок."""
    block: list[tuple[int, list[str]]] = []
    size = 0
    for number, cells in body:
        row_size = len(serialize_table_triplets([header, cells])) + 1
        if block and (size + row_size > BLOCK_CHARS or len(block) >= BLOCK_ROWS):
            yield block
            block, size = [], 0
        block.append((number, cells))
        size += row_size
    if block:
        yield block


def render_markdown(sheets: list[Sheet]) -> SheetRendition:
    """Аркуші → Markdown, кожен аркуш — окрема «сторінка» (розрив U+000C)."""
    pages: list[str] = []
    labels: dict[int, str] = {}
    warnings: list[str] = []
    total_rows = 0
    for sheet in sheets:
        lines: list[str] = []
        rows = _render_sheet(sheet, lines)
        if not rows:
            continue
        total_rows += rows
        pages.append("\n".join(lines).strip())
        labels[len(pages)] = f"арк. {sheet.name}"
        if sheet.truncated:
            warnings.append(
                f"Аркуш «{sheet.name}» обрізано до {MAX_ROWS_PER_SHEET} рядків."
            )
    # Розрив після КОЖНОГО аркуша, включно з останнім: інакше книга з одного
    # аркуша не мала б жодного розриву, і текстова проба нарізала б її на
    # «сторінки» по 1800 символів, для яких немає назви аркуша.
    markdown = "".join(page + PAGE_BREAK for page in pages)
    return SheetRendition(markdown=markdown, page_labels=labels, warnings=warnings,
                          sheets=len(pages), rows=total_rows)


def spreadsheet_to_markdown(path: str | Path, *, csv_name: str = "") -> SheetRendition:
    sheets, warnings = read_workbook(path, csv_name=csv_name)
    rendition = render_markdown(sheets)
    rendition.warnings[:0] = warnings
    return rendition
