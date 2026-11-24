"""Документ Word.

Оформлення наближене до службових документів (ДСТУ 4163:2020): А4, поля
30/15/20/20 мм, Times New Roman 14, абзацний відступ 1,25 см, вирівнювання за
шириною. Викладач несе такий файл далі — у методичку чи план заняття, — і
переформатовувати його вручну означало б звести користь нанівець.
"""

from __future__ import annotations

import io
from collections.abc import Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any

from app.export.outline import Block, Outline, plain, runs, strip_markers

if TYPE_CHECKING:
    from app.export.render import SourceRef

__all__ = ["render_docx"]

FONT = "Times New Roman"
MIN_SECTIONS_FOR_CONTENTS = 3
# Ширина тексту на сторінці: 21 см мінус поля 3 і 1,5 см.
TEXT_WIDTH_CM = 16.5
_LINK = "2E4A2B"          # темно-оливковий — колір посилань змісту
_HEADER_FILL = "E2E8DC"
_BAND_FILL = "F4F6F1"


def render_docx(
    outline: Outline, cited: Sequence[SourceRef], *, note: str
) -> tuple[bytes, dict[str, int]]:
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Cm, Pt, RGBColor

    doc = Document()
    _page_setup(doc, outline.title)
    _restyle(doc)

    title = doc.add_paragraph(style="Title")
    title.add_run(outline.title)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Зміст із переходами на розділи (Ctrl+клік у Word) — коли розділів є що гортати.
    headings = [plain(strip_markers(b.text)) or "…" for b in outline.blocks
                if b.kind == "heading" and b.level <= 2]
    if len(headings) >= MIN_SECTIONS_FOR_CONTENTS:
        _contents(doc, headings)

    sections = tables = rows = 0
    caption = outline.title
    for block in outline.blocks:
        if block.kind == "heading":
            # Назва документа — це «#»; розділи моделі — «##», підрозділи — «###».
            level = 1 if block.level <= 2 else (2 if block.level == 3 else 3)
            text = plain(strip_markers(block.text)) or "…"
            heading = doc.add_heading(text, level=level)
            caption = text
            if level == 1:
                sections += 1
                _bookmark(heading, f"_section{sections}", sections)
        elif block.kind == "paragraph":
            paragraph = doc.add_paragraph()
            _add_runs(paragraph, block.text)
            paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            paragraph.paragraph_format.first_line_indent = Cm(1.25)
        elif block.kind == "bullets":
            for index, item in enumerate(block.items):
                style = "List Bullet 2" if block.depth(index) else "List Bullet"
                _add_runs(doc.add_paragraph(style=style), item)
        elif block.kind == "numbered":
            # Нумерація вручну: «List Number» у Word наскрізна на весь документ,
            # і другий список продовжив би рахунок першого (4, 5, 6 …).
            number = 0
            for index, item in enumerate(block.items):
                nested = block.depth(index)
                number = number if nested else number + 1
                paragraph = doc.add_paragraph()
                paragraph.paragraph_format.left_indent = Cm(2.25 if nested else 1.25)
                paragraph.paragraph_format.first_line_indent = Cm(-0.6)
                paragraph.add_run("– " if nested else f"{number}. ")
                _add_runs(paragraph, item)
        elif block.kind == "table":
            tables += 1
            # Підпис над таблицею — як у службових документах: «Таблиця 1 – Назва».
            label = doc.add_paragraph()
            label.paragraph_format.keep_with_next = True
            label.paragraph_format.space_before = Pt(6)
            label.add_run(f"Таблиця {tables}").bold = True
            label.add_run(f" – {caption}")
            _table(doc, block)
            rows += len(block.rows)

    if cited:
        doc.add_heading("Джерела", level=1)
        for source in cited:
            paragraph = doc.add_paragraph()
            paragraph.paragraph_format.left_indent = Cm(1.0)
            paragraph.paragraph_format.first_line_indent = Cm(-1.0)
            paragraph.add_run(f"[{source.ordinal}] ").bold = True
            paragraph.add_run(source.full())

    tail = doc.add_paragraph()
    tail.paragraph_format.space_before = Pt(18)
    run = tail.add_run(note)
    run.italic = True
    run.font.size = Pt(10)
    run.font.color.rgb = RGBColor(0x59, 0x59, 0x59)

    props = doc.core_properties
    props.title = outline.title
    props.author = props.last_modified_by = "Асистент викладача"
    props.language = "uk-UA"
    props.comments = note
    props.created = props.modified = datetime.now()

    buffer = io.BytesIO()
    doc.save(buffer)
    stats = {"sections": sections, "tables": tables, "rows": rows}
    return buffer.getvalue(), {k: v for k, v in stats.items() if v}


def _add_runs(paragraph: Any, text: str) -> None:
    for chunk, bold, italic in runs(text):
        run = paragraph.add_run(chunk)
        if bold:
            run.bold = True
        if italic:
            run.italic = True


def _table(doc: Any, block: Block) -> None:
    """Таблиця на всю ширину тексту: шапка з тлом і повтором на кожній
    сторінці, смуги рядків, ширина стовпців — за обсягом тексту."""
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Cm, Pt

    width = max(1, len(block.header))
    all_rows = [block.header, *block.rows]
    table = doc.add_table(rows=len(all_rows), cols=width)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    weights = [min(60, max(6, max((len(r[c]) if c < len(r) else 0) for r in all_rows)))
               for c in range(width)]
    widths = _column_widths(weights, all_rows, width)
    for row_index, row in enumerate(all_rows):
        for col_index in range(width):
            cell = table.cell(row_index, col_index)
            cell.width = Cm(widths[col_index])
            text = row[col_index] if col_index < len(row) else ""
            paragraph = cell.paragraphs[0]
            paragraph.paragraph_format.first_line_indent = Pt(0)
            paragraph.paragraph_format.space_after = Pt(0)
            for chunk, bold, italic in runs(text):
                run = paragraph.add_run(chunk)
                run.bold = True if row_index == 0 else (bold or None)
                run.italic = italic or None
                run.font.size = Pt(12)
            fill = _HEADER_FILL if row_index == 0 else (_BAND_FILL if row_index % 2 == 0 else None)
            if fill:
                shading = OxmlElement("w:shd")
                shading.set(qn("w:val"), "clear")
                shading.set(qn("w:color"), "auto")
                shading.set(qn("w:fill"), fill)
                cell._tc.get_or_add_tcPr().append(shading)
    # Шапка повторюється на кожній сторінці довгої таблиці.
    header_row = table.rows[0]._tr.get_or_add_trPr()
    repeat = OxmlElement("w:tblHeader")
    repeat.set(qn("w:val"), "true")
    header_row.append(repeat)
    doc.add_paragraph()


def _column_widths(weights: list[int], rows: list[list[str]], width: int) -> list[float]:
    """Ширини стовпців (см): пропорційно обсягу тексту, але не вужче за
    найдовше слово стовпця — інакше Word ламає слово посередині
    («фрагментув / ання», «У / по / шу / ку»). Нестачу забирають у ширших стовпців."""
    widths = [TEXT_WIDTH_CM * w / sum(weights) for w in weights]
    floors = []
    for c in range(width):
        words = [w for row in rows if c < len(row) for w in plain(row[c]).split()]
        longest = max((len(w) for w in words), default=3)
        floors.append(min(5.0, 0.5 + 0.24 * longest))       # ~0,24 см на знак при 12 pt
    if sum(floors) >= TEXT_WIDTH_CM:
        return [TEXT_WIDTH_CM * f / sum(floors) for f in floors]
    deficit = sum(max(0.0, f - w) for f, w in zip(floors, widths, strict=True))
    surplus = [max(0.0, w - f) for f, w in zip(floors, widths, strict=True)]
    spare = sum(surplus)
    return [f if w <= f else w - deficit * s / spare
            for f, w, s in zip(floors, widths, surplus, strict=True)]


def _contents(doc: Any, headings: list[str]) -> None:
    """«Зміст» — посилання на закладки розділів. Без поля TOC навмисно: воно
    вимагає «Оновити поле» і в LibreOffice не працює, а посилання — одразу."""
    from docx.shared import Pt

    head = doc.add_paragraph()
    head.paragraph_format.space_before = Pt(6)
    head.add_run("Зміст").bold = True
    for index, text in enumerate(headings, start=1):
        paragraph = doc.add_paragraph()
        paragraph.paragraph_format.space_after = Pt(2)
        paragraph.add_run(f"{index}. ")
        _internal_link(paragraph, text, f"_section{index}")
    doc.add_paragraph()


def _internal_link(paragraph: Any, text: str, anchor: str) -> None:
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    link = OxmlElement("w:hyperlink")
    link.set(qn("w:anchor"), anchor)
    link.set(qn("w:history"), "1")
    run = OxmlElement("w:r")
    props = OxmlElement("w:rPr")
    color = OxmlElement("w:color")
    color.set(qn("w:val"), _LINK)
    underline = OxmlElement("w:u")
    underline.set(qn("w:val"), "single")
    props.append(color)
    props.append(underline)
    run.append(props)
    body = OxmlElement("w:t")
    body.set(qn("xml:space"), "preserve")
    body.text = text
    run.append(body)
    link.append(run)
    paragraph._p.append(link)


def _bookmark(paragraph: Any, name: str, number: int) -> None:
    """Закладка навколо заголовка — ціль посилання зі змісту."""
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    start = OxmlElement("w:bookmarkStart")
    start.set(qn("w:id"), str(number))
    start.set(qn("w:name"), name)
    end = OxmlElement("w:bookmarkEnd")
    end.set(qn("w:id"), str(number))
    element = paragraph._p
    # `w:pPr` мусить лишатися першим дочірнім елементом абзацу.
    element.insert(1 if element.pPr is not None else 0, start)
    element.append(end)


def _page_setup(doc: Any, title: str) -> None:
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Cm, Pt, RGBColor

    section = doc.sections[0]
    section.page_width, section.page_height = Cm(21.0), Cm(29.7)
    section.left_margin, section.right_margin = Cm(3.0), Cm(1.5)
    section.top_margin, section.bottom_margin = Cm(2.0), Cm(2.0)

    # Верхній колонтитул — назва документа, дрібно й сіро: аркуш, що випав
    # із папки, однаково видно, звідки він.
    header = section.header.paragraphs[0]
    header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    mark = header.add_run(title)
    mark.font.size = Pt(9)
    mark.font.color.rgb = RGBColor(0x80, 0x80, 0x80)

    # Номер сторінки внизу по центру — полем PAGE, щоб Word рахував сам.
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = footer.add_run()
    for kind, text in (("begin", None), ("instr", "PAGE"), ("end", None)):
        if kind == "instr":
            element = OxmlElement("w:instrText")
            element.set(qn("xml:space"), "preserve")
            element.text = text
        else:
            element = OxmlElement("w:fldChar")
            element.set(qn("w:fldCharType"), kind)
        run._r.append(element)


def _restyle(doc: Any) -> None:
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Pt

    normal = doc.styles["Normal"]
    _font(normal, 14)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.15
    # Мова тексту — українська: інакше Word перевіряє правопис англійським словником.
    rpr = normal.element.get_or_add_rPr()
    lang = OxmlElement("w:lang")
    lang.set(qn("w:val"), "uk-UA")
    rpr.append(lang)

    title = doc.styles["Title"]
    _font(title, 16, bold=True)
    title.paragraph_format.space_after = Pt(12)
    # Шаблон python-docx малює під назвою синю лінію — у службовому документі зайве.
    ppr = title.element.pPr
    if ppr is not None:
        border = ppr.find(qn("w:pBdr"))
        if border is not None:
            ppr.remove(border)

    for name, size in (("Heading 1", 14), ("Heading 2", 14), ("Heading 3", 14)):
        style = doc.styles[name]
        _font(style, size, bold=True)
        style.paragraph_format.space_before = Pt(12)
        style.paragraph_format.space_after = Pt(6)


def _font(style: Any, size: int, *, bold: bool | None = None) -> None:
    from docx.oxml.ns import qn
    from docx.shared import Pt, RGBColor

    style.font.name = FONT
    style.font.size = Pt(size)
    style.font.color.rgb = RGBColor(0, 0, 0)
    if bold is not None:
        style.font.bold = bold
    # Тематичні шрифти шаблону (asciiTheme…) Word ставить ВИЩЕ за явну назву,
    # тож без цього заголовки лишилися б Calibri Light, хоч би що тут писало.
    fonts = style.element.get_or_add_rPr().get_or_add_rFonts()
    for attr in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
        fonts.attrib.pop(qn(attr), None)
    for attr in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        fonts.set(qn(attr), FONT)
