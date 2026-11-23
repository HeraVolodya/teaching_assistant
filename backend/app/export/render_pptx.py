"""Презентація PowerPoint — інтерактивна, для показу на занятті.

Формат 16:9. Що робить її інтерактивною (усе офлайн, засобами самого PowerPoint):
  * слайд «Зміст» — кнопки розділів, клік у режимі показу веде на розділ;
  * кнопка «☰ Зміст» на кожному слайді — повернення до змісту;
  * тези з'являються по одній на клік — викладач веде розповідь, а не читає
    слайд наперед з аудиторією;
  * «Питання для самоперевірки» — відповідь з'являється лише на клік, тож
    слайд працює як усне опитування;
  * плавні переходи між слайдами, номер «3 / 9» на кожному.

ЗАГОЛОВКИ Й ТЕКСТ — У ПЛЕЙСХОЛДЕРАХ МАКЕТІВ, а не у вільних написах: тоді
PowerPoint бачить назви слайдів (структура, навігація, читачі екрана), а
викладач редагує слайди звичайними засобами.

ПЕРЕПОВНЕННЯ — РОЗБИТТЯМ, а не дрібним шрифтом: понад шість пунктів або ~700
символів ідуть на слайд «(продовження)», бо текст, що вилазить за край, у
показі просто зникає.

Анімації й переходи python-pptx не вміє — тут вони пишуться XML-ом PresentationML
(`p:timing`, `p:transition`) рівно так, як їх зберігає сам PowerPoint. Перевірено
відкриттям у PowerPoint: ефекти на місці (`MainSequence.Count`), файл без «відновлення».
"""

from __future__ import annotations

import io
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from app.export.outline import MARKER_RE, Outline, markers, plain, runs, strip_markers

if TYPE_CHECKING:
    from app.export.render import SourceRef

__all__ = ["QUIZ_TITLE", "render_pptx"]

MAX_ITEMS = 6
MAX_CHARS = 700
MAX_TABLE_ROWS = 8
MAX_TABLE_CHARS = 900
QUIZ_PER_SLIDE = 3
MIN_SECTIONS_FOR_CONTENTS = 3

# Широкий формат; макети шаблону python-pptx намальовані під 4:3 (10 дюймів).
SLIDE_W_IN, SLIDE_H_IN = 13.333, 7.5
TEMPLATE_W_IN = 10.0
NAV_W_IN = 1.45          # кнопка «☰ Зміст» у правому верхньому куті

_INK = (0x1F, 0x29, 0x33)
_MUTED = (0x5F, 0x6B, 0x76)
_ACCENT = (0x3F, 0x5A, 0x36)       # оливковий, без строкатості
_ROW_ALT = (0xF2, 0xF4, 0xEF)
_SOFT = (0xEC, 0xF0, 0xE8)          # тло кнопок і відповідей
_WHITE = (0xFF, 0xFF, 0xFF)

QUIZ_TITLE = "Питання для самоперевірки"
_QUIZ_HEADING = re.compile(
    r"питання\s+для\s+самоперевірки|контрольні\s+питання|перевір(?:те)?\s+себе|самоконтрол",
    re.IGNORECASE,
)
# «Питання? — Відповідь [n]» або «Питання? Відповідь: …».
_QA = re.compile(r"^(?P<q>.+?\?)\s*(?:[—–-]+|відповідь\s*:)\s*(?P<a>.+)$", re.IGNORECASE)
_QA_NO_MARK = re.compile(r"^(?P<q>.+?)\s+[—–]\s+(?P<a>.+)$")

_SLIDE_PREFIX = re.compile(r"^(?:слайд|slide)\s*\d+\s*[.:)—–-]*\s*", re.IGNORECASE)
# Межа речення — лише перед великою літерою: «с. 93», «рис. 4» не ріжемо.
_SENTENCE = re.compile(r"(?<=[.!?…])\s+(?=[«\"(]?[A-ZА-ЯІЇЄҐ])")
# Позначка підпункту в рядку плану слайда (символу \x00 у тексті моделі не буває).
_SUB = "\x00"


@dataclass(slots=True)
class _Plan:
    title: str
    items: list[str] = field(default_factory=list)
    table: tuple[list[str], list[list[str]]] | None = None
    # Пари «питання — відповідь» слайда самоперевірки.
    quiz: list[tuple[str, str]] | None = None
    # Маркери з заголовка розділу: модель ставить їх і туди («Зв'язок [1]»).
    title_markers: list[int] = field(default_factory=list)

    def text(self) -> str:
        parts = list(self.items)
        if self.table:
            parts += self.table[0] + [c for row in self.table[1] for c in row]
        if self.quiz:
            parts += [f"{q} {a}" for q, a in self.quiz]
        return "\n".join(parts)


def render_pptx(
    outline: Outline,
    sources: Mapping[int, SourceRef],
    cited: Sequence[SourceRef],
    *,
    day: str,
) -> tuple[bytes, dict[str, int]]:
    from pptx import Presentation
    from pptx.util import Inches

    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(SLIDE_W_IN), Inches(SLIDE_H_IN)
    _widen_layouts(prs)

    sections = _plan(outline)
    parts = [(section, part) for section in sections for part in _split(section)]
    with_contents = len(sections) >= MIN_SECTIONS_FOR_CONTENTS
    source_lines = [f"**[{s.ordinal}]** {s.full()}" for s in cited]
    source_slides = (len(source_lines) + 7) // 8
    total = 1 + int(with_contents) + len(parts) + source_slides
    number = 0

    def numbered(slide: Any) -> Any:
        nonlocal number
        number += 1
        _slide_number(slide, number, total)
        _transition(slide)
        return slide

    numbered(_title_slide(prs, outline.title, f"За матеріалами бази знань · {day}"))
    contents = numbered(_blank_titled(prs, "Зміст")) if with_contents else None

    first_slide: dict[int, Any] = {}
    for section, part in parts:
        if part.table is not None:
            slide = _table_slide(prs, part, sources)
        elif part.quiz is not None:
            slide = _quiz_slide(prs, part, sources)
        else:
            slide = _bullet_slide(prs, part, sources)
        first_slide.setdefault(id(section), slide)
        if contents is not None:
            _nav_button(slide, contents)
        numbered(slide)

    for index in range(0, len(source_lines), 8):
        title = "Джерела" if index == 0 else "Джерела (продовження)"
        slide = _bullet_slide(prs, _Plan(title=title, items=source_lines[index:index + 8]), {},
                              size=16, shrink_markers=False, animate=False)
        if contents is not None:
            _nav_button(slide, contents)
        numbered(slide)

    if contents is not None:
        _fill_contents(contents, [(s.title, first_slide[id(s)]) for s in sections])

    props = prs.core_properties
    props.title = outline.title
    props.author = props.last_modified_by = "Асистент викладача"
    props.language = "uk-UA"
    props.created = props.modified = datetime.now()

    buffer = io.BytesIO()
    prs.save(buffer)
    questions = sum(len(p.quiz or []) for _, p in parts)
    stats = {"slides": len(parts), "questions": questions}
    return buffer.getvalue(), {k: v for k, v in stats.items() if v}


# ------------------------------------------------------------------ план
def _plan(outline: Outline) -> list[_Plan]:
    plans: list[_Plan] = []
    current: _Plan | None = None
    last_title = outline.title

    def open_plan() -> _Plan:
        nonlocal current
        if current is None:
            current = _Plan(title=last_title)
            plans.append(current)
        return current

    for block in outline.blocks:
        if block.kind == "heading" and block.level <= 2:
            last_title = _slide_title(block.text) or outline.title
            current = _Plan(title=last_title, title_markers=markers(block.text))
            if _QUIZ_HEADING.search(last_title):
                current.title = QUIZ_TITLE
                current.quiz = []
            plans.append(current)
        elif block.kind == "heading":
            # Підзаголовок усередині розділу — жирний пункт, а не новий слайд.
            open_plan().items.append(f"**{plain(strip_markers(block.text))}**")
        elif block.kind == "paragraph":
            plan = open_plan()
            if plan.quiz is not None:
                plan.quiz.append(_question(block.text))
            else:
                plan.items.extend(_sentences(block.text))
        elif block.kind in ("bullets", "numbered"):
            plan = open_plan()
            if plan.quiz is not None:
                plan.quiz.extend(_question(item) for item in block.items)
            else:
                plan.items.extend(
                    f"{_SUB}{item}" if block.depth(index) else item
                    for index, item in enumerate(block.items)
                )
        elif block.kind == "table":
            plan = open_plan()
            if plan.items or plan.table is not None or plan.quiz is not None:
                plan = _Plan(title=last_title)
                plans.append(plan)
            plan.table = (block.header, block.rows)
            current = None   # текст після таблиці — уже наступний слайд
    return [p for p in plans if p.items or p.table is not None or p.quiz]


def _question(text: str) -> tuple[str, str]:
    """«Що таке ВП? — Ділянка місцевості… [1]» → (питання, відповідь)."""
    clean = text.strip()
    m = _QA.match(clean) or _QA_NO_MARK.match(clean)
    if m is not None:
        return m.group("q").strip(), m.group("a").strip()
    question, mark, rest = clean.partition("?")
    if mark and rest.strip(" —–-:"):
        # «Що таке ВП? Ділянка місцевості…» — без тире, але відповідь є.
        return f"{question}?", rest.strip(" —–-:")
    return clean, ""


def _slide_title(text: str) -> str:
    return _SLIDE_PREFIX.sub("", plain(strip_markers(text))).strip()


def _sentences(text: str) -> list[str]:
    """Довгий абзац на слайді — стіна тексту; ріжемо на речення-тези."""
    if len(text) <= 220:
        return [text]
    return [s.strip() for s in _SENTENCE.split(text) if s.strip()]


def _split(plan: _Plan) -> list[_Plan]:
    """Переповнений слайд → слайд і «(продовження)»."""
    out: list[_Plan] = []
    if plan.quiz is not None:
        for index in range(0, len(plan.quiz), QUIZ_PER_SLIDE):
            out.append(_Plan(title=plan.title, quiz=plan.quiz[index:index + QUIZ_PER_SLIDE]))
    elif plan.table is not None:
        header, rows = plan.table
        chunk: list[list[str]] = []
        size = 0
        for row in rows or [[]]:
            row_chars = sum(len(c) for c in row)
            if chunk and (len(chunk) >= MAX_TABLE_ROWS or size + row_chars > MAX_TABLE_CHARS):
                out.append(_Plan(title=plan.title, table=(header, chunk),
                                 title_markers=plan.title_markers))
                chunk, size = [], 0
            chunk.append(row)
            size += row_chars
        out.append(_Plan(title=plan.title, table=(header, chunk), title_markers=plan.title_markers))
    else:
        chunk_items: list[str] = []
        size = 0
        for item in plan.items:
            if chunk_items and (len(chunk_items) >= MAX_ITEMS or size + len(item) > MAX_CHARS):
                out.append(_Plan(title=plan.title, items=chunk_items, title_markers=plan.title_markers))
                chunk_items, size = [], 0
            chunk_items.append(item)
            size += len(item)
        out.append(_Plan(title=plan.title, items=chunk_items, title_markers=plan.title_markers))
    for index, part in enumerate(out):
        if index:
            part.title = f"{plan.title} (продовження)"
    return out


# ---------------------------------------------------------------- слайди
def _widen_layouts(prs: Any) -> None:
    """Розтягнути макети шаблону 4:3 на 16:9 — і для наших слайдів, і для тих,
    що викладач додасть сам: інакше новий слайд мав би заголовок у лівих 3/4."""
    factor = SLIDE_W_IN / TEMPLATE_W_IN
    holders = list(prs.slide_master.placeholders)
    for layout in prs.slide_layouts:
        holders.extend(layout.placeholders)
    for shape in holders:
        # Лише ті, що мають ВЛАСНІ координати: успадковані від майстра вже
        # розтягнуті разом із ним, і друге множення винесло б їх за край.
        if getattr(shape._element, "xfrm", None) is None:
            continue
        shape.left = int(shape.left * factor)
        shape.width = int(shape.width * factor)


def _rgb(color: tuple[int, int, int]) -> Any:
    from pptx.dml.color import RGBColor

    return RGBColor(*color)


def _place(shape: Any, left: float, top: float, width: float, height: float) -> None:
    from pptx.util import Inches

    shape.left, shape.top = Inches(left), Inches(top)
    shape.width, shape.height = Inches(width), Inches(height)


def _rect(slide: Any, left: float, top: float, width: float, height: float, *,
          color: tuple[int, int, int], rounded: bool = False) -> Any:
    from pptx.enum.shapes import MSO_SHAPE
    from pptx.util import Inches

    kind = MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE
    shape = slide.shapes.add_shape(kind, Inches(left), Inches(top), Inches(width), Inches(height))
    shape.fill.solid()
    shape.fill.fore_color.rgb = _rgb(color)
    shape.line.fill.background()
    shape.shadow.inherit = False
    if rounded:
        shape.adjustments[0] = 0.18
    return shape


def _write(paragraph: Any, text: str, size: int, *, color: tuple[int, int, int] = _INK,
           bold: bool = False, shrink_markers: bool = True) -> None:
    """Текст із розміткою: жирний/курсив — стилем, маркери [n] — дрібніше й сіріше.

    `shrink_markers=False` — там, де номер і є змістом: список джерел, стовпець
    «Джерело» таблиці.
    """
    from pptx.util import Pt

    for chunk, is_bold, is_italic in runs(text) or [("", False, False)]:
        last = 0
        for match in [*MARKER_RE.finditer(chunk), None]:
            end = match.start() if match else len(chunk)
            pieces = [(chunk[last:end], False)]
            if match:
                pieces.append((match.group(0), shrink_markers))
                last = match.end()
            for piece, small in pieces:
                if not piece:
                    continue
                run = paragraph.add_run()
                run.text = piece
                run.font.size = Pt(max(9, int(size * 0.7)) if small else size)
                run.font.bold = bold or is_bold
                run.font.italic = is_italic or None
                run.font.color.rgb = _rgb(_MUTED if small else color)


def _text_box(slide: Any, left: float, top: float, width: float, height: float, text: str,
              size: int, *, color: tuple[int, int, int] = _INK, bold: bool = False) -> Any:
    from pptx.util import Inches

    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    box.text_frame.word_wrap = True
    _write(box.text_frame.paragraphs[0], text, size, color=color, bold=bold)
    return box


def _set_title(slide: Any, text: str, *, size: int = 30) -> None:
    from pptx.enum.text import PP_ALIGN

    shape = slide.shapes.title
    # Праворуч лишаємо місце для кнопки «☰ Зміст».
    _place(shape, 0.6, 0.35, SLIDE_W_IN - 1.2 - NAV_W_IN - 0.2, 1.0)
    frame = shape.text_frame
    frame.clear()
    frame.word_wrap = True
    paragraph = frame.paragraphs[0]
    paragraph.alignment = PP_ALIGN.LEFT
    _write(paragraph, text, size if len(text) <= 48 else size - 4, bold=True)
    _rect(slide, 0.6, 1.38, 1.4, 0.06, color=_ACCENT)


def _title_slide(prs: Any, title: str, subtitle: str) -> Any:
    from pptx.enum.text import PP_ALIGN

    slide = prs.slides.add_slide(prs.slide_layouts[0])
    # Оливкова смуга ліворуч на всю висоту — «обкладинка» без строкатості.
    _rect(slide, 0, 0, 0.32, SLIDE_H_IN, color=_ACCENT)
    head = slide.shapes.title
    _place(head, 0.9, 2.0, SLIDE_W_IN - 1.8, 2.1)
    head.text_frame.clear()
    head.text_frame.word_wrap = True
    head.text_frame.paragraphs[0].alignment = PP_ALIGN.LEFT
    _write(head.text_frame.paragraphs[0], title, 40 if len(title) <= 60 else 32, bold=True)
    _rect(slide, 0.9, 4.25, 1.6, 0.08, color=_ACCENT)
    sub = slide.placeholders[1]
    _place(sub, 0.9, 4.5, SLIDE_W_IN - 1.8, 0.9)
    sub.text_frame.clear()
    sub.text_frame.paragraphs[0].alignment = PP_ALIGN.LEFT
    _write(sub.text_frame.paragraphs[0], subtitle, 18, color=_MUTED)
    return slide


def _blank_titled(prs: Any, title: str) -> Any:
    slide = prs.slides.add_slide(prs.slide_layouts[5])     # «Лише заголовок»
    _set_title(slide, title)
    return slide


def _fill_contents(slide: Any, entries: list[tuple[str, Any]]) -> None:
    """Кнопки розділів: у режимі показу клік веде на перший слайд розділу."""
    from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
    from pptx.util import Inches, Pt

    columns = 1 if len(entries) <= 6 else 2
    rows = (len(entries) + columns - 1) // columns
    gap = 0.14
    area_top, area_h = 1.65, SLIDE_H_IN - 1.65 - 0.75
    height = min(0.62, (area_h - gap * (rows - 1)) / rows)
    width = (SLIDE_W_IN - 1.2 - gap * (columns - 1)) / columns
    for index, (title, target) in enumerate(entries):
        col, row = divmod(index, rows)
        button = _rect(slide, 0.6 + col * (width + gap), area_top + row * (height + gap),
                       width, height, color=_SOFT, rounded=True)
        button.click_action.target_slide = target
        frame = button.text_frame
        frame.word_wrap = True
        frame.vertical_anchor = MSO_ANCHOR.MIDDLE
        frame.margin_left = Inches(0.2)
        paragraph = frame.paragraphs[0]
        paragraph.alignment = PP_ALIGN.LEFT
        number = paragraph.add_run()
        number.text = f"{index + 1}   "
        number.font.size, number.font.bold = Pt(16), True
        number.font.color.rgb = _rgb(_ACCENT)
        label = paragraph.add_run()
        label.text = title
        label.font.size = Pt(16 if len(title) <= 70 else 13)
        label.font.color.rgb = _rgb(_INK)
    _text_box(slide, 0.6, SLIDE_H_IN - 0.62, 8.0, 0.4,
              "У режимі показу (F5) натисніть розділ, щоб перейти до нього.", 11, color=_MUTED)


def _nav_button(slide: Any, contents: Any) -> None:
    from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
    from pptx.util import Pt

    button = _rect(slide, SLIDE_W_IN - 0.6 - NAV_W_IN, 0.5, NAV_W_IN, 0.42, color=_SOFT, rounded=True)
    button.click_action.target_slide = contents
    frame = button.text_frame
    frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    paragraph = frame.paragraphs[0]
    paragraph.alignment = PP_ALIGN.CENTER
    run = paragraph.add_run()
    run.text = "☰ Зміст"
    run.font.size, run.font.bold = Pt(12), True
    run.font.color.rgb = _rgb(_ACCENT)


def _slide_number(slide: Any, number: int, total: int) -> None:
    from pptx.enum.text import PP_ALIGN

    box = _text_box(slide, SLIDE_W_IN - 1.6, SLIDE_H_IN - 0.55, 1.0, 0.35, f"{number} / {total}",
                    10, color=_MUTED)
    box.text_frame.paragraphs[0].alignment = PP_ALIGN.RIGHT


def _bullet_slide(prs: Any, plan: _Plan, sources: Mapping[int, SourceRef], *,
                  size: int | None = None, shrink_markers: bool = True, animate: bool = True) -> Any:
    from pptx.util import Pt

    slide = prs.slides.add_slide(prs.slide_layouts[1])
    _set_title(slide, plan.title)
    body = slide.placeholders[1]
    _place(body, 0.6, 1.6, SLIDE_W_IN - 1.2, 5.1)
    frame = body.text_frame
    frame.clear()
    frame.word_wrap = True
    volume = sum(len(plain(i)) for i in plan.items)
    size = size or (24 if volume < 300 else 20 if volume < 520 else 18 if volume < 760 else 16)
    for index, item in enumerate(plan.items):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.space_after = Pt(6)
        nested = item.startswith(_SUB)
        paragraph.level = 1 if nested else 0
        _write(paragraph, item.removeprefix(_SUB), size - 2 if nested else size,
               shrink_markers=shrink_markers)
    _footer(slide, plan, sources)
    if animate and len(plan.items) > 1:
        # Тези — по одній на клік: викладач веде, аудиторія не читає наперед.
        _animate(slide, [(body.shape_id, index) for index in range(len(plan.items))])
    return slide


def _quiz_slide(prs: Any, plan: _Plan, sources: Mapping[int, SourceRef]) -> Any:
    """Питання видно одразу, відповідь — на клік: слайд працює як усне опитування."""
    from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
    from pptx.util import Inches

    assert plan.quiz is not None
    slide = _blank_titled(prs, plan.title)
    block = (SLIDE_H_IN - 1.6 - 0.75) / max(1, len(plan.quiz))
    answers: list[tuple[int, int | None]] = []
    for index, (question, answer) in enumerate(plan.quiz):
        top = 1.6 + index * block
        _text_box(slide, 0.6, top, SLIDE_W_IN - 1.2, 0.5, f"{index + 1}. {question}", 18, bold=True)
        if not answer:
            continue
        lines = max(1, -(-len(plain(answer)) // 105))      # ~105 знаків у рядку при 15 pt
        height = max(0.5, min(block - 0.7, 0.28 + 0.32 * lines))
        box = _rect(slide, 0.9, top + 0.55, SLIDE_W_IN - 1.5, height, color=_SOFT, rounded=True)
        frame = box.text_frame
        frame.word_wrap = True
        frame.vertical_anchor = MSO_ANCHOR.TOP
        frame.margin_left = frame.margin_right = Inches(0.18)
        frame.paragraphs[0].alignment = PP_ALIGN.LEFT
        _write(frame.paragraphs[0], f"**Відповідь:** {answer}", 15)
        answers.append((box.shape_id, None))
    _footer(slide, plan, sources)
    if answers:
        _animate(slide, answers)
    return slide


def _table_slide(prs: Any, plan: _Plan, sources: Mapping[int, SourceRef]) -> Any:
    from pptx.util import Inches, Pt

    assert plan.table is not None
    header, rows = plan.table
    width = max(1, len(header))
    slide = _blank_titled(prs, plan.title)
    total_w = SLIDE_W_IN - 1.2
    shape = slide.shapes.add_table(1 + len(rows), width, Inches(0.6), Inches(1.6),
                                   Inches(total_w), Inches(0.4 * (1 + len(rows))))
    table = shape.table
    size = 14 if width <= 3 else 12 if width <= 5 else 10
    # Ширина стовпців — за обсягом тексту, але не вужча за найдовше слово шапки:
    # інакше «Джерело» ламається на «Джерел / о».
    weights = [min(80, max(4, max((len(r[c]) if c < len(r) else 0) for r in [header, *rows])))
               for c in range(width)]
    scale = total_w / sum(weights)
    floors = [0.3 + 0.012 * size * max((len(w) for w in header[c].split()), default=4)
              if c < len(header) else 1.0 for c in range(width)]
    widths = [max(floor, w * scale) for floor, w in zip(floors, weights, strict=True)]
    shrink = total_w / sum(widths)
    for c, w in enumerate(widths):
        table.columns[c].width = Inches(w * shrink)
    for r, row in enumerate([header, *rows]):
        for c in range(width):
            cell = table.cell(r, c)
            cell.fill.solid()
            background = _ACCENT if r == 0 else (_ROW_ALT if r % 2 == 0 else _WHITE)
            cell.fill.fore_color.rgb = _rgb(background)
            frame = cell.text_frame
            frame.clear()
            frame.word_wrap = True
            paragraph = frame.paragraphs[0]
            paragraph.space_after = Pt(0)
            text = row[c] if c < len(row) else ""
            only_markers = bool(text.strip()) and not strip_markers(text)
            _write(paragraph, text, size, color=_WHITE if r == 0 else _INK,
                   bold=r == 0, shrink_markers=not only_markers)
    _footer(slide, plan, sources)
    return slide


def _footer(slide: Any, plan: _Plan, sources: Mapping[int, SourceRef]) -> None:
    from pptx.util import Inches, Pt

    used = [n for n in dict.fromkeys(plan.title_markers + markers(plan.text())) if n in sources]
    if not used:
        return
    names = [f"[{n}] {sources[n].short()}" for n in used[:3]]
    text = "Джерела: " + "; ".join(names) + ("; …" if len(used) > 3 else "")
    box = slide.shapes.add_textbox(Inches(0.6), Inches(SLIDE_H_IN - 0.62),
                                   Inches(SLIDE_W_IN - 2.6), Inches(0.4))
    frame = box.text_frame
    frame.word_wrap = True
    run = frame.paragraphs[0].add_run()
    run.text = text
    run.font.size = Pt(10)
    run.font.color.rgb = _rgb(_MUTED)


# ------------------------------------------------- анімації та переходи (XML)
_P = "http://schemas.openxmlformats.org/presentationml/2006/main"


def _transition(slide: Any) -> None:
    """Плавний перехід «Вицвітання» — стандартний `p:fade`, без розширень p14."""
    from pptx.oxml import parse_xml

    element = parse_xml(f'<p:transition xmlns:p="{_P}" spd="med"><p:fade/></p:transition>')
    _insert_after_color_map(slide, element)


def _animate(slide: Any, targets: list[tuple[int, int | None]]) -> None:
    """Поява на клік (ефект «Вицвітання»): абзац фігури або фігура цілком.

    Структура — та, що пише сам PowerPoint: кореневий вузол, головна
    послідовність і по групі «клік → ефект» на кожну ціль; у `bldLst` — як
    будувати фігуру (по абзацах чи цілком).
    """
    from pptx.oxml import parse_xml

    ids = iter(range(3, 10_000))
    clicks = []
    for spid, paragraph in targets:
        text_target = (f'<p:txEl><p:pRg st="{paragraph}" end="{paragraph}"/></p:txEl>'
                       if paragraph is not None else "")
        target = f'<p:spTgt spid="{spid}">{text_target}</p:spTgt>'
        click, delay, effect, visible, fade = (next(ids) for _ in range(5))
        clicks.append(
            f'<p:par><p:cTn id="{click}" fill="hold"><p:stCondLst><p:cond delay="indefinite"/>'
            f'</p:stCondLst><p:childTnLst><p:par><p:cTn id="{delay}" fill="hold"><p:stCondLst>'
            f'<p:cond delay="0"/></p:stCondLst><p:childTnLst><p:par><p:cTn id="{effect}" '
            f'presetID="10" presetClass="entr" presetSubtype="0" fill="hold" grpId="0" '
            f'nodeType="clickEffect"><p:stCondLst><p:cond delay="0"/></p:stCondLst><p:childTnLst>'
            f'<p:set><p:cBhvr><p:cTn id="{visible}" dur="1" fill="hold"><p:stCondLst>'
            f'<p:cond delay="0"/></p:stCondLst></p:cTn><p:tgtEl>{target}</p:tgtEl><p:attrNameLst>'
            f'<p:attrName>style.visibility</p:attrName></p:attrNameLst></p:cBhvr><p:to>'
            f'<p:strVal val="visible"/></p:to></p:set><p:animEffect transition="in" filter="fade">'
            f'<p:cBhvr><p:cTn id="{fade}" dur="500"/><p:tgtEl>{target}</p:tgtEl></p:cBhvr>'
            f'</p:animEffect></p:childTnLst></p:cTn></p:par></p:childTnLst></p:cTn></p:par>'
            f'</p:childTnLst></p:cTn></p:par>'
        )
    builds = []
    for spid in dict.fromkeys(spid for spid, _ in targets):
        by_paragraph = any(p is not None for s, p in targets if s == spid)
        builds.append(f'<p:bldP spid="{spid}" grpId="0" build="p"/>' if by_paragraph
                      else f'<p:bldP spid="{spid}" grpId="0" animBg="1"/>')
    timing = parse_xml(
        f'<p:timing xmlns:p="{_P}"><p:tnLst><p:par><p:cTn id="1" dur="indefinite" restart="never" '
        f'nodeType="tmRoot"><p:childTnLst><p:seq concurrent="1" nextAc="seek"><p:cTn id="2" '
        f'dur="indefinite" nodeType="mainSeq"><p:childTnLst>{"".join(clicks)}</p:childTnLst>'
        f'</p:cTn><p:prevCondLst><p:cond evt="onPrev" delay="0"><p:tgtEl><p:sldTgt/></p:tgtEl>'
        f'</p:cond></p:prevCondLst><p:nextCondLst><p:cond evt="onNext" delay="0"><p:tgtEl>'
        f'<p:sldTgt/></p:tgtEl></p:cond></p:nextCondLst></p:seq></p:childTnLst></p:cTn></p:par>'
        f'</p:tnLst><p:bldLst>{"".join(builds)}</p:bldLst></p:timing>'
    )
    _insert_after_color_map(slide, timing)


def _insert_after_color_map(slide: Any, element: Any) -> None:
    """Порядок дочірніх `p:sld` фіксований схемою: cSld, clrMapOvr, transition,
    timing, extLst. Перехід ставиться перед таймінгом, таймінг — у кінець
    (але перед extLst)."""
    root = slide._element
    tag = element.tag.rsplit("}", 1)[-1]
    existing = root.find(f"{{{_P}}}timing")
    if tag == "transition" and existing is not None:
        existing.addprevious(element)
        return
    ext = root.find(f"{{{_P}}}extLst")
    if ext is not None:
        ext.addprevious(element)
    else:
        root.append(element)
