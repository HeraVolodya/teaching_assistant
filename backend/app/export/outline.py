"""Markdown відповіді → нейтральна структура документа.

ЧОМУ MARKDOWN, А НЕ JSON-СХЕМА
Модель пише Markdown сама (заголовки, списки, таблиці), і чат уже вміє його
показувати (`frontend/src/lib/markdown.ts`). Файл збирається з ТОГО САМОГО
тексту, що збережено в повідомленні, тож завантажений документ завжди
збігається з тим, що викладач бачив у чаті й перевіряв за посиланнями.

Структурований вивід через `response_format` перевірено й відкинуто (заміри
23.11.2026, Gemma 4 E4B у LM Studio): перший токен через 20–25 с замість ~1 с
(граматику треба скомпілювати), один із трьох прогонів обірвався посеред рядка
й дав невалідний JSON, а таблиця видів вогню мала 4 рядки замість 10. Markdown
на тих самих джерелах: перший токен за 0,8–1 с, усі 10 рядків, і текст іде в
чат одразу, а не після останньої дужки.

РОЗБІР ПОБЛАЖЛИВИЙ. Мала модель пише Markdown «приблизно»: обгортає його в
```, ставить «**Розділ**» замість заголовка, дописує власний список джерел.
Усе це тут вирівнюється, бо інакше викладач отримав би файл із зірочками.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

__all__ = [
    "MARKER_RE",
    "Block",
    "Outline",
    "markers",
    "parse_markdown",
    "plain",
    "runs",
    "strip_markers",
    "strip_preamble",
]

BlockKind = Literal["heading", "paragraph", "bullets", "numbered", "table"]

# Той самий формат маркера, що й у `generation/citations.py`. Форму [1, 2]
# генерація вже розгорнула в [1][2] (`parse_citations`), тож тут лише одиничні.
MARKER_RE = re.compile(r"\[(\d{1,3})\]")


@dataclass(slots=True)
class Block:
    kind: BlockKind
    text: str = ""                                    # heading, paragraph
    level: int = 0                                    # heading: 1..6
    items: list[str] = field(default_factory=list)    # bullets, numbered
    # Рівень кожного пункту: 0 — верхній, 1 — підпункт. Глибше не розрізняємо:
    # `parse_citations` згортає відступи до одного пробілу, тож «є відступ» —
    # єдиний сигнал, що вцілів.
    depths: list[int] = field(default_factory=list)
    header: list[str] = field(default_factory=list)   # table
    rows: list[list[str]] = field(default_factory=list)

    def depth(self, index: int) -> int:
        return self.depths[index] if index < len(self.depths) else 0


@dataclass(slots=True)
class Outline:
    title: str
    blocks: list[Block]

    @property
    def has_content(self) -> bool:
        """Є що класти у файл: самі лише заголовки документом не є."""
        return any(
            (b.kind == "table" and (b.rows or b.header))
            or (b.kind in ("bullets", "numbered") and b.items)
            or (b.kind == "paragraph" and b.text.strip())
            for b in self.blocks
        )

    def text(self) -> str:
        """Увесь текст підряд — щоб зібрати маркери [n] для списку джерел."""
        parts = [self.title]
        for b in self.blocks:
            parts.append(b.text)
            parts.extend(b.items)
            parts.extend(b.header)
            parts.extend(cell for row in b.rows for cell in row)
        return "\n".join(parts)


# ------------------------------------------------------------------ рядки
_FENCE = re.compile(r"^\s*```")
_HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$")
_BOLD_LINE = re.compile(r"^\s*(?:\*\*|__)(.{2,160}?)(?:\*\*|__)\s*:?\s*$")
_BULLET = re.compile(r"^(\s*)[-*•+–]\s+(.*)$")
_NUMBERED = re.compile(r"^(\s*)\d{1,3}[.)]\s+(.*)$")
_TABLE_LINE = re.compile(r"^\s*\|")
_SEPARATOR_CELL = re.compile(r"^:?-{2,}:?$")
_HTML_BREAK = re.compile(r"<br\s*/?>", re.IGNORECASE)
_INDENTED = re.compile(r"^\s{2,}\S")

# Власний список джерел моделі. Справжній додає збирач файлу — з назвами й
# сторінками з бази, а не з пам'яті моделі.
_OWN_SOURCES = re.compile(
    r"^(?:джерела|список\s+джерел|використані\s+джерела|література|список\s+літератури|"
    r"джерела\s+інформації|посилання)\s*:?$",
    re.IGNORECASE,
)


_H1 = re.compile(r"^\s{0,3}#\s+\S")
_WRAPPER_FENCE = re.compile(r"^\s*```\s*(?:markdown|md)?\s*$", re.IGNORECASE)


def strip_preamble(text: str) -> str:
    """Відрізати вступ перед «# Назва» і обгортку ```markdown.

    Побачивши в проханні «файл PPTX», мала модель починає з «Я не можу
    створити файл формату PPTX… Однак ось структура презентації:» — і далі
    пише рівно те, що треба (живий випадок 23.11.2026). У файлі й в історії
    такий вступ лише плутає: файл же створено. Ріжемо лише кілька звичайних
    абзаців перед першим «# »; заголовки, списки чи таблиці перед ним — це вже
    зміст, і його не чіпаємо.
    """
    lines = [line for line in (text or "").split("\n") if not _WRAPPER_FENCE.match(line)]
    for index, line in enumerate(lines):
        if _H1.match(line):
            head = [h for h in lines[:index] if h.strip()]
            plain_head = all(
                not (_HEADING.match(h) or _BULLET.match(h) or _NUMBERED.match(h) or _TABLE_LINE.match(h))
                for h in head
            )
            if head and plain_head and len(head) <= 3:
                lines = lines[index:]
            break
    return "\n".join(lines).strip()


def parse_markdown(text: str, *, fallback_title: str = "") -> Outline:
    """Розібрати Markdown. Перший заголовок рівня 1 стає назвою документа."""
    text = strip_preamble(text)
    blocks: list[Block] = []
    paragraph: list[str] = []
    table: list[list[str]] = []

    def flush_paragraph() -> None:
        if paragraph:
            joined = " ".join(p.strip() for p in paragraph).strip()
            if joined:
                blocks.append(Block("paragraph", text=joined))
            paragraph.clear()

    def flush_table() -> None:
        if table:
            header, *rows = table
            width = len(header)
            fixed = [_fit_row(row, width) for row in rows]
            blocks.append(Block("table", header=header, rows=fixed))
            table.clear()

    def add_item(kind: BlockKind, item: str, indent: str) -> None:
        depth = 1 if indent else 0
        if blocks and blocks[-1].kind == kind:
            blocks[-1].items.append(item)
            blocks[-1].depths.append(depth)
        else:
            blocks.append(Block(kind, items=[item], depths=[depth]))

    for raw in (text or "").replace("\r\n", "\n").split("\n"):
        line = _HTML_BREAK.sub(" ", raw).rstrip()
        if _FENCE.match(line):
            # Обгортка ```markdown … ``` — не зміст.
            continue
        if not line.strip():
            flush_paragraph()
            flush_table()
            continue

        if _TABLE_LINE.match(line):
            flush_paragraph()
            cells = _cells(line)
            if cells and all(_SEPARATOR_CELL.match(c.replace(" ", "")) for c in cells if c):
                continue
            if cells:
                table.append(cells)
            continue
        flush_table()

        if m := _HEADING.match(line):
            flush_paragraph()
            blocks.append(Block("heading", text=m.group(2).strip(), level=len(m.group(1))))
            continue
        if m := _BOLD_LINE.match(line):
            # «**Загальні положення:**» окремим рядком — це підзаголовок.
            flush_paragraph()
            blocks.append(Block("heading", text=m.group(1).strip().rstrip(":"), level=3))
            continue
        if m := _BULLET.match(line):
            flush_paragraph()
            add_item("bullets", m.group(2).strip(), m.group(1))
            continue
        if m := _NUMBERED.match(line):
            flush_paragraph()
            add_item("numbered", m.group(2).strip(), m.group(1))
            continue
        if _INDENTED.match(line) and blocks and blocks[-1].kind in ("bullets", "numbered") \
                and not paragraph:
            # Продовження пункту списку на новому рядку.
            blocks[-1].items[-1] = f"{blocks[-1].items[-1]} {line.strip()}"
            continue
        paragraph.append(line)

    flush_paragraph()
    flush_table()

    blocks = _drop_own_sources(blocks)
    title = ""
    if blocks and blocks[0].kind == "heading" and blocks[0].level == 1:
        title = blocks.pop(0).text
    title = plain(strip_markers(title)) or plain(strip_markers(fallback_title)) or "Документ"
    return Outline(title=title, blocks=blocks)


def _cells(line: str) -> list[str]:
    body = line.strip()
    if body.startswith("|"):
        body = body[1:]
    if body.endswith("|"):
        body = body[:-1]
    return [c.strip() for c in body.split("|")]


def _fit_row(row: list[str], width: int) -> list[str]:
    """Рядок під ширину шапки: зайві комірки — в останню, відсутні — порожні."""
    if width <= 0:
        return row
    if len(row) > width:
        return [*row[: width - 1], " ".join(c for c in row[width - 1:] if c)]
    return row + [""] * (width - len(row))


def _drop_own_sources(blocks: list[Block]) -> list[Block]:
    out: list[Block] = []
    skipping_level: int | None = None
    for b in blocks:
        if b.kind == "heading":
            if skipping_level is not None and b.level <= skipping_level:
                skipping_level = None
            if skipping_level is None and _OWN_SOURCES.match(plain(strip_markers(b.text))):
                skipping_level = b.level
                continue
        if skipping_level is not None:
            continue
        if b.kind == "paragraph" and re.match(r"^\s*(?:\*\*)?джерела(?:\*\*)?\s*:", b.text, re.I) \
                and not MARKER_RE.sub("", b.text.split(":", 1)[1]).strip(" ,;."):
            # «Джерела: [1], [3]» наприкінці — дубль справжнього списку.
            continue
        out.append(b)
    return out


# --------------------------------------------------------------- інлайн
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?…»)])")
_MARKER_GROUP = re.compile(r"\s*\[\d{1,3}\](?:\s*[,;]?\s*\[\d{1,3}\])*")
_INLINE = re.compile(r"\*\*(.+?)\*\*|__(.+?)__|\*(?![\s*])(.+?)(?<![\s*])\*|`([^`]+)`")


# Формули LaTeX ($\alpha$, $V_0$): модель копіює їх із джерел, бо Docling
# витягує формули підручників саме так. Чат малює їх KaTeX-ом, а у Word,
# PowerPoint чи Excel лишився б сирий «$\alpha$ осі». Тут — найуживаніше
# символами Юнікоду; рідкісне просто втрачає службові знаки, а не текст.
_MATH = re.compile(r"\$\$?([^$\n]+?)\$\$?")
_MATH_SYMBOLS = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "Delta": "Δ", "epsilon": "ε",
    "varepsilon": "ε", "theta": "θ", "lambda": "λ", "mu": "μ", "pi": "π", "rho": "ρ",
    "sigma": "σ", "Sigma": "Σ", "tau": "τ", "phi": "φ", "varphi": "φ", "omega": "ω",
    "Omega": "Ω", "cdot": "·", "times": "×", "pm": "±", "approx": "≈", "le": "≤", "leq": "≤",
    "ge": "≥", "geq": "≥", "neq": "≠", "ne": "≠", "circ": "°", "degree": "°", "infty": "∞",
    "rightarrow": "→", "to": "→", "sim": "~", "div": "÷", "percent": "%", "%": "%",
}
_SUPERSCRIPT = str.maketrans("0123456789+-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻")
_SUBSCRIPT = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def _formula(match: re.Match[str]) -> str:
    expr = match.group(1)
    # «$5 і 10$» — це гроші, а не формула: формула має службові знаки або коротка.
    if not re.search(r"[\\^_{]", expr) and (re.search(r"\s", expr) or len(expr) > 12):
        return match.group(0)
    expr = re.sub(r"\\(?:text|mathrm|mathbf|mathit|operatorname|textbf|textit)\s*\{([^}]*)\}", r"\1", expr)
    expr = re.sub(r"\\frac\s*\{([^}]*)\}\s*\{([^}]*)\}", r"(\1)/(\2)", expr)
    expr = re.sub(r"\\sqrt\s*\{([^}]*)\}", r"√(\1)", expr)
    # Пробіл після команди в LaTeX — лише роздільник: «\Delta Д» — це «ΔД».
    expr = re.sub(r"\\([A-Za-z]+|%) ?", lambda m: _MATH_SYMBOLS.get(m.group(1), m.group(1)), expr)
    expr = re.sub(r"\^\{?([0-9+-]+)\}?", lambda m: m.group(1).translate(_SUPERSCRIPT), expr)
    expr = re.sub(r"_\{?([0-9]+)\}?", lambda m: m.group(1).translate(_SUBSCRIPT), expr)
    expr = re.sub(r"[_^]\{([^}]*)\}", r"\1", expr)
    expr = expr.replace("{", "").replace("}", "").replace("\\,", " ").replace("\\", "")
    return re.sub(r"\s+", " ", expr).strip()


def plain_math(text: str) -> str:
    """Формули $…$ → символи Юнікоду: «$\\alpha$ осі» → «α осі», «$V_0$» → «V₀»."""
    return _MATH.sub(_formula, text or "") if "$" in (text or "") else (text or "")


def markers(text: str) -> list[int]:
    """Номери джерел у порядку першої появи."""
    seen: list[int] = []
    for m in MARKER_RE.finditer(text or ""):
        n = int(m.group(1))
        if n not in seen:
            seen.append(n)
    return seen


def strip_markers(text: str) -> str:
    # Групу «[2], [3]» прибираємо разом із розділювачами — інакше від заголовка
    # «Топогеодезична підготовка [2], [3]» лишалась би кома в кінці.
    cleaned = _MARKER_GROUP.sub("", text or "")
    cleaned = _SPACE_BEFORE_PUNCT.sub(r"\1", cleaned)
    return re.sub(r"[ \t]{2,}", " ", cleaned).strip()


def plain(text: str) -> str:
    """Текст без розмітки: для назв файлів, заголовків, комірок Excel."""
    out = _INLINE.sub(lambda m: next(g for g in m.groups() if g is not None), plain_math(text))
    return re.sub(r"\s+", " ", out).strip()


def runs(text: str) -> list[tuple[str, bool, bool]]:
    """(шматок, жирний, курсив) — для Word і PowerPoint, де розмітка — це стиль."""
    text = plain_math(text)
    out: list[tuple[str, bool, bool]] = []
    last = 0
    for m in _INLINE.finditer(text or ""):
        if m.start() > last:
            out.append((text[last:m.start()], False, False))
        bold, bold_alt, italic, code = m.groups()
        if bold is not None or bold_alt is not None:
            out.append((bold if bold is not None else bold_alt, True, False))
        elif italic is not None:
            out.append((italic, False, True))
        else:
            out.append((code or "", False, False))
        last = m.end()
    if last < len(text or ""):
        out.append((text[last:], False, False))
    return [r for r in out if r[0]]
