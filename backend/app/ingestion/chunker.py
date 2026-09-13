"""Двоступеневе структурне чанкування. Бюджет — у СИМВОЛАХ.

Чому в символах, а не в токенах. Українська fertility коливається від 2.16
(Llama 4) до 3.90 (Qwen 3) токенів на слово — розкид 1.6–1.7×. Той самий
«1000 токенів» — це ~2000 українських символів під токенайзером Qwen, але
~3200 під Gemma. Токен-номінований чанк ТИХО змінює обсяг підручника вдвічі
при зміні embedding-моделі; символьний — ні.

Три матеріалізовані рівні (`ChunkLevel`):
  L0 картка документа — маршрутизація й диверсифікація, НЕ цитується
  L1 секція — стеля 6000 символів, зберігається, НЕ індексується, подається
     генератору через auto-merge (retrieve small, show medium)
  L2 листок — ціль 1400, м'який максимум 2200, жорсткий 3600, мінімум 250.
     ЄДИНИЙ рівень, що потрапляє в ANN-індекс і у FTS5.

Два рішення, які виглядають дрібними й не є такими:

  * Стек заголовків виштовхується за MARKDOWN-РІВНЕМ, а не за глибиною стека.
    Порт із NeoLens (`markdown_node_parser_plus.py:94-97`). Різниця видно
    рівно на стрибку H1→H3: при поп-за-глибиною H3 виштовхнув би H1 і шлях
    став би `//2.3. Балістика//` замість `//Розділ 2//2.3. Балістика//`.
  * Перекриття 0 на справжніх межах і 300 символів ЛИШЕ на рекурсивному
    фолбеку. Перекриття — компенсація за довільний розріз; там, де межа є
    заголовком або кінцем означення, воно чиста втрата: роздуває індекс на
    ~20% і створює майже-дублікати, що з'їдають слоти top-k.

`context_note` — СИРИЙ структурний контекст (назва документа), а не
LLM-конспект: абляція UNLP 2026 показала, що заміна сирого префікса на
згенерований конспект на 80 токенів ПОГІРШУЄ результат (0.9346 → 0.9177).
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from itertools import pairwise

from app.domain import BBox, Chapter, Chunk, ChunkLevel
from app.ingestion.docling_pipeline import (
    ElementKind,
    ParsedDocument,
    ParsedElement,
    ParseOptions,
)
from app.ingestion.normalize_uk import search_terms

__all__ = [
    "ChunkConfig",
    "ChunkTree",
    "chunk_document",
    "chunk_document_tree",
    "chunk_markdown",
    "build_chapters",
    "simhash64",
    "hamming64",
    "calibrate_ch_per_tok",
    "header_path_of",
    "sanitize_display",
    "QWEN_UK_FERTILITY",
    "BLOCK_PLACEHOLDER_RE",
]


# ------------------------------------------------------------------- конфіг
@dataclass(slots=True)
class ChunkConfig:
    """Бюджети з плану, §5. Змінювати лише разом із переіндексацією корпусу."""

    leaf_target: int = 1400
    leaf_soft_max: int = 2200
    leaf_hard_max: int = 3600
    leaf_min: int = 250
    section_max: int = 6000
    fallback_overlap: int = 300

    make_document_card: bool = True
    max_bboxes: int = 32
    language: str = "uk"
    document_id: str = ""
    collection_id: str = ""
    doc_type: str = ""
    ch_per_tok: float = 0.0        # 0 → відкалібрувати на самому документі
    tokenizer: Callable[[str], Sequence[object]] | None = None


# Українська fertility Qwen3 — 3.62–3.90 токена на слово. Беремо нижню межу:
# перевищити бюджет контексту гірше, ніж недовикористати його.
QWEN_UK_FERTILITY = 3.62


# ------------------------------------------------------- атомарні плейсхолдери
BLOCK_PLACEHOLDER_RE = re.compile(r"\[\[\[BLOCK:(\d+)\]\]\]")

# Блоки, які фізично неможливо розрізати: огорожі коду й Mermaid, теги таблиць,
# формул і описів рисунків із надмножинного Markdown.
_ATOMIC_SPAN_RE = re.compile(
    r"```.*?```"
    r"|<table\b[^>]*>.*?</table>"
    r"|<picture\b[^>]*>.*?</picture>"
    r"|<formula\b[^>]*>.*?</formula>",
    re.DOTALL,
)
# Колонтитули не мають їсти бюджет чанка (правило NeoLens «функція довжини,
# що ігнорує шум»).
_FURNITURE_RE = re.compile(
    r"<(header|footer|page_number)\b[^>]*>.*?</\1>|<page_separator\b[^>]*/>",
    re.DOTALL,
)


def _protect_atoms(text: str) -> tuple[str, list[str]]:
    """Замінити атомарні блоки плейсхолдерами перед рекурсивним поділом."""
    blocks: list[str] = []

    def replace(match: re.Match[str]) -> str:
        blocks.append(match.group(0))
        return f"[[[BLOCK:{len(blocks) - 1}]]]"

    return _ATOMIC_SPAN_RE.sub(replace, text), blocks


def _restore_atoms(text: str, blocks: Sequence[str]) -> str:
    def replace(match: re.Match[str]) -> str:
        index = int(match.group(1))
        return blocks[index] if 0 <= index < len(blocks) else match.group(0)

    return BLOCK_PLACEHOLDER_RE.sub(replace, text)


def _measure(text: str, blocks: Sequence[str] = ()) -> int:
    """Довжина, що ігнорує шум: плейсхолдери розпаковуються, колонтитули не рахуються."""
    expanded = _restore_atoms(text, blocks) if blocks else text
    return len(_FURNITURE_RE.sub("", expanded))


def sanitize_display(text: str) -> str:
    """Санітизація тіла, яке побачить генератор.

    В on-device POC NeoLens це була інтервенція з найвищим важелем узагалі:
    коміти «answers aren't great» → «sanitized context and much better answers».
    """
    cleaned = _FURNITURE_RE.sub("", text)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


# --------------------------------------------------------------------- утиліти
def header_path_of(titles: Sequence[str]) -> str:
    """`//Розділ 2//2.3. Балістика//`. Порожній шлях — рівно `//`."""
    clean = [t.strip().replace("/", "∕") for t in titles if t and t.strip()]
    return "//" + "//".join(clean) + "//" if clean else "//"


def simhash64(text: str, *, shingle: int = 3) -> int:
    """SimHash-64 по шинглах нормалізованих токенів — для дедуплікації.

    Повертається ЗНАКОВЕ 64-бітне число: SQLite не має беззнакового INTEGER,
    і 0.5% значень інакше переповнили б колонку при вставці.
    """
    tokens = search_terms(text)
    if not tokens:
        return 0
    grams = (
        [" ".join(tokens[i:i + shingle]) for i in range(len(tokens) - shingle + 1)]
        if len(tokens) >= shingle
        else [" ".join(tokens)]
    )
    weights = [0] * 64
    for gram in grams:
        digest = int.from_bytes(hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest(), "big")
        for bit in range(64):
            weights[bit] += 1 if (digest >> bit) & 1 else -1
    value = 0
    for bit in range(64):
        if weights[bit] > 0:
            value |= 1 << bit
    return value - (1 << 64) if value >= (1 << 63) else value


def hamming64(left: int, right: int) -> int:
    """Відстань Геммінга між двома SimHash. <= 3 — практично дублікат."""
    return bin((left ^ right) & ((1 << 64) - 1)).count("1")


def calibrate_ch_per_tok(
    text: str,
    *,
    tokenizer: Callable[[str], Sequence[object]] | None = None,
    sample_chars: int = 20000,
    fertility: float = QWEN_UK_FERTILITY,
) -> float:
    """Скільки символів припадає на токен у ЦЬОМУ документі.

    Калібрується на 20-тисячносимвольній вибірці. З токенайзером — виміром;
    без нього — з середньої довжини слова: ch_per_tok = (символів на слово,
    з розділювачем) / fertility. Для типової української прози це дає ~1.93,
    що збігається з арифметикою плану (2200 символів ≈ 1140 токенів Qwen).

    Значення не змінює бюджет чанка (він у символах за рішенням) — воно
    потрібне, щоб чесно перерахувати бюджет промпту в токени.
    """
    sample = text[:sample_chars]
    if not sample.strip():
        return 1.0 / fertility * 7.0
    if tokenizer is not None:
        count = len(tokenizer(sample))
        return max(0.5, len(sample) / count) if count else 1.93
    words = len(sample.split())
    if not words:
        return 1.93
    chars_per_word = len(sample) / words
    return min(4.0, max(1.2, chars_per_word / fertility))


# ----------------------------------------------------------- внутрішні одиниці
@dataclass(slots=True)
class _Unit:
    """Неподільна (для пакування) одиниця тіла секції."""

    text: str
    kind: ElementKind
    page_no: int
    char_from: int
    char_to: int
    self_ref: str = ""
    bbox: BBox | None = None
    meta: dict = field(default_factory=dict)

    @property
    def atomic(self) -> bool:
        return self.kind.is_atomic


@dataclass(slots=True)
class _Section:
    """Структурно обмежена секція: заголовок + тіло до наступного заголовка."""

    titles: list[str]
    level: int
    units: list[_Unit] = field(default_factory=list)

    @property
    def header_path(self) -> str:
        return header_path_of(self.titles)


@dataclass(slots=True)
class _Piece:
    """Готовий листок до збирання в `Chunk`."""

    text: str
    units: list[_Unit]
    char_from: int
    char_to: int


@dataclass(slots=True)
class ChunkTree:
    """Чанки плюс зв'язки «дитина → батько» за chunk_uid.

    `Chunk.parent_id` — це INTEGER у БД, а id з'являються лише після вставки.
    Тому чанкер віддає зв'язки за стабільними `chunk_uid`, а ingest-завдання
    після `ChunkRepo.insert_many` викликає `apply_parent_ids()`. Порядок у
    `chunks` гарантує, що батько вставляється раніше за дитину.
    """

    chunks: list[Chunk] = field(default_factory=list)
    parent_uid: dict[str, str] = field(default_factory=dict)
    ch_per_tok: float = 1.93

    def apply_parent_ids(self) -> int:
        """Проставити `parent_id` після вставки. Повертає кількість зв'язків."""
        by_uid = {c.chunk_uid: c for c in self.chunks}
        linked = 0
        for child_uid, parent_uid in self.parent_uid.items():
            child = by_uid.get(child_uid)
            parent = by_uid.get(parent_uid)
            if child is not None and parent is not None and parent.id is not None:
                child.parent_id = parent.id
                linked += 1
        return linked

    def leaves(self) -> list[Chunk]:
        return [c for c in self.chunks if c.level is ChunkLevel.LEAF]


# --------------------------------------------------------------- побудова секцій
def _units_from_elements(elements: Sequence[ParsedElement]) -> list[_Section]:
    """Розкласти елементи на секції, штовхаючи стек заголовків за рівнем."""
    stack: list[tuple[int, str]] = []
    sections: list[_Section] = [_Section(titles=[], level=0)]

    for el in elements:
        if el.kind.is_furniture:
            continue
        if el.kind in (ElementKind.HEADING, ElementKind.TITLE):
            level = el.level or 1
            # ГОЛОВНЕ: виштовхуємо за MARKDOWN-РІВНЕМ, а не за глибиною стека.
            # Саме це робить стрибок H1→H3 коректним.
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, el.text.strip()))
            sections.append(_Section(titles=[t for _, t in stack], level=level))
            continue
        text = el.text.strip()
        if not text:
            continue
        sections[-1].units.append(
            _Unit(
                text=text, kind=el.kind, page_no=el.page_no,
                char_from=el.char_from, char_to=el.char_to,
                self_ref=el.self_ref, bbox=el.bbox, meta=dict(el.meta),
            )
        )
    return [s for s in sections if s.units]


def _split_section(section: _Section, cfg: ChunkConfig) -> list[list[_Unit]]:
    """Розрізати задовгу секцію на L1-блоки по межах елементів (стеля 6000)."""
    blocks: list[list[_Unit]] = []
    current: list[_Unit] = []
    length = 0
    for unit in section.units:
        unit_len = len(unit.text)
        if current and length + unit_len + 2 > cfg.section_max:
            blocks.append(current)
            current, length = [], 0
        current.append(unit)
        length += unit_len + 2
    if current:
        blocks.append(current)
    return blocks


# ------------------------------------------------------------- рекурсивний поділ
def _split_by_separators(
    text: str,
    separators: Sequence[str],
    cfg: ChunkConfig,
    blocks: Sequence[str] = (),
) -> list[str]:
    """Рекурсивний поділ. Перекриття тут НЕ додається — воно накладається один
    раз зверху, інакше кожен рівень рекурсії множив би його.

    Бюджет міряється `_measure`, а не `len`: плейсхолдер `[[[BLOCK:3]]]` — це
    13 символів, а прихована за ним таблиця може бути на 1800. Без розпакування
    чанк із таблицею проковтнув би ще й повний абзац прози понад бюджет.
    """
    if _measure(text, blocks) <= cfg.leaf_soft_max or not separators:
        if _measure(text, blocks) <= cfg.leaf_hard_max or BLOCK_PLACEHOLDER_RE.search(text):
            return [text]
        # Останній рубіж: різати посимвольно. Сюди доходить лише текст без
        # жодного пробілу — таблиця-як-текст або злиплий OCR.
        step = max(1, cfg.leaf_target)
        return [text[i:i + step] for i in range(0, len(text), step)]

    sep = separators[0]
    fragments = text.split(sep)
    groups: list[str] = []
    current = ""
    for fragment in fragments:
        candidate = f"{current}{sep}{fragment}" if current else fragment
        if current and _measure(candidate, blocks) > cfg.leaf_soft_max:
            groups.append(current)
            current = fragment
        else:
            current = candidate
    if current:
        groups.append(current)

    out: list[str] = []
    for group in groups:
        if _measure(group, blocks) > cfg.leaf_soft_max:
            out.extend(_split_by_separators(group, separators[1:], cfg, blocks))
        else:
            out.append(group)
    return [g for g in out if g.strip()]


def _apply_overlap(parts: Sequence[str], overlap: int) -> list[str]:
    """300 символів перекриття — ЛИШЕ тут, на довільному розрізі."""
    if overlap <= 0 or len(parts) < 2:
        return list(parts)
    out = [parts[0]]
    for previous, part in pairwise(parts):
        # Плейсхолдер атомарного блоку з хвоста прибирається: продублювати цілу
        # таблицю чи Mermaid-схему в наступний чанк — це і є те роздування
        # індексу майже-дублікатами, проти якого існує політика перекриття.
        tail = BLOCK_PLACEHOLDER_RE.sub(" ", previous[-overlap:])
        # Не тягнути хвіст, що обривається посеред слова.
        space = tail.find(" ")
        if 0 <= space < len(tail) - 1:
            tail = tail[space + 1:]
        out.append(f"{tail.strip()} {part}".strip())
    return out


def _recursive_split(text: str, cfg: ChunkConfig) -> list[str]:
    """Поділ завеликої НЕатомарної одиниці із захистом атомарних блоків."""
    protected, blocks = _protect_atoms(text)
    parts = _split_by_separators(protected, ("\n\n", "\n", ". ", " "), cfg, blocks)
    parts = _apply_overlap(parts, cfg.fallback_overlap)
    return [_restore_atoms(p, blocks) for p in parts if p.strip()]


def _split_table_unit(unit: _Unit, cfg: ChunkConfig) -> list[str]:
    """Величезну таблицю різати МОЖНА — але тільки по рядках.

    Кожен рядок триплетної серіалізації самоописний (несе власні заголовки
    колонок), тому розріз між рядками не втрачає нічого. Розріз усередині
    рядка втратив би все, тому його тут немає.
    """
    lines = unit.text.split("\n")
    out: list[str] = []
    current: list[str] = []
    length = 0
    for line in lines:
        if current and length + len(line) + 1 > cfg.leaf_soft_max:
            out.append("\n".join(current))
            current, length = [], 0
        current.append(line)
        length += len(line) + 1
    if current:
        out.append("\n".join(current))
    return out


# ------------------------------------------------------------ пакування листків
def _pack_leaves(units: Sequence[_Unit], cfg: ChunkConfig) -> list[_Piece]:
    """Жадібне пакування по межах елементів до цілі 1400 / м'якого максимуму 2200."""
    pieces: list[_Piece] = []
    current: list[_Unit] = []
    length = 0

    def flush() -> None:
        nonlocal current, length
        if not current:
            return
        text = "\n\n".join(u.text for u in current)
        pieces.append(_Piece(
            text=text, units=list(current),
            char_from=min(u.char_from for u in current),
            char_to=max(u.char_to for u in current),
        ))
        current, length = [], 0

    for unit in units:
        unit_len = len(unit.text)

        # Одиниця, що сама не влазить у м'який максимум. Для НЕатомарних це
        # рекурсивний поділ із перекриттям 300; для атомарних — жорсткий
        # максимум 3600, і тільки таблиця може бути розрізана (по рядках).
        oversized = (
            unit_len > cfg.leaf_soft_max if not unit.atomic
            else unit_len > cfg.leaf_hard_max
        )
        if oversized:
            flush()
            if unit.kind is ElementKind.TABLE:
                parts = _split_table_unit(unit, cfg)
            elif unit.atomic:
                # Формула, рисунок, блок коду: жорсткий максимум тут свідомо
                # порушується — розрізаний атомарний блок марний.
                parts = [unit.text]
            else:
                parts = _recursive_split(unit.text, cfg)
            # Зсув усередині одиниці рахується накопиченням за мінусом
            # перекриття. Це наближення (розділювачі й відновлені блоки міняють
            # довжину на одиниці символів), але монотонне й достатнє: діапазон
            # символів потрібен для резолву в сторінки, а не для посимвольної
            # підсвітки — за неї відповідають bbox.
            offset = unit.char_from
            for part in parts:
                pieces.append(_Piece(
                    text=part, units=[unit],
                    char_from=offset, char_to=offset + len(part),
                ))
                offset += max(0, len(part) - cfg.fallback_overlap)
            continue

        if current and length + unit_len + 2 > cfg.leaf_soft_max:
            flush()
        current.append(unit)
        length += unit_len + 2
        if length >= cfg.leaf_target:
            # Ціль досягнуто на СПРАВЖНІЙ межі елемента → перекриття не потрібне.
            flush()
    flush()
    return pieces


def _looks_like_heading_only(text: str) -> bool:
    """Частина «лише із заголовків»: короткі рядки без розділових у кінці."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return True
    return all(len(ln) <= 80 and not ln.endswith((".", ":", ";", "!", "?")) for ln in lines)


def _merge_small(pieces: list[_Piece], cfg: ChunkConfig) -> list[_Piece]:
    """Приклеїти дрібні або суто заголовкові частини до сусідньої.

    Прибирає «сирітські» заголовки й обрізки на 40 символів, які інакше
    з'їдають слоти top-k, нічого не пояснюючи.
    """
    if len(pieces) < 2:
        return pieces
    out: list[_Piece] = []
    for piece in pieces:
        too_small = len(piece.text) < cfg.leaf_min or _looks_like_heading_only(piece.text)
        if out and too_small and len(out[-1].text) + len(piece.text) + 2 <= cfg.leaf_hard_max:
            previous = out[-1]
            out[-1] = _Piece(
                text=f"{previous.text}\n\n{piece.text}",
                units=previous.units + piece.units,
                char_from=min(previous.char_from, piece.char_from),
                char_to=max(previous.char_to, piece.char_to),
            )
            continue
        out.append(piece)

    # Останній прохід уперед: перша частина не мала попередника.
    if (
        len(out) >= 2
        and (len(out[0].text) < cfg.leaf_min or _looks_like_heading_only(out[0].text))
        and len(out[0].text) + len(out[1].text) + 2 <= cfg.leaf_hard_max
    ):
        merged = _Piece(
            text=f"{out[0].text}\n\n{out[1].text}",
            units=out[0].units + out[1].units,
            char_from=min(out[0].char_from, out[1].char_from),
            char_to=max(out[0].char_to, out[1].char_to),
        )
        out = [merged, *out[2:]]
    return out


# ------------------------------------------------------------- збирання Chunk
def _pages_of(units: Sequence[_Unit]) -> tuple[int, int]:
    pages = [u.page_no for u in units] or [1]
    return min(pages), max(pages)


def _bboxes_of(units: Sequence[_Unit], limit: int) -> list[BBox]:
    seen: set[tuple] = set()
    out: list[BBox] = []
    for unit in units:
        if unit.bbox is None:
            continue
        key = (unit.bbox.page, unit.bbox.left, unit.bbox.top, unit.bbox.right, unit.bbox.bottom)
        if key in seen:
            continue
        seen.add(key)
        out.append(unit.bbox)
        if len(out) >= limit:
            break
    return out


def _collect_refs(units: Sequence[_Unit]) -> tuple[dict, list, list]:
    pictures: dict[str, object] = {}
    tables: list[object] = []
    formulas: list[object] = []
    for unit in units:
        if unit.kind is ElementKind.PICTURE:
            pictures[unit.self_ref or f"pic-{len(pictures)}"] = {
                "page": unit.page_no,
                "classification": unit.meta.get("classification"),
                "description": unit.text[:400],
            }
        elif unit.kind is ElementKind.TABLE:
            tables.append({"ref": unit.self_ref, "page": unit.page_no,
                           "rows": len(unit.meta.get("grid") or [])})
        elif unit.kind is ElementKind.FORMULA:
            formulas.append({"ref": unit.self_ref, "page": unit.page_no, "latex": unit.text[:400]})
    return pictures, tables, formulas


def _document_card(parsed: ParsedDocument, cfg: ChunkConfig) -> str:
    """L0: сирий, детермінований, БЕЗ виклику LLM.

    Скелет змісту й перші абзаци — це той самий структурний контекст, який в
    абляції UNLP 2026 переміг згенерований конспект.
    """
    lines = [f"Документ: {parsed.title}"]
    if cfg.doc_type:
        lines.append(f"Тип: {cfg.doc_type}")
    lines.append(f"Мова: {parsed.language}. Сторінок: {parsed.page_count}.")
    headings = [h.text.strip() for h in parsed.headings() if h.text.strip()][:30]
    if headings:
        lines.append("Зміст: " + "; ".join(headings))
    body = next(
        (e.text.strip() for e in parsed.body_elements()
         if e.kind is ElementKind.PARAGRAPH and len(e.text.strip()) > 80),
        "",
    )
    if body:
        lines.append(body[:400])
    return "\n".join(lines)


def chunk_document_tree(
    parsed: ParsedDocument,
    cfg: ChunkConfig | None = None,
    *,
    document_id: str = "",
    collection_id: str = "",
) -> ChunkTree:
    """Повне чанкування документа з деревом батьківства."""
    cfg = cfg or ChunkConfig()
    document_id = document_id or cfg.document_id
    collection_id = collection_id or cfg.collection_id

    ch_per_tok = cfg.ch_per_tok or calibrate_ch_per_tok(
        parsed.full_text or "\n".join(e.text for e in parsed.elements),
        tokenizer=cfg.tokenizer,
    )
    # Сирий структурний контекст: назва документа. Шлях заголовків додає сам
    # `Chunk.build_embed_text`, тому дублювати його тут не можна.
    context_note = parsed.title.strip()

    tree = ChunkTree(ch_per_tok=ch_per_tok)
    ordinal = 0
    card: Chunk | None = None

    if cfg.make_document_card:
        card = Chunk(
            document_id=document_id, collection_id=collection_id, ordinal=ordinal,
            level=ChunkLevel.DOCUMENT_CARD,
            display_text=sanitize_display(_document_card(parsed, cfg)),
            context_note=context_note, header_path="//", language=cfg.language,
            page_from=1, page_to=max(1, parsed.page_count),
            page_label_from=parsed.label_for(1),
            page_label_to=parsed.label_for(max(1, parsed.page_count)),
            char_from=0, char_to=len(parsed.full_text),
        )
        card.simhash = simhash64(card.display_text)
        ordinal += 1

    sections = _units_from_elements(parsed.body_elements())

    # Спершу всі L1, потім усі L2 — щоб ordinal листків був НЕПЕРЕРВНИМ.
    # Від цього залежить `ChunkRepo.neighbours` (auto-merge шукає сусідів за
    # `ordinal BETWEEN ordinal-radius AND ordinal+radius`).
    sections_blocks: list[tuple[_Section, list[_Unit]]] = []
    for section in sections:
        for block in _split_section(section, cfg):
            sections_blocks.append((section, block))

    parents: list[Chunk] = []
    for section, block in sections_blocks:
        page_from, page_to = _pages_of(block)
        pictures, tables, formulas = _collect_refs(block)
        parent = Chunk(
            document_id=document_id, collection_id=collection_id, ordinal=ordinal,
            level=ChunkLevel.SECTION,
            display_text=sanitize_display("\n\n".join(u.text for u in block)),
            context_note=context_note, header_path=section.header_path, language=cfg.language,
            page_from=page_from, page_to=page_to,
            page_label_from=parsed.label_for(page_from), page_label_to=parsed.label_for(page_to),
            char_from=min(u.char_from for u in block), char_to=max(u.char_to for u in block),
            bboxes=_bboxes_of(block, cfg.max_bboxes),
            pictures=pictures, tables=tables, formulas=formulas,
        )
        parent.simhash = simhash64(parent.display_text)
        parents.append(parent)
        ordinal += 1

    leaves: list[Chunk] = []
    for parent, (section, block) in zip(parents, sections_blocks, strict=True):
        pieces = _merge_small(_pack_leaves(block, cfg), cfg)
        for index, piece in enumerate(pieces):
            page_from, page_to = _pages_of(piece.units)
            pictures, tables, formulas = _collect_refs(piece.units)
            leaf = Chunk(
                document_id=document_id, collection_id=collection_id, ordinal=ordinal,
                level=ChunkLevel.LEAF,
                display_text=sanitize_display(piece.text),
                context_note=context_note, header_path=section.header_path,
                language=cfg.language,
                page_from=page_from, page_to=page_to,
                page_label_from=parsed.label_for(page_from),
                page_label_to=parsed.label_for(page_to),
                char_from=piece.char_from, char_to=piece.char_to,
                bboxes=_bboxes_of(piece.units, cfg.max_bboxes),
                siblings_index=index, siblings_count=len(pieces),
                pictures=pictures, tables=tables, formulas=formulas,
            )
            leaf.simhash = simhash64(leaf.display_text)
            leaves.append(leaf)
            tree.parent_uid[leaf.chunk_uid] = parent.chunk_uid
            ordinal += 1

    if card is not None:
        tree.chunks.append(card)
        for parent in parents:
            tree.parent_uid[parent.chunk_uid] = card.chunk_uid
    tree.chunks.extend(parents)
    tree.chunks.extend(leaves)
    return tree


def chunk_document(
    parsed: ParsedDocument,
    cfg: ChunkConfig | None = None,
    *,
    document_id: str = "",
    collection_id: str = "",
) -> list[Chunk]:
    """Публічний інтерфейс модуля (див. docs/CONTRACT.md).

    Порядок результату: L0, далі всі L1, далі всі L2 — тобто батьки завжди
    раніше за дітей, а ordinal листків неперервний. Зв'язки батьківства беруть
    із `chunk_document_tree`, бо `parent_id` існує лише після вставки в БД.
    """
    return chunk_document_tree(
        parsed, cfg, document_id=document_id, collection_id=collection_id
    ).chunks


def chunk_markdown(
    text: str,
    *,
    title: str = "",
    cfg: ChunkConfig | None = None,
    document_id: str = "",
    collection_id: str = "",
) -> ChunkTree:
    """Зручний вхід із сирого Markdown — для тестів, конспектів і golden-файлів."""
    from app.ingestion.docling_pipeline import _parse_text_source

    parsed = _parse_text_source(text, ParseOptions(title=title), source_path=title or "markdown")
    return chunk_document_tree(parsed, cfg, document_id=document_id, collection_id=collection_id)


# ------------------------------------------------------------------- розділи
def build_chapters(parsed: ParsedDocument, document_id: str) -> list[Chapter]:
    """Дерево розділів у вигляді вкладених множин (lft/rgt).

    Саме така форма робить запит «усе під Розділом 2» range scan'ом по індексу,
    а не LIKE по `header_path`. `parent_id` тут — індекс у поверненому списку;
    справжні id проставляє `ChapterRepo.replace_for_document`.
    """
    chapters: list[Chapter] = []
    stack: list[tuple[int, int]] = []  # (markdown-рівень, індекс у chapters)
    counter = 0

    for element in parsed.elements:
        if element.kind not in (ElementKind.HEADING, ElementKind.TITLE):
            continue
        level = element.level or 1
        counter += 1
        while stack and stack[-1][0] >= level:
            _, closed = stack.pop()
            chapters[closed].rgt = counter
            counter += 1
        parent_index = stack[-1][1] if stack else None
        chapters.append(Chapter(
            id=None, document_id=document_id, parent_id=parent_index,
            level=level, title=element.text.strip(), lft=counter, rgt=counter,
        ))
        stack.append((level, len(chapters) - 1))

    while stack:
        counter += 1
        _, closed = stack.pop()
        chapters[closed].rgt = counter
    return chapters
