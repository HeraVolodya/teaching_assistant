"""Обгортка Docling + текстовий фолбек + stub.

Docling — ядро приймання (єдиний зрілий MIT-рушій із посторінковою атрибуцією,
типізованим документом і посторінковими оцінками впевненості), але його
дефолти для української небезпечні, а сам пакет тягне torch. Тому:

  * `import docling` — ОПЦІЙНИЙ і локальний усередині функцій. API-процес не
    має його бачити ніколи (див. docs/CONTRACT.md, правило 1);
  * без docling модуль працює: txt/md парсяться власним конвеєром, а
    ASISTENT_STUB=1 дає детермінований синтетичний документ для будь-чого.

П'ять дефолтів Docling, які тут знешкоджено (кожен — окрема задокументована
пастка з плану, §1):

  1. `OcrAutoOptions` — НІКОЛИ. Порожній `lang` резолвиться на Windows у
     RapidOCR із дефолтом ["chinese"] (китайські моделі на українському
     тексті!), а на macOS — в ocrmac із ["fr-FR","de-DE","es-ES","en-US"],
     де української немає взагалі. Завжди конструюємо опції явно.
  2. `HeadingHierarchyOptions(enabled=True)` — без нього КОЖЕН заголовок PDF
     отримує level=1, ієрархія стає пласкою і `header_path`, на якому
     тримається все чанкування, стає беззмістовним.
  3. `generate_parsed_pages=True` — без нього `use_style` у детекторі
     заголовків тихо не працює. Саме тихо: помилки немає, результат гірший.
  4. TableFormer V1 `mode=ACCURATE` — у V2 дві відкриті регресії (#3158, #3553).
     На macOS дефолт зсунуто у FAST: у Docling є жорсткий guard
     `if device == MPS: device = CPU`, тому TableFormer на кожному Mac працює
     на CPU, і різниця виміряна як 10.4 с проти 145.9 с (issue #3202).
  5. `do_formula_enrichment` і `do_picture_classification` вимкнені з коробки,
     а нам потрібні обидва.

І шосте, що не є дефолтом, але є архітектурним рішенням: `DocumentConverter.convert()`
НЕ має колбека прогресу. Тому документ ріжеться на вікна по 25 сторінок через
`page_range`, а результати зчеплюються `DoclingDocument.concatenate`. Це один
трюк, що дає прогрес, скасування, відновлюваність і обмежену пам'ять одразу.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from app.domain import BBox, IngestMode, OcrModeName, PageClass
from app.ingestion.normalize_uk import TEXT_PREPROC_VERSION
from app.ingestion.page_labels import PageLabelMap, resolve_page_labels
from app.ingestion.probe import ProbeReport, probe_pdf, probe_text, window_boundaries

__all__ = [
    "DOCLING_VERSION",
    "SUPPORTED_TEXT_SUFFIXES",
    "ElementKind",
    "ParseOptions",
    "ParsedDocument",
    "ParsedElement",
    "ParsedPage",
    "build_parse_profile",
    "is_stub_mode",
    "ocr_options_spec",
    "page_windows",
    "parse_document",
    "parse_profile_hash",
    "serialize_table_triplets",
]

log = logging.getLogger(__name__)

# Версія, під яку писався й перевірявся цей код. Входить у parse_profile_hash,
# тому оновлення Docling автоматично інвалідовує кеш документів.
DOCLING_VERSION = "2.126.0"

SUPPORTED_TEXT_SUFFIXES = frozenset({".txt", ".md", ".markdown", ".text"})

WINDOW_SIZE = 25
TEXT_PAGE_CHARS = 1800


def is_stub_mode() -> bool:
    """ASISTENT_STUB=1 — детермінований режим без жодної завантаженої моделі."""
    return os.environ.get("ASISTENT_STUB", "") == "1"


# --------------------------------------------------------------------- модель
class ElementKind(str, Enum):
    """Мітки елементів. Перекриваються з `DocItemLabel` Docling, але свої:
    модуль приймання не має протікати типами стороннього пакета далі за течією."""

    TITLE = "title"
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE = "table"
    PICTURE = "picture"
    FORMULA = "formula"
    CODE = "code"
    CAPTION = "caption"
    FOOTNOTE = "footnote"
    PAGE_HEADER = "page_header"
    PAGE_FOOTER = "page_footer"

    @property
    def is_atomic(self) -> bool:
        """Атомарні блоки фізично не можна розрізати чанкером."""
        return self in (
            ElementKind.TABLE, ElementKind.PICTURE, ElementKind.FORMULA, ElementKind.CODE
        )

    @property
    def is_furniture(self) -> bool:
        """Колонтитули не входять у тіло чанка і не їдять його бюджет."""
        return self in (ElementKind.PAGE_HEADER, ElementKind.PAGE_FOOTER)


@dataclass(slots=True)
class ParsedElement:
    """Один елемент документа з повною позиційною атрибуцією.

    `page_no` і `bbox` заповнюються з першого дня: без них підсвітка цитованого
    фрагмента в PDF неможлива, а доробка потім вимагає повної переіндексації.
    """

    kind: ElementKind
    text: str
    page_no: int = 1
    level: int | None = None          # рівень markdown-заголовка, 1..6
    bbox: BBox | None = None
    self_ref: str = ""
    char_from: int = 0
    char_to: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def is_atomic(self) -> bool:
        return self.kind.is_atomic


@dataclass(slots=True)
class ParsedPage:
    page_number: int
    width: float = 0.0
    height: float = 0.0
    page_label: str | None = None
    page_class: PageClass = PageClass.DIGITAL_CLEAN
    ocr_mode: OcrModeName = OcrModeName.DEFAULT
    cost_weight: float = 1.0
    parse_score: float | None = None
    layout_score: float | None = None
    table_score: float | None = None
    ocr_score: float | None = None
    lexicon_hit_rate: float | None = None
    cyrillic_ratio: float | None = None
    mojibake_ratio: float | None = None
    char_from: int = 0
    char_to: int = 0


@dataclass(slots=True)
class ParsedDocument:
    """Результат приймання. Канонічний артефакт: із нього чанки перевиводяться
    без повторного парсингу, тож експерименти з чанкуванням майже безкоштовні."""

    source_path: str = ""
    title: str = ""
    language: str = "uk"
    backend: str = "text"
    parse_profile_hash: str = ""
    pages: list[ParsedPage] = field(default_factory=list)
    elements: list[ParsedElement] = field(default_factory=list)
    page_labels: PageLabelMap = field(default_factory=PageLabelMap)
    full_text: str = ""
    warnings: list[str] = field(default_factory=list)
    probe: ProbeReport | None = None

    @property
    def page_count(self) -> int:
        return len(self.pages)

    def headings(self) -> list[ParsedElement]:
        return [e for e in self.elements if e.kind is ElementKind.HEADING]

    def tables(self) -> list[ParsedElement]:
        return [e for e in self.elements if e.kind is ElementKind.TABLE]

    def pictures(self) -> list[ParsedElement]:
        return [e for e in self.elements if e.kind is ElementKind.PICTURE]

    def formulas(self) -> list[ParsedElement]:
        return [e for e in self.elements if e.kind is ElementKind.FORMULA]

    def label_for(self, page_no: int) -> str:
        return self.page_labels.label_for(page_no)

    def body_elements(self) -> list[ParsedElement]:
        """Усе, крім колонтитулів. Саме це чанкує `chunker`."""
        return [e for e in self.elements if not e.kind.is_furniture]

    def to_markdown(self) -> str:
        """Надмножинний Markdown — артефакт для діагностики й золотих файлів.

        Не є входом чанкера: чанкер працює зі списком елементів, бо той несе
        page_no і bbox на кожному вузлі, а Markdown їх втрачає.
        """
        out: list[str] = []
        page = 0
        for el in self.elements:
            if el.page_no != page:
                page = el.page_no
                out.append(f"<page_separator page=\"{page}\" label=\"{self.label_for(page)}\"/>")
            if el.kind is ElementKind.HEADING or el.kind is ElementKind.TITLE:
                out.append("#" * (el.level or 1) + " " + el.text)
            elif el.kind is ElementKind.TABLE:
                out.append(f"<table id=\"{el.self_ref}\" page=\"{el.page_no}\">\n{el.text}\n</table>")
            elif el.kind is ElementKind.PICTURE:
                out.append(f"<picture id=\"{el.self_ref}\" page=\"{el.page_no}\">{el.text}</picture>")
            elif el.kind is ElementKind.FORMULA:
                out.append(f"<formula id=\"{el.self_ref}\">{el.text}</formula>")
            elif el.kind is ElementKind.PAGE_HEADER:
                out.append(f"<header>{el.text}</header>")
            elif el.kind is ElementKind.PAGE_FOOTER:
                out.append(f"<footer>{el.text}</footer>")
            else:
                out.append(el.text)
        return "\n\n".join(out)


@dataclass(slots=True)
class ParseOptions:
    """Опції приймання. Усе, що входить у `parse_profile_hash`, живе тут."""

    ingest_mode: IngestMode = IngestMode.FAST
    do_ocr: bool = True
    ocr_langs: tuple[str, ...] | None = None       # None → дефолт платформи
    ocr_scale: float | None = None                 # None → вивести з DPI сторінки
    table_mode: Literal["ACCURATE", "FAST"] | None = None   # None → дефолт платформи
    do_table_structure: bool = True
    do_cell_matching: bool = True
    # ЗА ЗАМОВЧУВАННЯМ ВИКЛЮЧЕНО, І ЦЕ НАВМИСНО.
    # `do_formula_enrichment` запускає CodeFormulaV2 — VLM — на КОЖНОМУ
    # FORMULA-регіоні. Виміряно на цьому проєкті (Apple Silicon, CPU):
    # 2.88 ток/с, ~25 с на зображення, 401 с на вікно з 25 сторінок, тобто
    # ~40 хвилин на 140-сторінковий звіт. План обіцяє «Швидко» = 0.4–0.7 с
    # на сторінку; з увімкненим збагаченням виходило 16 с.
    # Раніше тут стояло `True`, а конвеєр не перевизначав значення за режимом,
    # тому «Швидко» насправді був повільнішим за «Поглиблено» у сподіваннях
    # користувача. Тепер дороге збагачення вмикає ЛИШЕ `for_mode(DEEP)`.
    do_formula_enrichment: bool = False
    # Класифікація рисунків лишається: це маленький класифікатор, і він
    # потрібен саме щоб ГЕЙТУВАТИ дорогий VLM-опис (зрізає 60–80% викликів).
    do_picture_classification: bool = True
    heading_hierarchy: bool = True
    generate_parsed_pages: bool = True
    window_size: int = WINDOW_SIZE
    text_page_chars: int = TEXT_PAGE_CHARS
    num_threads: int = 0                            # 0 → cpu_count() - 1
    device: Literal["cpu", "cuda", "mps"] = "cpu"   # GPU зайнято LM Studio
    artifacts_path: Path | None = None
    max_pages: int | None = None
    title: str = ""
    language: str = "uk"
    cancel: Callable[[], bool] | None = None
    progress: Callable[[float, float], None] | None = None

    @classmethod
    def for_mode(cls, mode: IngestMode, **overrides: Any) -> ParseOptions:
        """Єдина точка, де режим приймання визначає склад конвеєра.

        «Швидко» мусить бути ШВИДКО: layout + OCR + таблиці, усе детерміноване.
        «Поглиблено» додає дорогі VLM-проходи — розпізнавання формул і опис
        рисунків, — і саме тому в інтерфейсі попереджає про тривалість.

        Раніше цієї функції не було, дефолти вмикали збагачення завжди, а
        конвеєр їх не перевизначав: «Швидко» на 140 сторінках працював ~40
        хвилин замість очікуваних ~2. Різниця між режимами існувала лише в
        назві кнопки.
        """
        deep = mode is IngestMode.DEEP
        params: dict[str, Any] = {
            "ingest_mode": mode,
            "do_formula_enrichment": deep,
            "do_picture_classification": True,
        }
        params.update(overrides)
        return cls(**params)

    def resolved_threads(self) -> int:
        return self.num_threads or max(1, (os.cpu_count() or 2) - 1)

    def resolved_table_mode(self) -> str:
        """На macOS TableFormer жорстко падає на CPU через guard у самому
        Docling (`if device == MPS: device = CPU`), тож ACCURATE там коштує
        в ~14 разів дорожче. Дефолт для Mac — FAST, з можливістю перекрити."""
        if self.table_mode is not None:
            return self.table_mode
        return "FAST" if sys.platform == "darwin" else "ACCURATE"


# ------------------------------------------------------- профіль і його хеш
def ocr_options_spec(opts: ParseOptions) -> dict[str, Any]:
    """ЯВНА специфікація OCR. Ніколи `auto`.

    Windows → RapidOCR PP-OCRv5 East-Slavic (uk/ru/be), backend onnxruntime.
    RapidOCR бере ЛИШЕ ПЕРШУ мову зі списку — тому список тут із одного
    елемента, а не з трьох.
    macOS  → Apple Vision через ocrmac; там мультимовність справжня.
    """
    if not opts.do_ocr:
        return {"engine": "none", "lang": []}
    if sys.platform == "darwin":
        langs = list(opts.ocr_langs or ("uk-UA", "ru-RU", "en-US"))
        return {"engine": "ocrmac", "lang": langs, "recognition": "accurate", "mode": "DEFAULT"}
    langs = list(opts.ocr_langs or ("eslav",))
    return {
        "engine": "rapidocr",
        "lang": langs,
        "backend": "onnxruntime",
        "mode": "DEFAULT",
        # scale за замовчуванням 3.0 (216 dpi). Для скану, що вже 300 dpi,
        # апскейл ПОГІРШУЄ розпізнавання, тож реальне значення виводиться з
        # рідного DPI сторінки у воркері; тут — лише декларація наміру.
        "scale": opts.ocr_scale or "native-dpi",
        "text_score": 0.5,
    }


def build_parse_profile(
    opts: ParseOptions, *, backend: str, docling_version: str | None = None
) -> dict[str, Any]:
    """Канонічний опис конвеєра парсингу — вхід для хешу кешу рівня документа."""
    return {
        "backend": backend,
        "docling_version": docling_version or DOCLING_VERSION,
        "ocr": ocr_options_spec(opts),
        "table": {
            "enabled": opts.do_table_structure,
            "model": "TableFormerV1",   # V2 має відкриті регресії #3158, #3553
            "mode": opts.resolved_table_mode(),
            "cell_matching": opts.do_cell_matching,
        },
        "formula_enrichment": opts.do_formula_enrichment,
        "picture_classification": opts.do_picture_classification,
        "heading_hierarchy": opts.heading_hierarchy,
        "generate_parsed_pages": opts.generate_parsed_pages,
        "ingest_mode": opts.ingest_mode.value,
        "window_size": opts.window_size,
        "device": opts.device,
        "text_preproc_version": TEXT_PREPROC_VERSION,
    }


def parse_profile_hash(profile: dict[str, Any]) -> str:
    """Стабільний хеш профілю.

    `sort_keys=True` обов'язковий: без нього порядок ключів словника став би
    частиною ключа кешу, і той самий конвеєр давав би різні хеші між запусками.
    Разом із `content_sha256` це те, що робить додавання того самого підручника
    до другого асистента миттєвим.
    """
    payload = json.dumps(profile, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def page_windows(page_count: int, size: int = WINDOW_SIZE,
                 probe: ProbeReport | None = None) -> list[tuple[int, int]]:
    """Межі вікон, 1-based включно. З пробою — зсунуті геть від таблиць."""
    if page_count <= 0:
        return []
    if probe is not None and probe.pages:
        return window_boundaries(probe, window=size)
    return [(lo, min(lo + size - 1, page_count)) for lo in range(1, page_count + 1, size)]


# ------------------------------------------------------- серіалізація таблиць
def serialize_table_triplets(
    grid: Sequence[Sequence[str]],
    *,
    caption: str = "",
    max_rows: int = 400,
) -> str:
    """Самоописні рядки таблиці: кожен рядок несе власні заголовки колонок.

    Це трюк NeoLens (`["колонка": "значення", ...]`), у Docling реалізований як
    `TripletTableSerializer`. Сенс: рядок переживає БУДЬ-ЯКИЙ розріз чанка, бо
    не залежить від того, чи потрапив у той самий чанк рядок заголовків.
    Класична табличка балістики, розрізана посередині, інакше перетворюється на
    колонку чисел без жодного натяку, що це дальність, а що — приціл.
    """
    rows = [list(r) for r in grid if any(str(c).strip() for c in r)]
    if not rows:
        return caption.strip()
    header = [str(c).strip() for c in rows[0]]
    lines: list[str] = []
    if caption.strip():
        lines.append(caption.strip())
    for row in rows[1:max_rows + 1]:
        cells = [str(c).strip() for c in row]
        row_label = cells[0] if cells else ""
        triplets = [
            f"{row_label}, {header[i] if i < len(header) else f'колонка {i + 1}'} = {cells[i]}"
            for i in range(1, len(cells))
            if cells[i]
        ]
        if not triplets and row_label:
            triplets = [row_label]
        if triplets:
            lines.append(". ".join(triplets) + ".")
    if len(rows) - 1 > max_rows:
        lines.append(f"[…ще {len(rows) - 1 - max_rows} рядків таблиці]")
    return "\n".join(lines)


# ---------------------------------------------------- текстовий/markdown шлях
_ATX_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_FENCE_RE = re.compile(r"^```(\w*)\s*$")
_LIST_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+\S")
_TABLE_SEP_RE = re.compile(r"^\s*\|?[\s:|-]+\|[\s:|-]*$")


def _split_markdown_table(block: list[str]) -> list[list[str]]:
    grid: list[list[str]] = []
    for line in block:
        if _TABLE_SEP_RE.match(line):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        grid.append(cells)
    return grid


def _page_of(offset: int, pages: list[ParsedPage]) -> int:
    for page in pages:
        if page.char_from <= offset < page.char_to:
            return page.page_number
    return pages[-1].page_number if pages else 1


def _parse_text_source(text: str, opts: ParseOptions, *, source_path: str) -> ParsedDocument:
    """Власний парсер txt/md. Працює завжди — без docling, без torch, без OCR.

    Це не «іграшковий» шлях: методичні розробки викладача часто приходять саме
    у вигляді текстових конспектів, і вони мають індексуватись повноцінно.
    """
    report = probe_text(text, page_chars=opts.text_page_chars, source_path=source_path)
    pages = [
        ParsedPage(
            page_number=p.page_number,
            page_class=p.page_class,
            ocr_mode=OcrModeName.NONE,
            cost_weight=p.cost_weight,
            lexicon_hit_rate=p.lexicon_hit_rate,
            cyrillic_ratio=p.cyrillic_ratio,
            mojibake_ratio=p.mojibake_ratio,
            char_from=p.char_from,
            char_to=p.char_to,
        )
        for p in report.pages
    ]

    elements: list[ParsedElement] = []
    lines = text.split("\n")
    offset = 0
    index = 0
    counter = 0
    buffer: list[str] = []
    buffer_start = 0

    def flush_paragraph() -> None:
        nonlocal buffer, buffer_start, counter
        body = "\n".join(buffer).strip()
        if body:
            counter += 1
            kind = ElementKind.LIST_ITEM if _LIST_RE.match(buffer[0]) else ElementKind.PARAGRAPH
            elements.append(ParsedElement(
                kind=kind, text=body, page_no=_page_of(buffer_start, pages),
                self_ref=f"#/texts/{counter}",
                char_from=buffer_start, char_to=buffer_start + len(body),
            ))
        buffer = []

    while index < len(lines):
        line = lines[index]
        line_start = offset
        offset += len(line) + 1

        heading = _ATX_RE.match(line)
        fence = _FENCE_RE.match(line)

        if heading:
            flush_paragraph()
            counter += 1
            elements.append(ParsedElement(
                kind=ElementKind.HEADING, text=heading.group(2), level=len(heading.group(1)),
                page_no=_page_of(line_start, pages), self_ref=f"#/texts/{counter}",
                char_from=line_start, char_to=line_start + len(line),
            ))
        elif fence:
            flush_paragraph()
            language = fence.group(1) or ""
            body: list[str] = []
            index += 1
            while index < len(lines) and not _FENCE_RE.match(lines[index]):
                body.append(lines[index])
                offset += len(lines[index]) + 1
                index += 1
            if index < len(lines):
                offset += len(lines[index]) + 1
            counter += 1
            elements.append(ParsedElement(
                kind=ElementKind.CODE, text="\n".join(body), page_no=_page_of(line_start, pages),
                self_ref=f"#/code/{counter}", char_from=line_start, char_to=offset,
                meta={"language": language},
            ))
        elif line.strip().startswith("|") and line.count("|") >= 2:
            flush_paragraph()
            block = [line]
            while index + 1 < len(lines) and lines[index + 1].strip().startswith("|"):
                index += 1
                block.append(lines[index])
                offset += len(lines[index]) + 1
            grid = _split_markdown_table(block)
            counter += 1
            elements.append(ParsedElement(
                kind=ElementKind.TABLE,
                text=serialize_table_triplets(grid),
                page_no=_page_of(line_start, pages), self_ref=f"#/tables/{counter}",
                char_from=line_start, char_to=offset,
                meta={"grid": grid, "markdown": "\n".join(block)},
            ))
        elif not line.strip():
            flush_paragraph()
        else:
            if not buffer:
                buffer_start = line_start
            buffer.append(line)
        index += 1
    flush_paragraph()

    title = opts.title or _guess_title(elements, source_path)
    doc = ParsedDocument(
        source_path=source_path,
        title=title,
        language=opts.language,
        backend="text",
        pages=pages,
        elements=elements,
        page_labels=resolve_page_labels(None, page_count=len(pages)),
        full_text=text,
        probe=report,
    )
    for page in doc.pages:
        page.page_label = doc.label_for(page.page_number)
    doc.parse_profile_hash = parse_profile_hash(
        build_parse_profile(opts, backend="text", docling_version="n/a")
    )
    return doc


def _guess_title(elements: Sequence[ParsedElement], source_path: str) -> str:
    for el in elements:
        if el.kind in (ElementKind.TITLE, ElementKind.HEADING):
            return el.text.strip()
    return Path(source_path).stem if source_path else "Без назви"


# ----------------------------------------------------------------- stub-шлях
def _parse_stub(path: Path, opts: ParseOptions) -> ParsedDocument:
    """Детермінований синтетичний документ для stub-режиму.

    Псевдовміст виводиться з імені файлу й розміру, тому один і той самий файл
    завжди дає ті самі чанки, ті самі chunk_uid і той самий індекс. Це дозволяє
    ганяти увесь UI і CI без жодної завантаженої моделі й без 1.5 ГБ артефактів.
    """
    stem = path.stem or "документ"
    seed = hashlib.blake2b(stem.encode("utf-8"), digest_size=4).hexdigest()
    body = "\n\n".join(
        [
            f"# {stem}",
            "Це заглушка приймання (ASISTENT_STUB=1). Реальний парсер не викликався, "
            "модель не завантажувалась, мережа не використовувалась.",
            "## Розділ 1. Загальні положення",
            f"Контрольна сума вмісту заглушки: {seed}. Текст навмисно детермінований, "
            "щоб чанки, ембединги та індекс відтворювались побайтово між запусками.",
            "## Розділ 2. Приклад таблиці",
            "| Параметр | Значення | Одиниця |",
            "| --- | --- | --- |",
            "| Дальність | 15300 | м |",
            "| Кут піднесення | 45 | град |",
            "### 2.1. Приклад формули",
            "Поправка на деривацію обчислюється як Z = f(D, V0).",
        ]
    )
    doc = _parse_text_source(body, opts, source_path=str(path))
    doc.backend = "stub"
    doc.title = opts.title or stem
    doc.warnings.append("Документ оброблено заглушкою ASISTENT_STUB=1 — вміст синтетичний.")
    doc.parse_profile_hash = parse_profile_hash(
        build_parse_profile(opts, backend="stub", docling_version="stub")
    )
    return doc


# ---------------------------------------------------------------- шлях Docling
def _apply(options: Any, name: str, value: Any, warnings: list[str]) -> None:
    """Виставити опцію конвеєра, голосно записавши невдачу.

    Docling рухає назви полів між мінорними версіями. Тиха відсутність опції —
    найгірший сценарій (наприклад, без `heading_hierarchy_options` уся ієрархія
    заголовків стає пласкою і ніхто цього не помічає), тому кожна невдача
    потрапляє у `ParsedDocument.warnings` і далі — в UI індексації.
    """
    if not hasattr(options, name):
        warnings.append(
            f"Опція Docling '{name}' відсутня у встановленій версії пакета — "
            f"значення {value!r} НЕ застосовано. Перевірте версію docling."
        )
        return
    setattr(options, name, value)


def build_pdf_pipeline_options(opts: ParseOptions, warnings: list[str]) -> Any:
    """Явні `PdfPipelineOptions`. Жодного `auto` — див. докстрінг модуля."""
    from docling.datamodel.pipeline_options import PdfPipelineOptions

    po = PdfPipelineOptions()
    _apply(po, "do_ocr", opts.do_ocr, warnings)
    _apply(po, "ocr_options", _build_ocr_options(opts, warnings), warnings)
    # Без generate_parsed_pages детектор заголовків за стилем тихо не працює.
    _apply(po, "generate_parsed_pages", opts.generate_parsed_pages, warnings)
    _apply(po, "do_table_structure", opts.do_table_structure, warnings)
    _apply(po, "do_formula_enrichment", opts.do_formula_enrichment, warnings)
    _apply(po, "do_picture_classification", opts.do_picture_classification, warnings)
    _apply(po, "generate_page_images", False, warnings)

    try:
        from docling.datamodel.pipeline_options import TableFormerMode, TableStructureOptions

        mode = TableFormerMode.ACCURATE if opts.resolved_table_mode() == "ACCURATE" else TableFormerMode.FAST
        _apply(po, "table_structure_options",
               TableStructureOptions(do_cell_matching=opts.do_cell_matching, mode=mode), warnings)
    except ImportError as exc:
        warnings.append(f"TableStructureOptions недоступні: {exc}")

    if opts.heading_hierarchy:
        try:
            from docling.datamodel.pipeline_options import HeadingHierarchyOptions

            _apply(po, "heading_hierarchy_options", HeadingHierarchyOptions(enabled=True), warnings)
        except ImportError:
            warnings.append(
                "HeadingHierarchyOptions недоступні у цій версії Docling: КОЖЕН заголовок "
                "отримає level=1, ієрархія стане пласкою, а header_path — беззмістовним."
            )

    try:
        from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions

        device = {"cpu": AcceleratorDevice.CPU, "cuda": AcceleratorDevice.CUDA,
                  "mps": AcceleratorDevice.MPS}[opts.device]
        _apply(po, "accelerator_options",
               AcceleratorOptions(device=device, num_threads=opts.resolved_threads()), warnings)
    except (ImportError, KeyError) as exc:
        warnings.append(f"AcceleratorOptions не застосовано: {exc}")

    if opts.artifacts_path is not None:
        # ВАЖЛИВО: це БАТЬКІВСЬКИЙ каталог, що містить теки <org>--<repo>, а не
        # тека конкретної моделі. Інакше Docling мовчки викликає
        # snapshot_download — тобто йде в мережу, і net_guard валить індексацію.
        _apply(po, "artifacts_path", str(opts.artifacts_path), warnings)

    return po


def _build_ocr_options(opts: ParseOptions, warnings: list[str]) -> Any:
    """НІКОЛИ `OcrAutoOptions` — див. пастку 1 у докстрінгу модуля."""
    spec = ocr_options_spec(opts)
    if spec["engine"] == "ocrmac":
        from docling.datamodel.pipeline_options import OcrMacOptions

        return OcrMacOptions(lang=spec["lang"], recognition="accurate")
    from docling.datamodel.pipeline_options import RapidOcrOptions

    kwargs: dict[str, Any] = {"lang": spec["lang"], "backend": spec["backend"]}
    if isinstance(opts.ocr_scale, (int, float)):
        kwargs["scale"] = float(opts.ocr_scale)
    else:
        warnings.append(
            "OCR scale не задано явно: буде використано дефолт RapidOCR 3.0 (216 dpi). "
            "Для сканів 300 dpi апскейл ПОГІРШУЄ розпізнавання — виведіть scale із рідного DPI."
        )
    return RapidOcrOptions(**kwargs)


_DOCLING_LABEL_MAP: dict[str, ElementKind] = {
    "title": ElementKind.TITLE,
    "section_header": ElementKind.HEADING,
    "paragraph": ElementKind.PARAGRAPH,
    "text": ElementKind.PARAGRAPH,
    "list_item": ElementKind.LIST_ITEM,
    "table": ElementKind.TABLE,
    "picture": ElementKind.PICTURE,
    "formula": ElementKind.FORMULA,
    "code": ElementKind.CODE,
    "caption": ElementKind.CAPTION,
    "footnote": ElementKind.FOOTNOTE,
    "page_header": ElementKind.PAGE_HEADER,
    "page_footer": ElementKind.PAGE_FOOTER,
}


def _element_from_docling(item: Any, counter: int) -> ParsedElement | None:
    """Перекласти `DocItem` у власний тип. Незнайомі мітки ігноруються."""
    label = getattr(getattr(item, "label", None), "value", None) or str(getattr(item, "label", ""))
    kind = _DOCLING_LABEL_MAP.get(label)
    if kind is None:
        return None

    page_no = 1
    bbox: BBox | None = None
    prov = list(getattr(item, "prov", []) or [])
    if prov:
        first = prov[0]
        page_no = int(getattr(first, "page_no", 1) or 1)
        raw = getattr(first, "bbox", None)
        if raw is not None:
            # Координати PDF: початок — НИЖНІЙ лівий кут. Фронтенд переводить
            # їх у viewport через page.getViewport().convertToViewportPoint().
            bbox = BBox(page=page_no, left=float(raw.l), top=float(raw.t),
                        right=float(raw.r), bottom=float(raw.b))

    meta: dict[str, Any] = {}
    if kind is ElementKind.TABLE:
        grid = _docling_table_grid(item)
        meta["grid"] = grid
        caption = ""
        try:
            caption = item.caption_text(doc=None) or ""
        except Exception:  # підпис необов'язковий
            caption = ""
        text = serialize_table_triplets(grid, caption=caption)
    elif kind is ElementKind.PICTURE:
        # Прив'язка опису до рисунка правильна ЗА ПОБУДОВОЮ: Docling породжує
        # обрізок із prov[0].bbox самого елемента, тому bbox-матчинг не потрібен.
        description = getattr(getattr(item, "meta", None), "description", "") or ""
        classification = getattr(getattr(item, "meta", None), "classification", None)
        meta["classification"] = getattr(classification, "predicted_class", None)
        text = description or "[рисунок без опису]"
    else:
        text = getattr(item, "text", "") or ""

    # Рівень заголовка. Назва документа — H1; розділи Docling нумеруються з 1,
    # тому зсуваємо їх на один рівень нижче назви. Якщо HeadingHierarchyOptions
    # вимкнено, тут БУДЕ рівно 2 на кожному заголовку — і це видно у warnings.
    level: int | None = None
    if kind is ElementKind.TITLE:
        level = 1
    elif kind is ElementKind.HEADING:
        raw_level = getattr(item, "level", None)
        level = min(6, int(raw_level) + 1) if raw_level is not None else 2

    return ParsedElement(
        kind=kind, text=text, page_no=page_no, level=level, bbox=bbox,
        self_ref=str(getattr(item, "self_ref", f"#/items/{counter}")), meta=meta,
    )


def _docling_table_grid(item: Any) -> list[list[str]]:
    data = getattr(item, "data", None)
    grid = getattr(data, "grid", None)
    if not grid:
        return []
    return [[str(getattr(cell, "text", "") or "") for cell in row] for row in grid]


def _parse_with_docling(path: Path, opts: ParseOptions) -> ParsedDocument:
    """Посторінкові вікна по 25 сторінок + зчеплення.

    `DocumentConverter.convert()` не має колбека прогресу, тому вікно — єдиний
    спосіб отримати прогрес, кооперативне скасування, відновлюваність і
    обмежену пам'ять. Пропускної здатності це не коштує: у бенчмарку Docling
    (#3442) розміри батчів 4/64/128/256 дали НУЛЬОВУ різницю — конвеєр не
    GPU-bound, він CPU/Python-bound.
    """
    import docling
    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import DocumentConverter, PdfFormatOption

    warnings: list[str] = []
    report = probe_pdf(path)
    page_count = report.page_count or 0
    if opts.max_pages:
        page_count = min(page_count, opts.max_pages)

    pipeline_options = build_pdf_pipeline_options(opts, warnings)
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )

    windows = page_windows(page_count, opts.window_size, report)
    total_weight = report.total_weight
    done_weight = 0.0
    docs: list[Any] = []

    for lo, hi in windows:
        if opts.cancel is not None and opts.cancel():
            warnings.append(f"Парсинг скасовано на сторінці {lo}.")
            break
        result = converter.convert(str(path), page_range=(lo, hi), raises_on_error=False)
        docs.append(result.document)
        done_weight += sum(p.cost_weight for p in report.pages if lo <= p.page_number <= hi) or (hi - lo + 1)
        if opts.progress is not None:
            opts.progress(done_weight, total_weight)

    if not docs:
        raise RuntimeError(f"Docling не повернув жодного вікна для {path}.")

    from docling_core.types.doc import DoclingDocument

    document = docs[0] if len(docs) == 1 else DoclingDocument.concatenate(docs)

    elements: list[ParsedElement] = []
    for counter, (item, _level) in enumerate(document.iterate_items(), start=1):
        element = _element_from_docling(item, counter)
        if element is not None:
            elements.append(element)

    # Колонтитули FURNITURE — сировина для регресії міток сторінок.
    header_texts: dict[int, list[str]] = {}
    for el in elements:
        if el.kind.is_furniture:
            header_texts.setdefault(el.page_no, []).append(el.text)

    labels = resolve_page_labels(path, page_count=page_count or len(report.pages),
                                 header_texts=header_texts)

    pages = [
        ParsedPage(
            page_number=p.page_number, width=p.width, height=p.height,
            page_label=labels.label_for(p.page_number), page_class=p.page_class,
            ocr_mode=p.ocr_mode, cost_weight=p.cost_weight,
            lexicon_hit_rate=p.lexicon_hit_rate, cyrillic_ratio=p.cyrillic_ratio,
            mojibake_ratio=p.mojibake_ratio, char_from=p.char_from, char_to=p.char_to,
        )
        for p in report.pages
    ]

    full_text, elements = _assign_char_spans(elements)
    doc = ParsedDocument(
        source_path=str(path),
        title=opts.title or getattr(document, "name", "") or _guess_title(elements, str(path)),
        language=opts.language,
        backend="docling",
        pages=pages,
        elements=elements,
        page_labels=labels,
        full_text=full_text,
        warnings=warnings,
        probe=report,
    )
    doc.parse_profile_hash = parse_profile_hash(
        build_parse_profile(opts, backend="docling",
                            docling_version=getattr(docling, "__version__", DOCLING_VERSION))
    )
    return doc


def _assign_char_spans(elements: list[ParsedElement]) -> tuple[str, list[ParsedElement]]:
    """Побудувати суцільний текст документа й записати діапазони символів.

    Позиція чанка зберігається як діапазон СИМВОЛІВ і резолвиться в сторінки —
    це рішення NeoLens, і воно переживає переверстку куди краще, ніж bbox.
    """
    parts: list[str] = []
    cursor = 0
    for el in elements:
        el.char_from = cursor
        parts.append(el.text)
        cursor += len(el.text) + 2
        el.char_to = el.char_from + len(el.text)
    return "\n\n".join(parts), elements


# ------------------------------------------------------------- точка входу
def parse_document(path: str | Path, opts: ParseOptions | None = None) -> ParsedDocument:
    """Публічний інтерфейс модуля (див. docs/CONTRACT.md).

    Порядок вибору бекенда:
      * ASISTENT_STUB=1 і файл не текстовий → детермінована заглушка;
      * .txt/.md → власний текстовий конвеєр (працює завжди);
      * .pdf і docling доступний → повний конвеєр Docling;
      * .pdf без docling → текстовий шар через pypdfium2, з чесним попередженням.
    """
    opts = opts or ParseOptions()
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in SUPPORTED_TEXT_SUFFIXES:
        return _parse_text_source(path.read_text(encoding="utf-8", errors="replace"),
                                  opts, source_path=str(path))

    if is_stub_mode():
        return _parse_stub(path, opts)

    if suffix == ".pdf":
        try:
            return _parse_with_docling(path, opts)
        except ImportError as exc:
            log.warning("Docling недоступний (%s) — деградую до текстового шару PDF.", exc)
            return _parse_pdf_text_layer(path, opts, reason=str(exc))

    raise ValueError(
        f"Непідтримуваний формат файлу: {suffix or path.name}. "
        f"Підтримуються PDF і текстові файли ({', '.join(sorted(SUPPORTED_TEXT_SUFFIXES))}). "
        f"DOCX слід спершу конвертувати у PDF через LibreOffice headless: "
        f"msword_backend Docling НІКОЛИ не створює ProvenanceItem, тобто цитата на "
        f"сторінку для DOCX неможлива в принципі."
    )


def _parse_pdf_text_layer(path: Path, opts: ParseOptions, *, reason: str) -> ParsedDocument:
    """Аварійний шлях: лише текстовий шар PDF через pypdfium2.

    Ані таблиць, ані формул, ані OCR, ані bbox на елемент — але документ
    індексується і цитата на сторінку працює. Це краще за відмову, і про
    деградацію повідомляється явно.
    """
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:
        raise RuntimeError(
            f"Ні Docling, ні pypdfium2 недоступні — PDF прочитати нічим ({exc}). "
            f"Встановіть залежності групи 'worker'."
        ) from exc

    report = probe_pdf(path)
    pdf = pdfium.PdfDocument(str(path))
    pages: list[ParsedPage] = []
    elements: list[ParsedElement] = []
    chunks_text: list[str] = []
    try:
        for probe in report.pages:
            page = pdf[probe.page_number - 1]
            textpage = page.get_textpage()
            try:
                text = textpage.get_text_bounded()
            finally:
                textpage.close()
                page.close()
            chunks_text.append(text)
            for index, block in enumerate(p for p in text.split("\n\n") if p.strip()):
                elements.append(ParsedElement(
                    kind=ElementKind.PARAGRAPH, text=block.strip(),
                    page_no=probe.page_number, self_ref=f"#/pdftext/{probe.page_number}/{index}",
                ))
            pages.append(ParsedPage(
                page_number=probe.page_number, width=probe.width, height=probe.height,
                page_class=probe.page_class, ocr_mode=probe.ocr_mode,
                cost_weight=probe.cost_weight, lexicon_hit_rate=probe.lexicon_hit_rate,
                cyrillic_ratio=probe.cyrillic_ratio, mojibake_ratio=probe.mojibake_ratio,
                char_from=probe.char_from, char_to=probe.char_to,
            ))
    finally:
        pdf.close()

    labels = resolve_page_labels(path, page_count=len(pages) or report.page_count)
    for page in pages:
        page.page_label = labels.label_for(page.page_number)
    full_text, elements = _assign_char_spans(elements)
    doc = ParsedDocument(
        source_path=str(path), title=opts.title or path.stem, language=opts.language,
        backend="pdfium-text", pages=pages, elements=elements, page_labels=labels,
        full_text=full_text, probe=report,
        warnings=[
            f"Docling недоступний ({reason}). Використано лише текстовий шар PDF: "
            f"таблиці, формули, OCR і рамки елементів відсутні."
        ],
    )
    doc.parse_profile_hash = parse_profile_hash(
        build_parse_profile(opts, backend="pdfium-text", docling_version="n/a")
    )
    return doc
