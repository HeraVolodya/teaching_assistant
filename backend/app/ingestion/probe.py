"""Тріаж рівня 0 — ~10 мс/сторінку, до того як увімкнеться будь-яка модель.

Навіщо він, якщо Docling має власний `ConfidenceReport`. Бо `parse_score`
детектує лише mojibake, а класична українська поломка інша: PDF, чия
`ToUnicode` CMap мапить кириличні гліфи на ВАЛІДНІ, але хибні кодпойнти.
Такий текст складається із законних українських літер — `parse_score` буде
1.0 при цілковито нечитабельному тексті. Єдиний спосіб це впіймати —
`lexicon_hit_rate` проти українського словника.

Проба дає три речі, кожна з яких потрібна далі за течією:
  * `PageClass` і `OcrModeName` на сторінку — план OCR (див. план, §1);
  * вагу вартості `w = 1 + 3·скан + 2·таблиця + 1·формула + 0.5·рисунки` —
    без неї прогрес брехливий, бо сканована таблична сторінка коштує в ~20
    разів більше за порожню;
  * межі посторінкових вікон, зсунуті так, щоб вікно не закінчувалось усередині
    таблиці.

`pypdfium2` — ОПЦІЙНИЙ імпорт: API-процес не має його тягнути, а stub-режим і
текстові джерела працюють без нього взагалі.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from app.domain import OcrModeName, PageClass, PageInfo
from app.ingestion.normalize_uk import cyrillic_ratio, normalize_uk, tokenize
from app.ingestion.uk_lexicon import lexicon_hit_rate

__all__ = [
    "CYRILLIC_EVIDENCE_THRESHOLD",
    "LEXICON_BROKEN_THRESHOLD",
    "MIN_CHARS_PER_PAGE",
    "MOJIBAKE_THRESHOLD",
    "PageProbe",
    "ProbeReport",
    "classify_page",
    "cost_weight",
    "looks_like_formula",
    "looks_like_table",
    "measure_page",
    "mojibake_ratio",
    "probe_pdf",
    "probe_text",
    "window_boundaries",
]

log = logging.getLogger(__name__)

# --------------------------------------------------------------------- пороги
# «lexicon_hit_rate < 0.45 при cyrillic_ratio > 0.3 → зламаний текстовий шар»
# (план, §1). Поріг калібрований під ПОВНИЙ словник; вбудований частотний
# список дає нижчі hit-rate, тому тут використано консервативніше значення,
# і рішення додатково вимагає достатньої кількості кириличних токенів.
LEXICON_BROKEN_THRESHOLD = 0.45
CYRILLIC_EVIDENCE_THRESHOLD = 0.3
MOJIBAKE_THRESHOLD = 0.02
# Менше 40 символів при помітному покритті растром — це скан, а не порожня
# сторінка. Число з плану, §1.
MIN_CHARS_PER_PAGE = 40
BITMAP_SCAN_THRESHOLD = 0.3
BITMAP_MIXED_THRESHOLD = 0.6

# Сліди зламаного вилучення тексту: явний U+FFFD і два діалекти «немапованого
# гліфа», що їх видають різні PDF-бекенди.
_GLYPH_RE = re.compile(r"/glyph<[^>]{0,16}>|/g\d+")
_PUA_RANGES = ((0xE000, 0xF8FF), (0xF0000, 0xFFFFD), (0x100000, 0x10FFFD))

_MATH_CHARS = frozenset("∑∫√±×÷≈≤≥≠∞∂∇°⋅·αβγδεθλμνπρστφχψωΔΩΣΠ")
_TABLE_WORD_RE = re.compile(r"таблиц|табл\.", re.IGNORECASE)
_FORMULA_WORD_RE = re.compile(r"формул|рівнянн", re.IGNORECASE)
_COLUMN_GAP_RE = re.compile(r"\S {2,}(?=\S)")  # lookahead: колонки рахуються всі, не через одну


def mojibake_ratio(text: str) -> float:
    """Частка символів, що є слідом зламаного вилучення тексту."""
    if not text:
        return 0.0
    bad = text.count("�")
    for match in _GLYPH_RE.finditer(text):
        bad += len(match.group(0))
    for ch in text:
        cp = ord(ch)
        # Private Use Area й керуючі символи — обидва є слідом зламаного
        # вилучення, тож рахуються однаково.
        if any(lo <= cp <= hi for lo, hi in _PUA_RANGES) or (
            cp < 0x20 and ch not in "\t\n\r\f"
        ):
            bad += 1
    return min(1.0, bad / len(text))


def looks_like_table(text: str) -> bool:
    """Дешева ознака таблиці на сторінці.

    Не намагається бути точною: її єдина роль — вага вартості й зсув межі
    вікна. Справжнє вилучення таблиць робить TableFormer.
    """
    if _TABLE_WORD_RE.search(text):
        return True
    aligned = sum(1 for line in text.splitlines() if len(_COLUMN_GAP_RE.findall(line)) >= 2)
    tabbed = sum(1 for line in text.splitlines() if line.count("\t") >= 2 or line.count("|") >= 2)
    return aligned >= 3 or tabbed >= 3


def looks_like_formula(text: str) -> bool:
    """Дешева ознака формули: математичні символи або пряма згадка."""
    if _FORMULA_WORD_RE.search(text):
        return True
    return any(ch in _MATH_CHARS for ch in text)


# ---------------------------------------------------------------- посторінково
@dataclass(slots=True)
class PageProbe:
    """Метрики й вердикт для однієї сторінки."""

    page_number: int
    width: float = 0.0
    height: float = 0.0
    chars_per_page: int = 0
    text_area_coverage: float = 0.0
    bitmap_coverage: float = 0.0
    cyrillic_ratio: float = 0.0
    mojibake_ratio: float = 0.0
    lexicon_hit_rate: float | None = None
    has_table: bool = False
    has_formula: bool = False
    picture_count: int = 0
    page_class: PageClass = PageClass.DIGITAL_CLEAN
    ocr_mode: OcrModeName = OcrModeName.DEFAULT
    cost_weight: float = 1.0
    char_from: int = 0
    char_to: int = 0

    def to_page_info(self, page_label: str | None = None) -> PageInfo:
        """Рядок таблиці `pages` — рівно те, що приймає `PageRepo.upsert_many`."""
        return PageInfo(
            page_number=self.page_number,
            page_label=page_label,
            char_from=self.char_from,
            char_to=self.char_to,
            width=self.width,
            height=self.height,
            page_class=self.page_class,
            ocr_mode=self.ocr_mode,
            cost_weight=self.cost_weight,
            lexicon_hit_rate=self.lexicon_hit_rate,
            cyrillic_ratio=self.cyrillic_ratio,
            mojibake_ratio=self.mojibake_ratio,
        )


@dataclass(slots=True)
class ProbeReport:
    """Результат тріажу документа."""

    source_path: str = ""
    page_count: int = 0
    pages: list[PageProbe] = field(default_factory=list)
    backend: str = "none"
    warnings: list[str] = field(default_factory=list)

    @property
    def total_weight(self) -> float:
        """Знаменник чесного прогрес-бару."""
        return sum(p.cost_weight for p in self.pages) or float(self.page_count)

    @property
    def dominant_class(self) -> PageClass:
        if not self.pages:
            return PageClass.DIGITAL_CLEAN
        counts: dict[PageClass, int] = {}
        for p in self.pages:
            counts[p.page_class] = counts.get(p.page_class, 0) + 1
        return max(counts, key=lambda k: counts[k])

    @property
    def needs_ocr(self) -> bool:
        return any(p.ocr_mode is not OcrModeName.NONE for p in self.pages)

    def scanned_share(self) -> float:
        if not self.pages:
            return 0.0
        scanned = sum(
            1 for p in self.pages
            if p.page_class in (PageClass.SCANNED, PageClass.DIGITAL_BROKEN)
        )
        return scanned / len(self.pages)


def classify_page(probe: PageProbe) -> tuple[PageClass, OcrModeName]:
    """Таблиця рішень із плану, §1. Порядок перевірок значущий.

    Головний рядок — передостанній: текстовий шар є, mojibake немає, але слова
    не словникові. Це та сама зламана `ToUnicode` CMap, яку Docling не ловить;
    текстові елементи PDF треба ВІДКИНУТИ і OCR-ити сторінку повністю.
    """
    if probe.chars_per_page < MIN_CHARS_PER_PAGE:
        if probe.bitmap_coverage >= BITMAP_SCAN_THRESHOLD:
            return PageClass.SCANNED, OcrModeName.FULL_PAGE
        # Порожня або суто титульна сторінка: OCR тут — чиста витрата часу.
        return PageClass.DIGITAL_CLEAN, OcrModeName.NONE

    if probe.mojibake_ratio > MOJIBAKE_THRESHOLD:
        return PageClass.DIGITAL_BROKEN, OcrModeName.FULL_PAGE

    if (
        probe.cyrillic_ratio > CYRILLIC_EVIDENCE_THRESHOLD
        and probe.lexicon_hit_rate is not None
        and probe.lexicon_hit_rate < LEXICON_BROKEN_THRESHOLD
    ):
        return PageClass.DIGITAL_BROKEN, OcrModeName.FULL_PAGE

    if probe.bitmap_coverage >= BITMAP_MIXED_THRESHOLD:
        return PageClass.MIXED, OcrModeName.LAYOUT_REGIONS
    if probe.bitmap_coverage >= BITMAP_SCAN_THRESHOLD:
        # Дефолтний PDF_AWARE_LAYOUT_REGIONS вирішує «чи потрібен OCR» на рівні
        # регіону, а не сторінки: цифровий підручник із трьома сканованими
        # вставками отримає OCR лише на цих трьох вставках.
        return PageClass.MIXED, OcrModeName.DEFAULT
    return PageClass.DIGITAL_CLEAN, OcrModeName.DEFAULT


def cost_weight(probe: PageProbe) -> float:
    """w = 1 + 3·скан + 2·таблиця + 1·формула + 0.5·рисунки.

    Кількість рисунків обмежена шістьма: сторінка з 200 декоративними гліфами
    не коштує в сто разів більше, вона коштує як сторінка зі схемами.
    """
    scanned = probe.page_class in (PageClass.SCANNED, PageClass.DIGITAL_BROKEN)
    return (
        1.0
        + 3.0 * float(scanned)
        + 2.0 * float(probe.has_table)
        + 1.0 * float(probe.has_formula)
        + 0.5 * float(min(probe.picture_count, 6))
    )


def measure_page(
    text: str,
    *,
    page_number: int,
    width: float = 0.0,
    height: float = 0.0,
    bitmap_coverage: float = 0.0,
    text_area_coverage: float = 0.0,
    picture_count: int = 0,
    char_from: int = 0,
) -> PageProbe:
    """Порахувати всі метрики сторінки з її тексту й геометрії.

    Чиста функція без PDF-залежностей — саме тому її можна тестувати без
    pypdfium2 і викликати для txt/md-джерел.
    """
    probe = PageProbe(
        page_number=page_number,
        width=width,
        height=height,
        chars_per_page=len(text.strip()),
        text_area_coverage=text_area_coverage,
        bitmap_coverage=bitmap_coverage,
        picture_count=picture_count,
        char_from=char_from,
        char_to=char_from + len(text),
    )
    probe.cyrillic_ratio = cyrillic_ratio(text)
    probe.mojibake_ratio = mojibake_ratio(text)
    probe.lexicon_hit_rate = lexicon_hit_rate(tokenize(normalize_uk(text)))
    probe.has_table = looks_like_table(text)
    probe.has_formula = looks_like_formula(text)
    probe.page_class, probe.ocr_mode = classify_page(probe)
    probe.cost_weight = cost_weight(probe)
    return probe


# ------------------------------------------------------------------- джерела
def probe_text(text: str, *, page_chars: int = 1800, source_path: str = "") -> ProbeReport:
    """Тріаж текстового джерела (txt/md).

    У текстового файлу немає сторінок. Розрив сторінки U+000C вважається
    авторитетним, якщо він є; інакше сторінки нарізаються по `page_chars` —
    це наближення, потрібне лише для того, щоб цитата мала якийсь номер.
    """
    raw_pages = text.split("\f") if "\f" in text else [
        text[i:i + page_chars] for i in range(0, max(len(text), 1), page_chars)
    ]
    report = ProbeReport(source_path=source_path, backend="text")
    cursor = 0
    for index, page_text in enumerate(raw_pages, start=1):
        report.pages.append(measure_page(page_text, page_number=index, char_from=cursor))
        cursor += len(page_text)
    report.page_count = len(report.pages)
    return report


def _object_bounds(obj: object) -> tuple[float, float, float, float] | None:
    """Рамка об'єкта сторінки, (left, bottom, right, top).

    pypdfium2 перейменував `PdfObject.get_pos()` на `get_bounds()` у 5.x.
    Мовчазне повернення нуля тут коштувало б дорого: `bitmap_coverage` став би
    завжди 0, кожна СКАНОВАНА сторінка класифікувалася б як DIGITAL_CLEAN і
    ЖОДЕН сканований підручник не отримав би OCR — без єдиного повідомлення.
    """
    for name in ("get_bounds", "get_pos"):
        getter = getattr(obj, name, None)
        if getter is None:
            continue
        left, bottom, right, top = getter()
        return float(left), float(bottom), float(right), float(top)
    return None


def _pdfium_page_stats(page: object, pdfium_c: object) -> tuple[float, int, str | None]:
    """Покриття растром, кількість зображень і, за потреби, текст попередження."""
    width, height = page.get_size()
    area = max(width * height, 1.0)
    covered = 0.0
    pictures = 0
    problem: str | None = None
    try:
        for obj in page.get_objects():
            if getattr(obj, "type", None) != pdfium_c.FPDF_PAGEOBJ_IMAGE:
                continue
            pictures += 1
            bounds = _object_bounds(obj)
            if bounds is None:
                problem = (
                    "pypdfium2 не має ні get_bounds(), ні get_pos() — покриття растром "
                    "порахувати неможливо, скановані сторінки не будуть розпізнані як скани."
                )
                continue
            left, bottom, right, top = bounds
            covered += max(0.0, right - left) * max(0.0, top - bottom)
    except Exception as exc:  # тріаж не має права валити індексацію
        return 0.0, 0, f"Не вдалося перелічити об'єкти сторінки ({exc})."
    return min(1.0, covered / area), pictures, problem


def probe_pdf(
    path: str | Path,
    *,
    page_range: tuple[int, int] | None = None,
) -> ProbeReport:
    """Тріаж PDF через pypdfium2. ~10 мс/сторінку.

    `page_range` — 1-based включно з обох боків, як у Docling.
    Якщо pypdfium2 недоступний, повертається порожній звіт із попередженням:
    конвеєр має деградувати, а не падати.
    """
    path = Path(path)
    report = ProbeReport(source_path=str(path), backend="pypdfium2")
    try:
        import pypdfium2 as pdfium
        import pypdfium2.raw as pdfium_c
    except ImportError:
        report.backend = "none"
        report.warnings.append(
            "pypdfium2 недоступний — тріаж рівня 0 пропущено. "
            "План OCR буде побудовано за замовчуванням, прогрес — лінійним."
        )
        log.warning(report.warnings[-1])
        return report

    pdf = pdfium.PdfDocument(str(path))
    try:
        total = len(pdf)
        first, last = page_range or (1, total)
        first = max(1, first)
        last = min(total, last)
        cursor = 0
        for physical in range(first, last + 1):
            page = pdf[physical - 1]
            try:
                width, height = page.get_size()
                textpage = page.get_textpage()
                try:
                    # get_text_bounded() у 5.x, get_text_range() у 4.x.
                    reader = getattr(textpage, "get_text_bounded", None) or textpage.get_text_range
                    text = reader()
                finally:
                    textpage.close()
                bitmap, pictures, problem = _pdfium_page_stats(page, pdfium_c)
                if problem and problem not in report.warnings:
                    report.warnings.append(problem)
                    log.warning(problem)
                report.pages.append(
                    measure_page(
                        text,
                        page_number=physical,
                        width=width,
                        height=height,
                        bitmap_coverage=bitmap,
                        picture_count=pictures,
                        char_from=cursor,
                    )
                )
                cursor += len(text)
            finally:
                page.close()
        report.page_count = total
    finally:
        pdf.close()
    return report


def window_boundaries(
    report: ProbeReport,
    *,
    window: int = 25,
    max_shift: int = 3,
) -> list[tuple[int, int]]:
    """Межі посторінкових вікон, зсунуті геть від таблиць.

    `DocumentConverter.convert()` не має колбека прогресу, тому єдиний спосіб
    отримати прогрес, скасування, відновлюваність і обмежену пам'ять — різати
    документ на вікна. Плата за це — таблиця, розірвана на межі вікна; тому
    межу зсуваємо на до `max_shift` сторінок, шукаючи сторінку без таблиці.
    """
    if report.page_count <= 0:
        return []
    has_table = {p.page_number: p.has_table for p in report.pages}
    windows: list[tuple[int, int]] = []
    start = 1
    while start <= report.page_count:
        end = min(start + window - 1, report.page_count)
        if end < report.page_count:
            for shift in range(0, max_shift + 1):
                candidate = end - shift
                if candidate <= start:
                    break
                # Таблиця перетинає розріз лише тоді, коли вона є і на
                # сторінці межі, і на наступній. Одна таблиця, що починається
                # одразу після межі, розрізу не заважає.
                if not (has_table.get(candidate, False) and has_table.get(candidate + 1, False)):
                    end = candidate
                    break
        windows.append((start, end))
        start = end + 1
    return windows
