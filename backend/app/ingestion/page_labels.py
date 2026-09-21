"""Резолвер ДРУКОВАНОЇ мітки сторінки.

Docling її не моделює зовсім: `PageItem` — це лише `{size, image, page_no}`, а
`ProvenanceItem.page_no` — фізичний індекс у файлі. Але викладач цитує «с. 147»,
а не «сто п'ятдесят другий аркуш PDF», і в підручнику з римською передмовою ці
два числа розходяться на десяток.

Порядок, у якому резолвер пробує джерела (план, §1):
  1. `/PageLabels` через pypdf — дерево нумерації, авторитетне; коректно
     обробляє «римська передмова, далі арабська»;
  2. регресія по колонтитулах — витягти цілі з `PAGE_HEADER`/`PAGE_FOOTER`,
     підігнати `label = physical + offset`, прийняти лише при згоді >= 80%;
  3. фолбек `label = physical`.

Цитата рендериться як «с. 147 (файл, стор. 152)», тому фізичний індекс не
губиться ніколи.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

__all__ = [
    "MIN_HEADER_AGREEMENT",
    "LabelSource",
    "PageLabelMap",
    "format_label",
    "from_roman",
    "labels_from_headers",
    "labels_from_pdf",
    "labels_from_ranges",
    "resolve_page_labels",
    "to_alpha",
    "to_roman",
]

log = logging.getLogger(__name__)

LabelSource = Literal["page_labels", "header_regression", "physical"]

# Регресія приймається лише при згоді 80% сторінок, що взагалі дали кандидата.
MIN_HEADER_AGREEMENT = 0.8
# І лише якщо кандидатів достатньо: три випадкові числа в колонтитулі згодні
# між собою значно частіше, ніж хотілося б.
MIN_HEADER_SAMPLES = 5


@dataclass(slots=True)
class PageLabelMap:
    """Мапа фізичний індекс (1-based) → друкована мітка."""

    labels: dict[int, str] = field(default_factory=dict)
    source: LabelSource = "physical"
    confidence: float = 0.0

    def label_for(self, physical: int) -> str:
        """Мітка сторінки; фолбек — сам фізичний індекс, ніколи не порожньо."""
        return self.labels.get(physical, str(physical))

    def as_dict(self) -> dict[int, str]:
        return dict(self.labels)

    def __len__(self) -> int:
        return len(self.labels)


# ------------------------------------------------------------------ нумерації
_ROMAN_VALUES: tuple[tuple[int, str], ...] = (
    (1000, "m"), (900, "cm"), (500, "d"), (400, "cd"), (100, "c"), (90, "xc"),
    (50, "l"), (40, "xl"), (10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i"),
)
_ROMAN_RE = re.compile(r"^m{0,4}(cm|cd|d?c{0,3})(xc|xl|l?x{0,3})(ix|iv|v?i{0,3})$", re.IGNORECASE)


def to_roman(number: int, *, upper: bool = False) -> str:
    """Римське число. PDF-стилі `r` (нижній регістр) і `R` (верхній)."""
    if number <= 0:
        return ""
    out: list[str] = []
    rest = number
    for value, glyph in _ROMAN_VALUES:
        while rest >= value:
            out.append(glyph)
            rest -= value
    text = "".join(out)
    return text.upper() if upper else text


def from_roman(text: str) -> int | None:
    """Розібрати римське число. None, якщо це не римське число."""
    s = text.strip()
    if not s or not _ROMAN_RE.match(s):
        return None
    values = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}
    total = 0
    previous = 0
    for ch in reversed(s.lower()):
        value = values[ch]
        total += value if value >= previous else -value
        previous = max(previous, value)
    return total or None


def to_alpha(number: int, *, upper: bool = False) -> str:
    """Стилі PDF `a`/`A`: a..z, потім aa..zz, потім aaa... (як у специфікації)."""
    if number <= 0:
        return ""
    index = number - 1
    letter = chr(ord("a") + index % 26)
    text = letter * (index // 26 + 1)
    return text.upper() if upper else text


def format_label(style: str | None, prefix: str, number: int) -> str:
    """Одна мітка за стилем /S, префіксом /P і номером."""
    if style in ("D", None, ""):
        body = str(number) if style == "D" else ""
    elif style == "r":
        body = to_roman(number)
    elif style == "R":
        body = to_roman(number, upper=True)
    elif style == "a":
        body = to_alpha(number)
    elif style == "A":
        body = to_alpha(number, upper=True)
    else:
        body = str(number)
    return f"{prefix}{body}"


def labels_from_ranges(
    ranges: Sequence[tuple[int, Mapping[str, Any]]],
    page_count: int,
) -> dict[int, str]:
    """Розгорнути дерево `/PageLabels` у мапу міток.

    `ranges` — пари (перший ФІЗИЧНИЙ індекс діапазону, 0-based; словник
    із ключами `S` — стиль, `P` — префікс, `St` — з якого номера починати).
    Саме ця структура робить «римська передмова, далі арабське тіло»
    тривіальною: два діапазони, {"S": "r"} і {"S": "D", "St": 1}.

    Функція чиста — тестується без pypdf і без PDF взагалі.
    """
    if page_count <= 0:
        return {}
    ordered = sorted(ranges, key=lambda item: item[0])
    if not ordered:
        return {}
    labels: dict[int, str] = {}
    for position, (start, spec) in enumerate(ordered):
        stop = ordered[position + 1][0] if position + 1 < len(ordered) else page_count
        style = spec.get("S")
        prefix = str(spec.get("P", "") or "")
        first_number = int(spec.get("St", 1) or 1)
        for offset, physical0 in enumerate(range(max(start, 0), min(stop, page_count))):
            labels[physical0 + 1] = format_label(style, prefix, first_number + offset)
    return labels


def labels_from_pdf(path: str | Path, page_count: int | None = None) -> PageLabelMap | None:
    """Джерело №1: `/PageLabels` через pypdf. None, якщо дерева немає.

    pypdf — ОПЦІЙНИЙ імпорт: API-процес його не тягне.
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        log.warning("pypdf недоступний — /PageLabels прочитати неможливо, переходжу до колонтитулів.")
        return None

    try:
        reader = PdfReader(str(path))
        total = page_count or len(reader.pages)
        raw = list(reader.page_labels)
    except Exception as exc:  # пошкоджений PDF не має валити індексацію
        log.warning("Не вдалося прочитати /PageLabels з %s: %s", path, exc)
        return None

    labels = {i + 1: str(value) for i, value in enumerate(raw[:total]) if str(value)}
    if not labels:
        return None
    # pypdf синтезує «1, 2, 3...», коли дерева немає. Такий результат не несе
    # інформації понад фізичний індекс — чесніше віддати його як фолбек, щоб
    # регресія по колонтитулах дістала свій шанс.
    if all(labels.get(i) == str(i) for i in range(1, total + 1)):
        return None
    return PageLabelMap(labels=labels, source="page_labels", confidence=1.0)


_INT_RE = re.compile(r"\d{1,4}")


def labels_from_headers(
    header_texts: Mapping[int, Sequence[str]],
    page_count: int,
    *,
    min_agreement: float = MIN_HEADER_AGREEMENT,
) -> PageLabelMap | None:
    """Джерело №2: регресія по колонтитулах.

    Ідея: у книзі друкована мітка й фізичний індекс різняться на СТАЛУ величину
    (передмова здвигає нумерацію рівно один раз). Витягаємо з колонтитулів усі
    цілі, для кожного рахуємо зсув `label - physical`, беремо моду і приймаємо
    її, лише якщо згодні >= 80% сторінок, що взагалі дали кандидата. Один
    випадковий «Розділ 3» у колонтитулі так не переважить сотню номерів.
    """
    offsets: Counter[int] = Counter()
    pages_with_candidate = 0
    supporting: dict[int, set[int]] = {}

    for physical, texts in header_texts.items():
        candidates: set[int] = set()
        for text in texts:
            for match in _INT_RE.finditer(text or ""):
                value = int(match.group(0))
                if 0 < value <= 9999:
                    candidates.add(value)
            roman = from_roman((text or "").strip())
            if roman:
                candidates.add(roman)
        if not candidates:
            continue
        pages_with_candidate += 1
        for value in candidates:
            offset = value - physical
            offsets[offset] += 1
            supporting.setdefault(offset, set()).add(physical)

    if pages_with_candidate < MIN_HEADER_SAMPLES or not offsets:
        return None

    best_offset, _ = offsets.most_common(1)[0]
    agreement = len(supporting[best_offset]) / pages_with_candidate
    if agreement < min_agreement:
        log.info(
            "Регресія по колонтитулах відхилена: згода %.2f < %.2f", agreement, min_agreement
        )
        return None

    labels = {physical: str(physical + best_offset) for physical in range(1, page_count + 1)
              if physical + best_offset > 0}
    return PageLabelMap(labels=labels, source="header_regression", confidence=agreement)


def resolve_page_labels(
    path: str | Path | None = None,
    *,
    page_count: int,
    header_texts: Mapping[int, Sequence[str]] | None = None,
) -> PageLabelMap:
    """Повний резолвер: /PageLabels → регресія → фолбек `label = physical`.

    Ніколи не кидає і ніколи не повертає порожню мапу: цитата без номера
    сторінки для викладача гірша за приблизний номер.
    """
    if path is not None:
        from_tree = labels_from_pdf(path, page_count)
        if from_tree is not None:
            return from_tree

    if header_texts:
        from_headers = labels_from_headers(header_texts, page_count)
        if from_headers is not None:
            return from_headers

    return PageLabelMap(
        labels={i: str(i) for i in range(1, page_count + 1)},
        source="physical",
        confidence=0.0,
    )
