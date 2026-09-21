"""Правдивість таблиць: поклітинний F1 (Віха 2 плану).

Балістична таблиця — це найдорожчий об'єкт у корпусі. Помилка в прозі коштує
незручності, помилка в комірці таблиці стрільби коштує неправильної поправки.
Тому таблиці міряються окремо від тексту й окремою метрикою.

ДВА F1, І САМЕ ЇХ РІЗНИЦЯ Є ДІАГНОЗОМ
------------------------------------
  * **позиційний** — комірка зараховується, лише якщо збіглися і текст, і
    координати (рядок, колонка);
  * **змістовний** — мультимножина текстів комірок без огляду на координати.

Якщо змістовний високий, а позиційний низький — рушій ПРОЧИТАВ значення, але
поламав структуру (з'їхав об'єднаний заголовок, розчепився стовпчик). Це
лікується іншим `TableFormerMode`, а не іншим OCR. Якщо низькі обидва — це
провал розпізнавання, і треба інший рушій або VLM-ремонт.

ПОРОГИ, ЩО ВЖЕ Є В СИСТЕМІ
--------------------------
План (§1) називає тригером VLM-ремонту таблицю, у якій **понад 30% комірок
порожні**. `TableScore.empty_ratio` рахує саме це, тож бейк-оф і рантайм
міряють одну й ту саму величину, а не дві схожі.
"""

from __future__ import annotations

import csv
import math
import re
import unicodedata
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

__all__ = [
    "EMPTY_CELL_REPAIR_THRESHOLD",
    "Grid",
    "TableScore",
    "cell_f1",
    "grid_from_html",
    "grid_from_markdown",
    "load_grid_csv",
    "normalize_cell",
    "score_table",
]

# Тригер VLM-ремонту сторінки з плану, §1.
EMPTY_CELL_REPAIR_THRESHOLD = 0.30

Grid = list[list[str]]

_APOSTROPHES = {"’": "'", "ʼ": "'", "´": "'", "′": "'", "`": "'"}
_DASHES = {"–": "-", "—": "-", "−": "-", "‑": "-"}


def normalize_cell(text: str, *, casefold: bool = True) -> str:
    """Нормалізувати вміст комірки перед порівнянням.

    Кома як десятковий роздільник НЕ зводиться до крапки: «15,3» і «15.3» —
    це різні написання, і рушій, що змінює роздільник, змінює значення в
    таблиці стрільби для того, хто читає її очима.
    """
    value = unicodedata.normalize("NFC", text or "")
    for src, dst in {**_APOSTROPHES, **_DASHES}.items():
        value = value.replace(src, dst)
    value = " ".join(value.split())
    return value.casefold() if casefold else value


def _cells(grid: Sequence[Sequence[str]], *, casefold: bool) -> dict[tuple[int, int], str]:
    out: dict[tuple[int, int], str] = {}
    for r, row in enumerate(grid):
        for c, value in enumerate(row):
            normalized = normalize_cell(value, casefold=casefold)
            if normalized:
                out[(r, c)] = normalized
    return out


def _f1(matched: int, gold: int, pred: int) -> tuple[float, float, float]:
    precision = matched / pred if pred else math.nan
    recall = matched / gold if gold else math.nan
    if not precision or not recall or math.isnan(precision) or math.isnan(recall):
        return precision, recall, 0.0 if (pred or gold) else math.nan
    return precision, recall, 2 * precision * recall / (precision + recall)


def cell_f1(
    gold: Sequence[Sequence[str]],
    predicted: Sequence[Sequence[str]],
    *,
    positional: bool = True,
    casefold: bool = True,
) -> tuple[float, float, float]:
    """(precision, recall, F1) по НЕПОРОЖНІХ комірках.

    Порожні комірки не входять у знаменник жодної метрики: у балістичних
    таблицях вони становлять до чверті сітки, і рушій, що видав порожню сітку
    правильного розміру, інакше отримав би пристойний бал ні за що.
    """
    gold_cells = _cells(gold, casefold=casefold)
    pred_cells = _cells(predicted, casefold=casefold)
    if positional:
        matched = sum(1 for key, value in gold_cells.items() if pred_cells.get(key) == value)
    else:
        gold_bag = Counter(gold_cells.values())
        pred_bag = Counter(pred_cells.values())
        matched = sum((gold_bag & pred_bag).values())
    return _f1(matched, len(gold_cells), len(pred_cells))


@dataclass(slots=True)
class TableScore:
    """Оцінка однієї таблиці одним режимом розпізнавання."""

    table: str
    engine: str
    gold_shape: tuple[int, int]
    pred_shape: tuple[int, int]
    positional_precision: float
    positional_recall: float
    positional_f1: float
    content_f1: float
    empty_ratio: float
    seconds: float = 0.0
    error: str = ""

    @property
    def shape_ok(self) -> bool:
        return self.gold_shape == self.pred_shape

    @property
    def structure_loss(self) -> float:
        """Скільки F1 втрачено САМЕ на структурі, а не на розпізнаванні."""
        if math.isnan(self.content_f1) or math.isnan(self.positional_f1):
            return math.nan
        return self.content_f1 - self.positional_f1

    @property
    def needs_vlm_repair(self) -> bool:
        """Той самий тригер, що й у рантаймі (план, §1)."""
        return self.empty_ratio > EMPTY_CELL_REPAIR_THRESHOLD

    def as_dict(self) -> dict[str, Any]:
        return {
            "таблиця": self.table,
            "режим": self.engine,
            "еталон": f"{self.gold_shape[0]}×{self.gold_shape[1]}",
            "розпізнано": f"{self.pred_shape[0]}×{self.pred_shape[1]}",
            "P": self.positional_precision,
            "R": self.positional_recall,
            "F1 позиц.": self.positional_f1,
            "F1 зміст.": self.content_f1,
            "втрата структури": self.structure_loss,
            "порожніх": self.empty_ratio,
            "с": self.seconds,
            "помилка": self.error,
        }


def score_table(
    name: str,
    engine: str,
    gold: Sequence[Sequence[str]],
    predicted: Sequence[Sequence[str]],
    *,
    seconds: float = 0.0,
    error: str = "",
) -> TableScore:
    """Порахувати обидва F1 і частку порожніх комірок."""
    p, r, f1 = cell_f1(gold, predicted, positional=True)
    _, _, content = cell_f1(gold, predicted, positional=False)
    total = sum(len(row) for row in predicted)
    filled = len(_cells(predicted, casefold=True))
    return TableScore(
        table=name,
        engine=engine,
        gold_shape=(len(gold), max((len(r_) for r_ in gold), default=0)),
        pred_shape=(len(predicted), max((len(r_) for r_ in predicted), default=0)),
        positional_precision=p,
        positional_recall=r,
        positional_f1=f1,
        content_f1=content,
        empty_ratio=(total - filled) / total if total else math.nan,
        seconds=seconds,
        error=error,
    )


# --------------------------------------------------------------- завантаження
def load_grid_csv(path: Path | str, *, delimiter: str = ",") -> Grid:
    """Еталонна таблиця з CSV. `utf-8-sig` — бо еталон роблять в Excel.

    УВАГА на десяткову кому: «0,002» і «Дальність, м» у CSV з комою-роздільником
    мусять бути в лапках, інакше рядок розпадеться на зайві колонки й еталон
    тихо стане іншим. Український Excel зберігає такі файли з `;` — тоді
    передавайте `delimiter=";"`.
    """
    with Path(path).open("r", encoding="utf-8-sig", newline="") as fh:
        return [list(row) for row in csv.reader(fh, delimiter=delimiter)]


_MD_SEPARATOR = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")


def grid_from_markdown(markdown: str) -> Grid:
    """Розібрати Markdown-таблицю (те, що видає `DoclingDocument.export_to_markdown`).

    Рядок-роздільник (`|---|---|`) відкидається: це розмітка, а не дані, і
    зарахований як рядок він зсунув би всі координати нижче на одиницю —
    тобто обнулив би позиційний F1 при ідеальному розпізнаванні.
    """
    grid: Grid = []
    for line in markdown.splitlines():
        stripped = line.strip()
        if not stripped or "|" not in stripped:
            continue
        if _MD_SEPARATOR.match(stripped):
            continue
        body = stripped.strip("|")
        grid.append([cell.strip() for cell in body.split("|")])
    return grid


class _TableHtmlParser(HTMLParser):
    """Мінімальний розбір `<table>`: `docling` експортує таблиці саме в HTML.

    `colspan`/`rowspan` РОЗГОРТАЮТЬСЯ у повторені комірки. Інакше об'єднаний
    заголовок «Дальність, м» над трьома колонками з'їдав би дві координати, і
    вся права частина таблиці зсувалася б — тобто типова балістична таблиця
    отримувала б нульовий позиційний F1 за правильного розпізнавання.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: Grid = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None
        self._colspan = 1
        self._rowspan = 1
        self._pending: dict[int, tuple[int, str]] = {}  # колонка → (скільки рядків, текст)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            values = dict(attrs)
            self._cell = []
            self._colspan = max(1, int(values.get("colspan") or 1))
            self._rowspan = max(1, int(values.get("rowspan") or 1))

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th") and self._cell is not None and self._row is not None:
            text = " ".join("".join(self._cell).split())
            for _ in range(self._colspan):
                self._row.append(text)
            if self._rowspan > 1:
                column = len(self._row) - 1
                self._pending[column] = (self._rowspan - 1, text)
            self._cell = None
        elif tag == "tr" and self._row is not None:
            for column, (left, text) in sorted(self._pending.items()):
                if left > 0 and column >= len(self._row):
                    self._row.extend("" for _ in range(column - len(self._row) + 1))
                    self._row[column] = text
            self._pending = {
                col: (left - 1, text) for col, (left, text) in self._pending.items() if left - 1 > 0
            }
            self.rows.append(self._row)
            self._row = None


def grid_from_html(html: str) -> Grid:
    """Сітка з HTML-таблиці."""
    parser = _TableHtmlParser()
    parser.feed(html)
    parser.close()
    return parser.rows
