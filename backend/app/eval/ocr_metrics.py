"""Метрики OCR: CER, WER і УКРАЇНСЬКІ ПЛУТАНИНИ.

Публікованої оцінки якості OCR українською не існує ДЛЯ ЖОДНОГО рушія — це
записано в реєстр ризиків НДР як ризик №1, і Віха 1 плану саме про те, щоб її
виміряти. Тому цей модуль — не допоміжний, а один із двох носіїв наукового
результату проєкту (другий — таблиця BM25 vs гібрид).

ЧОМУ CER І WER НЕДОСТАТНЬО
--------------------------
Два рушії з однаковим CER = 3% можуть бути придатні й непридатні відповідно.
Помилка «Гаубиця» → «Гаубица» коштує читабельності мало. Помилка
«і» (U+0456) → «i» (латинська, U+0069) не змінює ані вигляду сторінки, ані
CER помітно — але ЛАМАЄ ПОШУК: лема не побудується, FTS5 не знайде, а викладач
побачить порожню відповідь на питання, текст якого лежить у нього в
підручнику. Саме тому виправлення гомогліфів живе в `normalize_uk`, і саме
тому бейк-оф мусить рахувати ці пари окремо, а не ховати їх у середньому CER.

Класи, які рахуються (Віха 1 плану): `і/i`, `и/й`, `ї/i`, `є/e`, `ґ/г`,
апостроф, і загальний клас кирилично-латинських гомогліфів.

ВИРІВНЮВАННЯ
------------
Відстань редагування — точна (динамічне програмування у два рядки). Пари
підстановок беруться з `difflib.SequenceMatcher` із **`autojunk=False`**:
з увімкненим autojunk на послідовностях довших за 200 елементів символи, що
трапляються частіше за 1%, оголошуються «сміттям» — тобто на суцільному тексті
ігноруються пробіли й найчастіші літери, і вирівнювання стає безглуздим.
Це найтихіша пастка всього модуля.
"""

from __future__ import annotations

import difflib
import math
import re
import statistics
import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

__all__ = [
    "APOSTROPHES",
    "CONFUSION_CLASSES",
    "EngineReport",
    "PageScore",
    "aggregate",
    "align_pairs",
    "cer",
    "confusion_counts",
    "edit_distance",
    "normalize_for_scoring",
    "report_rows",
    "score_page",
    "wer",
]

# Усі варіанти апострофа, які трапляються в українських текстах і в OCR-виводі.
APOSTROPHES = frozenset("'’ʼ`´′‘")

# Пари подаються НЕВПОРЯДКОВАНО: рахується плутанина, а не напрямок. Напрямок
# зберігається окремо в `PageScore.substitutions`, коли він потрібен.
CONFUSION_CLASSES: dict[str, frozenset[frozenset[str]]] = {
    # Найдорожча пара: кирилична «і» U+0456 і латинська «i» U+0069 виглядають
    # однаково в будь-якому шрифті.
    "і/i": frozenset({
        frozenset({"і", "i"}), frozenset({"І", "I"}), frozenset({"і", "l"}),
        frozenset({"І", "l"}), frozenset({"і", "1"}), frozenset({"І", "1"}),
    }),
    "и/й": frozenset({frozenset({"и", "й"}), frozenset({"И", "Й"})}),
    "ї/i": frozenset({
        frozenset({"ї", "i"}), frozenset({"ї", "ï"}), frozenset({"ї", "і"}),
        frozenset({"Ї", "I"}), frozenset({"Ї", "І"}),
    }),
    "є/e": frozenset({
        frozenset({"є", "e"}), frozenset({"є", "с"}), frozenset({"Є", "E"}),
        frozenset({"є", "ε"}),
    }),
    "ґ/г": frozenset({frozenset({"ґ", "г"}), frozenset({"Ґ", "Г"})}),
    # Кирилиця → латиниця для літер, що збігаються за формою. Саме цей клас
    # тихо руйнує BM25: слово виглядає правильно й не знаходиться.
    "гомогліфи": frozenset({
        frozenset({"а", "a"}), frozenset({"А", "A"}), frozenset({"е", "e"}),
        frozenset({"Е", "E"}), frozenset({"о", "o"}), frozenset({"О", "O"}),
        frozenset({"с", "c"}), frozenset({"С", "C"}), frozenset({"р", "p"}),
        frozenset({"Р", "P"}), frozenset({"х", "x"}), frozenset({"Х", "X"}),
        frozenset({"у", "y"}), frozenset({"У", "Y"}), frozenset({"к", "k"}),
        frozenset({"К", "K"}), frozenset({"М", "M"}), frozenset({"Т", "T"}),
        frozenset({"В", "B"}), frozenset({"Н", "H"}),
    }),
}

_WORD_SPLIT = re.compile(r"\s+")
_SOFT_HYPHEN = "­"


@lru_cache(maxsize=1)
def _class_index() -> dict[frozenset[str], str]:
    """Пара символів → назва класу. Один словник замість шести перевірок."""
    index: dict[frozenset[str], str] = {}
    for name, pairs in CONFUSION_CLASSES.items():
        for pair in pairs:
            index.setdefault(pair, name)
    return index


def normalize_for_scoring(
    text: str,
    *,
    collapse_whitespace: bool = True,
    unify_apostrophes: bool = False,
    drop_apostrophes: bool = False,
    casefold: bool = False,
) -> str:
    """Нормалізація ПЕРЕД вимірюванням.

    За замовчуванням мінімальна: NFC, зняття м'якого переносу й згортання
    пробілів. Регістр НЕ зводиться — переплутаний регістр є помилкою OCR, і
    ховати його означало б завищити оцінку рушія.

    Два необов'язкові послаблення стосуються апострофа:
      * `unify_apostrophes` зводить `’ ʼ ´ ′ ` ‘` до `'` — тобто прощає лише
        СТИЛЬ апострофа;
      * `drop_apostrophes` прибирає апостроф зовсім — тобто прощає і його
        втрату («пять» замість «п'ять»), яка на практиці і є найчастішою.
    Різниця між звичайним CER і CER із `drop_apostrophes` — це повна ціна
    апострофа для конкретного рушія (`PageScore.apostrophe_cost`).
    """
    text = unicodedata.normalize("NFC", text).replace(_SOFT_HYPHEN, "")
    if drop_apostrophes:
        for ch in APOSTROPHES:
            text = text.replace(ch, "")
    elif unify_apostrophes:
        for ch in APOSTROPHES:
            text = text.replace(ch, "'")
    if collapse_whitespace:
        text = " ".join(text.split())
    if casefold:
        text = text.casefold()
    return text


# ------------------------------------------------------- відстань редагування
@lru_cache(maxsize=1)
def _rapidfuzz() -> Any:
    """`rapidfuzz` пришвидшує DP у сотні разів, але НЕ є залежністю.

    Бейк-оф — офлайн-експеримент на 40 сторінках, а не гарячий шлях; тягти
    заради нього ще одне бінарне колесо в закритий контур немає підстав.
    Якщо пакет уже стоїть у середовищі розробника — використовуємо.
    """
    try:
        from rapidfuzz.distance import Levenshtein  # type: ignore[import-not-found]
    except Exception:
        return None
    return Levenshtein


def edit_distance(a: Sequence[Any], b: Sequence[Any]) -> int:
    """Відстань Левенштейна. Точна, у два рядки пам'яті."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    lev = _rapidfuzz()
    if lev is not None and isinstance(a, str) and isinstance(b, str):
        return int(lev.distance(a, b))

    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb))
            )
        previous = current
    return previous[-1]


def cer(reference: str, hypothesis: str, **norm: Any) -> float:
    """Character Error Rate = відстань редагування / довжина еталона.

    Може перевищувати 1.0 — і це правильно: рушій, що видав удвічі більше
    сміття, ніж є тексту, має отримати CER > 1, а не «100%».
    """
    ref = normalize_for_scoring(reference, **norm)
    hyp = normalize_for_scoring(hypothesis, **norm)
    if not ref:
        return math.nan if not hyp else math.inf
    return edit_distance(ref, hyp) / len(ref)


def wer(reference: str, hypothesis: str, **norm: Any) -> float:
    """Word Error Rate — та сама відстань, але над списками слів."""
    ref = [w for w in _WORD_SPLIT.split(normalize_for_scoring(reference, **norm)) if w]
    hyp = [w for w in _WORD_SPLIT.split(normalize_for_scoring(hypothesis, **norm)) if w]
    if not ref:
        return math.nan if not hyp else math.inf
    return edit_distance(ref, hyp) / len(ref)


# ------------------------------------------------------------ вирівнювання
# Стеля для локального точного вирівнювання всередині блоку заміни. 250 тис.
# клітинок — це ~0.1 с на чистому Python; блок, більший за це, означає, що
# рушій видав зовсім інший текст, і поштучні пари там усе одно беззмістовні.
_BLOCK_CELL_LIMIT = 250_000


def _align_block(a: str, b: str) -> list[tuple[str, str]]:
    """Точне вирівнювання короткого блоку з відновленням шляху.

    Порожній рядок у парі означає вставку або вилучення: `("ї", "")` — рушій
    з'їв літеру, `("", "'")` — дописав апостроф.
    """
    if len(a) * len(b) > _BLOCK_CELL_LIMIT:
        # Позиційне парування — грубо, але чесно позначене в `raw`.
        pairs = [(a[i] if i < len(a) else "", b[i] if i < len(b) else "") for i in range(max(len(a), len(b)))]
        return pairs

    n, m = len(a), len(b)
    dist = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dist[i][0] = i
    for j in range(1, m + 1):
        dist[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dist[i][j] = min(
                dist[i - 1][j] + 1,
                dist[i][j - 1] + 1,
                dist[i - 1][j - 1] + (a[i - 1] != b[j - 1]),
            )

    pairs: list[tuple[str, str]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dist[i][j] == dist[i - 1][j - 1] + (a[i - 1] != b[j - 1]):
            pairs.append((a[i - 1], b[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and dist[i][j] == dist[i - 1][j] + 1:
            pairs.append((a[i - 1], ""))
            i -= 1
        else:
            pairs.append(("", b[j - 1]))
            j -= 1
    pairs.reverse()
    return pairs


def align_pairs(reference: str, hypothesis: str) -> list[tuple[str, str]]:
    """Пари «еталон → розпізнане» лише для розбіжних ділянок.

    Збіжні ділянки не повертаються: на сторінці підручника їх 97%, і вони
    роздули б результат у тридцять разів без жодної нової інформації.
    """
    matcher = difflib.SequenceMatcher(None, reference, hypothesis, autojunk=False)
    out: list[tuple[str, str]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag == "replace":
            out.extend(_align_block(reference[i1:i2], hypothesis[j1:j2]))
        elif tag == "delete":
            out.extend((ch, "") for ch in reference[i1:i2])
        elif tag == "insert":
            out.extend(("", ch) for ch in hypothesis[j1:j2])
    return out


def confusion_counts(
    reference: str, hypothesis: str, **norm: Any
) -> tuple[Counter[str], Counter[tuple[str, str]]]:
    """(лічильник іменованих класів, лічильник сирих підстановок).

    Апостроф рахується і як підстановка (`'` → `’`), і як вилучення
    (`п'ять` → `пять`), і як вставка. Вилучення тут не дрібниця: FTS5-токенайзер
    налаштований так, що апостроф — частина слова, тож «пять» і «п'ять» — це
    два різні токени, і половина запитів про боєприпаси перестає знаходитись.
    """
    ref = normalize_for_scoring(reference, **norm)
    hyp = normalize_for_scoring(hypothesis, **norm)
    classes: Counter[str] = Counter()
    raw: Counter[tuple[str, str]] = Counter()
    index = _class_index()

    for a, b in align_pairs(ref, hyp):
        if a == b:
            continue
        raw[(a, b)] += 1
        if a in APOSTROPHES or b in APOSTROPHES:
            classes["апостроф"] += 1
            continue
        if not a or not b:
            classes["вставка/вилучення"] += 1
            continue
        name = index.get(frozenset({a, b}))
        if name:
            classes[name] += 1
        else:
            classes["інші"] += 1
    return classes, raw


# ------------------------------------------------------------------- звіти
@dataclass(slots=True)
class PageScore:
    """Оцінка одного рушія на одній сторінці."""

    page: str
    engine: str
    cer: float
    wer: float
    cer_soft: float                     # CER, якщо апостроф ігнорувати зовсім
    ref_chars: int
    hyp_chars: int
    confusions: Counter[str] = field(default_factory=Counter)
    substitutions: Counter[tuple[str, str]] = field(default_factory=Counter)
    seconds: float = 0.0
    error: str = ""

    @property
    def apostrophe_cost(self) -> float:
        """Скільки CER коштує САМЕ апостроф — і стиль, і втрата.

        Різниця двох вимірів. Величина мала за модулем і величезна за
        наслідками: токенайзер `chunk_fts` тримає апостроф усередині слова,
        тож «пять» і «п'ять» — різні токени, і половина запитів про
        боєприпаси перестає знаходитись при бездоганному вигляді сторінки.
        """
        if math.isnan(self.cer) or math.isnan(self.cer_soft):
            return math.nan
        return self.cer - self.cer_soft

    def as_dict(self) -> dict[str, Any]:
        return {
            "page": self.page, "engine": self.engine,
            "cer": self.cer, "wer": self.wer, "cer_soft": self.cer_soft,
            "ref_chars": self.ref_chars, "hyp_chars": self.hyp_chars,
            "seconds": self.seconds, "error": self.error,
            **{f"conf:{k}": v for k, v in sorted(self.confusions.items())},
        }


def score_page(
    page: str,
    engine: str,
    reference: str,
    hypothesis: str,
    *,
    seconds: float = 0.0,
    error: str = "",
) -> PageScore:
    """Порахувати всі метрики однієї сторінки одного рушія."""
    classes, raw = confusion_counts(reference, hypothesis)
    return PageScore(
        page=page,
        engine=engine,
        cer=cer(reference, hypothesis),
        wer=wer(reference, hypothesis),
        cer_soft=cer(reference, hypothesis, drop_apostrophes=True),
        ref_chars=len(normalize_for_scoring(reference)),
        hyp_chars=len(normalize_for_scoring(hypothesis)),
        confusions=classes,
        substitutions=raw,
        seconds=seconds,
        error=error,
    )


@dataclass(slots=True)
class EngineReport:
    """Підсумок по рушію на всьому наборі сторінок."""

    engine: str
    pages: int
    mean_cer: float
    median_cer: float
    mean_wer: float
    median_wer: float
    total_seconds: float
    confusions: Counter[str] = field(default_factory=Counter)
    top_substitutions: list[tuple[str, str, int]] = field(default_factory=list)
    failures: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "рушій": self.engine,
            "сторінок": self.pages,
            "CER сер.": self.mean_cer,
            "CER мед.": self.median_cer,
            "WER сер.": self.mean_wer,
            "WER мед.": self.median_wer,
            "с/стор.": self.total_seconds / self.pages if self.pages else math.nan,
            "збоїв": self.failures,
            **{k: v for k, v in sorted(self.confusions.items())},
        }


def aggregate(scores: Iterable[PageScore]) -> list[EngineReport]:
    """Звести посторінкові оцінки по рушіях.

    Медіана поруч із середнім навмисно: одна сторінка, на якій рушій вивалив
    нісенітницю (CER 3.4), зсуває середнє по 40 сторінках на 8 пунктів. Медіана
    показує типову сторінку, середнє — вартість хвоста; для рішення «який рушій
    брати» потрібні обидві.
    """
    by_engine: dict[str, list[PageScore]] = {}
    for score in scores:
        by_engine.setdefault(score.engine, []).append(score)

    out: list[EngineReport] = []
    for engine, items in sorted(by_engine.items()):
        good = [s for s in items if not s.error and not math.isnan(s.cer)]
        cers = [s.cer for s in good]
        wers = [s.wer for s in good if not math.isnan(s.wer)]
        confusions: Counter[str] = Counter()
        subs: Counter[tuple[str, str]] = Counter()
        for s in good:
            confusions.update(s.confusions)
            subs.update(s.substitutions)
        out.append(
            EngineReport(
                engine=engine,
                pages=len(good),
                mean_cer=statistics.fmean(cers) if cers else math.nan,
                median_cer=statistics.median(cers) if cers else math.nan,
                mean_wer=statistics.fmean(wers) if wers else math.nan,
                median_wer=statistics.median(wers) if wers else math.nan,
                total_seconds=sum(s.seconds for s in items),
                confusions=confusions,
                top_substitutions=[(a, b, n) for (a, b), n in subs.most_common(15)],
                failures=sum(1 for s in items if s.error),
            )
        )
    out.sort(key=lambda r: (math.inf if math.isnan(r.mean_cer) else r.mean_cer))
    return out


def report_rows(reports: Sequence[EngineReport]) -> list[dict[str, Any]]:
    """Рядки таблиці з однаковим набором колонок для всіх рушіїв."""
    rows = [r.as_dict() for r in reports]
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    return [{k: row.get(k, 0) for k in keys} for row in rows]
