"""Українська морфологія ДО пошукового рушія.

Ключовий факт, який визначає всю конструкцію: жоден пошуковий рушій не має
українського стемера. Список Snowball має 28 мов, української серед них немає,
і FTS5, Tantivy, LanceDB та DuckDB FTS усі успадковують той самий список. Тому
лематизація робиться тут, у Python, на етапі індексації, а в індекс іде вже
потік лем — після чого вибір BM25-рушія майже не має значення.

Три потоки на виході (рівно три колонки `chunk_fts`):
    lemmas — нормальні форми, зі стоп-листом; головна колонка BM25 (вага 1.0)
    forms  — нормалізовані словоформи як є, без стоп-листа (вага 0.4);
             рятує там, де аналізатор помилився з лемою
    codes  — позначення: Д-30, 2С1, ДСТУ 3008:2015 (вага 2.5). НІКОЛИ не
             лематизуються: «Д-30» не має нормальної форми, а будь-яка спроба
             її знайти дає сміття

Кеш словоформ обов'язковий. Аналізатор має викликатись раз на УНІКАЛЬНУ
словоформу, а не раз на токен: на 100 тис. чанків це ~10^6 викликів (30–90 с)
замість ~10^8 (години). Саме для цього в схемі є таблиця `lemma_cache`.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from app.ingestion.normalize_uk import (
    expand_compounds,
    normalize_uk,
    tokenize,
)
from app.ingestion.uk_lexicon import STOPWORDS

__all__ = [
    "DESIGNATION_RE",
    "UK_STOPWORDS",
    "LemmaStream",
    "MorphBackend",
    "FallbackBackend",
    "PyMorphyBackend",
    "SimplemmaBackend",
    "MemoryLemmaCache",
    "SqliteLemmaCache",
    "Lemmatizer",
    "lemmatize",
    "analyze_text",
    "default_lemmatizer",
    "is_designation",
]

log = logging.getLogger(__name__)

# Стоп-лист живе в модулі даних `uk_lexicon`, бо він потрібен і словниковій
# перевірці проби (стоп-слова — найчастотніші токени тексту), і тут.
UK_STOPWORDS = STOPWORDS

# ------------------------------------------------------------------ позначення
# Ці токени йдуть у колонку `codes` і НІКОЛИ не лематизуються.
# «Голими» дозволені лише NATO і STANAG: «ТУ» та «EN» збігаються зі звичайними
# словами («ту гармату», «en»), і без вимоги числа поруч вони дали б потік
# хибних спрацювань.
_STD_NUMBERED = r"ДСТУ|ГОСТ|ДБН|СОУ|ОСТ|ТУ|ТКП|ISO|IEC|EN|MIL-STD|AQAP|STANAG|NATO|AR|FM"
_STD_BARE = r"NATO|STANAG"

DESIGNATION_RE = re.compile(
    rf"""(?xi)
      (?:\b(?:{_STD_NUMBERED})\b\s*[-–—]?\s*
         \d[\w./–-]*(?:\s*:\s*\d{{2,4}})?)                         # ДСТУ 3008:2015 · ISO 9001
    | (?:\b(?:{_STD_BARE})\b)                                     # голі NATO / STANAG
    | (?:\b\d+[.,]\d+(?:[.,]\d+)*\b)                              # десяткові: 3,14 · 1.2.3
    | (?:\b[^\W\d_]{{1,4}}[-–—]\d+[^\W_]*\b)                        # Д-30 · БМ-21 · МТ-12
    | (?:\b\d+[^\W\d_]{{1,4}}\d*\b)                               # 2С1 · 2А36 · 152мм
    | (?:\b[^\W\d_]{{1,4}}\d{{2,}}\b)                             # Д30 (без дефіса)
    """,
)

_DIGITS_ONLY_RE = re.compile(r"^\d+$")


def is_designation(token: str) -> bool:
    """Чи є ОКРЕМИЙ токен позначенням (повний збіг)."""
    return DESIGNATION_RE.fullmatch(token) is not None


# --------------------------------------------------------------------- потоки
@dataclass(slots=True)
class LemmaStream:
    """Три потоки під три колонки `chunk_fts`."""

    lemmas: list[str] = field(default_factory=list)
    forms: list[str] = field(default_factory=list)
    codes: list[str] = field(default_factory=list)
    backend: str = "fallback"

    def as_fts_columns(self) -> tuple[str, str, str]:
        """Рівно те, що подається у `ChunkRepo.index_fts(chunk_id, ...)`."""
        return " ".join(self.lemmas), " ".join(self.forms), " ".join(self.codes)

    def __bool__(self) -> bool:
        return bool(self.lemmas or self.forms or self.codes)


# -------------------------------------------------------------------- бекенди
class MorphBackend(Protocol):
    """Мінімальний контракт морфологічного аналізатора."""

    name: str

    def normal_forms(self, surface: str, lang: str) -> list[str]:
        """До двох нормальних форм словоформи, найкраща першою."""
        ...


class FallbackBackend:
    """Деградація без морфології: лема == словоформа.

    Свідомо чесна: пошук працює, просто гірше на словозміні. Це та поведінка,
    яку отримує stub-режим і будь-яка машина, куди не доїхали словники.
    """

    name = "fallback"

    def normal_forms(self, surface: str, lang: str) -> list[str]:
        return [surface]


class PyMorphyBackend:
    """pymorphy3 + pymorphy3-dicts-uk (і -dicts-ru для російських фрагментів).

    Той самий стек, що посів 2-ге місце на UNLP 2026 Shared Task з українського
    document QA. Аналізатори створюються ліниво й кешуються: конструктор читає
    словник із диска і коштує сотні мілісекунд.
    """

    name = "pymorphy3"

    def __init__(self) -> None:
        import pymorphy3  # локальний імпорт: пакет опційний

        self._pymorphy3 = pymorphy3
        self._analyzers: dict[str, object] = {}

    def _analyzer(self, lang: str):  # тип належить опційному пакету
        key = "ru" if lang == "ru" else "uk"
        analyzer = self._analyzers.get(key)
        if analyzer is None:
            analyzer = self._pymorphy3.MorphAnalyzer(lang=key)
            self._analyzers[key] = analyzer
        return analyzer

    def normal_forms(self, surface: str, lang: str) -> list[str]:
        parses = self._analyzer(lang).parse(surface)
        out: list[str] = []
        for parse in parses:
            form = parse.normal_form
            if form and form not in out:
                out.append(form)
            if len(out) == 2:
                break
        return out or [surface]


class SimplemmaBackend:
    """simplemma — фолбек на випадок, коли немає pymorphy3 або його словників."""

    name = "simplemma"

    def __init__(self) -> None:
        import simplemma  # локальний імпорт: пакет опційний

        self._simplemma = simplemma

    def normal_forms(self, surface: str, lang: str) -> list[str]:
        code = "ru" if lang == "ru" else "uk"
        try:
            return [self._simplemma.lemmatize(surface, lang=code)]
        except (ValueError, KeyError):
            return [surface]


_backend_singleton: MorphBackend | None = None


def _load_backend() -> MorphBackend:
    """Обрати найкращий доступний аналізатор. Логувати деградацію чесно."""
    global _backend_singleton
    if _backend_singleton is not None:
        return _backend_singleton
    for factory, why in (
        (PyMorphyBackend, "pymorphy3"),
        (SimplemmaBackend, "simplemma"),
    ):
        try:
            _backend_singleton = factory()  # type: ignore[assignment]
            return _backend_singleton
        except Exception as exc:  # будь-яка проблема пакета = деградація, не збій
            log.warning(
                "Морфологічний аналізатор %s недоступний (%s). "
                "Продовжую без лематизації — якість пошуку за словозміною буде нижчою.",
                why, exc,
            )
    _backend_singleton = FallbackBackend()
    return _backend_singleton


# ----------------------------------------------------------------------- кеші
class MemoryLemmaCache:
    """Кеш у пам'яті процесу. Використовується у тестах і в разових прогонах."""

    def __init__(self) -> None:
        self._data: dict[tuple[str, str], tuple[str, str | None]] = {}

    def get_many(self, surfaces: Iterable[str], lang: str) -> dict[str, tuple[str, str | None]]:
        return {s: self._data[(s, lang)] for s in surfaces if (s, lang) in self._data}

    def put_many(self, items: dict[str, tuple[str, str | None]], lang: str) -> None:
        for surface, value in items.items():
            self._data[(surface, lang)] = value


class SqliteLemmaCache:
    """Кеш у таблиці `lemma_cache` — переживає перезапуск воркера.

    Саме він перетворює 10^8 викликів аналізатора на 10^6: індексація
    100-тисячного корпусу бачить кожну словоформу вперше рівно один раз за
    всю історію бази, а не один раз на документ.
    """

    _BATCH = 400  # межа на кількість параметрів у SQLite — 999 за замовчуванням

    def __init__(self, con: sqlite3.Connection) -> None:
        self.con = con

    def get_many(self, surfaces: Iterable[str], lang: str) -> dict[str, tuple[str, str | None]]:
        out: dict[str, tuple[str, str | None]] = {}
        unique = list(dict.fromkeys(surfaces))
        for i in range(0, len(unique), self._BATCH):
            batch = unique[i:i + self._BATCH]
            marks = ",".join("?" * len(batch))
            rows = self.con.execute(
                f"SELECT surface, lemma, lemma2 FROM lemma_cache"
                f" WHERE lang=? AND surface IN ({marks})",
                (lang, *batch),
            ).fetchall()
            for row in rows:
                out[row[0]] = (row[1], row[2])
        return out

    def put_many(self, items: dict[str, tuple[str, str | None]], lang: str) -> None:
        if not items:
            return
        self.con.executemany(
            "INSERT INTO lemma_cache (surface, lang, lemma, lemma2) VALUES (?,?,?,?)"
            " ON CONFLICT(surface, lang) DO UPDATE SET"
            " lemma=excluded.lemma, lemma2=excluded.lemma2",
            [(surface, lang, lemma, lemma2) for surface, (lemma, lemma2) in items.items()],
        )


# ----------------------------------------------------------------- лематизатор
class Lemmatizer:
    """Перетворює текст або токени на три потоки для FTS5."""

    def __init__(
        self,
        *,
        backend: MorphBackend | None = None,
        cache: MemoryLemmaCache | SqliteLemmaCache | None = None,
        stopwords: frozenset[str] = UK_STOPWORDS,
    ) -> None:
        self.backend = backend or _load_backend()
        self.cache = cache if cache is not None else MemoryLemmaCache()
        self.stopwords = stopwords

    # -- внутрішнє ---------------------------------------------------------
    def _lemmas_for(self, surfaces: Sequence[str], lang: str) -> dict[str, tuple[str, str | None]]:
        """Один виклик аналізатора на УНІКАЛЬНУ словоформу — і жодного зайвого."""
        unique = list(dict.fromkeys(surfaces))
        known = self.cache.get_many(unique, lang)
        missing = [s for s in unique if s not in known]
        fresh: dict[str, tuple[str, str | None]] = {}
        for surface in missing:
            forms = self.backend.normal_forms(surface, lang) or [surface]
            lemma = forms[0]
            lemma2 = forms[1] if len(forms) > 1 and forms[1] != lemma else None
            fresh[surface] = (lemma, lemma2)
        if fresh:
            self.cache.put_many(fresh, lang)
            known.update(fresh)
        return known

    # -- публічне ----------------------------------------------------------
    def lemmatize(self, tokens: Sequence[str], lang: str = "uk") -> LemmaStream:
        """Розкласти вже нарізані токени на три потоки.

        Токени вважаються нормалізованими; якщо ні — нормалізація застосується
        тут, бо розбіжність нормалізації індексу й запиту тихо руйнує BM25.
        """
        codes: list[str] = []
        forms: list[str] = []
        word_surfaces: list[str] = []

        for raw in tokens:
            token = normalize_uk(raw).strip()
            if not token:
                continue
            if is_designation(token) or (len(token) >= 2 and _DIGITS_ONLY_RE.match(token)):
                # Позначення й голі числа — лексичні якорі. У `forms` вони теж
                # потрібні, бо запит може прийти без екранування колонки codes.
                codes.append(token)
                forms.append(token)
                continue
            for part in expand_compounds(token):
                forms.append(part)
                word_surfaces.append(part)

        table = self._lemmas_for(word_surfaces, lang)
        lemmas: list[str] = []
        for surface in word_surfaces:
            lemma, lemma2 = table.get(surface, (surface, None))
            # Стоп-лист фільтрується ЛИШЕ тут: у `forms` службові слова
            # лишаються, бо на фразових запитах вони інколи несуть сенс.
            if lemma not in self.stopwords:
                lemmas.append(lemma)
            if lemma2 and lemma2 not in self.stopwords:
                lemmas.append(lemma2)

        return LemmaStream(lemmas=lemmas, forms=forms, codes=codes, backend=self.backend.name)

    def analyze(self, text: str, lang: str = "uk") -> LemmaStream:
        """Повний конвеєр із сирого тексту. Основна точка входу для індексації.

        Позначення виймаються з ТЕКСТУ, а не з токенів, бо «ДСТУ 3008:2015» —
        це три токени, і жодна поштокенна перевірка його не впізнає.
        """
        normalized = normalize_uk(text)
        codes: list[str] = []
        spans: list[tuple[int, int]] = []
        for match in DESIGNATION_RE.finditer(normalized):
            codes.append(re.sub(r"\s+", " ", match.group(0)).strip())
            spans.append(match.span())

        # Вирізати позначення з тексту, щоб їхні уламки не потрапили в леми.
        if spans:
            pieces: list[str] = []
            cursor = 0
            for start, end in spans:
                pieces.append(normalized[cursor:start])
                pieces.append(" ")
                cursor = end
            pieces.append(normalized[cursor:])
            body = "".join(pieces)
        else:
            body = normalized

        stream = self.lemmatize(tokenize(body), lang)
        # Позначення додаються попереду: порядок у колонці неважливий для BM25,
        # але детермінований порядок робить діагностику відтворюваною.
        stream.codes = codes + stream.codes
        stream.forms = codes + stream.forms
        return stream


_default_lemmatizer: Lemmatizer | None = None


def default_lemmatizer(con: sqlite3.Connection | None = None) -> Lemmatizer:
    """Спільний лематизатор процесу. З'єднання вмикає стійкий кеш словоформ."""
    global _default_lemmatizer
    if con is not None:
        return Lemmatizer(cache=SqliteLemmaCache(con))
    if _default_lemmatizer is None:
        _default_lemmatizer = Lemmatizer()
    return _default_lemmatizer


def lemmatize(
    tokens: Sequence[str],
    lang: str = "uk",
    *,
    lemmatizer: Lemmatizer | None = None,
) -> LemmaStream:
    """Публічний інтерфейс модуля (див. docs/CONTRACT.md)."""
    return (lemmatizer or default_lemmatizer()).lemmatize(tokens, lang)


def analyze_text(
    text: str,
    lang: str = "uk",
    *,
    lemmatizer: Lemmatizer | None = None,
) -> LemmaStream:
    """Те саме, але із сирого тексту — з коректним вийманням позначень."""
    return (lemmatizer or default_lemmatizer()).analyze(text, lang)
