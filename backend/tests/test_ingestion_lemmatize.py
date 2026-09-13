"""Гейти української морфології перед пошуковим рушієм.

Два твердження тут коштують найдорожче, якщо їх зламати:
  * позначення (Д-30, ДСТУ 3008:2015) НІКОЛИ не лематизуються — інакше
    артилерійський корпус перестає знаходитись за індексами й кодами;
  * аналізатор викликається раз на УНІКАЛЬНУ словоформу — інакше індексація
    100 тис. чанків займає години замість хвилин.

Морфологічні твердження перевіряються на ВПРОВАДЖЕНОМУ бекенді, а не на
pymorphy3: тест має проходити і на машині без словників, і у stub-режимі.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from app.ingestion.lemmatize import (
    DESIGNATION_RE,
    UK_STOPWORDS,
    FallbackBackend,
    Lemmatizer,
    MemoryLemmaCache,
    SqliteLemmaCache,
    analyze_text,
    is_designation,
    lemmatize,
)

MIGRATION = Path(__file__).resolve().parents[1] / "app" / "db" / "migrations" / "0001_initial.sql"


class CountingBackend:
    """Детермінований «аналізатор»: відрізає закінчення й рахує виклики."""

    name = "counting"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def normal_forms(self, surface: str, lang: str) -> list[str]:
        self.calls.append(surface)
        return [surface[:-1] if len(surface) > 4 and surface[-1] in "иіїаяоеєуюь" else surface]


@pytest.fixture()
def counting() -> CountingBackend:
    return CountingBackend()


@pytest.fixture()
def lemmatizer(counting: CountingBackend) -> Lemmatizer:
    return Lemmatizer(backend=counting, cache=MemoryLemmaCache())


@pytest.fixture()
def con(tmp_path: Path) -> sqlite3.Connection:
    c = sqlite3.connect(tmp_path / "t.db")
    c.executescript(MIGRATION.read_text(encoding="utf-8"))
    return c


# ------------------------------------------------------------------ позначення
@pytest.mark.parametrize(
    "token",
    ["д-30", "2с1", "бм-21", "мт-12", "3,14", "1.2.3", "д30", "152мм", "nato", "stanag"],
)
def test_designations_are_recognised(token: str) -> None:
    assert is_designation(token), token


@pytest.mark.parametrize("word", ["гармата", "ту", "en", "снаряд", "балістика"])
def test_ordinary_words_are_not_designations(word: str) -> None:
    """«ТУ» і «EN» без числа — звичайні слова; голими вони дали б потік хиб."""
    assert not is_designation(word)


def test_designations_never_reach_the_lemmatizer(lemmatizer: Lemmatizer,
                                                 counting: CountingBackend) -> None:
    stream = lemmatizer.analyze("Гармата Д-30 відповідає ДСТУ 3008:2015, калібр 122,5 мм.")
    assert "д-30" in stream.codes
    assert "дсту 3008:2015" in stream.codes
    assert "122,5" in stream.codes
    assert not any(c.startswith("д-3") for c in stream.lemmas)
    # Аналізатор не бачив жодного позначення.
    assert all("д-30" not in call and call != "дсту" for call in counting.calls)


def test_multi_token_standard_designation_is_kept_whole(lemmatizer: Lemmatizer) -> None:
    """«ДСТУ 3008:2015» — три токени; поштокенна перевірка його б не впізнала."""
    stream = lemmatizer.analyze("Оформлення за ДСТУ 3008:2015 обов'язкове.")
    assert "дсту 3008:2015" in stream.codes


def test_designation_regex_is_anchored_on_whole_token() -> None:
    assert DESIGNATION_RE.fullmatch("д-30")
    assert not DESIGNATION_RE.fullmatch("гармата")


# ------------------------------------------------------------------------ кеш
def test_analyzer_is_called_once_per_unique_surface(lemmatizer: Lemmatizer,
                                                    counting: CountingBackend) -> None:
    """10^6 викликів замість 10^8 — саме заради цього існує lemma_cache."""
    tokens = ["гармати", "гармати", "гармата", "гармати", "снаряда", "снаряда"]
    lemmatizer.lemmatize(tokens)
    assert sorted(counting.calls) == ["гармата", "гармати", "снаряда"]


def test_cache_survives_between_calls(lemmatizer: Lemmatizer, counting: CountingBackend) -> None:
    lemmatizer.lemmatize(["гармати"])
    lemmatizer.lemmatize(["гармати", "гармати"])
    assert counting.calls == ["гармати"]


def test_sqlite_cache_round_trip(con: sqlite3.Connection, counting: CountingBackend) -> None:
    """Кеш переживає перезапуск воркера: другий лематизатор аналізатор не кличе."""
    first = Lemmatizer(backend=counting, cache=SqliteLemmaCache(con))
    first.lemmatize(["гармати", "снаряда"])
    assert len(counting.calls) == 2

    second = Lemmatizer(backend=counting, cache=SqliteLemmaCache(con))
    second.lemmatize(["гармати", "снаряда"])
    assert len(counting.calls) == 2, "кеш у lemma_cache не спрацював"

    rows = con.execute("SELECT surface, lang, lemma FROM lemma_cache ORDER BY surface").fetchall()
    assert [r[0] for r in rows] == ["гармати", "снаряда"]
    assert all(r[1] == "uk" for r in rows)


# ------------------------------------------------------------------ стоп-лист
def test_stopwords_are_filtered_only_from_lemmas(lemmatizer: Lemmatizer) -> None:
    stream = lemmatizer.lemmatize(["і", "не", "гармата"])
    assert "і" in stream.forms and "не" in stream.forms
    assert "і" not in stream.lemmas and "не" not in stream.lemmas
    # «гармат» — лема від фіктивного аналізатора фікстури, не від pymorphy3.
    assert stream.lemmas == ["гармат"]


def test_stoplist_has_reasonable_size() -> None:
    """~250 службових слів. Різке падіння означає, що список зіпсували."""
    assert 200 <= len(UK_STOPWORDS) <= 400


# ---------------------------------------------------------------- три потоки
def test_three_streams_map_onto_fts_columns(lemmatizer: Lemmatizer) -> None:
    lemmas, forms, codes = lemmatizer.analyze("Гармата Д-30 стріляє.").as_fts_columns()
    assert isinstance(lemmas, str) and isinstance(forms, str) and isinstance(codes, str)
    assert "д-30" in codes
    assert "гармата" in forms


def test_compound_words_reach_both_lemmas_and_forms(lemmatizer: Lemmatizer) -> None:
    stream = lemmatizer.analyze("військово-технічний")
    assert "військово-технічний" in stream.forms
    assert "технічний" in stream.forms


def test_module_level_entry_points_work_without_configuration() -> None:
    """`lemmatize` і `analyze_text` мусять працювати «з коробки» — і без словників."""
    stream = lemmatize(["гармата"], "uk")
    assert stream.forms == ["гармата"]
    assert analyze_text("Гармата Д-30.").codes == ["д-30"]


def test_fallback_backend_degrades_honestly() -> None:
    """Без морфології лема == словоформа; пошук працює, просто гірше."""
    stream = Lemmatizer(backend=FallbackBackend(), cache=MemoryLemmaCache()).lemmatize(["гармати"])
    assert stream.lemmas == ["гармати"]
    assert stream.backend == "fallback"


def test_query_and_index_use_the_same_normalisation(lemmatizer: Lemmatizer) -> None:
    """Індексований текст і запит мусять дати ті самі леми."""
    indexed = lemmatizer.analyze("Гар-\nмата Д–30")
    queried = lemmatizer.analyze("гармата Д-30")
    assert indexed.codes == queried.codes
    assert indexed.lemmas == queried.lemmas
