"""Гейти для українського повнотекстового пошуку.

Кожне твердження тут — припущення, від якого залежить якість пошуку і яке
мовчки ламається при зміні токенайзера. Жодне з них не є очевидним.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

MIGRATION = Path(__file__).resolve().parents[1] / "app" / "db" / "migrations" / "0001_initial.sql"


def fts_term(term: str) -> str:
    """Єдиний дозволений спосіб подати термін у MATCH.

    Без цього запит, що містить "Д-30", парситься як оператор NOT, а ":" / "*" /
    "^" / "NEAR" змінюють семантику або кидають виняток.
    """
    return '"' + term.replace('"', '""') + '"'


@pytest.fixture()
def con(tmp_path: Path) -> sqlite3.Connection:
    c = sqlite3.connect(tmp_path / "t.db")
    c.execute("PRAGMA page_size = 8192")
    c.executescript(MIGRATION.read_text(encoding="utf-8"))
    return c


def _match(con: sqlite3.Connection, term: str, table: str = "chunk_fts") -> int:
    row = con.execute(
        f"SELECT count(*) FROM {table} WHERE {table} MATCH ?", (fts_term(term),)
    ).fetchone()
    return row[0]


@pytest.mark.parametrize(
    "term",
    ["п'ять", "об'єкт", "м'який", "гармата", "військово-технічний", "Д-30", "2С1"],
)
def test_tokenchars_keep_ukrainian_words_whole(con: sqlite3.Connection, term: str) -> None:
    """Апостроф і дефіс мають бути tokenchars, інакше слово розривається навпіл."""
    con.execute(
        "INSERT INTO chunk_fts(rowid, lemmas, forms, codes) VALUES (1,?,?,?)",
        ("п'ять об'єкт м'який гармата військово-технічний", "", "Д-30 2С1"),
    )
    assert _match(con, term) == 1


def test_remove_diacritics_2_does_not_fold_cyrillic(con: sqlite3.Connection) -> None:
    """`remove_diacritics 2` зачіпає лише латиницю.

    Якщо це колись зміниться, ї→і та ґ→г зіллються, і українські запити почнуть
    знаходити не те. Це найтихіша з можливих поломок.
    """
    con.execute("INSERT INTO chunk_fts(rowid, lemmas, forms, codes) VALUES (1,?,'','')", ("їжак ґанок є",))
    assert _match(con, "їжак") == 1
    assert _match(con, "ежак") == 0, "ї було згорнуто в е — токенайзер змінився"
    assert _match(con, "ґанок") == 1
    assert _match(con, "ганок") == 0, "ґ було згорнуто в г — токенайзер змінився"


def test_unescaped_designation_is_a_syntax_trap(con: sqlite3.Connection) -> None:
    """Голий 'Д-30' у MATCH — це синтаксична помилка, а не пошук.

    Тест фіксує, ЧОМУ fts_term() обов'язковий, щоб ніхто не «спростив» його потім.
    """
    con.execute("INSERT INTO chunk_fts(rowid, lemmas, forms, codes) VALUES (1,'','','Д-30')")
    with pytest.raises(sqlite3.OperationalError):
        con.execute("SELECT count(*) FROM chunk_fts WHERE chunk_fts MATCH ?", ("Д-30",)).fetchone()


def test_trigram_index_tolerates_missing_hyphen(con: sqlite3.Connection) -> None:
    """Технік напише «Д30», у підручнику написано «Д-30»."""
    con.execute("INSERT INTO code_fts(rowid, codes) VALUES (1, 'Д-30 2С1 ДСТУ3008')")
    assert _match(con, "Д-3", table="code_fts") == 1
    assert _match(con, "ДСТУ3008", table="code_fts") == 1


def test_contentless_delete_is_enabled(con: sqlite3.Connection) -> None:
    """Без contentless_delete=1 переіндексація документа неможлива."""
    con.execute("INSERT INTO chunk_fts(rowid, lemmas, forms, codes) VALUES (1,'гармата','','')")
    con.execute("DELETE FROM chunk_fts WHERE rowid = 1")
    assert _match(con, "гармата") == 0


def test_schema_carries_bbox_and_page_from_day_one(con: sqlite3.Connection) -> None:
    """Без bbox_json підсвітка цитованого фрагмента неможлива, а доробка потім
    вимагає повної переіндексації корпусу."""
    cols = {r[1] for r in con.execute("PRAGMA table_info(chunks)")}
    for required in ("bbox_json", "page_from", "page_to", "page_label_from", "chunk_uid"):
        assert required in cols, f"chunks.{required} відсутня — дороге до відкату рішення"


def test_facets_are_indexed_for_mandatory_metadata_filtering(con: sqlite3.Connection) -> None:
    """Стаття автора називає фільтрацію за метаданими обов'язковою, не додатковою."""
    idx = {r[1] for r in con.execute("PRAGMA index_list(chunk_facets)")}
    assert len(idx) >= 5, f"замало індексів на chunk_facets: {idx}"
