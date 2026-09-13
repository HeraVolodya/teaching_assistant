"""Наскрізний гейт приймання: файл → чанки → SQLite → FTS5 → знайдено.

Модульні тести перевіряють кожен крок окремо; цей перевіряє, що вихід
приймання ФІЗИЧНО влазить у схему і що запит українською справді знаходить
потрібний фрагмент. Саме тут ловляться розбіжності типів, переповнення
simhash і хибне екранування термінів FTS5.

Працює без docling, без torch і без жодної моделі — тобто в CI.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from app.db.repositories import ChunkRepo, PageRepo
from app.domain import (
    Assistant,
    ChunkLevel,
    Collection,
    Document,
    new_id,
)
from app.ingestion import (
    ParseOptions,
    analyze_text,
    chunk_document_tree,
    parse_document,
    probe_text,
    search_terms,
)
from app.ingestion.lemmatize import Lemmatizer, MemoryLemmaCache, SqliteLemmaCache

MIGRATION = Path(__file__).resolve().parents[1] / "app" / "db" / "migrations" / "0001_initial.sql"

TEXTBOOK = """# Основи стрільби артилерії

Навчальний посібник для курсантів Інституту РВіА.

## Розділ 1. Гармата Д-30

Гармата Д-30 — 122-мм буксирована гаубиця. Її максимальна дальність стрільби
осколково-фугасним снарядом становить 15 300 м. Кут піднесення ствола —
від мінус 7 до плюс 70 градусів.

### 1.1. Таблиці стрільби

| Дальність | Приціл | Поправка |
| --- | --- | --- |
| 1000 | 12 | 0,5 |
| 2000 | 24 | 1,2 |

## Розділ 2. Поправка на деривацію

Деривація — це відхилення снаряда вбік від площини стрільби через обертання
навколо поздовжньої осі. Поправка на деривацію обчислюється за таблицями
стрільби згідно з ДСТУ 3008:2015.
"""


def fts_term(term: str) -> str:
    """Єдиний дозволений спосіб подати термін у MATCH (docs/CONTRACT.md, правило 6)."""
    return '"' + term.replace('"', '""') + '"'


@pytest.fixture()
def con(tmp_path: Path) -> sqlite3.Connection:
    c = sqlite3.connect(tmp_path / "assistant.db")
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys = ON")
    c.executescript(MIGRATION.read_text(encoding="utf-8"))
    return c


@pytest.fixture()
def ids(con: sqlite3.Connection) -> tuple[str, str]:
    assistant = Assistant(id=new_id(), name="Артилерія")
    con.execute(
        "INSERT INTO assistants (id,name,config_json) VALUES (?,?,?)",
        (assistant.id, assistant.name, assistant.config.to_json()),
    )
    collection = Collection(
        id=new_id(), assistant_id=assistant.id, name="Основна",
        embedding_model_id="qwen3-embedding-0.6b", embedding_model_key="k", dim=1024,
    )
    con.execute(
        "INSERT INTO collections (id,assistant_id,name,embedding_model_id,"
        "embedding_model_key,dim) VALUES (?,?,?,?,?,?)",
        (collection.id, collection.assistant_id, collection.name,
         collection.embedding_model_id, collection.embedding_model_key, collection.dim),
    )
    document = Document(
        id=new_id(), collection_id=collection.id, title="Основи стрільби",
        original_name="posibnyk.md", stored_path="docs/x.md", content_sha256="0" * 64,
    )
    con.execute(
        "INSERT INTO documents (id,collection_id,title,original_name,stored_path,"
        "content_sha256) VALUES (?,?,?,?,?,?)",
        (document.id, document.collection_id, document.title, document.original_name,
         document.stored_path, document.content_sha256),
    )
    return document.id, collection.id


def test_end_to_end_ingest_and_find(tmp_path: Path, con: sqlite3.Connection,
                                    ids: tuple[str, str]) -> None:
    document_id, collection_id = ids
    source = tmp_path / "posibnyk.md"
    source.write_text(TEXTBOOK, encoding="utf-8")

    # 1. Парсинг — без docling, без torch.
    parsed = parse_document(source, ParseOptions(title="Основи стрільби"))
    assert parsed.backend == "text"
    assert parsed.parse_profile_hash

    # 2. Сторінки з тріажу лягають у таблицю pages.
    PageRepo(con).upsert_many(
        document_id,
        [p.to_page_info(parsed.label_for(p.page_number)) for p in probe_text(TEXTBOOK).pages],
    )
    assert PageRepo(con).label_map(document_id)

    # 3. Чанкування.
    tree = chunk_document_tree(parsed, document_id=document_id, collection_id=collection_id)
    assert tree.leaves()

    # 4. Вставка + зв'язки батьківства + FTS.
    repo = ChunkRepo(con)
    repo.insert_many(tree.chunks)
    assert tree.apply_parent_ids() == len(tree.parent_uid)
    con.executemany(
        "UPDATE chunks SET parent_id=? WHERE id=?",
        [(c.parent_id, c.id) for c in tree.chunks if c.parent_id is not None],
    )

    lemmatizer = Lemmatizer(cache=SqliteLemmaCache(con))
    for chunk in tree.chunks:
        if not chunk.is_indexable or chunk.id is None:
            continue
        lemmas, forms, codes = lemmatizer.analyze(chunk.display_text).as_fts_columns()
        repo.index_fts(chunk.id, lemmas, forms, codes)

    # 5. Запит українською знаходить потрібний фрагмент.
    terms = " ".join(fts_term(t) for t in search_terms("деривація"))
    rows = con.execute(
        "SELECT rowid, -bm25(chunk_fts, 1.0, 0.4, 2.5) AS s FROM chunk_fts"
        " WHERE chunk_fts MATCH ? ORDER BY rank LIMIT 5",
        (terms,),
    ).fetchall()
    assert rows, "запит «деривація» не знайшов нічого"
    best = repo.get(rows[0]["rowid"])
    assert best is not None
    assert "деривац" in best.display_text.lower()
    assert best.level is ChunkLevel.LEAF
    assert best.page_label_from

    # 6. Позначення знаходиться і в codes, і в триграмній мікро-таблиці.
    code_rows = con.execute(
        "SELECT rowid FROM code_fts WHERE code_fts MATCH ?", (fts_term("Д-30"),)
    ).fetchall()
    assert code_rows, "позначення Д-30 не потрапило у code_fts"

    # Триграмна мікро-таблиця дає підрядковий пошук по позначеннях: технік
    # шукає «Д-3», у підручнику написано «Д-30».
    partial = con.execute(
        "SELECT rowid FROM code_fts WHERE code_fts MATCH ?", (fts_term("Д-3"),)
    ).fetchall()
    assert partial


def test_chunks_survive_the_schema_constraints(con: sqlite3.Connection,
                                               ids: tuple[str, str]) -> None:
    """simhash — знаковий int64, bbox — JSON, chunk_uid — UNIQUE."""
    document_id, collection_id = ids

    from app.ingestion.docling_pipeline import _parse_text_source

    doc = _parse_text_source(TEXTBOOK, ParseOptions(title="Т"), source_path="t.md")
    tree = chunk_document_tree(doc, document_id=document_id, collection_id=collection_id)
    ChunkRepo(con).insert_many(tree.chunks)

    stored = con.execute("SELECT simhash, chunk_uid FROM chunks").fetchall()
    assert len(stored) == len(tree.chunks)
    for row in stored:
        assert -(2 ** 63) <= row["simhash"] < 2 ** 63
        assert len(row["chunk_uid"]) == 32


def test_lemma_cache_table_is_actually_used(con: sqlite3.Connection) -> None:
    """Кеш словоформ — те, що перетворює 10^8 викликів аналізатора на 10^6."""
    lemmatizer = Lemmatizer(cache=SqliteLemmaCache(con))
    lemmatizer.analyze(TEXTBOOK)
    count = con.execute("SELECT count(*) FROM lemma_cache").fetchone()[0]
    assert count > 20


def test_designations_reach_the_codes_column_not_the_lemmas() -> None:
    stream = analyze_text(TEXTBOOK, lemmatizer=Lemmatizer(cache=MemoryLemmaCache()))
    assert "д-30" in stream.codes
    assert "дсту 3008:2015" in stream.codes
    assert not any("дсту 3008" in lemma for lemma in stream.lemmas)


def test_stub_mode_runs_the_whole_pipeline_without_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ASISTENT_STUB=1: увесь конвеєр працює без жодної завантаженої моделі."""
    monkeypatch.setenv("ASISTENT_STUB", "1")
    fake_pdf = tmp_path / "підручник.pdf"
    fake_pdf.write_bytes(b"%PDF-1.7 not a real pdf")

    parsed = parse_document(fake_pdf)
    assert parsed.backend == "stub"
    assert parsed.warnings

    tree = chunk_document_tree(parsed, document_id="d", collection_id="c")
    assert tree.leaves()
    # Детермінованість: той самий файл — ті самі chunk_uid.
    again = chunk_document_tree(parse_document(fake_pdf), document_id="d", collection_id="c")
    assert [c.chunk_uid for c in tree.chunks] == [c.chunk_uid for c in again.chunks]
