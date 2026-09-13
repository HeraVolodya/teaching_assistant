"""Векторний індекс: збірка, публікація, mmap-читання, деградація до точного пошуку.

Головне твердження файлу: індекс НІКОЛИ не є джерелом істини й НІКОЛИ не має
права зламати чат. Канонічні вектори лежать у SQLite; граф — кеш, що будь-якої
миті може виявитись відсутнім, застарілим або зібраним іншим бекендом. У всіх
трьох випадках пошук мусить повернути правильну відповідь, лише повільніше.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.db.repositories import ChunkRepo, CollectionRepo
from app.domain import Chunk, ChunkLevel
from app.retrieval.vector_index import (
    CollectionIndex,
    ExactVectors,
    IndexParams,
    VectorIndex,
    build_collection_index,
    compute_build_id,
    index_path_for,
    usearch_available,
)
from tests.helpers_retrieval import build_corpus


# --------------------------------------------------------------- build_id
def test_build_id_is_stable_for_identical_inputs() -> None:
    kwargs = dict(
        embedding_model_key="k", metric="cos", connectivity=32,
        expansion_add=200, max_chunk_id=100, count=90,
    )
    assert compute_build_id(**kwargs) == compute_build_id(**kwargs)


@pytest.mark.parametrize(
    "override",
    [
        {"embedding_model_key": "інший"},
        {"metric": "ip"},
        {"connectivity": 16},
        {"expansion_add": 128},
        {"max_chunk_id": 101},
        {"count": 89},
        {"schema_version": 2},
    ],
)
def test_build_id_changes_on_every_meaningful_parameter(override: dict) -> None:
    """Кожен параметр, що робить старий граф несумісним, мусить бути в хеші.

    Пропущений параметр — це тихо неправильні результати пошуку: граф є, він
    відкривається, він повертає сусідів — просто не тих.
    """
    base = dict(
        embedding_model_key="k", metric="cos", connectivity=32,
        expansion_add=200, max_chunk_id=100, count=90,
    )
    assert compute_build_id(**base) != compute_build_id(**{**base, **override})


def test_index_path_is_versioned_per_collection_and_model() -> None:
    """Дві embedding-моделі мусять співіснувати; міграція — зміна вказівника."""
    a = index_path_for("/idx", "coll-1", "a" * 64)
    b = index_path_for("/idx", "coll-1", "b" * 64)
    c = index_path_for("/idx", "coll-2", "a" * 64)
    assert a != b != c
    assert a.parts[-3] == "coll-1"
    assert a.name == "index.usearch"


# ------------------------------------------------------------ точний пошук
def test_exact_search_returns_sorted_hits() -> None:
    vecs = ExactVectors(
        keys=np.array([7, 8, 9], dtype=np.uint64),
        vectors=np.array([[1, 0], [0.7, 0.7], [0, 1]], dtype=np.float32),
    )
    hits = vecs.search(np.array([1, 0], dtype=np.float32), 2)
    assert [h.key for h in hits] == [7, 8]
    assert hits[0].score == pytest.approx(1.0)


def test_exact_search_honours_the_allowed_set() -> None:
    """Точний скан по відфільтрованій множині — рівень L2 стратегії §7."""
    vecs = ExactVectors(
        keys=np.array([7, 8, 9], dtype=np.uint64),
        vectors=np.array([[1, 0], [0.7, 0.7], [0, 1]], dtype=np.float32),
    )
    hits = vecs.search(np.array([1, 0], dtype=np.float32), 3, allowed={8, 9})
    assert [h.key for h in hits] == [8, 9]


def test_exact_search_on_empty_matrix() -> None:
    vecs = ExactVectors.from_rows([], dim=4)
    assert vecs.search(np.zeros(4, dtype=np.float32), 5) == []


# ------------------------------------------------- запис / читання файлу
@pytest.mark.parametrize("prefer_usearch", [True, False])
def test_write_and_search_roundtrip(tmp_path: Path, prefer_usearch: bool) -> None:
    """Обидва бекенди мусять давати той самий результат.

    Фолбек існує не заради екзотики: без нього «немає колеса usearch» ставало б
    аварією, а «індекс ще не побудований» — помилкою в чаті.
    """
    params = IndexParams(dim=3)
    vectors = np.array([[1, 0, 0], [0, 1, 0], [0.6, 0.8, 0]], dtype=np.float32)
    backend = VectorIndex.write(
        tmp_path / "i.usearch", params, [11, 22, 33], vectors, prefer_usearch=prefer_usearch
    )
    assert backend == ("usearch" if prefer_usearch and usearch_available() else "mmap")

    with VectorIndex(tmp_path / "i.usearch", params, prefer_usearch=prefer_usearch) as index:
        assert len(index) == 3
        hits = index.search(np.array([1, 0, 0], dtype=np.float32), 2)
        assert hits[0].key == 11
        assert hits[0].score == pytest.approx(1.0, abs=1e-2)


def test_keys_are_chunk_ids_verbatim(tmp_path: Path) -> None:
    """Ключ USearch = `chunks.id`, без жодного перемапування.

    Будь-яка проміжна таблиця відповідності — це другий стан, який може
    розійтися з БД після часткової переіндексації.
    """
    params = IndexParams(dim=2)
    VectorIndex.write(
        tmp_path / "i.usearch", params, [4242], np.array([[1, 0]], dtype=np.float32)
    )
    with VectorIndex(tmp_path / "i.usearch", params) as index:
        assert index.search(np.array([1, 0], dtype=np.float32), 1)[0].key == 4242


def test_foreign_index_format_falls_back_instead_of_raising(tmp_path: Path) -> None:
    """Індекс, зібраний на машині БЕЗ usearch, має читатися машиною З usearch.

    `Index.restore` кидає ValueError на чужому файлі, а не повертає None.
    Дати цьому винятку піднятись означало б вимагати переіндексації всього
    корпусу лише через різницю в середовищах.
    """
    params = IndexParams(dim=2)
    VectorIndex.write(
        tmp_path / "i.usearch", params, [1, 2],
        np.array([[1, 0], [0, 1]], dtype=np.float32), prefer_usearch=False,
    )
    with VectorIndex(tmp_path / "i.usearch", params, prefer_usearch=True) as index:
        assert index.backend == "mmap"
        assert index.search(np.array([1, 0], dtype=np.float32), 1)[0].key == 1


def test_search_applies_the_allowed_postfilter(tmp_path: Path) -> None:
    params = IndexParams(dim=2)
    VectorIndex.write(
        tmp_path / "i.usearch", params, [1, 2],
        np.array([[1, 0], [0, 1]], dtype=np.float32),
    )
    with VectorIndex(tmp_path / "i.usearch", params) as index:
        hits = index.search(np.array([1, 0], dtype=np.float32), 1, allowed={2}, overfetch=4)
        assert [h.key for h in hits] == [2]


def test_key_and_vector_count_mismatch_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="не збігається"):
        VectorIndex.write(
            tmp_path / "i.usearch", IndexParams(dim=2), [1, 2],
            np.array([[1, 0]], dtype=np.float32),
        )


def test_wrong_dimension_is_rejected(tmp_path: Path) -> None:
    """Розмірність не збігається з колекцією → зупинитись, а не «якось шукати»."""
    with pytest.raises(ValueError, match="Розмірність"):
        VectorIndex.write(
            tmp_path / "i.usearch", IndexParams(dim=4), [1],
            np.array([[1, 0]], dtype=np.float32),
        )


def test_missing_file_says_what_to_do(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="перебудуйте"):
        VectorIndex(tmp_path / "нема.usearch", IndexParams(dim=2)).open()


def test_publish_replaces_atomically_after_close(tmp_path: Path) -> None:
    """Читач ЗАКРИВАЄТЬСЯ перед підміною — інакше на Windows це PermissionError.

    `view=True` тримає відкритий дескриптор; на POSIX `os.replace` над ним
    працює, на Windows — блокується. Порядок «close → replace» єдиний, що
    коректний на обох.
    """
    params = IndexParams(dim=2)
    final = tmp_path / "index.usearch"
    VectorIndex.write(final, params, [1], np.array([[1, 0]], dtype=np.float32))
    reader = VectorIndex(final, params).open()
    reader.close()

    tmp = tmp_path / ".build.usearch"
    VectorIndex.write(tmp, params, [2], np.array([[0, 1]], dtype=np.float32))
    VectorIndex.publish(tmp, final)

    assert not tmp.exists()
    with VectorIndex(final, params) as index:
        assert index.search(np.array([0, 1], dtype=np.float32), 1)[0].key == 2


# ------------------------------------------------------------ збірка з БД
def test_build_from_database_bumps_generation(tmp_path: Path) -> None:
    corpus = build_corpus(tmp_path)
    report = build_collection_index(corpus.db, corpus.collection_id, index_dir=corpus.index_dir)

    assert report.count == corpus.leaf_count()
    assert report.dim == corpus.provider.dim
    assert report.index_generation == 1
    assert report.path.exists()

    with corpus.db.connection() as con:
        collection = CollectionRepo(con).get(corpus.collection_id)
    assert collection is not None
    assert collection.build_id == report.build_id
    assert collection.dirty is False


def test_reader_uses_the_graph_when_build_id_matches(tmp_path: Path) -> None:
    corpus = build_corpus(tmp_path)
    build_collection_index(corpus.db, corpus.collection_id, index_dir=corpus.index_dir)

    with CollectionIndex(corpus.db, corpus.collection_id, index_dir=corpus.index_dir) as index:
        assert index.is_stale is False
        assert index.backend in {"usearch", "mmap"}
        assert len(index) == corpus.leaf_count()
        query = corpus.provider.embed_queries(["деривація снаряда"])[0]
        assert index.search(query, 3)


def test_stale_index_degrades_to_exact_search_instead_of_failing(tmp_path: Path) -> None:
    """Додали документ — граф застарів. Це НЕ помилка, це повільніший режим.

    Саме тому «індекс застарів» ніколи не перетворюється на повідомлення про
    помилку в чаті: перебудова — справа воркера, а відповідати треба зараз.
    """
    corpus = build_corpus(tmp_path)
    build_collection_index(corpus.db, corpus.collection_id, index_dir=corpus.index_dir)

    reader = CollectionIndex(corpus.db, corpus.collection_id, index_dir=corpus.index_dir)
    reader.refresh()
    assert reader.is_stale is False

    document_id = corpus.doc("Основи балістики")
    with corpus.db.transaction() as con:
        chunks = ChunkRepo(con)
        extra = Chunk(
            document_id=document_id,
            collection_id=corpus.collection_id,
            ordinal=9999,
            level=ChunkLevel.LEAF,
            display_text="Таблиці стрільби містять поправку на деривацію у тисячних.",
            header_path="//Розділ 9//",
        )
        chunks.insert_many([extra])
        assert extra.id is not None
        vector = corpus.provider.embed_documents([extra.display_text])[0]
        chunks.set_embedding(extra.id, vector, corpus.provider.model_key)

    reader.refresh()
    assert reader.is_stale is True
    assert reader.backend == "exact"
    assert len(reader) == corpus.leaf_count() + 1

    query = corpus.provider.embed_queries(["поправка на деривацію у тисячних"])[0]
    hits = reader.search(query, 3)
    assert hits and hits[0].key == extra.id
    reader.close()


def test_missing_index_file_is_a_normal_state(tmp_path: Path) -> None:
    """Індекс іще не побудовано — пошук усе одно мусить працювати."""
    corpus = build_corpus(tmp_path)
    with CollectionIndex(corpus.db, corpus.collection_id, index_dir=corpus.index_dir) as index:
        assert index.backend == "exact"
        assert len(index) == corpus.leaf_count()
        query = corpus.provider.embed_queries(["гаубиця Д-30"])[0]
        assert index.search(query, 2)


def test_unknown_collection_is_a_loud_error(tmp_path: Path) -> None:
    corpus = build_corpus(tmp_path)
    with pytest.raises(KeyError, match="не знайдено"):
        build_collection_index(corpus.db, "немає-такої", index_dir=corpus.index_dir)


def test_rebuild_is_idempotent_for_the_same_content(tmp_path: Path) -> None:
    """Той самий вміст → той самий build_id; змінюється лише generation."""
    corpus = build_corpus(tmp_path)
    first = build_collection_index(corpus.db, corpus.collection_id, index_dir=corpus.index_dir)
    second = build_collection_index(corpus.db, corpus.collection_id, index_dir=corpus.index_dir)
    assert first.build_id == second.build_id
    assert second.index_generation == first.index_generation + 1
