"""Спільний мініатюрний корпус для тестів пошуку.

Корпус навмисно артилерійський і навмисно різнорідний: чотири документи різних
типів, років і мов, серед них — точний дублікат абзацу з першого підручника в
конспекті лекцій. Саме на цьому наборі перевіряються всі властивості, заради
яких модуль існує: метаданні фільтри, дедуплікація, диверсифікація джерел,
екранування FTS5 і auto-merge до L1-батька.

Усе будується у stub-режимі (`create_provider(stub=True)`) — жодної завантаженої
моделі. Заглушка ембедингів дає косинус, приблизно рівний лексичному перекриттю,
тому dense-гілка тут не лотерея, і тести перевіряють реальний конвеєр, а не
власні моки.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.db.database import Database
from app.db.repositories import AssistantRepo, ChunkRepo, CollectionRepo, DocumentRepo
from app.domain import (
    Assistant,
    AssistantConfig,
    Chunk,
    ChunkLevel,
    Collection,
    Document,
    RetrievedChunk,
    new_id,
)
from app.embeddings.provider import create_provider

__all__ = [
    "CORPUS",
    "Corpus",
    "DocSpec",
    "add_collection",
    "build_corpus",
    "make_retrieved",
    "text_simhash",
]


def text_simhash(text: str) -> int:
    """Детермінований 64-бітний SimHash-сурогат для тестів.

    Зсув на біт вправо обов'язковий: SQLite INTEGER — ЗНАКОВЕ 64-бітне число,
    і чесний беззнаковий SimHash із увімкненим старшим бітом кидає
    `OverflowError` на вставці. Справжній модуль чанкування має рахуватися з
    тим самим обмеженням.
    """
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") >> 1


@dataclass(frozen=True, slots=True)
class DocSpec:
    title: str
    doc_type: str
    language: str
    year: int
    bodies: tuple[str, ...]


# Один розділ на документ: усі листки — рідні брати, тож ланцюги впорядкування
# й auto-merge до L1-батька справді мають на чому спрацювати.
CORPUS: tuple[DocSpec, ...] = (
    DocSpec(
        title="Основи балістики",
        doc_type="textbook",
        language="uk",
        year=2019,
        bodies=(
            "Деривація снаряда — це відхилення снаряда від площини стрільби "
            "внаслідок обертання снаряда навколо власної осі.",
            "Поправка на деривацію обчислюється за таблицями стрільби окремо "
            "для кожної гармати та кожного заряду.",
            "Початкова швидкість снаряда залежить від температури заряду та "
            "зносу каналу ствола гармати.",
            "Кут підвищення визначається за таблицями стрільби відповідно до "
            "дальності до цілі та обраного заряду.",
        ),
    ),
    DocSpec(
        title="Методична розробка: гаубиця Д-30",
        doc_type="methodical",
        language="uk",
        year=2021,
        bodies=(
            "Гаубиця Д-30 калібру 122 мм має максимальну дальність стрільби "
            "15300 метрів осколково-фугасним снарядом.",
            "Розрахунок гаубиці Д-30 складається з шести осіб, бойова маса "
            "гармати становить 3200 кг.",
            "Поправка на деривацію для Д-30 береться з таблиць стрільби, "
            "оформлених згідно ДСТУ 3008:2015.",
        ),
    ),
    DocSpec(
        title="Field Artillery Manual",
        doc_type="manual",
        language="en",
        year=2015,
        bodies=(
            "Field artillery projectile drift is caused by the spin of the "
            "projectile around its longitudinal axis.",
            "The howitzer M119 has a maximum range of 14000 metres with the "
            "standard propelling charge.",
        ),
    ),
    # Конспект несе ДОСЛІВНИЙ абзац із підручника — це матеріал для перевірки
    # дедуплікації майже-дублікатів між документами.
    DocSpec(
        title="Конспект лекцій з балістики",
        doc_type="lecture",
        language="uk",
        year=2019,
        bodies=(
            "Деривація снаряда — це відхилення снаряда від площини стрільби "
            "внаслідок обертання снаряда навколо власної осі.",
        ),
    ),
)


@dataclass(slots=True)
class Corpus:
    """Усе, що потрібно тесту, щоб звернутися до побудованої колекції."""

    db: Database
    provider: Any
    collection_id: str
    assistant_id: str
    index_dir: Path
    document_ids: dict[str, str] = field(default_factory=dict)   # назва → id
    leaf_ids: dict[str, list[int]] = field(default_factory=dict)  # назва → id листків
    parent_ids: dict[str, int] = field(default_factory=dict)      # назва → id L1-батька

    def doc(self, title: str) -> str:
        return self.document_ids[title]

    def leaves(self, title: str) -> list[int]:
        return self.leaf_ids[title]

    def leaf_count(self) -> int:
        return sum(len(v) for v in self.leaf_ids.values())


def build_corpus(tmp_path: Path, *, specs: tuple[DocSpec, ...] = CORPUS) -> Corpus:
    """Створити БД, колекцію, документи, чанки, вектори й FTS-індекс."""
    db = Database(tmp_path / "assistant.db")
    provider = create_provider(stub=True)
    index_dir = tmp_path / "idx"

    corpus = Corpus(
        db=db,
        provider=provider,
        collection_id="",
        assistant_id="",
        index_dir=index_dir,
    )

    with db.transaction() as con:
        assistant = Assistant(id=new_id(), name="Артилерія", config=AssistantConfig())
        AssistantRepo(con).create(assistant)
        collection = Collection(
            id=new_id(),
            assistant_id=assistant.id,
            name="Основна",
            embedding_model_id="qwen3-embedding-0.6b",
            embedding_model_key=provider.model_key,
            dim=provider.dim,
        )
        CollectionRepo(con).create(collection)
        corpus.assistant_id = assistant.id
        corpus.collection_id = collection.id
        _populate(con, corpus, collection, specs)

    return corpus


def add_collection(corpus: Corpus, *, name: str, specs: tuple[DocSpec, ...]) -> str:
    """Додати ДРУГОГО асистента з власною колекцією в ту саму БД.

    Потрібно, щоб перевірити головне твердження рівня L0: ізоляція асистентів
    фізична (один файл індексу на колекцію), а не через метаданий фільтр.
    """
    with corpus.db.transaction() as con:
        assistant = Assistant(id=new_id(), name=name, config=AssistantConfig())
        AssistantRepo(con).create(assistant)
        collection = Collection(
            id=new_id(),
            assistant_id=assistant.id,
            name=name,
            embedding_model_id="qwen3-embedding-0.6b",
            embedding_model_key=corpus.provider.model_key,
            dim=corpus.provider.dim,
        )
        CollectionRepo(con).create(collection)
        _populate(con, corpus, collection, specs)
    return collection.id


def _populate(
    con: Any, corpus: Corpus, collection: Collection, specs: tuple[DocSpec, ...]
) -> None:
    """Наповнити колекцію документами: L1-батько + L2-листки, вектори, FTS."""
    documents = DocumentRepo(con)
    chunks = ChunkRepo(con)
    provider = corpus.provider
    ordinal = 0

    for spec in specs:
        document = Document(
            id=new_id(),
            collection_id=collection.id,
            title=spec.title,
            original_name=f"{spec.title}.pdf",
            stored_path=f"docs/{new_id()}.pdf",
            content_sha256=hashlib.sha256(spec.title.encode()).hexdigest(),
            doc_type=spec.doc_type,
            language=spec.language,
            year=spec.year,
            page_count=len(spec.bodies),
        )
        documents.create(document)
        corpus.document_ids[spec.title] = document.id

        parent = Chunk(
            document_id=document.id,
            collection_id=collection.id,
            ordinal=ordinal,
            level=ChunkLevel.SECTION,
            display_text=" ".join(spec.bodies),
            header_path="//Розділ 1//",
            language=spec.language,
            page_from=1,
            page_to=len(spec.bodies),
        )
        ordinal += 1
        chunks.insert_many([parent])
        assert parent.id is not None
        corpus.parent_ids[spec.title] = parent.id

        leaves: list[Chunk] = []
        for i, body in enumerate(spec.bodies):
            leaves.append(
                Chunk(
                    document_id=document.id,
                    collection_id=collection.id,
                    ordinal=ordinal,
                    level=ChunkLevel.LEAF,
                    display_text=body,
                    header_path="//Розділ 1//",
                    parent_id=parent.id,
                    chapter_id=None,
                    language=spec.language,
                    page_from=i + 1,
                    page_to=i + 1,
                    page_label_from=str(i + 1),
                    page_label_to=str(i + 1),
                    siblings_index=i,
                    siblings_count=len(spec.bodies),
                    simhash=text_simhash(body),
                )
            )
            ordinal += 1
        chunks.insert_many(leaves)
        corpus.leaf_ids[spec.title] = [c.id for c in leaves if c.id is not None]

        vectors = provider.embed_documents([c.display_text for c in leaves])
        chunks.set_embeddings(
            [(c.id, v) for c, v in zip(leaves, vectors, strict=True) if c.id is not None],
            provider.model_key,
        )
        for chunk in leaves:
            assert chunk.id is not None
            chunks.index_fts(chunk.id, *_fts_columns(chunk.display_text, spec.language))


def _fts_columns(text: str, language: str) -> tuple[str, str, str]:
    """Леми / словоформи / коди для `chunk_fts`.

    Викликаємо справжній лематизатор модуля приймання — саме так наповнює
    індекс воркер, і саме проти цього має зійтися запит. Якщо модуля ще немає
    (він пишеться паралельно), відступаємо до словоформ: BM25 стає слабшим,
    але тест лишається чесним.
    """
    try:
        from app.ingestion.lemmatize import analyze_text
    except Exception:  # pragma: no cover — лише поки модуль приймання відсутній
        import re

        forms = re.findall(r"(?u)[^\W_]+", text.casefold())
        return " ".join(forms), " ".join(forms), ""
    return analyze_text(text, language).as_fts_columns()


def make_retrieved(
    *,
    chunk_id: int,
    document_id: str,
    score: float,
    text: str = "текст",
    header_path: str = "//Розділ 1//",
    siblings_index: int = 0,
    simhash: int | None = None,
    title: str = "",
) -> RetrievedChunk:
    """Синтетичний `RetrievedChunk` для тестів чистих функцій.

    `rerank_score` виставляється явно, бо саме він домінує в `.score` і саме
    на ньому працюють диверсифікація й впорядкування.
    """
    chunk = Chunk(
        id=chunk_id,
        document_id=document_id,
        collection_id="c",
        ordinal=chunk_id,
        level=ChunkLevel.LEAF,
        display_text=text,
        header_path=header_path,
        siblings_index=siblings_index,
        simhash=simhash,
    )
    return RetrievedChunk(
        chunk=chunk,
        document_title=title or document_id,
        rerank_score=score,
        fused_score=score,
    )
