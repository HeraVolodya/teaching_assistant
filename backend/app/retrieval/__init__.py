"""Векторний індекс і гібридний пошук.

Публічний контракт модуля (docs/CONTRACT.md):
    `VectorIndex` — build / search / mmap
    `HybridRetriever.retrieve(query, config, collection_id)
        -> tuple[list[RetrievedChunk], RetrievalDebug]`

Імпорти навмисно НЕ підняті сюди. `hybrid` тягне numpy і — за наявності —
`app.ingestion` (лематизація запиту), а `vector_index` потрібен воркеру
індексації окремо від пошуку. Імпортуйте підмодулі явно:

    from app.retrieval.vector_index import CollectionIndex, build_collection_index
    from app.retrieval.hybrid import HybridRetriever, MetadataFilter

Підмодулі:
    `vector_index` — USearch HNSW як mmap-сайдкар + точний фолбек із SQLite
    `fusion`       — зважений RRF з автопідйомом sparse на лексичному якорі
    `diversity`    — дедуплікація SimHash+косинус і квота на документ
    `reorder`      — порядок читання всередині блоку, U-форма між блоками
    `hybrid`       — конвеєр §7 плану цілком
"""

from __future__ import annotations

__all__: list[str] = []
