"""Приймання й парсинг документів.

Публічний інтерфейс модуля (docs/CONTRACT.md):
    parse_document(path, opts) -> ParsedDocument
    chunk_document(parsed, cfg) -> list[Chunk]
    normalize_uk(text)          -> str
    lemmatize(tokens, lang)     -> LemmaStream

Жоден імпорт тут не тягне torch, docling чи rapidocr: усі важкі залежності
завантажуються ліниво всередині функцій, тому API-процес може імпортувати
`app.ingestion` і лишитись із холодним стартом ~0.6 с.
"""

from __future__ import annotations

from app.ingestion.chunker import (
    ChunkConfig,
    ChunkTree,
    build_chapters,
    calibrate_ch_per_tok,
    chunk_document,
    chunk_document_tree,
    chunk_markdown,
    simhash64,
)
from app.ingestion.docling_pipeline import (
    ElementKind,
    ParsedDocument,
    ParsedElement,
    ParsedPage,
    ParseOptions,
    build_parse_profile,
    parse_document,
    parse_profile_hash,
    serialize_table_triplets,
)
from app.ingestion.lemmatize import (
    DESIGNATION_RE,
    UK_STOPWORDS,
    LemmaStream,
    Lemmatizer,
    SqliteLemmaCache,
    analyze_text,
    lemmatize,
)
from app.ingestion.normalize_uk import (
    TEXT_PREPROC_VERSION,
    normalize_uk,
    search_terms,
    tokenize,
)
from app.ingestion.page_labels import PageLabelMap, resolve_page_labels
from app.ingestion.probe import PageProbe, ProbeReport, probe_pdf, probe_text

__all__ = [
    "DESIGNATION_RE",
    # нормалізація й морфологія
    "TEXT_PREPROC_VERSION",
    "UK_STOPWORDS",
    "ChunkConfig",
    "ChunkTree",
    "ElementKind",
    "LemmaStream",
    "Lemmatizer",
    "PageLabelMap",
    "PageProbe",
    "ParseOptions",
    "ParsedDocument",
    "ParsedElement",
    "ParsedPage",
    "ProbeReport",
    "SqliteLemmaCache",
    "analyze_text",
    "build_chapters",
    "build_parse_profile",
    "calibrate_ch_per_tok",
    # чанкування
    "chunk_document",
    "chunk_document_tree",
    "chunk_markdown",
    "lemmatize",
    "normalize_uk",
    # парсинг
    "parse_document",
    "parse_profile_hash",
    # тріаж і мітки сторінок
    "probe_pdf",
    "probe_text",
    "resolve_page_labels",
    "search_terms",
    "serialize_table_triplets",
    "simhash64",
    "tokenize",
]
