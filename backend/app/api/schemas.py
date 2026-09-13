"""Схеми запитів і відповідей HTTP-шару.

ІМЕНА ПОЛІВ — camelCase НА ДРОТІ, snake_case У PYTHON.
Фронтенд — React/TypeScript, і `session_id` у TS-коді читається як чуже тіло.
`alias_generator=to_camel` плюс `populate_by_name=True` дає обидва написання
на вході й рівно camelCase на виході — тобто ніхто не мусить пам'ятати, у
який бік конвертувати.

Доменні типи (`AssistantConfig`, `Chunk`, `Citation`) — dataclass'и з
`app/domain.py`, і вони СВІДОМО не переписані в pydantic: подвоєння схеми
означало б два джерела правди про конфіг асистента, який є версійованими
даними НДР (Завдання 4). Тут — лише тонкі перекладачі.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

from app.domain import (
    Assistant,
    AssistantConfig,
    Citation,
    Collection,
    Document,
    PageInfo,
)

__all__ = [
    "ApiModel", "AssistantIn", "AssistantOut", "CollectionOut", "DocumentOut",
    "PageOut", "SessionIn", "SessionOut", "MessageOut", "ChatIn", "FeedbackIn",
    "ReingestIn", "DownloadIn", "PromptPreviewOut", "assistant_out",
    "collection_out", "document_out", "page_out", "citation_out", "config_from_dict",
]


class ApiModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel, populate_by_name=True, extra="ignore"
    )


# ------------------------------------------------------------- асистенти
class AssistantIn(ApiModel):
    name: str = Field(min_length=1, max_length=120)
    description: str = ""
    colour: str = "#4f46e5"
    emoji: str = "📘"
    # Конфіг приймається як довільний словник і фільтрується по відомих полях:
    # старий фронтенд не має ламатися об нове поле, а новий — об старий бекенд.
    config: dict[str, Any] | None = None


class CollectionOut(ApiModel):
    id: str
    name: str
    embedding_model_id: str
    dim: int
    index_generation: int
    dirty: bool
    documents: int = 0
    chunks: int = 0


class AssistantOut(ApiModel):
    id: str
    name: str
    description: str
    colour: str
    emoji: str
    config: dict[str, Any]
    config_version: int
    collections: list[CollectionOut] = Field(default_factory=list)


class PromptPreviewOut(ApiModel):
    """Скомпільований системний промпт — лише для читання.

    Пороги виконуються В КОДІ, а не в промпті (контракт, правило 8), тому
    прев'ю показує і те, і те: викладач мусить бачити, що «висока
    впевненість» — це число 0.55 у ретривері, а не ввічливе прохання моделі.
    """
    system: str
    persona: str
    rules_tokens: int
    persona_tokens: int
    confidence_threshold: float
    min_supporting_chunks: int
    final_top_k: int
    max_per_document: int
    on_missing_information: str
    sample: str = ""


# ------------------------------------------------------------- документи
class DocumentOut(ApiModel):
    id: str
    collection_id: str
    title: str
    original_name: str
    doc_type: str
    language: str
    year: int | None
    author: str | None
    page_count: int
    ingest_mode: str
    status: str
    error_code: str | None
    error_detail: str | None
    quality_grade: str | None
    size_bytes: int | None = None
    chunks: int = 0
    job: dict[str, Any] | None = None


class PageOut(ApiModel):
    page_number: int
    page_label: str | None
    width: float
    height: float
    page_class: str
    ocr_mode: str
    cost_weight: float
    lexicon_hit_rate: float | None
    cyrillic_ratio: float | None
    mojibake_ratio: float | None
    parse_score: float | None
    layout_score: float | None
    table_score: float | None
    ocr_score: float | None
    needs_repair: bool


class ReingestIn(ApiModel):
    mode: Literal["FAST", "DEEP"] = "FAST"


# ------------------------------------------------------------------ чат
class SessionIn(ApiModel):
    assistant_id: str
    title: str = ""


class SessionOut(ApiModel):
    id: str
    assistant_id: str
    title: str
    created_at: str
    updated_at: str
    messages: int = 0


class MessageOut(ApiModel):
    id: str
    session_id: str
    role: str
    content: str
    abstained: bool
    model_id: str | None
    ttft_ms: int | None
    tokens_out: int | None
    created_at: str
    citations: list[dict[str, Any]] = Field(default_factory=list)


class ChatIn(ApiModel):
    session_id: str
    message: str = Field(min_length=1)
    detailed: bool = False
    document_ids: list[str] = Field(default_factory=list)
    doc_types: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    year_from: int | None = None
    year_to: int | None = None


class FeedbackIn(ApiModel):
    message_id: str | None = None
    verdict: Literal["up", "down"]
    chunk_uid: str | None = None
    note: str = ""


class DownloadIn(ApiModel):
    model: str = Field(min_length=1)


# ------------------------------------------------------------ перекладачі
def config_from_dict(raw: dict[str, Any] | None) -> AssistantConfig:
    """Тільки відомі поля. Невідомі ігноруються мовчки — так конфіг
    переживає і старіший фронтенд, і новіший."""
    if not raw:
        return AssistantConfig()
    known = set(AssistantConfig.__dataclass_fields__)
    return AssistantConfig(**{k: v for k, v in raw.items() if k in known})


def assistant_out(assistant: Assistant, collections: list[CollectionOut]) -> AssistantOut:
    return AssistantOut(
        id=assistant.id,
        name=assistant.name,
        description=assistant.description,
        colour=assistant.colour,
        emoji=assistant.emoji,
        config=asdict(assistant.config),
        config_version=assistant.config_version,
        collections=collections,
    )


def collection_out(collection: Collection, *, documents: int = 0, chunks: int = 0) -> CollectionOut:
    return CollectionOut(
        id=collection.id,
        name=collection.name,
        embedding_model_id=collection.embedding_model_id,
        dim=collection.dim,
        index_generation=collection.index_generation,
        dirty=collection.dirty,
        documents=documents,
        chunks=chunks,
    )


def document_out(
    document: Document, *, size_bytes: int | None = None, chunks: int = 0,
    job: dict[str, Any] | None = None,
) -> DocumentOut:
    return DocumentOut(
        id=document.id,
        collection_id=document.collection_id,
        title=document.title,
        original_name=document.original_name,
        doc_type=document.doc_type,
        language=document.language,
        year=document.year,
        author=document.author,
        page_count=document.page_count,
        ingest_mode=document.ingest_mode.value,
        status=document.status.value,
        error_code=document.error_code,
        error_detail=document.error_detail,
        quality_grade=document.quality_grade.value if document.quality_grade else None,
        size_bytes=size_bytes,
        chunks=chunks,
        job=job,
    )


def page_out(page: PageInfo) -> PageOut:
    return PageOut(
        page_number=page.page_number,
        page_label=page.page_label,
        width=page.width,
        height=page.height,
        page_class=page.page_class.value,
        ocr_mode=page.ocr_mode.value,
        cost_weight=page.cost_weight,
        lexicon_hit_rate=page.lexicon_hit_rate,
        cyrillic_ratio=page.cyrillic_ratio,
        mojibake_ratio=page.mojibake_ratio,
        parse_score=page.parse_score,
        layout_score=page.layout_score,
        table_score=page.table_score,
        ocr_score=page.ocr_score,
        needs_repair=page.needs_vlm_repair(),
    )


def citation_out(citation: Citation) -> dict[str, Any]:
    """Цитата для UI. `bboxes` йдуть як є — фронтенд перетворює їх у
    координати viewport через `page.getViewport().convertToViewportPoint()`."""
    return {
        "ordinal": citation.ordinal,
        "chunkUid": citation.chunk_uid,
        "documentId": citation.document_id,
        "documentTitle": citation.document_title,
        "pageFrom": citation.page_from,
        "pageTo": citation.page_to,
        "pageLabel": citation.page_label,
        "quote": citation.quote,
        "bboxes": citation.bboxes,
        "language": citation.language,
    }
