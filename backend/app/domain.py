"""Доменні типи — спільний контракт для всіх шарів застосунку.

Усе, що перетинає межу між модулями, описано тут. Модулі не імпортують типи
один в одного: приймання не знає про пошук, пошук не знає про генерацію.

Три рівні чанків (див. план, §5):
    L0 — картка документа: маршрутизація й диверсифікація, НЕ цитується
    L1 — батьківська секція: показується генератору через auto-merge, НЕ індексується
    L2 — листок: ЄДИНИЙ рівень, що потрапляє в ANN-індекс і у FTS5
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

__all__ = [
    "ChunkLevel", "PageClass", "OcrModeName", "DocStatus", "JobType", "JobState",
    "IngestMode", "QualityGrade", "BBox", "PageInfo", "Chapter", "Chunk",
    "Document", "Collection", "Assistant", "AssistantConfig", "RetrievedChunk",
    "RetrievalDebug", "Citation", "AnswerChunk", "GenerationResult",
    "new_id", "chunk_uid_for", "utcnow",
]


def new_id() -> str:
    return uuid.uuid4().hex


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def chunk_uid_for(document_id: str, ordinal: int, text: str) -> str:
    """Контентний, стабільний ідентифікатор чанка — 32 hex.

    Стабільність важлива: збережена відповідь має лишатись розв'язною після
    переіндексації, доки текст не змінився. У промпті модель бачить НЕ це, а
    порядковий номер [n] (див. план, §8): 8 hex — це 32 біти, тобто ~69%
    ймовірності колізії на 100 тис. чанків.
    """
    h = hashlib.blake2b(digest_size=16)
    h.update(document_id.encode())
    h.update(str(ordinal).encode())
    h.update(text.encode())
    return h.hexdigest()


# --------------------------------------------------------------------- енуми
class ChunkLevel(str, Enum):
    DOCUMENT_CARD = "L0"
    SECTION = "L1"
    LEAF = "L2"


class PageClass(str, Enum):
    DIGITAL_CLEAN = "DIGITAL_CLEAN"
    DIGITAL_BROKEN = "DIGITAL_BROKEN"   # текстовий шар є, але хибний (ToUnicode CMap)
    SCANNED = "SCANNED"
    MIXED = "MIXED"


class OcrModeName(str, Enum):
    DEFAULT = "DEFAULT"                 # PDF_AWARE_LAYOUT_REGIONS
    FULL_PAGE = "FULL_PAGE"
    LAYOUT_REGIONS = "LAYOUT_REGIONS"
    NONE = "NONE"


class DocStatus(str, Enum):
    QUEUED = "QUEUED"
    PROBING = "PROBING"
    PARSING = "PARSING"
    ENRICHING = "ENRICHING"
    CHUNKING = "CHUNKING"
    INDEXING = "INDEXING"
    READY = "READY"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class IngestMode(str, Enum):
    FAST = "FAST"   # детермінований Docling; рисунки зберігаються, але не описуються
    DEEP = "DEEP"   # + VLM-опис рисунків + ремонт провалених сторінок


class QualityGrade(str, Enum):
    POOR = "POOR"
    FAIR = "FAIR"
    GOOD = "GOOD"
    EXCELLENT = "EXCELLENT"

    @classmethod
    def from_score(cls, score: float) -> "QualityGrade":
        if score < 0.5:
            return cls.POOR
        if score < 0.8:
            return cls.FAIR
        if score < 0.9:
            return cls.GOOD
        return cls.EXCELLENT


class JobType(str, Enum):
    PROBE = "PROBE"
    PARSE = "PARSE"
    ENRICH_FIGURES = "ENRICH_FIGURES"
    VLM_REPAIR = "VLM_REPAIR"
    CHUNK = "CHUNK"
    INDEX = "INDEX"


class JobState(str, Enum):
    QUEUED = "QUEUED"
    LEASED = "LEASED"
    RUNNING = "RUNNING"
    DONE = "DONE"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


# ------------------------------------------------------------------ значення
@dataclass(frozen=True, slots=True)
class BBox:
    """Прямокутник у координатах PDF (початок — НИЖНІЙ лівий кут, як у PDF).

    Перетворення у viewport робить фронтенд через
    page.getViewport().convertToViewportPoint().
    """
    page: int
    left: float
    top: float
    right: float
    bottom: float

    def as_dict(self) -> dict[str, Any]:
        return {"page": self.page, "l": self.left, "t": self.top,
                "r": self.right, "b": self.bottom}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BBox":
        return cls(page=int(d["page"]), left=float(d["l"]), top=float(d["t"]),
                   right=float(d["r"]), bottom=float(d["b"]))


@dataclass(slots=True)
class PageInfo:
    page_number: int                 # ФІЗИЧНИЙ індекс у файлі, 1-based
    page_label: str | None = None    # ДРУКОВАНА мітка: "2-1", "147", "xii"
    char_from: int = 0
    char_to: int = 0
    width: float = 0.0
    height: float = 0.0
    page_class: PageClass = PageClass.DIGITAL_CLEAN
    ocr_mode: OcrModeName = OcrModeName.DEFAULT
    cost_weight: float = 1.0
    lexicon_hit_rate: float | None = None
    cyrillic_ratio: float | None = None
    mojibake_ratio: float | None = None
    parse_score: float | None = None
    layout_score: float | None = None
    table_score: float | None = None
    ocr_score: float | None = None

    def needs_vlm_repair(self) -> bool:
        """Тригери рівня 2 (див. план, §1)."""
        if self.parse_score is not None and self.parse_score < 0.5:
            return True
        if self.layout_score is not None and self.layout_score < 0.6:
            return True
        if self.table_score is not None and self.table_score < 0.5:
            return True
        if (self.lexicon_hit_rate is not None and self.lexicon_hit_rate < 0.45
                and (self.cyrillic_ratio or 0) > 0.3):
            return True
        return False


@dataclass(slots=True)
class Chapter:
    id: int | None
    document_id: str
    parent_id: int | None
    level: int
    title: str
    lft: int
    rgt: int


@dataclass(slots=True)
class Chunk:
    """Одиниця бази знань.

    Чотири похідні текстові варіанти (покращення проти двох у NeoLens):
      embed_text   — контекст + шлях заголовків + тіло; ІНДЕКСУЄТЬСЯ
      rerank_text  — короткий шлях + тіло, БЕЗ картки документа (крос-енкодери
                     чутливі до шуму в префіксі, а стала картка їсть їхній бюджет)
      display_text — санітизоване тіло; саме це бачить генератор
      body_lemmas  — леми для BM25 (живуть у chunk_fts, не тут)
    """
    document_id: str
    collection_id: str
    ordinal: int
    level: ChunkLevel
    display_text: str
    embed_text: str = ""
    rerank_text: str = ""
    context_note: str = ""
    header_path: str = "//"
    language: str = "uk"
    id: int | None = None
    chunk_uid: str = ""
    parent_id: int | None = None
    chapter_id: int | None = None
    page_from: int | None = None
    page_to: int | None = None
    page_label_from: str | None = None
    page_label_to: str | None = None
    char_from: int | None = None
    char_to: int | None = None
    bboxes: list[BBox] = field(default_factory=list)
    siblings_index: int = 0
    siblings_count: int = 1
    pictures: dict[str, Any] = field(default_factory=dict)
    tables: list[Any] = field(default_factory=list)
    formulas: list[Any] = field(default_factory=list)
    simhash: int | None = None

    def __post_init__(self) -> None:
        if not self.chunk_uid:
            self.chunk_uid = chunk_uid_for(self.document_id, self.ordinal, self.display_text)
        if not self.embed_text:
            self.embed_text = self.build_embed_text()
        if not self.rerank_text:
            self.rerank_text = self.build_rerank_text()

    def build_embed_text(self) -> str:
        """Контекст СИРИЙ і структурний, не згенерований конспект.

        Абляція UNLP 2026: заміна сирого 128-токенного префікса документа на
        LLM-конспект на 80 токенів ПОГІРШУЄ результат (0.9346 → 0.9177), і
        довший конспект гірший за коротший.
        """
        parts = [p for p in (self.context_note, self.header_path.strip("/").replace("//", " › "),
                             self.display_text) if p and p.strip()]
        return "\n".join(parts)

    def build_rerank_text(self) -> str:
        tail = self.header_path.strip("/").split("//")[-1] if self.header_path.strip("/") else ""
        return f"{tail}\n{self.display_text}".strip()

    @property
    def is_indexable(self) -> bool:
        """Лише листки потрапляють у векторний і повнотекстовий індекси.

        Батьки зберігаються для реконструкції й auto-merge (правило NeoLens:
        батько розділеної секції лишається плейсхолдером, але виключений з індексу).
        """
        return self.level is ChunkLevel.LEAF

    def citation_label(self) -> str:
        """Те, що бачить викладач: друкована мітка, а не фізичний індекс."""
        a = self.page_label_from or (str(self.page_from) if self.page_from else None)
        b = self.page_label_to or (str(self.page_to) if self.page_to else None)
        if a and b and a != b:
            return f"с. {a}–{b}"
        return f"с. {a}" if a else ""


@dataclass(slots=True)
class Document:
    id: str
    collection_id: str
    title: str
    original_name: str
    stored_path: str
    content_sha256: str
    parse_profile_hash: str | None = None
    doc_type: str = "textbook"
    language: str = "uk"
    year: int | None = None
    author: str | None = None
    page_count: int = 0
    ingest_mode: IngestMode = IngestMode.FAST
    status: DocStatus = DocStatus.QUEUED
    error_code: str | None = None
    error_detail: str | None = None
    quality_grade: QualityGrade | None = None


@dataclass(slots=True)
class Collection:
    id: str
    assistant_id: str
    name: str
    embedding_model_id: str
    embedding_model_key: str
    dim: int
    metric: str = "cos"
    build_id: str | None = None
    index_generation: int = 0
    dirty: bool = False


@dataclass(slots=True)
class AssistantConfig:
    """Конфіг асистента як ВЕРСІЙОВАНІ ДАНІ, не код (Завдання 4 НДР).

    Ключовий принцип із плану: пороги виконуються В КОДІ, а не в промпті.
    Промптом неможливо надійно змусити 12B-модель відмовитись; порогом — можна.
    """
    instructions: str = ""
    topics_covered: list[str] = field(default_factory=list)
    topics_refused: list[str] = field(default_factory=list)
    # no_information | general_knowledge_marked | refuse
    on_missing_information: Literal["no_information", "general_knowledge_marked", "refuse"] = "no_information"
    # low | medium | high → поріг скора реранкера + мінімум опорних фрагментів
    confidence_required: Literal["low", "medium", "high"] = "medium"
    answer_language: Literal["match_question", "always_uk"] = "match_question"
    always_cite_pages: bool = True

    # --- пошук (див. план, §7). Це параметр №1 для налаштування harness'ом. ---
    dense_top_k: int = 60
    sparse_top_k: int = 60
    rerank_top_k: int = 50
    final_top_k: int = 5
    max_per_document: int = 2       # → щонайменше 3 різні документи у фінальній п'ятірці
    min_distinct_documents: int = 2
    relative_score_floor: float = 0.6
    rrf_k: int = 60
    weight_dense: float = 1.0
    weight_sparse: float = 0.6
    weight_char_ngram: float = 0.4

    # --- генерація ---
    temperature: float = 0.2
    # Бюджет спільний із ЛАНЦЮЖКОМ МІРКУВАНЬ, а не лише з видимою відповіддю.
    # Gemma 4 (і будь-яка reasoning-модель) спершу генерує `reasoning_content`,
    # і лише потім `content`. Замір на реальному питанні: 550 токенів міркування
    # з 600 → `finish_reason: length`, на відповідь лишилось 41 символ, обрізаний
    # на півслові. Довший промпт з'їдав усі 600, і користувач отримував ПОРОЖНЮ
    # відповідь без жодної помилки: клієнт рахує лише дельти `content`, тож стрім
    # завершувався штатно з `tokens_out=0`.
    # Вимкнути міркування через API не можна: LM Studio ігнорує і `reasoning: off`,
    # і `reasoning.effort: none`, і `chat_template_kwargs.enable_thinking: false`.
    # Єдиний важіль — бюджет. 2200 = ~600 на відповідь + запас на міркування.
    max_tokens: int = 2200
    repeat_penalty: float = 1.08    # НІКОЛИ вище: високий штраф калічить українську словозміну
    prompt_tier: Literal["compact", "full"] = "compact"

    def confidence_threshold(self) -> float:
        return {"low": 0.15, "medium": 0.35, "high": 0.55}[self.confidence_required]

    def min_supporting_chunks(self) -> int:
        return {"low": 1, "medium": 1, "high": 2}[self.confidence_required]

    def to_json(self) -> str:
        from dataclasses import asdict
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> "AssistantConfig":
        data = json.loads(raw) if raw else {}
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass(slots=True)
class Assistant:
    id: str
    name: str
    description: str = ""
    colour: str = "#4f46e5"
    emoji: str = "📘"
    config: AssistantConfig = field(default_factory=AssistantConfig)
    config_version: int = 1


# ------------------------------------------------------------------- пошук
@dataclass(slots=True)
class RetrievedChunk:
    chunk: Chunk
    document_title: str
    dense_score: float | None = None
    sparse_score: float | None = None
    ngram_score: float | None = None
    fused_score: float = 0.0
    rerank_score: float | None = None
    ordinal_in_prompt: int | None = None   # це і є [n], яке бачить модель

    @property
    def score(self) -> float:
        return self.rerank_score if self.rerank_score is not None else self.fused_score


@dataclass(slots=True)
class RetrievalDebug:
    """Те, що показує екран «Чому ця відповідь»."""
    query: str
    dense_count: int = 0
    sparse_count: int = 0
    ngram_count: int = 0
    fused_count: int = 0
    reranked_count: int = 0
    final_count: int = 0
    distinct_documents: int = 0
    abstained: bool = False
    abstain_confidence: float | None = None
    latency_ms: dict[str, float] = field(default_factory=dict)
    candidates: list[dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> str:
        from dataclasses import asdict
        return json.dumps(asdict(self), ensure_ascii=False)


@dataclass(slots=True)
class Citation:
    ordinal: int          # [n], яке модель написала у відповіді
    chunk_uid: str
    document_id: str
    document_title: str
    page_from: int | None
    page_to: int | None
    page_label: str
    quote: str
    bboxes: list[dict[str, Any]] = field(default_factory=list)
    language: str = "uk"


@dataclass(slots=True)
class AnswerChunk:
    """Одна порція стрімінгу."""
    kind: Literal["token", "citations", "debug", "error", "done"]
    text: str = ""
    payload: Any = None


@dataclass(slots=True)
class GenerationResult:
    text: str
    citations: list[Citation] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    abstained: bool = False
    debug: RetrievalDebug | None = None
    ttft_ms: int | None = None
    tokens_out: int | None = None
    model_id: str | None = None
