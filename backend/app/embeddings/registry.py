"""Реєстр embedding-моделей.

Це найтихіше місце для помилки в усій системі. Кожна родина має власний
контракт форматування, і якщо переплутати — вектори виглядають правдоподібно,
жоден тест не падає, а recall тихо стає випадковим.

Тому контракт тут — ДАНІ, а не розкидані по коду рядки, і `selftest.py`
перевіряє його емпірично при кожному старті.

Ключові факти, що визначили дефолт (Qwen3-Embedding-0.6B):
  * Незалежний український юридичний бенчмарк, 203 запити / 11 516 чанків:
    Recall@5 — OpenAI-3-small 78.3%, BGE-M3 81.3%, Qwen3-4B 89.7%,
    Qwen3-8B@2048 93.1%. MRR — 0.765 / 0.802 / 0.891 / 0.921.
  * UNLP 2026 Shared Task (Львів): 0.6B відстав від 8B лише на 0.6 п. public /
    0.9 п. private, а найбільшим важелем був не розмір ембедера, а структурний
    контекст (+8.4 п.) і реранкінг (+9.8 п. Recall@1).
  * Ліцензійний фільтр: державна НДР із розповсюджуваним бінарником →
    лише Apache-2.0 / MIT. Це знімає jina-v4/v5 (CC-BY-NC / Qwen Research).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum

__all__ = ["Pooling", "Tier", "EmbeddingModel", "REGISTRY", "DEFAULT_MODEL_ID", "get", "model_key"]

# Версія конвеєра нормалізації українського тексту (app/ingestion/normalize_uk.py).
# Входить у ключ моделі: зміна нормалізації робить старі вектори несумісними.
TEXT_PREPROC_VERSION = 1


class Pooling(str, Enum):
    LAST_TOKEN = "last_token"
    CLS = "cls"
    MEAN = "mean"


class Tier(str, Enum):
    SMALL = "small"      # слабкий CPU
    DEFAULT = "default"
    BEST = "best"        # >= 24 ГБ VRAM


@dataclass(frozen=True, slots=True)
class EmbeddingModel:
    id: str
    revision: str
    dim: int
    max_seq: int
    pooling: Pooling
    licence: str
    tier: Tier
    onnx_file: str
    # `{query}` / `{task}` підставляються. Порожній рядок = префікса немає.
    query_template: str = ""
    doc_template: str = ""
    requires_eos: bool = False
    normalize: bool = True
    notes: str = ""
    aliases: tuple[str, ...] = field(default_factory=tuple)

    def format_query(self, query: str, task: str) -> str:
        if not self.query_template:
            return query
        return self.query_template.format(query=query, task=task)

    def format_document(self, text: str, title: str = "") -> str:
        if not self.doc_template:
            return text
        return self.doc_template.format(text=text, title=title or "none")


# Українське інструкційне завдання, адаптоване з системи-переможця UNLP 2026.
# Qwen документує 1–5% приросту retrieval від задачо-специфічної інструкції.
UK_RETRIEVAL_TASK = (
    "Given a question in Ukrainian about military or technical teaching materials, "
    "retrieve the passage from the textbook or methodical development that answers it"
)


REGISTRY: dict[str, EmbeddingModel] = {
    # ---------------------------------------------------------------- дефолт
    "qwen3-embedding-0.6b": EmbeddingModel(
        id="Qwen/Qwen3-Embedding-0.6B",
        revision="main",
        dim=1024,                      # MRL 32..1024
        max_seq=32768,
        pooling=Pooling.LAST_TOKEN,
        licence="Apache-2.0",
        tier=Tier.DEFAULT,
        onnx_file="model_int8.onnx",
        # УВАГА: картка Qwen показує "Query:{query}" БЕЗ пробілу.
        # Не ділити цей рядок з іншими родинами — копіювати побайтово з картки.
        query_template="Instruct: {task}\nQuery:{query}",
        doc_template="",               # документ — сирий текст, без префікса
        requires_eos=True,             # last-token pooling без EOS = pooling не того токена
        notes="Дефолт. Прямі українські виміри; єдиний розмір, що працює і в LM Studio (#1647).",
        aliases=("qwen3-0.6b", "default"),
    ),
    # ---------------------------------------------------------------- швидкий
    "granite-embedding-97m-r2": EmbeddingModel(
        id="ibm-granite/granite-embedding-97m-multilingual-r2",
        revision="main",
        dim=384,
        max_seq=8192,
        pooling=Pooling.CLS,
        licence="Apache-2.0",
        tier=Tier.SMALL,
        onnx_file="model_int8.onnx",
        query_template="",             # префіксів немає ЗОВСІМ
        doc_template="",
        notes="Українська в 52 мовах посиленої підтримки. 2534 док/с на H100 (batch 512).",
        aliases=("granite-97m", "small"),
    ),
    # ---------------------------------------------------------------- якість
    "qwen3-embedding-4b": EmbeddingModel(
        id="Qwen/Qwen3-Embedding-4B",
        revision="main",
        dim=2560,
        max_seq=32768,
        pooling=Pooling.LAST_TOKEN,
        licence="Apache-2.0",
        tier=Tier.BEST,
        onnx_file="model_int8.onnx",
        query_template="Instruct: {task}\nQuery:{query}",
        doc_template="",
        requires_eos=True,
        notes="Той самий контракт, що й 0.6B — перемикання рівня є зміною конфіга, не коду. "
              "Потребує >=24 ГБ VRAM, інакше затримка запиту йде в 1.5-3 с.",
        aliases=("qwen3-4b", "best"),
    ),
    # ------------------------------------------------- кандидати бейк-офу дня 1
    "bge-m3": EmbeddingModel(
        id="BAAI/bge-m3",
        revision="main",
        dim=1024,
        max_seq=8192,
        pooling=Pooling.CLS,
        licence="MIT",
        tier=Tier.DEFAULT,
        onnx_file="model.onnx",
        query_template="",
        doc_template="",
        notes="Порівняльна точка. Її multi-vector голова відкинута: ColBERT дає 1024 виміри "
              "НА ТОКЕН, тобто ~100 ГБ на 100k чанків.",
    ),
    "pplx-embed-context-0.6b": EmbeddingModel(
        id="perplexity-ai/pplx-embed-context-v1-0.6b",
        revision="main",
        dim=1024,
        max_seq=32768,
        pooling=Pooling.LAST_TOKEN,
        licence="MIT",
        tier=Tier.DEFAULT,
        onnx_file="model_int8.onnx",
        query_template="Instruct: {task}\nQuery:{query}",
        doc_template="",
        requires_eos=True,
        notes="Кандидат бейк-офу: споживає контекст документа нативно через вкладені масиви "
              "чанків. Використовувався призером UNLP 2026. Перевірити проти дефолту на "
              "реальному корпусі перед перемиканням.",
    ),
    "multilingual-e5-large": EmbeddingModel(
        id="intfloat/multilingual-e5-large",
        revision="main",
        dim=1024,
        max_seq=512,                   # жорстко 512 — замалий для наших листків
        pooling=Pooling.MEAN,
        licence="MIT",
        tier=Tier.SMALL,
        onnx_file="model_int8.onnx",
        # Літеральні префікси на КОЖНОМУ тексті, обидві сторони.
        # Картка: "Yes, this is how the model is trained, otherwise you will see
        # a performance degradation."
        query_template="query: {query}",
        doc_template="passage: {text}",
        notes="Порівняльна точка. max_seq 512 дискваліфікує її як основну для наших чанків.",
    ),
}

DEFAULT_MODEL_ID = "qwen3-embedding-0.6b"

_ALIASES: dict[str, str] = {
    alias: key for key, m in REGISTRY.items() for alias in (*m.aliases, key)
}


def get(name: str | None = None) -> EmbeddingModel:
    key = _ALIASES.get((name or DEFAULT_MODEL_ID).lower())
    if key is None:
        known = ", ".join(sorted(REGISTRY))
        raise KeyError(f"Невідома embedding-модель {name!r}. Відомі: {known}")
    return REGISTRY[key]


def model_key(model: EmbeddingModel, *, text_preproc_version: int = TEXT_PREPROC_VERSION) -> str:
    """Ключ версіювання векторів.

    Узагальнює наявний у NeoLens ключ кешу sha256(model_class + joined_titles).
    Саме він робить міграцію між рівнями неруйнівною: вектори зберігаються з
    ключем, і два ключі можуть співіснувати, доки міграційний гейт не
    підтвердить нерегресію Recall@10 / MRR / nDCG на золотому наборі.
    """
    payload = "|".join(
        (
            model.id,
            model.revision,
            str(model.dim),
            model.pooling.value,
            "norm" if model.normalize else "raw",
            model.query_template,
            model.doc_template,
            f"eos={int(model.requires_eos)}",
            f"preproc={text_preproc_version}",
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
