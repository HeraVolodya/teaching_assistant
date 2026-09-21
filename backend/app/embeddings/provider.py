"""Єдиний інтерфейс ембедингів для всього застосунку.

Решта системи НІКОЛИ не знає, що всередині — сесія ONNX Runtime чи заглушка.
Вона бачить лише `embed_queries` / `embed_documents` / `dim` / `model_key`.

Чому саме такий поділ на два методи, а не один `embed(texts)`:
контракт форматування асиметричний. Qwen3 подає запит як
`"Instruct: {task}\\nQuery:{query}"`, а документ — сирим текстом. Помилка тут
не падає, не логується і не ловиться жодним тестом типів: вектори виглядають
цілком правдоподібно, а recall тихо стає випадковим. Тому шаблони живуть у
`registry.py` як ДАНІ, застосовуються рівно в одному місці (тут) і емпірично
перевіряються `selftest.py` при кожному старті.

Друге правило: ембедер працює у ВЛАСНОМУ процесі через onnxruntime, а не через
LM Studio (план, §4). Auto-Evict у LM Studio вивантажить чат-модель, щойно
хтось попросить ембединг; Qwen3-Embedding 4B/8B GGUF там просто ламаються
(issue #1647); а при last-token pooling відсутній EOS дає pooling не того
токена — абсолютно тихо.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

import numpy as np

from app.embeddings import registry
from app.embeddings.registry import UK_RETRIEVAL_TASK, EmbeddingModel

__all__ = [
    "STUB_ENV",
    "BaseEmbeddingProvider",
    "EmbeddingProvider",
    "EncodeStats",
    "Side",
    "create_provider",
    "default_model_dir",
    "l2_normalize",
    "stub_enabled",
]

Side = Literal["query", "document"]

STUB_ENV = "ASISTENT_STUB"
MODEL_DIR_ENV = "ASISTENT_EMBED_MODEL_DIR"
MAX_TOKENS_ENV = "ASISTENT_EMBED_MAX_TOKENS"
TOKEN_BUDGET_ENV = "ASISTENT_EMBED_TOKEN_BUDGET"

# Стеля довжини входу. Жорсткий максимум чанка — 3600 символів (контракт, §4),
# що під токенайзером Qwen — близько 1800 токенів. 2048 лишає запас і водночас
# не дає одному аномальному чанку роздути паддінг усього батча до 32k.
DEFAULT_MAX_INPUT_TOKENS = 2048

# Бюджет батча в ТОКЕНАХ, не в рядках: батч із 64 рядків по 1500 токенів і батч
# із 64 рядків по 20 токенів відрізняються за вартістю у 75 разів.
DEFAULT_TOKEN_BUDGET = 16384


def stub_enabled() -> bool:
    """Чи ввімкнено детерміновані заглушки (`ASISTENT_STUB=1`)."""
    raw = os.environ.get(STUB_ENV, "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    """L2-нормалізація по рядках; нульовий рядок лишається нульовим.

    Ділення на нуль тут не абстрактна небезпека: порожній чи повністю
    відфільтрований текст дає нульовий вектор, а `nan` у векторі отруює весь
    HNSW-граф при перебудові індексу.
    """
    mat = np.asarray(mat, dtype=np.float32)
    if mat.ndim == 1:
        mat = mat.reshape(1, -1)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return (mat / norms).astype(np.float32)


@dataclass(slots=True)
class EncodeStats:
    """Телеметрія кодування — саме звідси береться рядок «обрізано N чанків».

    Обрізання МАЄ бути видимим: мовчазне відсікання хвоста чанка — це втрата
    саме тієї частини означення, заради якої чанк і індексували.
    """

    texts: int = 0
    tokens: int = 0
    truncated: int = 0
    batches: int = 0
    max_tokens_seen: int = 0

    def note(self, *, tokens: int, truncated: bool) -> None:
        self.texts += 1
        self.tokens += tokens
        self.max_tokens_seen = max(self.max_tokens_seen, tokens)
        if truncated:
            self.truncated += 1

    def reset(self) -> None:
        self.texts = 0
        self.tokens = 0
        self.truncated = 0
        self.batches = 0
        self.max_tokens_seen = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "texts": self.texts,
            "tokens": self.tokens,
            "truncated": self.truncated,
            "batches": self.batches,
            "max_tokens_seen": self.max_tokens_seen,
        }


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Контракт, проти якого пишуть пошук, індексація й оцінювання."""

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        """Матриця (n, dim), float32, L2-нормована; шаблон запиту застосовано."""
        ...

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Матриця (n, dim), float32, L2-нормована; шаблон документа застосовано."""
        ...

    @property
    def dim(self) -> int:
        ...

    @property
    def model_key(self) -> str:
        ...


class BaseEmbeddingProvider(ABC):
    """Спільна частина: форматування за реєстром, нормалізація, статистика.

    Обидва бекенди (ONNX і заглушка) успадковують саме її, тому шаблон префікса
    застосовується побайтово однаково — інакше заглушка тестувала б не той код,
    що працює в постачанні.
    """

    def __init__(self, model: EmbeddingModel, *, task: str = UK_RETRIEVAL_TASK) -> None:
        self._model = model
        self._task = task
        self.stats = EncodeStats()

    # ------------------------------------------------------------ властивості
    @property
    def model(self) -> EmbeddingModel:
        return self._model

    @property
    def dim(self) -> int:
        return self._model.dim

    @property
    def model_key(self) -> str:
        return registry.model_key(self._model)

    @property
    def task(self) -> str:
        return self._task

    @property
    @abstractmethod
    def max_input_tokens(self) -> int:
        """Скільки токенів реально приймає сесія (з урахуванням стелі рантайму)."""

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Довжина тексту в токенах ЦІЄЇ моделі. Використовує selftest, §4.4."""

    @abstractmethod
    def _encode_formatted(self, texts: list[str], *, side: Side) -> np.ndarray:
        """Закодувати вже відформатовані рядки. Нормалізацію робить `encode`.

        `side` передається саме сюди, а не виводиться з тексту, бо бекенд мусить
        знати, ЯКИЙ контракт форматування він мав отримати. Заглушка на цьому
        будує імітацію «без префікса вектор деградує» (див. `stub_backend`), а
        ONNX-бекенд — вибір pooling'у й діагностику EOS.
        """

    # -------------------------------------------------------------- публічне
    def format_text(self, text: str, *, side: Side, apply_template: bool = True) -> str:
        """Застосувати шаблон реєстру ПОБАЙТОВО.

        У Qwen `"Query:{query}"` — без пробілу після двокрапки. Це не помилка
        друку в реєстрі, а те, що написано в картці моделі; «виправлення»
        цього пробілу — тихий регрес recall.
        """
        if not apply_template:
            return text
        if side == "query":
            return self._model.format_query(text, self._task)
        return self._model.format_document(text)

    def encode(
        self,
        texts: list[str],
        *,
        side: Side,
        apply_template: bool = True,
    ) -> np.ndarray:
        """Основний вхід. `apply_template=False` потрібен лише самотесту (§4.3)."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        formatted = [self.format_text(t, side=side, apply_template=apply_template) for t in texts]
        out = self._encode_formatted(formatted, side=side)
        out = np.asarray(out, dtype=np.float32)
        if out.shape[0] != len(texts):
            raise RuntimeError(
                f"Бекенд ембедингів повернув {out.shape[0]} векторів на {len(texts)} текстів."
            )
        if out.shape[1] != self.dim:
            raise RuntimeError(
                f"Розмірність вектора {out.shape[1]} не збігається з реєстром ({self.dim}) "
                f"для моделі {self._model.id}."
            )
        return l2_normalize(out) if self._model.normalize else out

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self.encode(list(texts), side="query")

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self.encode(list(texts), side="document")

    def close(self) -> None:  # noqa: B027 — свідомо НЕ абстрактний
        """Звільнити сесію. Заглушці нічого звільняти.

        Порожня реалізація навмисна: більшість провайдерів не тримають ресурсів,
        і вимагати від кожного писати `pass` означало б шум без користі.
        """


# ------------------------------------------------------------------- фабрика
def default_model_dir(model: EmbeddingModel, models_dir: Path | None = None) -> Path:
    """Де лежать ваги.

    Ніколи не кеш HuggingFace: у закритому контурі моделі приїжджають із USB і
    лежать у каталозі даних застосунку (`config.Paths.models_dir`), звідки їх
    видно й адміністратору, і інсталятору.
    """
    override = os.environ.get(MODEL_DIR_ENV, "").strip()
    if override:
        return Path(override).expanduser()
    if models_dir is None:
        from app.config import Paths

        models_dir = Paths.resolve().models_dir
    key = next((k for k, m in registry.REGISTRY.items() if m is model), model.id.replace("/", "--"))
    return Path(models_dir) / "embeddings" / key


def create_provider(
    model_name: str | None = None,
    stub: bool | None = None,
    *,
    model_dir: Path | None = None,
    models_dir: Path | None = None,
    task: str = UK_RETRIEVAL_TASK,
    max_input_tokens: int | None = None,
    token_budget: int | None = None,
) -> EmbeddingProvider:
    """Створити провайдер ембедингів.

    `stub=None` означає «дивись на `ASISTENT_STUB`». Явне `stub=True/False`
    потрібне лише тестам і скриптам оцінювання.
    """
    model = registry.get(model_name)
    use_stub = stub_enabled() if stub is None else stub

    if use_stub:
        from app.embeddings.stub_backend import StubEmbeddingProvider

        return StubEmbeddingProvider(model, task=task, max_input_tokens=max_input_tokens)

    from app.embeddings.onnx_backend import OnnxEmbeddingProvider

    path = model_dir if model_dir is not None else default_model_dir(model, models_dir)
    return OnnxEmbeddingProvider(
        model,
        model_dir=path,
        task=task,
        max_input_tokens=max_input_tokens,
        token_budget=token_budget,
    )
