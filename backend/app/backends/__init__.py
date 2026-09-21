"""Шар LLM: LM Studio, заглушка, проба заліза, драбина моделей.

Публічний інтерфейс модуля (див. docs/CONTRACT.md):
    LlmBackend            — протокол: chat_stream / health / load_model / list_models
    probe_hardware()      -> Hardware
    recommend_model(hw)   -> ModelSpec
    get_backend()         -> LlmBackend      (заглушка при ASISTENT_STUB=1)

Цей пакет НЕ імпортує torch, docling чи rapidocr — він живе в API-процесі,
холодний старт якого мусить лишатись близько 0.6 с.
"""

from __future__ import annotations

from app.backends.base import (
    BackendHealth,
    ChatParams,
    LlmBackend,
    LlmCancelled,
    LlmError,
    LlmProtocolError,
    LlmTimeout,
    LlmUnavailable,
    LoadConfig,
    Message,
    Messages,
    ModelInfo,
    ModelLoadRefused,
    ModelNotLoaded,
)
from app.backends.hardware_probe import (
    Hardware,
    MachineCalibration,
    VramEstimate,
    estimate_vram,
    kv_cache_bytes,
    plan_gpu_offload,
    probe_hardware,
)
from app.backends.lmstudio_client import LmStudioClient
from app.backends.model_registry import (
    GenerationPlan,
    ModelSpec,
    plan_generation,
    recommend_model,
)
from app.backends.stub_backend import StubBackend, is_stub_enabled

__all__ = [
    "BackendHealth",
    "ChatParams",
    "GenerationPlan",
    "Hardware",
    "LlmBackend",
    "LlmCancelled",
    "LlmError",
    "LlmProtocolError",
    "LlmTimeout",
    "LlmUnavailable",
    "LmStudioClient",
    "LoadConfig",
    "MachineCalibration",
    "Message",
    "Messages",
    "ModelInfo",
    "ModelLoadRefused",
    "ModelNotLoaded",
    "ModelSpec",
    "StubBackend",
    "VramEstimate",
    "estimate_vram",
    "get_backend",
    "is_stub_enabled",
    "kv_cache_bytes",
    "plan_generation",
    "plan_gpu_offload",
    "probe_hardware",
    "recommend_model",
]


async def get_backend(*, base_url: str | None = None) -> LlmBackend:
    """Єдина фабрика бекенда для всього застосунку.

    ASISTENT_STUB=1 має пріоритет над усім: у CI не має значення, чи стоїть
    на агенті LM Studio, і випадкове потрапляння реального бекенда в тест
    зробило б його недетермінованим.
    """
    if is_stub_enabled():
        return StubBackend()
    client = await LmStudioClient.discover() if base_url is None else LmStudioClient(base_url)
    if client is None:
        raise LlmUnavailable("Автовиявлення не знайшло LM Studio на 127.0.0.1.")
    return client
