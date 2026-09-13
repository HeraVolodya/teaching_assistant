"""Драбина генератора: чи справді залізо обирає ту модель, що в плані §10."""

from __future__ import annotations

import pytest

from app.backends.base import LoadConfig
from app.backends.hardware_probe import GIB, GpuInfo, Hardware
from app.backends.model_registry import (
    LADDER,
    REGISTRY,
    ModelSpec,
    plan_generation,
    recommend_model,
)


def nvidia(gib: float, *, ram_gib: float = 32.0) -> Hardware:
    return Hardware(
        platform="win32", machine="AMD64", cpu_count=8, ram_bytes=int(ram_gib * GIB),
        gpus=(GpuInfo(name="NVIDIA", total_bytes=int(gib * GIB), vendor="nvidia"),),
    )


def apple(unified_gib: float) -> Hardware:
    usable = int(unified_gib * 0.70 * GIB)
    return Hardware(
        platform="darwin", machine="arm64", cpu_count=10,
        ram_bytes=int(unified_gib * GIB),
        gpus=(GpuInfo(name="Apple", total_bytes=usable, vendor="apple"),),
        is_apple_silicon=True, unified_memory=True,
    )


def cpu_only() -> Hardware:
    return Hardware(platform="win32", machine="AMD64", cpu_count=8,
                    ram_bytes=16 * GIB, gpus=())


def test_default_rung_is_mamaylm_12b_at_16k() -> None:
    """12 ГіБ — наш дефолт: MamayLM 12B Q4_K_M, 16k, повне вивантаження."""
    plan = plan_generation(nvidia(12))
    assert plan.spec.key == "mamaylm-12b-q4km"
    assert plan.n_ctx == 16384
    assert plan.gpu_offload == 1.0
    assert plan.estimate.fits


def test_8_gib_drops_to_8k_and_partial_offload() -> None:
    plan = plan_generation(nvidia(8, ram_gib=16))
    assert plan.spec.key == "mamaylm-12b-q4km"
    assert plan.n_ctx == 8192
    assert 0.0 < plan.gpu_offload < 1.0
    assert plan.estimate.fits
    assert any("знижено" in w for w in plan.warnings)


def test_cpu_only_falls_to_gemma_e4b() -> None:
    plan = plan_generation(cpu_only())
    assert plan.cpu_only
    assert plan.spec.key == "gemma-4-e4b-it-qat"
    assert plan.n_ctx == 8192


def test_16_gib_prefers_q8_12b_over_low_quant_27b() -> None:
    """Q8 на 12B майже напевно кращий за Q3 на 27B — не гнатись за параметрами."""
    plan = plan_generation(nvidia(16))
    assert plan.spec.key == "mamaylm-12b-q8"


def test_unconfirmed_build_is_skipped_by_default() -> None:
    """GGUF 27B не підтверджено → 24 ГіБ падають на попередній щабель,
    а не показують викладачеві модель, якої може не існувати."""
    plan = plan_generation(nvidia(24))
    assert plan.spec.availability == "confirmed"
    assert plan.spec.key == "mamaylm-12b-q8"
    assert any("не підтверджена" in w for w in plan.warnings)

    allowed = plan_generation(nvidia(24), allow_unconfirmed=True)
    assert allowed.spec.key == "mamaylm-27b-q4km"


def test_apple_silicon_uses_mlx_entry_not_gguf() -> None:
    """Офіційної MLX-збірки MamayLM немає — на Apple беремо базову Gemma."""
    plan = plan_generation(apple(32))
    assert plan.spec.engine == "mlx"
    assert plan.spec.mlx_repo
    assert plan.spec.gguf_repo is None
    assert plan.gpu_offload == 1.0    # часткового офлоаду в MLX не існує


def test_apple_with_little_memory_falls_to_cpu_rung() -> None:
    plan = plan_generation(apple(8))
    assert plan.spec.key == "gemma-4-e4b-it-qat"


def test_recommend_model_matches_contract_signature() -> None:
    spec = recommend_model(nvidia(12))
    assert isinstance(spec, ModelSpec)
    assert spec.key == "mamaylm-12b-q4km"


def test_mlx_load_payload_drops_llama_cpp_only_parameters() -> None:
    """Найдорожча помилка інтеграції: слати eval_batch_size у MLX-рушій."""
    llama = LoadConfig(model="m", context_length=8192, engine="llama.cpp",
                       gpu_offload=0.78).to_payload()
    mlx = LoadConfig(model="m", context_length=8192, engine="mlx").to_payload()

    assert llama["eval_batch_size"] == 512
    assert llama["flash_attention"] is True
    assert llama["offload_kv_cache_to_gpu"] is True
    assert llama["gpu_offload"] == 0.78
    assert llama["echo_load_config"] is True

    assert set(mlx) == {"model", "context_length", "echo_load_config"}


@pytest.mark.parametrize("rung", LADDER)
def test_every_rung_points_at_a_known_model(rung) -> None:
    assert rung.model_key in REGISTRY


def test_registry_documents_unverified_architectures() -> None:
    """Чесність важливіша за красу: там, де параметри архітектури не
    підтверджені, це має бути видно в даних, а не лише в чиїйсь голові."""
    unverified = {k for k, m in REGISTRY.items() if not m.arch_verified}
    assert "mamaylm-12b-q4km" not in unverified     # цей підтверджено планом
    assert unverified, "прапорець arch_verified має реально використовуватись"


def test_measured_file_size_overrides_table_estimate() -> None:
    """W — РЕАЛЬНИЙ розмір GGUF на диску, а не наш прогноз."""
    spec = REGISTRY["mamaylm-12b-q4km"].with_measured_file_bytes(int(7.05 * GIB))
    assert spec.file_bytes == int(7.05 * GIB)
    assert spec.key == "mamaylm-12b-q4km"


def test_gemma12_layer_split_is_8_global_40_local() -> None:
    spec = REGISTRY["mamaylm-12b-q4km"]
    assert spec.n_global_layers == 8
    assert spec.n_local_layers == 40
