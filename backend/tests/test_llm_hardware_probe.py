"""Арифметика VRAM — те, що відділяє «працює» від «OOM посеред відповіді».

Головні перевірки: формула §10 на прикладах 8 і 12 ГіБ, SWA проти повної
уваги, і незламне правило «запас < 1.0 ГіБ — не приймати».
"""

from __future__ import annotations

import pytest

from app.backends.hardware_probe import (
    GIB,
    MIN_HEADROOM_GIB,
    GpuInfo,
    Hardware,
    MachineCalibration,
    estimate_vram,
    kv_cache_bytes,
    plan_gpu_offload,
    probe_hardware,
)

# Gemma-3-12B (= MamayLM v2): підтверджені планом параметри архітектури.
GEMMA12 = dict(n_layers=48, n_kv_heads=8, head_dim=256, swa_window=1024, global_every=6)


def windows_gpu(gib: float) -> Hardware:
    return Hardware(
        platform="win32", machine="AMD64", cpu_count=8, ram_bytes=32 * GIB,
        gpus=(GpuInfo(name="NVIDIA", total_bytes=int(gib * GIB), vendor="nvidia",
                      is_primary_display=True),),
    )


# --------------------------------------------------------------------- KV
def test_kv_per_layer_per_token_is_8_kib_at_f16() -> None:
    """2·n_kv·head_dim·b = 2·8·256·2.0 = 8 КіБ на шар на токен."""
    one = kv_cache_bytes(n_ctx=1, n_layers=1, n_kv_heads=8, head_dim=256, kv_dtype="f16")
    assert one == 8 * 1024


def test_swa_cuts_kv_at_16k_by_4_6x() -> None:
    """KV@16k: 1.31 ГіБ із SWA проти 6.00 ГіБ без неї (план, §10)."""
    with_swa = kv_cache_bytes(n_ctx=16384, kv_dtype="f16", **GEMMA12)
    without = kv_cache_bytes(n_ctx=16384, kv_dtype="f16",
                             **{**GEMMA12, "swa_window": None, "global_every": None})
    assert with_swa / GIB == pytest.approx(1.31, abs=0.01)
    assert without / GIB == pytest.approx(6.00, abs=0.01)
    assert without / with_swa == pytest.approx(4.57, abs=0.05)


@pytest.mark.parametrize(("dtype", "expected_gib"), [("f16", 1.3125),
                                                     ("q8_0", 0.697),
                                                     ("q4_0", 0.369)])
def test_kv_quantisation_scales_linearly(dtype: str, expected_gib: float) -> None:
    kv = kv_cache_bytes(n_ctx=16384, kv_dtype=dtype, **GEMMA12)
    assert kv / GIB == pytest.approx(expected_gib, abs=0.005)


def test_unknown_kv_dtype_is_loud() -> None:
    with pytest.raises(ValueError, match="Невідомий тип KV-кешу"):
        kv_cache_bytes(n_ctx=1024, kv_dtype="q3_k", **GEMMA12)


# ------------------------------------------------------- формула на 12 ГіБ
def test_vram_formula_reproduces_plan_numbers_at_12_gib() -> None:
    """MamayLM 12B Q4_K_M @ 16k, f16 KV → 9.61 ГіБ, запас +1.39 ГіБ."""
    hw = windows_gpu(12)
    est = estimate_vram(
        weights_bytes=int(7.3 * GIB), n_ctx=16384, total_vram_bytes=hw.vram_bytes,
        kv_dtype="f16", cuda_context_bytes=hw.cuda_context_bytes,
        display_reserve_bytes=hw.display_reserve_bytes, **GEMMA12,
    )
    # W + KV + C + O = 7.30 + 1.31 + 0.60 + 0.40
    assert est.model_side_bytes / GIB == pytest.approx(9.61, abs=0.01)
    assert est.headroom_gib == pytest.approx(1.39, abs=0.01)
    assert est.fits


def test_full_attention_at_16k_does_not_fit_12_gib() -> None:
    """Без SWA ті самі 16k дають +6 ГіБ KV — конфігурація мусить відпасти."""
    hw = windows_gpu(12)
    est = estimate_vram(
        weights_bytes=int(7.3 * GIB), n_ctx=16384, total_vram_bytes=hw.vram_bytes,
        cuda_context_bytes=hw.cuda_context_bytes,
        display_reserve_bytes=hw.display_reserve_bytes,
        **{**GEMMA12, "swa_window": None, "global_every": None},
    )
    assert not est.fits


# -------------------------------------------------------- формула на 8 ГіБ
def test_8_gib_needs_partial_offload_and_keeps_headroom() -> None:
    """На 8 ГіБ 12B цілком не влазить: потрібне часткове вивантаження шарів."""
    hw = windows_gpu(8)
    full = estimate_vram(
        weights_bytes=int(7.3 * GIB), n_ctx=8192, total_vram_bytes=hw.vram_bytes,
        cuda_context_bytes=hw.cuda_context_bytes,
        display_reserve_bytes=hw.display_reserve_bytes, **GEMMA12,
    )
    assert not full.fits

    offload = plan_gpu_offload(
        weights_bytes=int(7.3 * GIB), n_ctx=8192, total_vram_bytes=hw.vram_bytes,
        cuda_context_bytes=hw.cuda_context_bytes,
        display_reserve_bytes=hw.display_reserve_bytes, **GEMMA12,
    )
    # План називає 0.78; наше правило «запас >= 1.0 ГіБ» суворіше, тому
    # реальна цифра НИЖЧА — і це навмисно.
    assert 0.5 < offload <= 0.78
    partial = estimate_vram(
        weights_bytes=int(7.3 * GIB), n_ctx=8192, total_vram_bytes=hw.vram_bytes,
        gpu_offload=offload, cuda_context_bytes=hw.cuda_context_bytes,
        display_reserve_bytes=hw.display_reserve_bytes, **GEMMA12,
    )
    assert partial.fits
    assert partial.headroom_gib >= MIN_HEADROOM_GIB


def test_headroom_below_one_gib_is_rejected() -> None:
    """Незламне правило: 0.9 ГіБ запасу — це не «майже», це майбутній OOM."""
    hw = windows_gpu(12)
    est = estimate_vram(
        weights_bytes=int(8.7 * GIB), n_ctx=8192, total_vram_bytes=hw.vram_bytes,
        cuda_context_bytes=hw.cuda_context_bytes,
        display_reserve_bytes=hw.display_reserve_bytes, **GEMMA12,
    )
    assert 0 < est.headroom_gib < MIN_HEADROOM_GIB
    assert not est.fits


def test_display_reserve_only_on_windows_primary_gpu() -> None:
    win = windows_gpu(12)
    linux = Hardware(platform="linux", machine="x86_64", cpu_count=8, ram_bytes=32 * GIB,
                     gpus=(GpuInfo(name="NVIDIA", total_bytes=12 * GIB, vendor="nvidia"),))
    assert win.display_reserve_bytes == int(1.0 * GIB)
    assert linux.display_reserve_bytes == int(0.4 * GIB)


def test_calibration_multiplier_applies_to_kv() -> None:
    """Баг LM Studio #1129 може змінити реальний KV у рази — поправка з
    `lms load --estimate-only` мусить проходити крізь формулу."""
    hw = windows_gpu(12)
    common = dict(weights_bytes=int(7.3 * GIB), n_ctx=16384,
                  total_vram_bytes=hw.vram_bytes,
                  cuda_context_bytes=hw.cuda_context_bytes,
                  display_reserve_bytes=hw.display_reserve_bytes, **GEMMA12)
    base = estimate_vram(**common)
    doubled = estimate_vram(**common, calibration=MachineCalibration(kv_multiplier=2.0))
    assert doubled.kv_bytes == pytest.approx(base.kv_bytes * 2, rel=1e-6)
    assert not doubled.fits


# ------------------------------------------------------------------- проба
def test_probe_hardware_never_raises_and_reports_something() -> None:
    hw = probe_hardware()
    assert hw.cpu_count >= 1
    assert hw.platform
    assert isinstance(hw.describe(), str)


def test_apple_unified_memory_is_capped_at_usable_fraction(monkeypatch) -> None:
    """Уніфікована пам'ять: придатних ~70%, а не всі 100%."""
    import app.backends.hardware_probe as hp

    monkeypatch.setattr(hp.sys, "platform", "darwin")
    monkeypatch.setattr(hp.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(hp, "_nvidia_gpus", lambda: [])
    monkeypatch.setattr(hp, "_ram_bytes", lambda: 32 * GIB)

    hw = hp.probe_hardware()
    assert hw.unified_memory and hw.is_apple_silicon
    assert hw.vram_bytes == int(32 * GIB * hp.APPLE_USABLE_FRACTION)
    # Пам'ять уніфікована → окремого резерву дисплея немає.
    assert hw.display_reserve_bytes == 0
    assert hw.cuda_context_bytes == 0
