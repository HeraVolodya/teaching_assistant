"""Драбина генеративних моделей — ДАНІ, не код.

Чому родина Gemma, а не Qwen, попри те, що Qwen — наш вибір для ембедингів і
реранкінгу. Це РІЗНІ задачі, і доказова база розходиться з інтуїцією:

  * UkrQualBench (25 моделей, ELO за швейцарською системою, серпень 2026)
    міряє мовну природність української — русизми, орфографію, відмінки:
        gemma-4-31b-it            ELO 1582.9, русизмів 0.2 /1k
        MamayLM-Gemma-3-12B-IT    ELO 1503.1, русизмів 1.6 /1k
        gemma-3-27b-it            ELO 1421.8
        Qwen3-32B                 ELO 1236.3, русизмів 3.9 /1k
  * Токенізаційний податок: українська fertility у Gemma — 2.35 токена на
    слово, у Qwen3 — 3.62-3.90. Qwen споживає на ~60% БІЛЬШЕ токенів на той
    самий український документ, тобто «8k контексту» на Qwen вміщає приблизно
    те, що 5k на Gemma.

Разом: Qwen — хибна родина для генератора і правильна для пошуку (там прямі
українські виміри UNLP 2026). Не переплутати.

Поле `arch_verified` — чесність щодо арифметики: для моделей, чиї параметри
архітектури не підтверджені первинним джерелом, KV-оцінка є ОЦІНКОЮ, і при
першому запуску її треба звірити з `lms load --estimate-only` (баг LM Studio
#1129 конфлікту SWA і context-shift може змінити числа в рази).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from app.backends.hardware_probe import (
    GIB,
    Hardware,
    MachineCalibration,
    VramEstimate,
    estimate_vram,
    plan_gpu_offload,
)

__all__ = [
    "APPLE_LADDER",
    "DEFAULT_MODEL_KEY",
    "LADDER",
    "REGISTRY",
    "GenerationPlan",
    "LadderRung",
    "ModelSpec",
    "SamplingDefaults",
    "get",
    "plan_generation",
    "recommend_model",
]


@dataclass(frozen=True, slots=True)
class SamplingDefaults:
    """Параметри проти розбігання малих моделей (план, §10).

    `repeat_penalty` 1.05-1.10 і НІКОЛИ вище: сильний штраф за повтори калічить
    українську словозміну — вона легітимно повторює морфемні токени.
    """
    temperature: float = 0.2
    top_p: float = 0.95
    top_k: int = 64
    repeat_penalty: float = 1.08
    max_tokens: int = 600


@dataclass(frozen=True, slots=True)
class ModelSpec:
    key: str
    title: str
    quant: str
    file_bytes: int                  # W — має бути РЕАЛЬНИМ розміром GGUF на диску
    n_layers: int
    n_kv_heads: int
    head_dim: int
    engine: Literal["llama.cpp", "mlx"] = "llama.cpp"
    gguf_repo: str | None = None
    gguf_file: str | None = None
    mlx_repo: str | None = None
    swa_window: int | None = None    # None → звичайна повна увага
    global_every: int | None = None  # кожен n-й шар — глобальний (решта локальні)
    has_mmproj: bool = False
    max_context: int = 32768
    sampling: SamplingDefaults = SamplingDefaults()
    licence: str = "Gemma Terms of Use"
    availability: Literal["confirmed", "unconfirmed"] = "confirmed"
    arch_verified: bool = True
    notes: str = ""

    @property
    def n_global_layers(self) -> int:
        if not (self.swa_window and self.global_every and self.global_every > 1):
            return self.n_layers
        return max(1, self.n_layers // self.global_every)

    @property
    def n_local_layers(self) -> int:
        return self.n_layers - self.n_global_layers

    def with_measured_file_bytes(self, size_bytes: int) -> ModelSpec:
        """W — реальний розмір файлу. Щойно GGUF на диску, беремо його розмір,
        а не наш табличний прогноз: між ревізіями квантизації різниця сягає
        сотень мегабайтів, а в нас бюджет запасу лише 1 ГіБ."""
        return replace(self, file_bytes=size_bytes)


# ------------------------------------------------------------------- реєстр
REGISTRY: dict[str, ModelSpec] = {
    # ------------------------------------------------------------ CPU-рівень
    "gemma-4-e4b-it-qat": ModelSpec(
        key="gemma-4-e4b-it-qat",
        title="Gemma-4-E4B-it QAT q4_0",
        quant="q4_0",
        file_bytes=int(4.0 * GIB),
        n_layers=35, n_kv_heads=4, head_dim=256,
        gguf_repo="google/gemma-4-E4B-it-qat-GGUF",
        swa_window=512, global_every=5,
        max_context=32768,
        arch_verified=False,
        notes="Єдиний рівень для машин без прискорювача. QAT-квантизація втрачає "
              "менше за посттренувальну. 4-8 ток/с на CPU — повільно, але працює.",
    ),
    # --------------------------------------------------------- наш дефолт 12 ГіБ
    "mamaylm-12b-q4km": ModelSpec(
        key="mamaylm-12b-q4km",
        title="MamayLM-Gemma-3-12B-IT-v2.0 Q4_K_M",
        quant="Q4_K_M",
        file_bytes=int(7.3 * GIB),
        # Підтверджено планом, §10: L=48, n_kv=8, head_dim=256, вікно 1024,
        # 8 глобальних / 40 локальних → 8 КіБ на шар на токен при f16.
        n_layers=48, n_kv_heads=8, head_dim=256,
        swa_window=1024, global_every=6,
        gguf_repo="INSAIT-Institute/MamayLM-Gemma-3-12B-IT-v2.0-GGUF",
        gguf_file="MamayLM-Gemma-3-12B-IT-v2.0.Q4_K_M.gguf",
        max_context=32768,
        notes="Дефолт. Найкраща українська серед моделей, що вміщаються у 12 ГіБ: "
              "ELO 1503.1 при 1.6 русизму на 1000 слів.",
    ),
    # ----------------------------------------------------------------- 16 ГіБ
    "mamaylm-12b-q8": ModelSpec(
        key="mamaylm-12b-q8",
        title="MamayLM-Gemma-3-12B-IT-v2.0 Q8_0",
        quant="Q8_0",
        file_bytes=int(12.5 * GIB),
        n_layers=48, n_kv_heads=8, head_dim=256,
        swa_window=1024, global_every=6,
        gguf_repo="INSAIT-Institute/MamayLM-Gemma-3-12B-IT-v2.0-GGUF",
        gguf_file="MamayLM-Gemma-3-12B-IT-v2.0.Q8_0.gguf",
        max_context=32768,
        notes="Q8 на 12B майже напевно кращий за Q3 на 27B — не гнатись за "
              "кількістю параметрів у низькі кванти.",
    ),
    # ----------------------------------------------------------------- 24 ГіБ
    "mamaylm-27b-q4km": ModelSpec(
        key="mamaylm-27b-q4km",
        title="MamayLM-Gemma-3-27B-v2.0 Q4_K_M",
        quant="Q4_K_M",
        file_bytes=int(16.5 * GIB),
        n_layers=62, n_kv_heads=16, head_dim=128,
        swa_window=1024, global_every=6,
        gguf_repo="INSAIT-Institute/MamayLM-Gemma-3-27B-v2.0-GGUF",
        max_context=32768,
        availability="unconfirmed",   # GGUF-збірку НЕ підтверджено — перевірити перед показом
        arch_verified=False,
        notes="НАЯВНІСТЬ GGUF НЕ ПІДТВЕРДЖЕНО. Показувати лише після успішної "
              "перевірки через /api/v1/models; інакше падати на 16-гігабайтний рівень.",
    ),
    # ---------------------------------------------------------------- 32 ГіБ+
    "gemma-4-31b-it-qat": ModelSpec(
        key="gemma-4-31b-it-qat",
        title="Gemma-4-31B-it QAT q4_0",
        quant="q4_0",
        file_bytes=int(18.5 * GIB),
        n_layers=62, n_kv_heads=16, head_dim=128,
        swa_window=1024, global_every=6,
        gguf_repo="google/gemma-4-31b-it-qat-GGUF",
        max_context=32768,
        arch_verified=False,
        notes="Найвищий виміряний ELO серед відкритих ваг на українській — 1582.9.",
    ),
    # --------------------------------------------------------- Apple Silicon
    "gemma-4-12b-mlx-4bit": ModelSpec(
        key="gemma-4-12b-mlx-4bit",
        title="Gemma-4-12B-it MLX 4-bit",
        quant="4bit",
        file_bytes=int(7.0 * GIB),
        n_layers=48, n_kv_heads=8, head_dim=256,
        engine="mlx",
        # Регістр значущий: канонічний ID — з великою «B». HuggingFace редиректить
        # (307) із малої, а от `lms get` понижує рядок і падає з «artifact does not
        # exist», тож підказка в майстрі першого запуску має бути точною.
        mlx_repo="mlx-community/gemma-4-12B-it-4bit",
        swa_window=1024, global_every=6,
        max_context=32768,
        arch_verified=False,
        notes="ОФІЦІЙНОЇ MLX-збірки MamayLM НЕМАЄ — на Apple Silicon беремо базову "
              "Gemma. Тому реєстр і несе окремі поля gguf_repo та mlx_repo: це не "
              "одна модель у двох форматах, а два різні записи.",
    ),
}

DEFAULT_MODEL_KEY = "mamaylm-12b-q4km"


@dataclass(frozen=True, slots=True)
class LadderRung:
    """Щабель драбини: скільки пам'яті треба і яку конфігурацію тоді брати."""
    min_vram_bytes: int
    model_key: str
    n_ctx: int
    gpu_offload: float = 1.0
    kv_dtype: str = "f16"
    note: str = ""


# Драбина з плану, §10. Порядок — від найпотужнішого до найслабшого.
LADDER: tuple[LadderRung, ...] = (
    LadderRung(int(32 * GIB), "gemma-4-31b-it-qat", 16384, 1.0, "f16",
               "Найвищий український ELO серед відкритих ваг."),
    LadderRung(int(24 * GIB), "mamaylm-27b-q4km", 16384, 1.0, "f16",
               "Наявність GGUF не підтверджена — перевіряти в рантаймі."),
    LadderRung(int(16 * GIB), "mamaylm-12b-q8", 16384, 1.0, "f16",
               "Q8 на 12B кращий за низький квант 27B."),
    LadderRung(int(12 * GIB), "mamaylm-12b-q4km", 16384, 1.0, "f16",
               "ДЕФОЛТ: 9.61 ГіБ споживання, запас +1.39 ГіБ."),
    # 0.78 — значення з плану. Реальна цифра рахується plan_gpu_offload() і буде
    # НИЖЧОЮ, бо наше правило «запас >= 1.0 ГіБ» суворіше за guardrail LM Studio.
    # Беремо мінімум із двох: краще трохи повільніше, ніж OOM посеред відповіді.
    LadderRung(int(8 * GIB), "mamaylm-12b-q4km", 8192, 0.78, "f16",
               "Часткове вивантаження шарів на GPU; решта — на CPU."),
    LadderRung(0, "gemma-4-e4b-it-qat", 8192, 0.0, "f16",
               "Без прискорювача. 4-8 токенів за секунду."),
)

# Apple Silicon: пам'ять уніфікована, «часткового офлоаду» не існує —
# або модель вміщається в придатні ~70% пам'яті, або ні.
APPLE_LADDER: tuple[LadderRung, ...] = (
    LadderRung(int(20 * GIB), "gemma-4-12b-mlx-4bit", 16384, 1.0, "f16",
               "MLX 4-bit, повний контекст."),
    LadderRung(int(10 * GIB), "gemma-4-12b-mlx-4bit", 8192, 1.0, "f16",
               "MLX 4-bit, контекст 8k — цього достатньо для заземленої відповіді."),
    LadderRung(0, "gemma-4-e4b-it-qat", 8192, 1.0, "f16",
               "Замало уніфікованої пам'яті для 12B."),
)


@dataclass(frozen=True, slots=True)
class GenerationPlan:
    """Що саме завантажувати і з якими параметрами."""
    spec: ModelSpec
    n_ctx: int
    gpu_offload: float
    kv_dtype: str
    estimate: VramEstimate
    rung_note: str = ""
    warnings: tuple[str, ...] = ()
    cpu_only: bool = False

    @property
    def model_ref(self) -> str:
        """Ідентифікатор для LM Studio: MLX-репозиторій або GGUF-репозиторій."""
        if self.spec.engine == "mlx" and self.spec.mlx_repo:
            return self.spec.mlx_repo
        return self.spec.gguf_repo or self.spec.key

    def describe(self) -> str:
        return (f"{self.spec.title}, контекст {self.n_ctx}, "
                f"GPU {self.gpu_offload:.0%}. {self.estimate.describe()}")


def get(key: str) -> ModelSpec:
    spec = REGISTRY.get(key)
    if spec is None:
        known = ", ".join(sorted(REGISTRY))
        raise KeyError(f"Невідома генеративна модель {key!r}. Відомі: {known}")
    return spec


def _estimate_for(
    spec: ModelSpec, rung: LadderRung, hw: Hardware, *,
    gpu_offload: float, n_ctx: int, calibration: MachineCalibration | None,
) -> VramEstimate:
    return estimate_vram(
        weights_bytes=spec.file_bytes,
        n_ctx=n_ctx,
        n_layers=spec.n_layers,
        n_kv_heads=spec.n_kv_heads,
        head_dim=spec.head_dim,
        total_vram_bytes=hw.vram_bytes,
        kv_dtype=rung.kv_dtype,
        swa_window=spec.swa_window,
        global_every=spec.global_every,
        gpu_offload=gpu_offload,
        has_mmproj=spec.has_mmproj,
        cuda_context_bytes=hw.cuda_context_bytes,
        display_reserve_bytes=hw.display_reserve_bytes,
        calibration=calibration,
    )


def plan_generation(
    hw: Hardware,
    *,
    calibration: MachineCalibration | None = None,
    allow_unconfirmed: bool = False,
    context_ladder: tuple[int, ...] = (16384, 8192, 4096),
) -> GenerationPlan:
    """Обрати найпотужнішу конфігурацію, що лишає >= 1.0 ГіБ запасу.

    Порядок поступок: спершу нижчий контекст, потім часткове вивантаження
    шарів, і лише тоді — щабель нижче. Саме такий порядок зберігає якість:
    менший контекст коштує історії чату, менший офлоад коштує швидкості, а
    менша модель коштує української мови, і це найдорожча втрата з трьох.
    """
    ladder = APPLE_LADDER if hw.unified_memory else LADDER
    vram = hw.vram_bytes
    warnings: list[str] = []

    for rung in ladder:
        if vram < rung.min_vram_bytes:
            continue
        spec = get(rung.model_key)

        if spec.availability == "unconfirmed" and not allow_unconfirmed:
            warnings.append(
                f"{spec.title}: наявність збірки не підтверджена, пропускаємо щабель."
            )
            continue

        if rung.min_vram_bytes == 0 or vram == 0:
            # CPU-рівень: VRAM ні до чого, обмежує RAM. Оцінку робимо для звіту.
            est = _estimate_for(spec, rung, hw, gpu_offload=0.0, n_ctx=rung.n_ctx,
                                calibration=calibration)
            if hw.ram_bytes and spec.file_bytes + 2 * GIB > hw.ram_bytes:
                warnings.append(
                    f"Оперативної пам'яті {hw.ram_bytes / GIB:.0f} ГіБ може не вистачити "
                    f"для {spec.title} ({spec.file_bytes / GIB:.1f} ГіБ)."
                )
            return GenerationPlan(
                spec=spec, n_ctx=rung.n_ctx, gpu_offload=0.0, kv_dtype=rung.kv_dtype,
                estimate=est, rung_note=rung.note, warnings=tuple(warnings), cpu_only=True,
            )

        contexts = tuple(c for c in (rung.n_ctx, *context_ladder)
                         if c <= min(rung.n_ctx, spec.max_context))
        for n_ctx in dict.fromkeys(contexts):     # унікальні, порядок збережено
            full = _estimate_for(spec, rung, hw, gpu_offload=1.0, n_ctx=n_ctx,
                                 calibration=calibration)
            if full.fits:
                return GenerationPlan(
                    spec=spec, n_ctx=n_ctx, gpu_offload=1.0, kv_dtype=rung.kv_dtype,
                    estimate=full, rung_note=rung.note, warnings=tuple(warnings),
                )
            if hw.unified_memory:
                continue      # часткового офлоаду в MLX немає — лише нижчий контекст
            safe = plan_gpu_offload(
                weights_bytes=spec.file_bytes, n_ctx=n_ctx, n_layers=spec.n_layers,
                n_kv_heads=spec.n_kv_heads, head_dim=spec.head_dim,
                total_vram_bytes=vram, kv_dtype=rung.kv_dtype,
                swa_window=spec.swa_window, global_every=spec.global_every,
                has_mmproj=spec.has_mmproj, cuda_context_bytes=hw.cuda_context_bytes,
                display_reserve_bytes=hw.display_reserve_bytes, calibration=calibration,
            )
            offload = min(rung.gpu_offload, safe) if rung.gpu_offload < 1.0 else safe
            if offload <= 0.0:
                continue
            est = _estimate_for(spec, rung, hw, gpu_offload=offload, n_ctx=n_ctx,
                                calibration=calibration)
            if est.fits:
                if offload < rung.gpu_offload:
                    warnings.append(
                        f"Частку шарів на GPU знижено з {rung.gpu_offload:.0%} до "
                        f"{offload:.0%}, щоб зберегти запас >= 1.0 ГіБ."
                    )
                return GenerationPlan(
                    spec=spec, n_ctx=n_ctx, gpu_offload=offload, kv_dtype=rung.kv_dtype,
                    estimate=est, rung_note=rung.note, warnings=tuple(warnings),
                )

    # Драбина завжди має закінчуватись щаблем 0 → сюди не потрапляємо.
    spec = get("gemma-4-e4b-it-qat")
    est = _estimate_for(spec, LADDER[-1], hw, gpu_offload=0.0, n_ctx=8192,
                        calibration=calibration)
    warnings.append("Жодна конфігурація не пройшла перевірку запасу — лишається CPU.")
    return GenerationPlan(spec=spec, n_ctx=8192, gpu_offload=0.0, kv_dtype="f16",
                          estimate=est, warnings=tuple(warnings), cpu_only=True)


def recommend_model(
    hw: Hardware,
    *,
    calibration: MachineCalibration | None = None,
    allow_unconfirmed: bool = False,
) -> ModelSpec:
    """Публічний контракт модуля: залізо → специфікація моделі.

    Повну конфігурацію (контекст, офлоад, розклад VRAM) дає `plan_generation`.
    """
    return plan_generation(
        hw, calibration=calibration, allow_unconfirmed=allow_unconfirmed
    ).spec
