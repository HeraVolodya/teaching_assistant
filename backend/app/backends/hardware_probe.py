"""Проба заліза й арифметика VRAM.

Мета — відповісти на одне питання: «яку найпотужнішу модель дозволяє ця
машина, не впавши в OOM посеред відповіді викладачеві».

Три джерела правди про залізо, кожне з причиною:

  * NVIDIA → `nvidia-smi --query-gpu=name,memory.total,memory.used`.
    Він іде з ДРАЙВЕРОМ, тобто присутній усюди, де є карта, і не потребує
    CUDA toolkit. НІКОЛИ не WMI `Win32_VideoController.AdapterRAM`: це поле
    типу uint32 і воно ОБРІЗАЄТЬСЯ на 4 ГБ — 12-гігабайтна карта звітує 4 ГБ,
    і драбина моделей мовчки обирає найслабшу конфігурацію.
  * Apple Silicon → `sysctl -n hw.memsize`. Пам'ять уніфікована, окремої VRAM
    немає; придатними для моделі вважаємо ~70% (діапазон 65-75%), решта —
    система, WindowServer і сам застосунок.
  * RAM → stdlib (`os.sysconf` на POSIX, `GlobalMemoryStatusEx` через ctypes
    на Windows). psutil у залежностях немає і не потрібен.

Формула розміру (план, §10):  VRAM = W + KV(N) + C + M + O + D
  W — РЕАЛЬНИЙ розмір GGUF на диску, а не оцінка за кількістю параметрів;
  KV(N) — кеш ключів/значень на N токенів контексту;
  C ≈ 0.6 ГіБ буфер обчислень (діапазон 0.5-0.8);
  M ≈ 0.9 ГіБ mmproj — лише для VLM;
  O ≈ 0.4 ГіБ контекст CUDA (нуль на Apple і на CPU);
  D ≈ 1.0 ГіБ резерв дисплея на ПЕРВИННОМУ GPU Windows (діапазон 0.8-1.2).

Для sliding-window моделей (Gemma 3/4):
  KV_swa(N) = 2·n_kv·head_dim·b·[L_global·N + L_local·min(N, W_win)]
b — байтів на елемент: f16 = 2.0, q8_0 ≈ 1.0625, q4_0 ≈ 0.5625.

НІКОЛИ не приймати конфігурацію із запасом < 1.0 ГіБ: Windows забере VRAM під
композитор DWM і застосунок впаде посеред відповіді. Через відкритий баг
LM Studio #1129 (конфлікт SWA і context-shift) SWA-числам не можна довіряти
наосліп — при першому запуску прогнати `lms load --estimate-only` і зберегти
поправку як калібрування машини (`MachineCalibration`).
"""

from __future__ import annotations

import math
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Literal

__all__ = [
    "COMPUTE_BUFFER_GIB",
    "CUDA_CONTEXT_GIB",
    "DISPLAY_RESERVE_GIB",
    "GIB",
    "KV_BYTES_PER_ELEMENT",
    "MIN_HEADROOM_GIB",
    "MMPROJ_GIB",
    "GpuInfo",
    "Hardware",
    "MachineCalibration",
    "VramEstimate",
    "estimate_vram",
    "kv_cache_bytes",
    "plan_gpu_offload",
    "probe_hardware",
]

GIB = 1024 ** 3

# Байтів на один елемент KV-кешу за типом кванта.
KV_BYTES_PER_ELEMENT: dict[str, float] = {
    "f32": 4.0,
    "f16": 2.0,
    "bf16": 2.0,
    "q8_0": 1.0625,   # 32 значення по 8 біт + масштаб fp16 → 34/32 байта
    "q4_0": 0.5625,   # 32 значення по 4 біти + масштаб fp16 → 18/32 байта
}

COMPUTE_BUFFER_GIB = 0.6     # C
CUDA_CONTEXT_GIB = 0.4       # O
DISPLAY_RESERVE_GIB = 1.0    # D — лише первинний GPU під Windows
MMPROJ_GIB = 0.9             # M — лише VLM
MIN_HEADROOM_GIB = 1.0       # незламний мінімум запасу

# Частка уніфікованої пам'яті Apple, яку можна віддати моделі.
APPLE_USABLE_FRACTION = 0.70


@dataclass(frozen=True, slots=True)
class GpuInfo:
    name: str
    total_bytes: int
    used_bytes: int = 0
    vendor: Literal["nvidia", "apple", "amd", "intel", "unknown"] = "unknown"
    index: int = 0
    is_primary_display: bool = True

    @property
    def free_bytes(self) -> int:
        return max(0, self.total_bytes - self.used_bytes)

    @property
    def total_gib(self) -> float:
        return self.total_bytes / GIB


@dataclass(frozen=True, slots=True)
class Hardware:
    platform: str
    machine: str
    cpu_count: int
    ram_bytes: int
    gpus: tuple[GpuInfo, ...] = ()
    is_apple_silicon: bool = False
    unified_memory: bool = False
    detail: str = ""

    @property
    def best_gpu(self) -> GpuInfo | None:
        return max(self.gpus, key=lambda g: g.total_bytes) if self.gpus else None

    @property
    def vram_bytes(self) -> int:
        """Скільки пам'яті ПРИСКОРЮВАЧА взагалі є. 0 → тільки CPU."""
        gpu = self.best_gpu
        return gpu.total_bytes if gpu else 0

    @property
    def display_reserve_bytes(self) -> int:
        """Резерв дисплея D.

        Тільки Windows на первинному GPU: DWM-композитор забирає VRAM у фоні,
        і саме він з'їдає «запас», який на папері виглядав достатнім.
        На Apple пам'ять уніфікована — резерв уже врахований у 70%.
        """
        gpu = self.best_gpu
        if gpu is None or self.unified_memory:
            return 0
        if self.platform == "win32" and gpu.is_primary_display:
            return int(DISPLAY_RESERVE_GIB * GIB)
        if self.platform == "linux" and gpu.is_primary_display:
            return int(0.4 * GIB)
        return 0

    @property
    def cuda_context_bytes(self) -> int:
        gpu = self.best_gpu
        if gpu is None or gpu.vendor != "nvidia":
            return 0
        return int(CUDA_CONTEXT_GIB * GIB)

    def describe(self) -> str:
        gpu = self.best_gpu
        gpu_text = f"{gpu.name}, {gpu.total_gib:.1f} ГіБ" if gpu else "без прискорювача"
        return (f"{self.platform}/{self.machine}, {self.cpu_count} ядер, "
                f"{self.ram_bytes / GIB:.1f} ГіБ RAM, {gpu_text}")


@dataclass(frozen=True, slots=True)
class MachineCalibration:
    """Поправка до нашої арифметики, зміряна на конкретній машині.

    Через баг LM Studio #1129 обчислення KV для SWA-моделей може розходитись
    із реальністю в рази. `lms load --estimate-only` — авторитетне джерело;
    його результат зберігається як множник і застосовується до KV.
    """
    kv_multiplier: float = 1.0
    overhead_bytes: int = 0
    source: str = "default"


@dataclass(frozen=True, slots=True)
class VramEstimate:
    """Розклад VRAM по доданках — саме так його показує екран діагностики."""
    weights_bytes: int
    kv_bytes: int
    compute_bytes: int
    mmproj_bytes: int
    cuda_context_bytes: int
    display_reserve_bytes: int
    total_bytes: int
    n_ctx: int
    kv_dtype: str
    gpu_offload: float = 1.0

    @property
    def model_side_bytes(self) -> int:
        """Усе, що споживає сама модель: W + KV + C + M + O. Без резерву дисплея."""
        return (self.weights_bytes + self.kv_bytes + self.compute_bytes
                + self.mmproj_bytes + self.cuda_context_bytes)

    @property
    def required_bytes(self) -> int:
        """Повна формула §10: W + KV + C + M + O + D."""
        return self.model_side_bytes + self.display_reserve_bytes

    @property
    def headroom_bytes(self) -> int:
        return self.total_bytes - self.required_bytes

    @property
    def headroom_gib(self) -> float:
        return self.headroom_bytes / GIB

    @property
    def fits(self) -> bool:
        """Запас менший за 1.0 ГіБ — це НЕ «майже вміщається», це майбутній OOM."""
        return self.headroom_gib >= MIN_HEADROOM_GIB

    def describe(self) -> str:
        return (f"W {self.weights_bytes / GIB:.2f} + KV {self.kv_bytes / GIB:.2f} + "
                f"C {self.compute_bytes / GIB:.2f} + M {self.mmproj_bytes / GIB:.2f} + "
                f"O {self.cuda_context_bytes / GIB:.2f} + D {self.display_reserve_bytes / GIB:.2f} = "
                f"{self.required_bytes / GIB:.2f} ГіБ, запас {self.headroom_gib:+.2f} ГіБ")


# ------------------------------------------------------------------ KV-кеш
def kv_cache_bytes(
    *,
    n_ctx: int,
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    kv_dtype: str = "f16",
    swa_window: int | None = None,
    global_every: int | None = None,
    gpu_offload: float = 1.0,
) -> int:
    """Розмір KV-кешу в байтах.

    Звичайна модель:  2·n_kv·head_dim·b·L·N
    Sliding-window (Gemma 3/4): локальні шари ніколи не тримають більше за
    вікно, тож KV перестає рости з контекстом майже повністю. Для
    Gemma-3-12B (= MamayLM v2) при 16k це 1.31 ГіБ проти 6.00 ГіБ без SWA —
    різниця в 4.6 раза, і саме вона робить 16k можливими на 12 ГіБ.
    """
    try:
        b = KV_BYTES_PER_ELEMENT[kv_dtype]
    except KeyError:
        raise ValueError(
            f"Невідомий тип KV-кешу {kv_dtype!r}. Відомі: {sorted(KV_BYTES_PER_ELEMENT)}"
        ) from None
    if n_ctx <= 0 or n_layers <= 0:
        return 0

    per_layer_per_token = 2 * n_kv_heads * head_dim * b   # K і V

    if swa_window and global_every and global_every > 1:
        n_global = max(1, n_layers // global_every)
        n_local = n_layers - n_global
        tokens = n_global * n_ctx + n_local * min(n_ctx, swa_window)
    else:
        tokens = n_layers * n_ctx

    total = per_layer_per_token * tokens
    # llama.cpp тримає KV тих шарів, що на GPU; решта — у RAM.
    return int(total * max(0.0, min(1.0, gpu_offload)))


def estimate_vram(
    *,
    weights_bytes: int,
    n_ctx: int,
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    total_vram_bytes: int,
    kv_dtype: str = "f16",
    swa_window: int | None = None,
    global_every: int | None = None,
    gpu_offload: float = 1.0,
    has_mmproj: bool = False,
    cuda_context_bytes: int | None = None,
    display_reserve_bytes: int = 0,
    calibration: MachineCalibration | None = None,
) -> VramEstimate:
    """Формула §10 у зборі. `weights_bytes` — розмір GGUF НА ДИСКУ."""
    cal = calibration or MachineCalibration()
    offload = max(0.0, min(1.0, gpu_offload))

    kv = kv_cache_bytes(
        n_ctx=n_ctx, n_layers=n_layers, n_kv_heads=n_kv_heads, head_dim=head_dim,
        kv_dtype=kv_dtype, swa_window=swa_window, global_every=global_every,
        gpu_offload=offload,
    )
    kv = int(kv * cal.kv_multiplier)

    return VramEstimate(
        weights_bytes=int(weights_bytes * offload),
        kv_bytes=kv,
        compute_bytes=int(COMPUTE_BUFFER_GIB * GIB) + cal.overhead_bytes,
        mmproj_bytes=int(MMPROJ_GIB * GIB) if has_mmproj else 0,
        cuda_context_bytes=(int(CUDA_CONTEXT_GIB * GIB)
                            if cuda_context_bytes is None else cuda_context_bytes),
        display_reserve_bytes=display_reserve_bytes,
        total_bytes=total_vram_bytes,
        n_ctx=n_ctx,
        kv_dtype=kv_dtype,
        gpu_offload=offload,
    )


def plan_gpu_offload(
    *,
    weights_bytes: int,
    n_ctx: int,
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    total_vram_bytes: int,
    kv_dtype: str = "f16",
    swa_window: int | None = None,
    global_every: int | None = None,
    has_mmproj: bool = False,
    cuda_context_bytes: int | None = None,
    display_reserve_bytes: int = 0,
    calibration: MachineCalibration | None = None,
) -> float:
    """Максимальна частка шарів на GPU, за якої запас лишається >= 1.0 ГіБ.

    Крок 0.01 — рівно та гранульованість, яку приймає `--gpu` у LM Studio.
    Повертає 0.0, якщо навіть порожня конфігурація не лишає запасу: тоді
    драбина моделей мусить спускатись на рівень нижче, а не «якось втиснути».
    """
    def head(fraction: float) -> float:
        return estimate_vram(
            weights_bytes=weights_bytes, n_ctx=n_ctx, n_layers=n_layers,
            n_kv_heads=n_kv_heads, head_dim=head_dim, total_vram_bytes=total_vram_bytes,
            kv_dtype=kv_dtype, swa_window=swa_window, global_every=global_every,
            gpu_offload=fraction, has_mmproj=has_mmproj,
            cuda_context_bytes=cuda_context_bytes,
            display_reserve_bytes=display_reserve_bytes, calibration=calibration,
        ).headroom_gib

    if head(1.0) >= MIN_HEADROOM_GIB:
        return 1.0
    if head(0.0) < MIN_HEADROOM_GIB:
        return 0.0
    # Монотонна за побудовою (обидва доданки лінійні за часткою) → бінарний пошук.
    lo, hi = 0.0, 1.0
    for _ in range(12):
        mid = (lo + hi) / 2
        if head(mid) >= MIN_HEADROOM_GIB:
            lo = mid
        else:
            hi = mid
    return math.floor(lo * 100) / 100


# ------------------------------------------------------------------ проба
def _run(cmd: list[str], timeout: float = 5.0) -> str | None:
    """Тонка обгортка над subprocess. Ніколи не кидає — проба заліза не має
    права завалити старт застосунку на екзотичній машині."""
    exe = shutil.which(cmd[0])
    if exe is None:
        return None
    try:
        out = subprocess.run(
            [exe, *cmd[1:]], capture_output=True, text=True, timeout=timeout, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def _nvidia_gpus() -> list[GpuInfo]:
    raw = _run([
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.used",
        "--format=csv,noheader,nounits",
    ])
    if not raw:
        return []
    gpus: list[GpuInfo] = []
    for i, line in enumerate(raw.strip().splitlines()):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            total_mib, used_mib = int(float(parts[1])), int(float(parts[2]))
        except ValueError:
            continue
        gpus.append(GpuInfo(
            name=parts[0],
            total_bytes=total_mib * 1024 * 1024,
            used_bytes=used_mib * 1024 * 1024,
            vendor="nvidia",
            index=i,
            # Первинним вважаємо GPU 0: саме на ньому DWM тримає композитор.
            is_primary_display=(i == 0),
        ))
    return gpus


def _ram_bytes() -> int:
    if sys.platform == "win32":
        try:
            import ctypes

            class _MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = _MemoryStatusEx()
            status.dwLength = ctypes.sizeof(_MemoryStatusEx)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))  # type: ignore[attr-defined]
            return int(status.ullTotalPhys)
        except Exception:
            return 0
    if sys.platform == "darwin":
        raw = _run(["sysctl", "-n", "hw.memsize"])
        if raw and raw.strip().isdigit():
            return int(raw.strip())
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return 0


def probe_hardware() -> Hardware:
    """Єдина точка входу. Ніколи не кидає, ніколи не ходить у мережу."""
    plat = sys.platform
    machine = platform.machine()
    ram = _ram_bytes()
    apple_silicon = plat == "darwin" and machine in ("arm64", "aarch64")

    gpus: list[GpuInfo] = _nvidia_gpus()
    unified = False
    detail = ""

    if not gpus and apple_silicon:
        # Уніфікована пам'ять: окремої VRAM немає, придатних ~65-75%.
        usable = int(ram * APPLE_USABLE_FRACTION)
        gpus = [GpuInfo(
            name=f"Apple {machine} (уніфікована пам'ять)",
            total_bytes=usable, used_bytes=0, vendor="apple", index=0,
            is_primary_display=True,
        )]
        unified = True
        detail = (f"Уніфікована пам'ять: {ram / GIB:.0f} ГіБ усього, "
                  f"{usable / GIB:.0f} ГіБ придатних для моделі.")
    elif not gpus:
        detail = ("Прискорювача не виявлено (nvidia-smi відсутній або не відповів). "
                  "Працюватимемо на CPU.")

    return Hardware(
        platform=plat,
        machine=machine,
        cpu_count=os.cpu_count() or 1,
        ram_bytes=ram,
        gpus=tuple(gpus),
        is_apple_silicon=apple_silicon,
        unified_memory=unified,
        detail=detail,
    )
