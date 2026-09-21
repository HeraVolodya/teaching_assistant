"""Спільний контракт LLM-бекендів.

Межа LLM навмисно тонка: усе, що знає решта застосунку, — це чотири методи
(`health`, `list_models`, `load_model`, `chat_stream`) і кілька значень-даних.
Саме завдяки цьому CI без GPU працює: заглушка з `stub_backend.py` підставляється
замість LM Studio, і застосунок різниці не бачить (див. план, «Верифікація»).

Тайм-аути тут — не «на око». Кожен обґрунтований:
  * підключення 2 с — LM Studio або слухає локально, або ні; чекати нічого;
  * завантаження моделі 180 с — 12B із холодного диска це 30-90 с, із кешу
    сторінок 5-15 с; 180 с покриває найгірший випадок із запасом;
  * TTFT 120 с холодна / 30 с тепла — холодний prefill на CPU легітимно довгий;
  * і головне — ТАЙМАУТ ПРОСТОЮ МІЖ ТОКЕНАМИ 60 с, а НЕ глобальний таймаут
    запиту. Заземлена відповідь на 600 токенів при 4 ток/с на CPU триває понад
    дві хвилини цілком легітимно; глобальний таймаут убив би саме ті відповіді,
    заради яких застосунок існує.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

__all__ = [
    "CONNECT_TIMEOUT_S",
    "CONTEXT_LADDER",
    "DEFAULT_STOP",
    "IDLE_BETWEEN_TOKENS_S",
    "LOAD_TIMEOUT_S",
    "TTFT_COLD_S",
    "TTFT_WARM_S",
    "BackendHealth",
    "ChatParams",
    "LlmBackend",
    "LlmCancelled",
    "LlmError",
    "LlmProtocolError",
    "LlmTimeout",
    "LlmUnavailable",
    "LoadConfig",
    "Message",
    "Messages",
    "ModelInfo",
    "ModelLoadRefused",
    "ModelNotLoaded",
    "Role",
    "join_messages",
    "stream_with_idle_timeout",
]

Role = Literal["system", "user", "assistant"]
Message = dict[str, str]
Messages = list[Message]

# ------------------------------------------------------------------ тайм-аути
CONNECT_TIMEOUT_S: float = 2.0
LOAD_TIMEOUT_S: float = 180.0
TTFT_COLD_S: float = 120.0
TTFT_WARM_S: float = 30.0
IDLE_BETWEEN_TOKENS_S: float = 60.0

# Драбина контексту. Спускатись по ній САМЕ на помилці modelLoadingGuardrails:
# оцінка пам'яті LM Studio буває завищена вдвічі (22.9 ГБ проти реальних 11.5),
# а еквівалента кнопки «Load anyway» через REST не існує (lms#499, #1631).
# НІКОЛИ не редагувати конфіг LM Studio за користувача — лише просити менше.
CONTEXT_LADDER: tuple[int, ...] = (32768, 16384, 8192, 4096)

# Стоп-послідовності проти розбігання малих моделей (план, §10).
DEFAULT_STOP: tuple[str, ...] = ("\n\nПитання:", "\n\nДЖЕРЕЛА", "\nUser:", "<end_of_turn>")


# -------------------------------------------------------------------- помилки
class LlmError(RuntimeError):
    """Базова помилка шару LLM. Повідомлення — українською, бо доходять до UI."""


class LlmUnavailable(LlmError):
    """LM Studio не знайдено або він не відповідає."""

    def __init__(self, detail: str = "") -> None:
        super().__init__(
            "LM Studio не відповідає на 127.0.0.1. Запустіть LM Studio і увімкніть "
            "локальний сервер (вкладка «Developer» → «Start Server»)."
            + (f" Деталі: {detail}" if detail else "")
        )


class LlmTimeout(LlmError):
    """Модель мовчала довше за дозволений простій між токенами."""

    def __init__(self, seconds: float, *, stage: str = "генерація") -> None:
        super().__init__(
            f"Модель не надіслала жодного токена за {seconds:.0f} с ({stage}). "
            "Імовірно, її вивантажено або LM Studio перевантажено."
        )
        self.seconds = seconds
        self.stage = stage


class ModelLoadRefused(LlmError):
    """LM Studio відхилив завантаження.

    Причин дві, і плутати їх не можна:
      * `modelLoadingGuardrails` — не вистачає пам'яті. Тут допомагає драбина
        контексту, і лише тут.
      * помилка запиту (`invalid_arguments`, `invalid_request`) — наприклад,
        неправильний ідентифікатор моделі. Драбина контексту не допоможе НІКОЛИ,
        і обіцяти її — брехати користувачеві.

    Спостережено: застосунок відправив коротке гасло `gemma-4-12b-mlx-4bit`
    замість повного шляху репозиторію, LM Studio відповів
    `Invalid model name format`, а повідомлення бадьоро повідомило
    «Спробуємо менший контекст» і пішло по колу.
    """

    def __init__(self, model: str, context_length: int, detail: str = "") -> None:
        self.model = model
        self.context_length = context_length
        self.detail = detail
        self.retryable_with_smaller_context = self._is_memory_refusal(detail)

        if self.retryable_with_smaller_context:
            head = (
                f"LM Studio відмовився завантажити «{model}» із контекстом "
                f"{context_length}: не вистачає пам'яті. Спробуємо менший контекст."
            )
        elif context_length:
            head = (
                f"LM Studio відхилив запит на завантаження «{model}» "
                f"із контекстом {context_length}. Це помилка запиту, а не браку "
                f"пам'яті — менший контекст не допоможе."
            )
        else:
            head = (
                f"LM Studio відхилив запит щодо моделі «{model}». Найімовірніше, "
                f"ідентифікатор моделі неправильний: потрібен повний шлях "
                f"репозиторію, наприклад «mlx-community/gemma-4-12b-it-4bit»."
            )
        super().__init__(f"{head} Деталі: {detail or 'без деталей'}")

    @staticmethod
    def _is_memory_refusal(detail: str) -> bool:
        """Чи це справді брак пам'яті, а не зіпсований запит.

        ПОРЯДОК ПЕРЕВІРОК ТУТ КРИТИЧНИЙ. LM Studio віддає `invalid_request_error`
        як ЗАГАЛЬНИЙ тип для будь-якої відмови 400 — включно з відмовою
        гардрейла через пам'ять. Спостережено на живій машині: тіло містило
        одночасно `"type": "invalid_request_error"` і
        `"Model loading was stopped due to insufficient system resources"`.
        Перевірка «спершу заперечні маркери» бачила `invalid_request`, вважала
        це зіпсованим запитом і показувала «ідентифікатор моделі неправильний»
        на цілком правильному ідентифікаторі — тобто рівно та сама хиба
        діагностики, яку цей клас мав виправити, лише з іншим знаком.

        Тому маркери пам'яті мають ПРІОРИТЕТ: вони конкретні, а
        `invalid_request_error` не несе інформації взагалі.
        """
        lowered = (detail or "").lower()
        memory = ("insufficient system resources", "insufficient memory",
                  "not enough memory", "guardrail", "would likely overload",
                  "out of memory", "insufficient resources")
        if any(m in lowered for m in memory):
            return True
        # Лише тепер — ознаки зіпсованого запиту. `invalid_request_error` сюда
        # НЕ входить: він стоїть у кожній відмові й нічого не розрізняє.
        bad_request = ("invalid_arguments", "invalid model name", "unrecognized_keys",
                       "model not found", "unknown model", "no such model")
        # Якщо це не зіпсований запит — причина невідома, і тоді пробуємо менший
        # контекст: одна зайва спроба дешевша за хибну впевненість у діагнозі.
        return not any(m in lowered for m in bad_request)


class NoChatModelReady(LlmError):
    """У LM Studio немає ЖОДНОЇ придатної мовної моделі в пам'яті.

    Відрізняється від `LlmUnavailable` (сервер не відповідає) і від
    `ModelNotLoaded` (конкретна модель зникла посеред роботи). Це третій,
    найчастіший на першому запуску стан: сервер працює, моделі завантажені
    на диск, але жодна не піднята в пам'ять — `loaded_instances` порожній
    у кожної.

    Текст навмисно веде до дії, а не описує внутрішній стан: викладач має
    зрозуміти, що натиснути, і не мусить знати слова «інстанс».
    """

    def __init__(self, downloaded: tuple[str, ...] = (), detail: str = "") -> None:
        if downloaded:
            names = ", ".join(f"«{d}»" for d in downloaded[:4])
            tail = f" Доступні на диску: {names}." if names else ""
            hint = (
                "Оберіть модель у LM Studio (вкладка «Developer» → «Select a model to load») "
                f"або виконайте `lms load <модель>`.{tail}"
            )
        else:
            hint = (
                "У LM Studio немає жодної завантаженої мовної моделі. Завантажте її на "
                "вкладці «Discover» — для української рекомендовано "
                "MamayLM-Gemma-3-12B-IT-v2.0, Q4_K_M."
            )
        super().__init__(f"Мовна модель не піднята в пам'ять. {hint}"
                         + (f" Деталі: {detail}" if detail else ""))
        self.downloaded = downloaded


class ModelNotLoaded(LlmError):
    """Модель зникла з пам'яті посеред роботи (Auto-Evict, ручне вивантаження)."""

    def __init__(self, model: str) -> None:
        super().__init__(
            f"Модель «{model}» більше не завантажена в LM Studio. "
            "Повторюємо завантаження з тією самою конфігурацією."
        )
        self.model = model


class LlmCancelled(LlmError):
    """Користувач перервав відповідь."""

    def __init__(self) -> None:
        super().__init__("Генерацію відповіді скасовано.")


class LlmProtocolError(LlmError):
    """Відповідь не схожа на OpenAI-сумісну — інша версія або інший сервер."""


# --------------------------------------------------------------------- дані
@dataclass(frozen=True, slots=True)
class ChatParams:
    """Параметри генерації. Дефолти — з плану, §10.

    `repeat_penalty` НІКОЛИ не піднімати вище 1.10: високий штраф за повтори
    калічить українську словозміну, бо українська легітимно повторює морфемні
    токени («артилерійського дивізіону артилерійської бригади»).
    """
    temperature: float = 0.2
    max_tokens: int = 600
    top_p: float = 0.95
    top_k: int = 64
    repeat_penalty: float = 1.08
    stop: tuple[str, ...] = DEFAULT_STOP
    seed: int | None = None
    # НІКОЛИ truncateMiddle / rollingWindow: вони ТИХО видаляють ваші докази.
    # Потрібна жорстка помилка, яку можна перехопити і перейти в map-reduce.
    context_overflow_policy: Literal["stopAtLimit"] = "stopAtLimit"

    def __post_init__(self) -> None:
        if self.repeat_penalty > 1.10:
            raise ValueError(
                f"repeat_penalty={self.repeat_penalty} завеликий. Стеля — 1.10: "
                "сильніший штраф калічить українську словозміну."
            )

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "stop": list(self.stop),
            # LM Studio приймає обидва написання; llama-server розуміє repeat_penalty.
            "repeat_penalty": self.repeat_penalty,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
        }
        if self.seed is not None:
            payload["seed"] = self.seed
        return payload


@dataclass(frozen=True, slots=True)
class LoadConfig:
    """Точна конфігурація завантаження.

    Зберігається як є й перевидається БАЙТ-У-БАЙТ, якщо модель вивантажили
    посеред відповіді. Відтворювати «приблизно ту саму» конфігурацію — це
    тихо змінити довжину контексту й отримати іншу поведінку на пів сесії.

    УВАГА, per-engine whitelist: `eval_batch_size`, `flash_attention`,
    `offload_kv_cache_to_gpu` — параметри ЛИШЕ для llama.cpp. MLX-рушій
    (macOS) відхиляє їх помилкою. Тому payload формується від рушія.
    """
    model: str
    context_length: int
    engine: Literal["llama.cpp", "mlx", "unknown"] = "llama.cpp"
    eval_batch_size: int = 512
    flash_attention: bool = True
    offload_kv_cache_to_gpu: bool = True
    gpu_offload: float | None = None      # 0.0..1.0, частка шарів на GPU
    ttl_seconds: int | None = None
    echo_load_config: bool = True

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "context_length": self.context_length,
            "echo_load_config": self.echo_load_config,
        }
        if self.ttl_seconds is not None:
            payload["ttl"] = self.ttl_seconds
        if self.engine == "mlx":
            # MLX не має ні eval_batch_size, ні flash_attention, ні окремого
            # KV-офлоаду: пам'ять уніфікована. Надсилання їх дає 400.
            return payload
        payload.update(
            eval_batch_size=self.eval_batch_size,
            flash_attention=self.flash_attention,
            offload_kv_cache_to_gpu=self.offload_kv_cache_to_gpu,
        )
        if self.gpu_offload is not None:
            payload["gpu_offload"] = round(self.gpu_offload, 2)
        return payload

    def with_context(self, context_length: int) -> LoadConfig:
        from dataclasses import replace
        return replace(self, context_length=context_length)


# Хвости, якими постачальники позначають ФОРМАТ і КВАНТИЗАЦІЮ, а не саму модель.
# `gemma-4-12b-it` у LM Studio і `mlx-community/gemma-4-12B-it-4bit` у реєстрі —
# це один файл на диску під двома різними іменами.
_FORMAT_SUFFIXES = (
    "-gguf", "-mlx", "-onnx", "-qat",
    "-2bit", "-3bit", "-4bit", "-5bit", "-6bit", "-8bit",
)


def normalize_model_key(key: str) -> str:
    """Ключ моделі без префікса організації, регістру і хвоста квантизації."""
    base = key.rsplit("/", 1)[-1].strip().lower()
    changed = True
    while changed:
        changed = False
        for suffix in _FORMAT_SUFFIXES:
            if base.endswith(suffix) and len(base) > len(suffix):
                base = base[: -len(suffix)]
                changed = True
    return base


@dataclass(frozen=True, slots=True)
class ModelInfo:
    key: str
    display_name: str = ""
    kind: Literal["llm", "vlm", "embedding", "unknown"] = "unknown"
    engine: str = ""
    loaded: bool = False
    max_context: int | None = None
    loaded_context: int | None = None
    quantization: str | None = None
    size_bytes: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class BackendHealth:
    ok: bool
    base_url: str | None = None
    # v1 — єдина поверхня, що повідомляє стан завантаження, максимальний
    # контекст і квантизацію. v0 і openai — деградовані фолбеки.
    api_version: Literal["v1", "v0", "openai", "stub", "none"] = "none"
    models: tuple[ModelInfo, ...] = ()
    detail: str = ""

    @property
    def loaded_models(self) -> tuple[ModelInfo, ...]:
        return tuple(m for m in self.models if m.loaded)

    @property
    def chat_ready(self) -> bool:
        """Чи можна ПРЯМО ЗАРАЗ згенерувати відповідь.

        `ok` означає лише «сервер відповідає». Цього недостатньо, і саме через
        цю різницю інтерфейс показував зелений стан, тоді як кожен запит падав
        із «No models loaded»: список моделей — це те, що завантажено НА ДИСК,
        а не те, що піднято В ПАМ'ЯТЬ.

        Єдиний кандидат на диску теж вважається готовим: LM Studio піднімає
        його через JIT, і це справжній ідентифікатор, а не вигадка. Неоднозначні
        випадки (кілька моделей, жодна не піднята) готовими НЕ є — вибір має
        зробити викладач, а не ми за нього.
        """
        if not self.ok:
            return False
        if self.api_version == "stub":
            return True
        chat = [m for m in self.models if m.kind in ("llm", "vlm", "unknown") and m.key]
        if any(m.loaded for m in chat):
            return True
        if self.api_version != "v1":
            # v0 і openai-поверхня не повідомляють стан завантаження взагалі,
            # тож «не знаю» тлумачимо як «спробуємо»: краще зрозуміла помилка
            # від самого сервера, ніж наша хибна відмова.
            return bool(chat)
        return len(chat) == 1

    @property
    def readiness_detail(self) -> str:
        """Чому саме не готово — текстом, який можна показати викладачеві."""
        if not self.ok:
            return "LM Studio не відповідає."
        if self.chat_ready:
            return ""
        chat = [m for m in self.models if m.kind in ("llm", "vlm", "unknown") and m.key]
        if not chat:
            return ("У LM Studio немає жодної мовної моделі. Завантажте її на вкладці "
                    "«Discover» — для української рекомендовано MamayLM-Gemma-3-12B-IT-v2.0, Q4_K_M.")
        names = ", ".join(f"«{m.key}»" for m in chat[:4])
        return (f"Мовна модель не піднята в пам'ять. Оберіть одну з наявних ({names}) "
                f"у LM Studio → «Developer», або виконайте `lms load <модель>`.")

    def find(self, key: str) -> ModelInfo | None:
        """Спершу точний збіг, далі — нормалізований.

        LM Studio показує модель під ВЛАСНИМ коротким ключем (`gemma-4-12b-it`),
        а реєстр несе ідентифікатор репозиторію
        (`mlx-community/gemma-4-12B-it-4bit`). Це той самий файл на диску, але
        два різні рядки, тож точний збіг для рекомендованої драбиною моделі не
        спрацьовував НІКОЛИ: застосунок бачив модель у списку й водночас
        вважав, що її немає.
        """
        for m in self.models:
            if m.key == key:
                return m
        target = normalize_model_key(key)
        if not target:
            return None
        for m in self.models:
            if normalize_model_key(m.key) == target:
                return m
        return None


# ------------------------------------------------------------------ протокол
@runtime_checkable
class LlmBackend(Protocol):
    """Усе, що застосунок знає про LLM.

    `chat_stream` — асинхронний генератор рядків-дельт (НЕ корутина, що
    повертає ітератор): це дозволяє закрити HTTP-стрім простим виходом із
    циклу, чим і реалізується скасування.
    """

    name: str

    async def health(self) -> BackendHealth: ...

    async def list_models(self) -> list[ModelInfo]: ...

    async def load_model(self, cfg: LoadConfig) -> ModelInfo: ...

    def chat_stream(
        self,
        messages: Messages,
        *,
        params: ChatParams | None = None,
        model: str | None = None,
        cancel: asyncio.Event | None = None,
        warm: bool = True,
    ) -> AsyncIterator[str]: ...


# ------------------------------------------------------------- допоміжне
async def stream_with_idle_timeout(
    source: AsyncIterator[str],
    *,
    first_timeout: float,
    idle_timeout: float,
) -> AsyncIterator[str]:
    """Обгортка, що реалізує ТАЙМАУТ ПРОСТОЮ, а не глобальний таймаут.

    Перший елемент чекаємо `first_timeout` (TTFT — холодний prefill буває
    хвилинним), кожен наступний — `idle_timeout`. Сумарна тривалість не
    обмежена: довга заземлена відповідь — це нормальна робота, а не збій.
    """
    it = source.__aiter__()
    deadline = first_timeout
    stage = "перший токен"
    while True:
        try:
            item = await asyncio.wait_for(it.__anext__(), timeout=deadline)
        except StopAsyncIteration:
            return
        except TimeoutError as exc:
            raise LlmTimeout(deadline, stage=stage) from exc
        yield item
        deadline = idle_timeout
        stage = "простій між токенами"


def join_messages(messages: Sequence[Message]) -> str:
    """Плоский текст промпту — для заглушки, логів і оцінки бюджету токенів."""
    return "\n\n".join(f"{m.get('role', '')}: {m.get('content', '')}" for m in messages)
