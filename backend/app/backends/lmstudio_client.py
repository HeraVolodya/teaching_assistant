"""Клієнт LM Studio 0.4.x — сирий HTTP, без SDK.

ЧОМУ НЕ `lmstudio-python`: SDK прив'язує застосунок до внутрішнього
WebSocket-протоколу й до мажорної версії LM Studio. Гарячий шлях на сирому
HTTP проти OpenAI-сумісної поверхні заодно робить запасний варіант майже
безкоштовним: `llama-server` із llama.cpp говорить БАЙТ-У-БАЙТ тим самим
діалектом, тож підміна бекенда — це зміна base_url, а не переписування коду.

Розподіл поверхонь (план, §10):
  * генерація       → POST /v1/chat/completions, stream: true (OpenAI-сумісна)
  * виявлення/стан  → GET  /api/v1/models  — ЛИШЕ v1 повідомляє стан
                       завантаження, максимальний контекст і квантизацію;
                       фолбеки /api/v0/models і /v1/models деградовані
  * завантаження    → POST /api/v1/models/load — єдина REST-поверхня з
                       context_length
  * встановлення    → POST /api/v1/models/download + GET .../download/status —
                       найбільший UX-виграш v1: викладач ніколи не відкриває
                       вкладку Discover

ЧОТИРИ ПАСТКИ, НАВКОЛО ЯКИХ СПРОЄКТОВАНО ЦЕЙ ФАЙЛ:

1. `modelLoadingGuardrails` відхиляє завантаження без еквівалента «Load
   anyway» через REST (lms#499, #1631), і його оцінка буває завищена вдвічі
   (22.9 ГБ проти реальних 11.5). Відповідь — драбина контексту
   32k → 16k → 8k → 4k САМЕ на цій помилці. Конфіг LM Studio за користувача
   НІКОЛИ не редагувати: це його програма, не наша.
2. KV-кеш скидається після простою (#1861) — тому тайм-аут TTFT для «холодного»
   запиту вчетверо більший за «теплий».
3. Auto-Evict тримає в пам'яті одну JIT-завантажену модель, тож будь-який
   сторонній запит може вивантажити нашу LLM посеред відповіді. Точна
   конфігурація завантаження зберігається і перевидається БАЙТ-У-БАЙТ, спроба
   рівно одна — інакше цикл падінь замаскує справжню причину.
4. Скасування стріму (#1203) може не зупинити генерацію одразу. Тому id запиту
   позначається мертвим, і всі пізні токени ігноруються на нашому боці.

`logprobs` У LM STUDIO НЕ РЕАЛІЗОВАНО: параметри приймаються, а
`choices[0].logprobs` завжди NULL. Наслідок — перплексійна оцінка впевненості
й детекція галюцинацій за ентропією НЕДОСТУПНІ. Заміна: скор реранкера плюс
механічна перевірка цитат (`app/generation/citations.py`). Не витрачати час
на спроби це обійти.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import AsyncIterator, Callable, Sequence
from pathlib import Path
from typing import Any

import httpx

from app.backends.base import (
    CONNECT_TIMEOUT_S,
    CONTEXT_LADDER,
    IDLE_BETWEEN_TOKENS_S,
    LOAD_TIMEOUT_S,
    TTFT_COLD_S,
    TTFT_WARM_S,
    BackendHealth,
    ChatParams,
    LlmError,
    LlmProtocolError,
    LlmTimeout,
    LlmUnavailable,
    LoadConfig,
    Messages,
    ModelInfo,
    ModelLoadRefused,
    ModelNotLoaded,
    NoChatModelReady,
    stream_with_idle_timeout,
)

__all__ = [
    "DEFAULT_PORTS",
    "LmStudioClient",
    "candidate_base_urls",
    "lmstudio_config_path",
    "read_configured_port",
]

log = logging.getLogger("asistent.lmstudio")

# Порти-кандидати після конфігу: 1234 — дефолт LM Studio, 1235 — типовий
# другий інстанс.
DEFAULT_PORTS: tuple[int, ...] = (1234, 1235)

# Маркери в тілі помилки, що означають «guardrail відхилив», а не «зламався».
_GUARDRAIL_MARKERS = ("guardrail", "modelloadingguardrails", "not enough memory",
                      "insufficient memory", "insufficient system resources")
# Маркери «моделі більше немає в пам'яті».
_UNLOADED_MARKERS = ("model not loaded", "no model loaded", "model_not_found",
                     "is not loaded", "no models loaded")


def lmstudio_config_path() -> Path:
    """~/.lmstudio/.internal/http-server-config.json.

    На Windows Path.home() резолвиться в %USERPROFILE%, тому окремої гілки
    не потрібно — але шлях свідомо саме такий, а не «десь у ProgramData».
    """
    return Path.home() / ".lmstudio" / ".internal" / "http-server-config.json"


def read_configured_port(path: Path | None = None) -> int | None:
    """Порт із конфігу LM Studio. Файл читаємо, НІКОЛИ не пишемо."""
    p = path or lmstudio_config_path()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    port = data.get("port") if isinstance(data, dict) else None
    if isinstance(port, int) and 1 <= port <= 65535:
        return port
    return None


def candidate_base_urls(config_path: Path | None = None) -> list[str]:
    """Кандидати в порядку спадання довіри. Тільки 127.0.0.1: мережевий гард
    усе одно заблокує будь-що інше, і це навмисно."""
    ports: list[int] = []
    env_port = os.environ.get("ASISTENT_LMSTUDIO_PORT")
    if env_port and env_port.isdigit():
        ports.append(int(env_port))
    configured = read_configured_port(config_path)
    if configured:
        ports.append(configured)
    ports.extend(DEFAULT_PORTS)
    seen: list[int] = []
    for p in ports:
        if p not in seen:
            seen.append(p)
    return [f"http://127.0.0.1:{p}" for p in seen]


def _parse_model(raw: dict[str, Any]) -> ModelInfo:
    """Нормалізація трьох різних форм відповіді в один тип.

    v1 несе state/max_context/quantization; v0 — частину; /v1/models —
    лише id. Тому `loaded` для OpenAI-поверхні лишається False: краще
    «не знаю», ніж вигадане «завантажено».
    """
    key = str(raw.get("model_key") or raw.get("id") or raw.get("key") or "")
    state = str(raw.get("state") or raw.get("status") or "")
    kind = str(raw.get("type") or raw.get("model_type") or "unknown").lower()
    if kind not in ("llm", "vlm", "embedding"):
        kind = "vlm" if raw.get("vision") else ("llm" if state or key else "unknown")
    ctx = raw.get("max_context_length") or raw.get("max_context") or raw.get("context_length")
    loaded_ctx = raw.get("loaded_context_length") or raw.get("context_length")

    # `loaded_instances` — ЄДИНЕ поле, яким LM Studio 0.4.x повідомляє, що модель
    # справді в пам'яті. Поля `state` у цій відповіді НЕМА ЗОВСІМ (перевірено на
    # живому сервері), тож `loaded` завжди виходило False. Наслідок був гірший за
    # просто неточність: перевірка здоров'я бачила «сервер відповідає, дві моделі
    # у списку» й показувала зелений стан, а КОЖЕН запит генерації падав із
    # «No models loaded». Список моделей — це те, що ЗАВАНТАЖЕНО НА ДИСК, а не
    # те, що піднято в пам'ять.
    instances = raw.get("loaded_instances")
    if isinstance(instances, list) and instances:
        loaded = True
        first = instances[0] if isinstance(instances[0], dict) else {}
        loaded_ctx = first.get("context_length") or loaded_ctx
    else:
        loaded = state.lower() in ("loaded", "loading") or bool(raw.get("is_loaded"))
        if isinstance(instances, list):
            loaded_ctx = None   # список є і він порожній → у пам'яті нічого немає

    # `quantization` у v1 — це ОБ'ЄКТ {"name": "MXFP4", "bits_per_weight": 4},
    # тож str() давав `{'name': 'MXFP4', ...}` у полі, яке показується користувачу.
    quant = raw.get("quantization")
    if isinstance(quant, dict):
        quant_name = str(quant.get("name") or "") or None
    else:
        quant_name = str(quant) if quant else None

    # `capabilities.reasoning.allowed_options` — лише v1 (LM Studio ≥ 0.4.8). Саме
    # за ним вирішується, чи можна попросити модель не міркувати: моделі без цієї
    # можливості LM Studio відповідає помилкою «does not support reasoning
    # configuration», тож слати параметр навмання не можна.
    caps = raw.get("capabilities")
    reasoning = caps.get("reasoning") if isinstance(caps, dict) else None
    options = reasoning.get("allowed_options") if isinstance(reasoning, dict) else None
    reasoning_options = tuple(str(o) for o in options) if isinstance(options, list) else ()

    return ModelInfo(
        key=key,
        display_name=str(raw.get("display_name") or raw.get("name") or key),
        kind=kind,                                     # type: ignore[arg-type]
        # `format` (mlx|gguf) — саме те, що визначає рушій, і саме за ним
        # вирішується, чи слати llama.cpp-параметри і чи має сенс драбина контексту.
        engine=str(raw.get("engine") or raw.get("runtime") or raw.get("format") or ""),
        loaded=loaded,
        max_context=int(ctx) if isinstance(ctx, (int, float)) else None,
        loaded_context=int(loaded_ctx) if isinstance(loaded_ctx, (int, float)) else None,
        quantization=quant_name,
        size_bytes=(int(raw["size_bytes"]) if isinstance(raw.get("size_bytes"), int) else None),
        reasoning_options=reasoning_options,
        raw=raw,
    )


def _models_from_payload(payload: Any) -> list[ModelInfo]:
    if isinstance(payload, dict):
        items = payload.get("data") or payload.get("models") or []
    elif isinstance(payload, list):
        items = payload
    else:
        items = []
    return [_parse_model(i) for i in items if isinstance(i, dict)]


class LmStudioClient:
    """Бекенд LM Studio. Реалізує протокол `LlmBackend`."""

    name = "lmstudio"

    def __init__(
        self,
        base_url: str | None = None,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        idle_timeout_s: float = IDLE_BETWEEN_TOKENS_S,
        ttft_cold_s: float = TTFT_COLD_S,
        ttft_warm_s: float = TTFT_WARM_S,
        load_timeout_s: float = LOAD_TIMEOUT_S,
        connect_timeout_s: float = CONNECT_TIMEOUT_S,
    ) -> None:
        self.base_url = (base_url or candidate_base_urls()[0]).rstrip("/")
        self._transport = transport
        self.idle_timeout_s = idle_timeout_s
        self.ttft_cold_s = ttft_cold_s
        self.ttft_warm_s = ttft_warm_s
        self.load_timeout_s = load_timeout_s
        self.connect_timeout_s = connect_timeout_s
        self._client: httpx.AsyncClient | None = None
        # Точна конфігурація, застосована при старті. Перевидається БАЙТ-У-БАЙТ.
        self._last_load_config: LoadConfig | None = None
        self.api_version: str = "none"
        # Баг #1203: стрім може не зупинитись одразу після закриття. Пізні
        # токени мертвих запитів мовчки відкидаються.
        self._dead_requests: set[str] = set()
        # Останній список моделей (оновлює кожен `health()` — UI опитує його
        # постійно): звідси клієнт знає можливості моделі, не роблячи зайвого запиту.
        self._models: dict[str, ModelInfo] = {}
        # Моделі, що відхилили `reasoning_effort` попри оголошену можливість:
        # більше його їм не шлемо, щоб не платити зайвим запитом на кожне питання.
        self._no_reasoning_cfg: set[str] = set()

    # ------------------------------------------------------------ службове
    def _http(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                transport=self._transport,
                timeout=httpx.Timeout(
                    connect=self.connect_timeout_s,
                    # read — це тайм-аут МІЖ порціями, а не на весь запит.
                    read=max(self.ttft_cold_s, self.idle_timeout_s) + 5.0,
                    write=30.0,
                    pool=self.connect_timeout_s,
                ),
                headers={"Content-Type": "application/json"},
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None

    async def __aenter__(self) -> LmStudioClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    # ---------------------------------------------------------- виявлення
    @classmethod
    async def discover(
        cls,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        config_path: Path | None = None,
        **kwargs: Any,
    ) -> LmStudioClient | None:
        """Знайти живий LM Studio. None → не запущений."""
        for url in candidate_base_urls(config_path):
            client = cls(url, transport=transport, **kwargs)
            health = await client.health()
            if health.ok:
                return client
            await client.aclose()
        return None

    async def health(self) -> BackendHealth:
        """Три поверхні по спаданню інформативності.

        `/api/v1/models` перша не тому, що новіша, а тому, що ЛИШЕ вона каже,
        чи модель уже в пам'яті. Без цього неможливо відрізнити «холодний
        старт на 90 с» від «зависло».
        """
        for path, version in (("/api/v1/models", "v1"),
                              ("/api/v0/models", "v0"),
                              ("/v1/models", "openai")):
            try:
                resp = await self._http().get(path, timeout=httpx.Timeout(
                    connect=self.connect_timeout_s, read=5.0, write=5.0, pool=5.0))
            except httpx.HTTPError:
                continue
            if resp.status_code != 200:
                continue
            try:
                models = _models_from_payload(resp.json())
            except ValueError:
                continue
            self.api_version = version
            self._models = {m.key: m for m in models if m.key}
            return BackendHealth(
                ok=True, base_url=self.base_url, api_version=version,  # type: ignore[arg-type]
                models=tuple(models),
                detail="" if version == "v1" else
                       f"Поверхня {version}: стан завантаження моделей недоступний.",
            )
        return BackendHealth(ok=False, base_url=self.base_url, api_version="none",
                             detail="Жодна з поверхонь /api/v1, /api/v0, /v1 не відповіла.")

    async def list_models(self) -> list[ModelInfo]:
        health = await self.health()
        if not health.ok:
            raise LlmUnavailable(health.detail)
        return list(health.models)

    # -------------------------------------------------------- завантаження
    async def load_model(self, cfg: LoadConfig) -> ModelInfo:
        """Одна спроба з ТОЧНО заданою конфігурацією.

        `echo_load_config: true` — щоб у лог потрапило те, що реально
        застосувалось, а не те, що ми просили: LM Studio мовчки коригує
        частину параметрів.
        """
        try:
            resp = await self._http().post(
                "/api/v1/models/load", json=cfg.to_payload(),
                timeout=httpx.Timeout(connect=self.connect_timeout_s,
                                      read=self.load_timeout_s,
                                      write=30.0, pool=self.connect_timeout_s),
            )
        except httpx.HTTPError as exc:
            raise LlmUnavailable(str(exc)) from exc

        if resp.status_code >= 400:
            body = resp.text or ""
            if any(m in body.lower() for m in _GUARDRAIL_MARKERS):
                raise ModelLoadRefused(cfg.model, cfg.context_length, body[:400])
            raise ModelLoadRefused(cfg.model, cfg.context_length,
                                   f"HTTP {resp.status_code}: {body[:400]}")

        self._last_load_config = cfg
        try:
            payload = resp.json()
        except ValueError:
            payload = {}
        raw = payload.get("model") if isinstance(payload, dict) else None
        info = _parse_model(raw if isinstance(raw, dict) else {"model_key": cfg.model})
        return info

    async def load_with_context_ladder(
        self,
        cfg: LoadConfig,
        *,
        ladder: Sequence[int] = CONTEXT_LADDER,
        on_step: Callable[[int, str], None] | None = None,
    ) -> tuple[ModelInfo, LoadConfig]:
        """Драбина 32k → 16k → 8k → 4k САМЕ на браку пам'яті.

        Починаємо з бажаного контексту, далі йдемо лише ВНИЗ по драбині.
        Будь-яка інша помилка (немає такої моделі, зіпсований ідентифікатор,
        рушій не запущено) прокидається одразу: спускати контекст на ній
        безглуздо і лише приховає причину. Раніше код ловив УСІ відмови
        однаково й радісно повідомляв «Спробуємо менший контекст» у відповідь
        на `Invalid model name format`, роблячи чотири марні спроби.

        ВАЖЛИВО ПРО MLX: на Apple Silicon драбина безсила. Перевірено на живій
        машині — `lms load --estimate-only` для MLX-моделі дає ту саму оцінку
        (15.78 ГіБ) і при 32768, і при 8192, і при 4096: оцінювач MLX ігнорує
        довжину контексту. Єдиний вихід там — менша модель, тож не марнуємо
        хвилини на спуск.
        """
        steps = [cfg.context_length]
        if getattr(cfg, "engine", None) != "mlx":
            steps.extend(c for c in ladder if c < cfg.context_length)
        last: ModelLoadRefused | None = None
        for ctx in steps:
            attempt = cfg.with_context(ctx)
            try:
                info = await self.load_model(attempt)
            except ModelLoadRefused as exc:
                last = exc
                if not exc.retryable_with_smaller_context:
                    raise  # не брак пам'яті — спуск не допоможе
                if on_step is not None:
                    on_step(ctx, str(exc))
                continue
            return info, attempt
        raise last or ModelLoadRefused(cfg.model, cfg.context_length,
                                       "Драбина контексту вичерпана.")

    async def unload_model(self, model_key: str) -> bool:
        """Явне вивантаження. Потрібне для перемикання LLM ↔ VLM на 8-12 ГіБ:
        вони фізично не співіснують, і покладатись на Auto-Evict — це отримати
        OOM у випадковий момент замість керованої паузи."""
        try:
            resp = await self._http().post("/api/v1/models/unload",
                                           json={"model": model_key})
        except httpx.HTTPError:
            return False
        return resp.status_code < 400

    # ------------------------------------------------------- завантаження з мережі
    async def download_model(self, model_key: str) -> str:
        """POST /api/v1/models/download → id завантаження.

        УВАГА: у закритому контурі цей шлях недоступний і має падати голосно —
        мережевий гард заблокує вихід LM Studio? Ні: у мережу йде LM Studio,
        а не ми. Тому рішення про завантаження ухвалює ВИКЛАДАЧ явною дією в
        майстрі першого запуску, а в offline-постачанні кнопка прихована.
        """
        try:
            resp = await self._http().post("/api/v1/models/download",
                                           json={"model": model_key})
        except httpx.HTTPError as exc:
            raise LlmUnavailable(str(exc)) from exc
        if resp.status_code >= 400:
            raise ModelLoadRefused(model_key, 0, resp.text[:400])
        data = resp.json()
        did = data.get("download_id") or data.get("id") if isinstance(data, dict) else None
        if not did:
            raise LlmProtocolError(
                "LM Studio не повернув ідентифікатор завантаження — імовірно, "
                "версія без /api/v1. Завантажте модель вручну через вкладку Discover."
            )
        return str(did)

    async def download_status(self, download_id: str) -> dict[str, Any]:
        resp = await self._http().get(f"/api/v1/models/download/status/{download_id}")
        if resp.status_code >= 400:
            raise LlmProtocolError(f"Стан завантаження недоступний: HTTP {resp.status_code}")
        data = resp.json()
        return data if isinstance(data, dict) else {}

    async def download_and_wait(
        self,
        model_key: str,
        *,
        on_progress: Callable[[float, str], None] | None = None,
        poll_interval_s: float = 1.0,
        timeout_s: float = 3600.0,
    ) -> dict[str, Any]:
        """Завантажити модель із прогресом у власному UI.

        Це найбільший UX-виграш v1 API: викладач ніколи не відкриває вкладку
        Discover і не гадає, який файл із двадцяти йому потрібен.
        """
        did = await self.download_model(model_key)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        while loop.time() < deadline:
            status = await self.download_status(did)
            state = str(status.get("status") or status.get("state") or "")
            progress = float(status.get("progress") or 0.0)
            if on_progress is not None:
                on_progress(progress, state)
            if state.lower() in ("completed", "done", "finished"):
                return status
            if state.lower() in ("failed", "error", "cancelled"):
                raise ModelLoadRefused(model_key, 0, json.dumps(status, ensure_ascii=False)[:400])
            await asyncio.sleep(poll_interval_s)
        raise LlmTimeout(timeout_s, stage="завантаження моделі")

    # ------------------------------------------------------------- генерація
    async def chat_stream(
        self,
        messages: Messages,
        *,
        params: ChatParams | None = None,
        model: str | None = None,
        cancel: asyncio.Event | None = None,
        warm: bool = True,
    ) -> AsyncIterator[str]:
        """Стрім дельт відповіді.

        Відновлення після вивантаження моделі: повторюємо РІВНО ОДИН раз і
        лише якщо не встигли віддати жодного токена — інакше користувач
        побачить склеєні дві половини різних відповідей.
        """
        emitted = False
        retry_cold = False
        try:
            async for delta in self._chat_stream_once(
                messages, params=params, model=model, cancel=cancel, warm=warm
            ):
                emitted = True
                yield delta
            return
        except ModelNotLoaded:
            if emitted or self._last_load_config is None:
                raise
        except LlmTimeout:
            # Теплий TTFT вичерпано, а не віддано ЖОДНОГО токена. Найімовірніша
            # причина — LM Studio вивантажив модель за власним JIT-TTL і зараз
            # піднімає її заново: для 12B це приблизно 30 с, тобто рівно межа
            # TTFT_WARM_S. `ModelNotLoaded` сюди не долітає: сервер запит уже
            # ПРИЙНЯВ і мовчки вантажить модель, тож із погляду клієнта це
            # звичайний таймаут. Без цієї гілки перше питання після паузи
            # падало у викладача завжди, а друге те саме — спрацьовувало.
            if emitted or not warm:
                raise
            retry_cold = True

        if retry_cold:
            log.info(
                "Теплий TTFT вичерпано без жодного токена — найімовірніше JIT-"
                "завантаження; повторюю один раз із холодним таймаутом %.0f с.",
                self.ttft_cold_s,
            )
            async for delta in self._chat_stream_once(
                messages, params=params, model=model, cancel=cancel, warm=False
            ):
                yield delta
            return

        # Перевидаємо ТОЧНО ту конфігурацію, що була застосована при старті.
        await self.load_model(self._last_load_config)
        async for delta in self._chat_stream_once(
            messages, params=params, model=model, cancel=cancel, warm=False
        ):
            yield delta

    async def _resolve_chat_model(self) -> str:
        """Ім'я моделі для генерації — з реального стану сервера.

        Порядок: уже піднята в пам'ять LLM → JIT-завантаження єдиної наявної →
        гучна помилка з підказкою. Вигаданих значень немає й бути не може:
        саме вони давали помилку, що вказувала не на ту причину.
        """
        models = await self.list_models()
        chat = [m for m in models if m.kind in ("llm", "vlm", "unknown") and m.key]

        ready = [m for m in chat if m.loaded]
        if ready:
            # Найбільший корисний контекст серед піднятих.
            ready.sort(key=lambda m: (m.loaded_context or m.max_context or 0), reverse=True)
            return ready[0].key

        if len(chat) == 1:
            # Єдиний однозначний кандидат — довіряємо JIT-завантаженню LM Studio.
            # Це не «вигадування»: ідентифікатор справжній, і якщо пам'яті не
            # хватить, гардрейл відповість зрозумілою помилкою про ресурси.
            log.info("Жодна модель не піднята; пробую JIT для єдиної наявної: %s", chat[0].key)
            return chat[0].key

        raise NoChatModelReady(tuple(m.key for m in chat))

    # ------------------------------------------------------ можливості моделі
    def _model_info(self, key: str) -> ModelInfo | None:
        """Модель з останнього списку: спершу точний ключ, далі нормалізований
        (ключ LM Studio і ідентифікатор репозиторію — той самий файл)."""
        info = self._models.get(key)
        if info is not None:
            return info
        return BackendHealth(ok=True, models=tuple(self._models.values())).find(key)

    async def _reasoning_off_supported(self, model_key: str) -> bool:
        """Чи можна попросити цю модель не міркувати.

        Лише за оголошеною можливістю (`capabilities.reasoning.allowed_options`):
        моделі без неї LM Studio на параметр відповідає помилкою. Якщо модель ще
        не бачили (явно задана в налаштуваннях, `health()` ще не викликався) —
        один запит списку моделей, далі працює кеш.
        """
        if model_key in self._no_reasoning_cfg:
            return False
        info = self._model_info(model_key)
        if info is None:
            try:
                await self.list_models()
            except LlmError:
                return False
            info = self._model_info(model_key)
        return info is not None and "off" in info.reasoning_options

    async def ensure_loaded(self, model: str | None = None) -> str | None:
        """Підняти модель у пам'ять ДО питання. Повертає її ключ або None.

        JIT-завантаження LM Studio (15–22 с для Gemma 4 E4B на цій лінійці
        ноутбуків) інакше лягає на перше питання після старту застосунку або
        після того, як LM Studio вивантажив модель за власним TTL простою.
        Викликається паралельно з пошуком і при відкритті чату, тож час
        завантаження ховається за роботою, яку все одно треба зробити.

        Механізм — той самий JIT, що й у першого питання (крихітний запит на
        один токен), а не `/api/v1/models/load` з власною конфігурацією: конфіг
        LM Studio належить викладачеві, і ми його не перевизначаємо. Ніколи не
        кидає: невдалий розігрів не має ламати відповідь, яка його не потребує.
        """
        try:
            models = await self.list_models()
        except LlmError:
            return None
        key = model or (self._last_load_config.model if self._last_load_config else None)
        if key:
            info = self._model_info(key)
        else:
            chat = [m for m in models if m.kind in ("llm", "vlm", "unknown") and m.key]
            loaded = [m for m in chat if m.loaded]
            # Кілька моделей і жодна не піднята — вибір за викладачем, не за нами.
            info = loaded[0] if loaded else (chat[0] if len(chat) == 1 else None)
        if info is None:
            return None
        if info.loaded:
            return info.key

        payload: dict[str, Any] = {
            "model": info.key,
            "messages": [{"role": "user", "content": "."}],
            "max_tokens": 1,
            "stream": False,
        }
        if "off" in info.reasoning_options and info.key not in self._no_reasoning_cfg:
            payload["reasoning_effort"] = "none"
        log.info("Піднімаю модель %s заздалегідь (JIT), щоб не чекати на першому питанні.",
                 info.key)
        try:
            resp = await self._http().post(
                "/v1/chat/completions", json=payload,
                timeout=httpx.Timeout(connect=self.connect_timeout_s,
                                      read=self.load_timeout_s,
                                      write=30.0, pool=self.connect_timeout_s),
            )
        except httpx.HTTPError as exc:
            log.info("Попереднє завантаження %s не вдалося: %s", info.key, exc)
            return None
        if resp.status_code >= 400:
            log.info("Попереднє завантаження %s не вдалося: HTTP %s %s",
                     info.key, resp.status_code, resp.text[:200])
            return None
        return info.key

    async def _chat_stream_once(
        self,
        messages: Messages,
        *,
        params: ChatParams | None,
        model: str | None,
        cancel: asyncio.Event | None,
        warm: bool,
    ) -> AsyncIterator[str]:
        p = params or ChatParams()
        model_key = model or (self._last_load_config.model if self._last_load_config else None)
        if not model_key:
            # НІКОЛИ не підставляти вигаданий ідентифікатор. Раніше тут стояло
            # `model_key or "local-model"`, і LM Studio відповідав «No models
            # loaded» — повідомлення, яке вказує на порожню пам'ять, тоді як
            # справжня причина була в тому, що ми не назвали модель.
            # Дві помилки з однаковим текстом — найдорожчий вид діагностики.
            model_key = await self._resolve_chat_model()

        # МІРКУВАННЯ ВИМИКАЮТЬСЯ САМЕ `reasoning_effort: "none"`. LM Studio 0.4.x
        # мовчки ігнорує на цій поверхні і `reasoning: "off"`, і `reasoning.effort`,
        # і `chat_template_kwargs.enable_thinking` — перевірено на живому сервері,
        # саме тому раніше вважалося, що вимкнути їх «неможливо». Замір на Gemma 4
        # E4B з тим самим промптом: перший видимий токен 23.3 с → 0.84 с.
        send_reasoning = p.reasoning == "off" and await self._reasoning_off_supported(model_key)

        first_timeout = self.ttft_warm_s if warm else self.ttft_cold_s
        request_id = f"{id(messages):x}-{len(messages)}"

        for attempt in (1, 2):
            payload: dict[str, Any] = {
                "model": model_key,
                "messages": messages,
                "stream": True,
                **p.to_payload(),
            }
            # НІКОЛИ truncateMiddle: воно тихо викидає ваші докази з середини
            # промпту, і ви отримуєте впевнену відповідь без джерел.
            payload["context_overflow_policy"] = p.context_overflow_policy
            if send_reasoning:
                payload["reasoning_effort"] = "none"

            try:
                async with self._http().stream(
                    "POST", "/v1/chat/completions", json=payload,
                    timeout=httpx.Timeout(
                        connect=self.connect_timeout_s,
                        read=max(first_timeout, self.idle_timeout_s) + 5.0,
                        write=30.0, pool=self.connect_timeout_s),
                ) as resp:
                    if resp.status_code >= 400:
                        body = (await resp.aread()).decode("utf-8", "replace")
                        lowered = body.lower()
                        if send_reasoning and attempt == 1 and "reasoning" in lowered:
                            # Модель оголосила можливість, але сервер параметр
                            # відхилив. Відповідь без нього все одно краща за
                            # помилку: повторюємо рівно один раз і запам'ятовуємо.
                            # Токенів ще не було, тож склеїти дві відповіді неможливо.
                            log.warning(
                                "LM Studio відхилив reasoning_effort для %s (%s); "
                                "повторюю без нього.", model_key, body[:200])
                            self._no_reasoning_cfg.add(model_key)
                            send_reasoning = False
                            continue
                        if any(m in lowered for m in _UNLOADED_MARKERS):
                            raise ModelNotLoaded(model_key or "?")
                        if any(m in lowered for m in _GUARDRAIL_MARKERS):
                            raise ModelLoadRefused(model_key or "?", 0, body[:400])
                        raise LlmProtocolError(
                            f"LM Studio повернув HTTP {resp.status_code}: {body[:300]}")

                    stream = stream_with_idle_timeout(
                        resp.aiter_lines(),
                        first_timeout=first_timeout,
                        idle_timeout=self.idle_timeout_s,
                    )
                    async for line in stream:
                        if cancel is not None and cancel.is_set():
                            # Баг #1203: сервер може ще якийсь час слати токени.
                            # Позначаємо запит мертвим і виходимо — вихід із
                            # `async with` закриває HTTP-стрім.
                            self._dead_requests.add(request_id)
                            return
                        for delta in _iter_sse_deltas(line):
                            yield delta
                    return
            except httpx.ReadTimeout as exc:
                raise LlmTimeout(self.idle_timeout_s, stage="очікування токена") from exc
            except httpx.HTTPError as exc:
                raise LlmUnavailable(str(exc)) from exc


def _iter_sse_deltas(line: str) -> list[str]:
    """Розбір одного рядка SSE. Порожні рядки й `[DONE]` — не помилка."""
    if not line or not line.startswith("data:"):
        return []
    data = line[5:].strip()
    if not data or data == "[DONE]":
        return []
    try:
        chunk = json.loads(data)
    except ValueError:
        return []
    out: list[str] = []
    for choice in chunk.get("choices", []) or []:
        delta = choice.get("delta") or {}
        text = delta.get("content")
        if text:
            out.append(text)
        # Нестрімінгова форма інколи проривається у стрім-відповідь.
        message = choice.get("message") or {}
        if not text and message.get("content"):
            out.append(message["content"])
    return out
