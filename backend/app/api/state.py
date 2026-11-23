"""Спільний стан API-процесу: БД, шина подій, черга, моделі, наглядач.

ЖОДЕН ВАЖКИЙ ІМПОРТ НЕ ВІДБУВАЄТЬСЯ НА СТАРТІ.
`import torch` коштує 1.5–4 с на Windows; в API це означає 6+ секунд до
чутливості вікна, і викладач вирішує, що застосунок зламався. Тому в цьому
процесі немає ні torch, ні docling, ні rapidocr — жодного, ніколи. Ембедер і
реранкер тут МОЖНА: це прямі сесії `onnxruntime` (14 МБ), і вони потрібні на
гарячому шляху запиту.

ВІДСУТНІСТЬ МОДЕЛІ — НЕ ПРИЧИНА НЕ ЗАПУСТИТИСЬ.
Ембедер і реранкер створюються ЛІНИВО і в try/except. Машина без розпакованих
ваг мусить піднятися, показати майстер першого запуску й пояснити, чого бракує;
падіння на старті залишило б викладача з вікном, яке просто не відкривається.
Усе, що не вдалося створити, потрапляє в `problems` і видно в `/api/health`.

ЛІНИВО — НЕ ЗНАЧИТЬ «НА ПЕРШОМУ ПИТАННІ».
Одразу після старту `warm_models()` піднімає ті самі ліниві моделі у фоновому
потоці (старт API від цього не сповільнюється), а `warm_llm()` просить LM Studio
підняти мовну модель, щойно відкрито чат, і паралельно з пошуком кожного ходу.
Інакше перше питання платило ~7 с за ONNX-сесії і 15–22 с за JIT-завантаження LLM.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from app.api.events import EventBus
from app.config import Paths
from app.db.database import Database
from app.jobs.queue import JobQueue
from app.settings import Settings

__all__ = ["Services", "build_services"]

log = logging.getLogger("asistent.api")

# Здоров'я LM Studio кешуємо на дві секунди: /health опитується поллінгом з
# UI, а кожен промах виявлення порту — це системний виклик і таймаут.
HEALTH_TTL_S = 2.0


@dataclass(slots=True)
class Problem:
    code: str
    message: str
    hint: str = ""

    def as_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message, "hint": self.hint}


@dataclass
class Services:
    settings: Settings
    paths: Paths
    db: Database
    events: EventBus
    queue: JobQueue
    started_at: float = field(default_factory=time.monotonic)
    problems: list[Problem] = field(default_factory=list)

    _provider: Any = None
    _reranker: Any = None
    _retriever: Any = None
    _backend: Any = None
    _health_cache: tuple[float, Any] | None = None
    supervisor: Any = None
    watcher: Any = None
    # Цикл подій процесу — щоб планувати розігрів LLM із синхронних обробників,
    # які FastAPI виконує в пулі потоків.
    loop: asyncio.AbstractEventLoop | None = None
    # Ліниві моделі створюються під замком: фоновий розігрів і перше питання
    # інакше одночасно підняли б ДВІ копії ембедера й реранкера (~2 ГБ RAM).
    # RLock — бо `retriever` всередині бере `provider` і `reranker`.
    _models_lock: Any = field(default_factory=threading.RLock)
    _llm_warm_task: asyncio.Task[Any] | None = None
    _background: set[asyncio.Task[Any]] = field(default_factory=set)

    # ------------------------------------------------------------ ембедер
    @property
    def provider(self) -> Any | None:
        """Провайдер ембедингів або None, якщо ваг немає."""
        with self._models_lock:
            if self._provider is None:
                from app.embeddings.provider import create_provider

                try:
                    # `models_dir` — явно: без нього ваги шукаються в типовому
                    # каталозі, а не в тому, який задав `ASISTENT_DATA_DIR`.
                    self._provider = create_provider(
                        self.settings.embedding_model,
                        stub=self.settings.stub,
                        models_dir=self.paths.models_dir,
                    )
                except Exception as exc:
                    self._note(
                        "EMBEDDER_MISSING",
                        f"Модель ембедингів недоступна: {exc}",
                        "Розпакуйте моделі з носія в каталог models/embeddings або "
                        "запустіть із ASISTENT_STUB=1 для роботи без моделей.",
                    )
                    self._provider = False
            return self._provider or None

    # ----------------------------------------------------------- реранкер
    @property
    def reranker(self) -> Any | None:
        """Реранкер або None. Пошук зобов'язаний працювати і без нього.

        Типово (`rerank_with="llm"`) кандидатів оцінює та сама модель LM Studio,
        що відповідає: на GPU це ~2.5 с проти ~80 с CPU-крос-енкодера, і на
        питаннях «що таке X» вона краще відрізняє визначення від згадки. CPU-
        крос-енкодер лишається запасним шляхом і створюється лише тоді, коли
        LM Studio не відповів, — до того процес не тримає його 1.2 ГБ ваг.
        """
        if not self.settings.use_reranker:
            return None
        with self._models_lock:
            if self._reranker is None:
                if self.settings.stub or self.settings.rerank_with == "onnx":
                    self._reranker = self._onnx_reranker() or False
                else:
                    from app.rerank.llm_reranker import LlmListwiseReranker

                    self._reranker = LlmListwiseReranker(
                        base_url=self.settings.lmstudio_url,
                        model=self.settings.lmstudio_model,
                        fallback=self._onnx_reranker,
                        context_tokens=self.settings.context_tokens,
                    )
            return self._reranker or None

    def _onnx_reranker(self) -> Any | None:
        """CPU-крос-енкодер (або заглушка в stub-режимі). None — ваг немає."""
        from app.rerank.reranker import create_reranker

        try:
            return create_reranker(
                self.settings.rerank_model,
                stub=self.settings.stub,
                models_dir=self.paths.models_dir,
            )
        except Exception as exc:
            self._note(
                "RERANKER_MISSING",
                f"Реранкер недоступний: {exc}",
                "Пошук працюватиме без реранкінгу — це помітно гірша якість "
                "відповідей, але не відмова.",
            )
            return None

    # ----------------------------------------------------------- ретривер
    @property
    def retriever(self) -> Any | None:
        with self._models_lock:
            if self._retriever is None:
                provider = self.provider
                if provider is None:
                    return None
                from app.retrieval.hybrid import HybridRetriever

                self._retriever = HybridRetriever(
                    self.db,
                    provider=provider,
                    index_dir=self.paths.index_dir,
                    reranker=self.reranker,
                )
            return self._retriever

    # ------------------------------------------------------------ розігрів
    def warm_models(self) -> None:
        """Підняти ембедер і реранкер ДО першого питання. Синхронно — у потоці.

        Перше питання інакше платило ~3.5 с за кожну ONNX-сесію плюс перший
        прохід (виділення арен ORT) і завантаження словників pymorphy. Ніколи
        не кидає: відсутні ваги — це `problems` у /api/health, а не падіння.
        """
        started = time.perf_counter()
        try:
            retriever = self.retriever
            if retriever is None:
                return
            retriever.provider.embed_queries(["розігрів"])
            # LLM-реранкер не розігріваємо: це запит до LM Studio, тобто
            # завантаження мовної моделі на старті, від якого ми свідомо відмовились.
            reranker = retriever.reranker
            if reranker is not None and getattr(reranker, "name", "") != "llm":
                reranker.score("розігрів", ["розігрів"])
            from app.retrieval.hybrid import analyze_query

            analyze_query("розігрів моделей")
            log.info("Ембедер і реранкер розігріто за %.1f с.", time.perf_counter() - started)
        except Exception:
            log.warning("Розігрів моделей пошуку не вдався", exc_info=True)

    async def warm_llm(self) -> None:
        """Підняти LLM у LM Studio заздалегідь. Одна спільна задача на процес.

        Кличеться на початку кожного ходу (паралельно з пошуком) і при
        відкритті чату. Повторні виклики, поки розігрів триває, чекають ту
        саму задачу, а не шлють LM Studio нових запитів. Ніколи не кидає.
        """
        task = self._llm_warm_task
        if task is None or task.done():
            task = asyncio.create_task(self._warm_llm_once(), name="llm-warmup")
            self._llm_warm_task = task
        await asyncio.shield(task)

    async def _warm_llm_once(self) -> None:
        try:
            backend = await self.backend()
            ensure = getattr(backend, "ensure_loaded", None)
            if ensure is not None:
                await ensure(self.settings.lmstudio_model)
        except Exception:
            log.info("Розігрів LLM не вдався", exc_info=True)

    def schedule_llm_warmup(self) -> None:
        """Те саме з синхронного коду (обробник у пулі потоків): не чекаємо."""
        loop = self.loop
        if loop is None or loop.is_closed():
            return
        try:
            asyncio.run_coroutine_threadsafe(self.warm_llm(), loop)
        except RuntimeError:
            log.debug("Цикл подій уже зупинено — розігрів LLM пропущено")

    def spawn(self, coro: Any, *, name: str) -> asyncio.Task[Any]:
        """Фонова задача з утриманням посилання (інакше GC може її прибити)."""
        task = asyncio.create_task(coro, name=name)
        self._background.add(task)
        task.add_done_callback(self._background.discard)
        return task

    # ------------------------------------------------------------- LLM
    async def backend(self) -> Any | None:
        """Бекенд LLM або None, якщо LM Studio не знайдено."""
        if self._backend is None:
            # Рішення про заглушку ухвалюють НАЛАШТУВАННЯ, а не змінна
            # середовища: інакше тест, який виставив `stub=True` у Settings,
            # мовчки пішов би шукати справжній LM Studio на машині розробника
            # і став би недетермінованим.
            if self.settings.stub:
                from app.backends.stub_backend import StubBackend

                self._backend = StubBackend()
                return self._backend

            from app.backends import get_backend

            try:
                self._backend = await get_backend(base_url=self.settings.lmstudio_url)
            except Exception as exc:
                self._note(
                    "LMSTUDIO_UNAVAILABLE",
                    f"LM Studio недоступний: {exc}",
                    "Запустіть LM Studio і ввімкніть локальний сервер "
                    "(Developer → Start Server) або скористайтесь кнопкою в майстрі.",
                )
                self._backend = False
        return self._backend or None

    def reset_backend(self) -> None:
        """Після `lms server start` або ручного запуску — спробувати ще раз."""
        self._backend = None
        self._health_cache = None
        self.problems = [p for p in self.problems if p.code != "LMSTUDIO_UNAVAILABLE"]

    async def llm_health(self) -> dict[str, Any]:
        now = time.monotonic()
        if self._health_cache is not None and (now - self._health_cache[0]) < HEALTH_TTL_S:
            return self._health_cache[1]
        backend = await self.backend()
        if backend is None:
            payload: dict[str, Any] = {
                "ok": False,
                "chatReady": False,
                "detail": "LM Studio не виявлено на 127.0.0.1.",
                "apiVersion": "none",
                "models": [],
            }
        else:
            try:
                health = await backend.health()
                # `ok` і `chatReady` — РІЗНІ речі, і плутати їх коштувало
                # зеленого індикатора при повністю неробочій генерації:
                # `ok` = сервер відповідає, `chatReady` = модель у пам'яті.
                payload = {
                    "ok": bool(health.ok),
                    "chatReady": bool(health.chat_ready),
                    "baseUrl": health.base_url,
                    "apiVersion": health.api_version,
                    "detail": health.detail or health.readiness_detail,
                    "models": [
                        {
                            "key": m.key, "loaded": m.loaded, "kind": m.kind,
                            "engine": m.engine, "maxContext": m.max_context,
                            "loadedContext": m.loaded_context,
                            "quantization": m.quantization, "sizeBytes": m.size_bytes,
                        }
                        for m in health.models
                    ],
                }
            except Exception as exc:
                payload = {"ok": False, "detail": str(exc), "apiVersion": "none", "models": []}
        self._health_cache = (now, payload)
        return payload

    # ------------------------------------------------------------ службове
    def _note(self, code: str, message: str, hint: str = "") -> None:
        if any(p.code == code for p in self.problems):
            return
        log.warning("[%s] %s", code, message)
        self.problems.append(Problem(code, message, hint))

    def emit(self, event_type: str, data: dict[str, Any]) -> None:
        self.events.publish(event_type, data)

    @property
    def uptime_seconds(self) -> float:
        return round(time.monotonic() - self.started_at, 1)

    def close(self) -> None:
        for name in ("_retriever", "_reranker", "_provider", "_backend"):
            obj = getattr(self, name, None)
            closer = getattr(obj, "close", None) if obj else None
            if callable(closer):
                try:
                    result = closer()
                    if asyncio.iscoroutine(result):
                        result.close()
                except Exception:
                    log.debug("Помилка закриття %s", name, exc_info=True)
        self.events.close()
        try:
            self.db.close()
        except Exception:
            log.debug("Помилка закриття БД", exc_info=True)


def build_services(settings: Settings) -> Services:
    """Створити стан застосунку. Виконує міграції БД."""
    paths = settings.paths()
    db = Database(paths.db_path)
    events = EventBus(buffer=settings.event_buffer)
    return Services(
        settings=settings,
        paths=paths,
        db=db,
        events=events,
        queue=JobQueue(db),
    )
