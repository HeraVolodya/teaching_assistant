"""Єдиний мультиплексований SSE-канал застосунку.

ЧОМУ SSE, А НЕ IPC ОБОЛОНКИ
---------------------------
Це свідоме архітектурне рішення, а не зручність. `invoke()` Tauri і IPC
Electron прив'язали б увесь realtime-шар до конкретної оболонки; SSE поверх
звичайного HTTP працює однаково в Tauri, в Electron і в браузері на
`vite dev`. Саме це робить майбутню міграцію в багатокафедральну веб-платформу
ВИДАЛЕННЯМ оболонки, а не переписуванням застосунку (план, §0). Ціна рішення —
цей файл; ціна протилежного — переписати чат, індикатор індексації й майстер
першого запуску при кожній зміні оболонки.

ОДИН КАНАЛ, А НЕ П'ЯТЬ
----------------------
Браузери історично обмежують кількість одночасних HTTP/1.1-з'єднань на
походження шістьма. Окремий SSE-канал на прогрес, на чат і на завантаження
моделі з'їв би половину бюджету й залишив би застосунок без з'єднань саме
тоді, коли він одночасно індексує, відповідає і качає модель. Тому канал
один, а події типізовані.

LAST-EVENT-ID
-------------
WebView перепідключається сам, і без відновлення викладач втрачає рівно ті
події, що сталися під час обриву, — тобто найчастіше `doc.ready`. Кільцевий
буфер віддає все, що сталося після зазначеного id.
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

__all__ = [
    "Event",
    "EventBus",
    "JOB_PROGRESS",
    "JOB_FAILED",
    "DOC_READY",
    "CHAT_TOKEN",
    "CHAT_CITATIONS",
    "CHAT_DEBUG",
    "CHAT_DONE",
    "CHAT_ERROR",
    "MODEL_DOWNLOAD",
    "WORKER_STATUS",
    "EVENT_TYPES",
    "sse_frame",
    "parse_last_event_id",
]

# --- типи подій ---------------------------------------------------------
# П'ять обов'язкових із завдання плюс три, без яких UI не збирається:
# chat.debug (етап map-reduce), chat.done (кінець ходу), model.download
# (майстер першого запуску) і worker.status (індикатор «індексація активна»).
JOB_PROGRESS = "job.progress"        # {jobId, docId, stage, current, total, etaSeconds}
JOB_FAILED = "job.failed"            # {jobId, docId, errorCode, message, hint}
DOC_READY = "doc.ready"              # {docId, chunks, tables, formulas}
CHAT_TOKEN = "chat.token"            # {messageId, delta}
CHAT_CITATIONS = "chat.citations"    # {messageId, citations:[...], merged:[...]}
CHAT_DEBUG = "chat.debug"            # {messageId, stage, ...}
CHAT_DONE = "chat.done"              # {messageId, abstained, tokensOut, ttftMs}
CHAT_ERROR = "chat.error"            # {messageId, errorCode, message}
MODEL_DOWNLOAD = "model.download"    # {downloadId, model, percent, state}
WORKER_STATUS = "worker.status"      # {mode, running, active}

EVENT_TYPES: tuple[str, ...] = (
    JOB_PROGRESS, JOB_FAILED, DOC_READY,
    CHAT_TOKEN, CHAT_CITATIONS, CHAT_DEBUG, CHAT_DONE, CHAT_ERROR,
    MODEL_DOWNLOAD, WORKER_STATUS,
)


@dataclass(frozen=True, slots=True)
class Event:
    id: int
    type: str
    data: dict[str, Any]

    def to_sse(self) -> str:
        return sse_frame(self.type, self.data, event_id=self.id)


def sse_frame(event: str, data: Any, *, event_id: int | None = None) -> str:
    """Один кадр SSE.

    `ensure_ascii=False` обов'язковий: інакше кожна українська відповідь
    роздувається втричі в `\\uXXXX`-екрануванні, і на стрімінгу це видно.
    Переводи рядків усередині JSON неможливі (json.dumps їх екранує), тож
    багаторядкового `data:` тут не буває за побудовою.
    """
    payload = json.dumps(data, ensure_ascii=False, default=_fallback)
    head = f"id: {event_id}\n" if event_id is not None else ""
    return f"{head}event: {event}\ndata: {payload}\n\n"


def _fallback(obj: Any) -> Any:
    """Останній рубіж серіалізації: dataclass'и доменного шару."""
    from dataclasses import asdict, is_dataclass

    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, (set, frozenset, tuple)):
        return list(obj)
    return str(obj)


class EventBus:
    """Публікація подій у всіх підписників плюс кільцевий буфер для повтору.

    Потокобезпечність не факультативна: у режимі `inline` конвеєр індексації
    крутиться в пулі потоків, і `asyncio.Queue.put_nowait` з чужого потоку —
    це тиха втрата події або пошкоджений стан циклу. Тому публікація з
    не-loop-потоку проходить через `call_soon_threadsafe`.
    """

    def __init__(self, *, buffer: int = 512) -> None:
        self._buffer: deque[Event] = deque(maxlen=max(16, buffer))
        self._subscribers: set[asyncio.Queue[Event]] = set()
        self._lock = threading.Lock()
        self._next_id = 1
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: int | None = None

    # ------------------------------------------------------------ життєвий цикл
    def attach_loop(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        """Запам'ятати цикл, у якому живуть черги підписників."""
        self._loop = loop or asyncio.get_running_loop()
        self._loop_thread = threading.get_ident()

    def close(self) -> None:
        with self._lock:
            self._subscribers.clear()
            self._loop = None

    # ------------------------------------------------------------ публікація
    def publish(self, event_type: str, data: dict[str, Any]) -> Event:
        with self._lock:
            event = Event(id=self._next_id, type=event_type, data=data)
            self._next_id += 1
            self._buffer.append(event)
            targets = list(self._subscribers)

        if not targets:
            return event
        loop = self._loop
        if loop is not None and threading.get_ident() != self._loop_thread:
            # Публікація з потоку індексації: доставку виконує сам цикл.
            loop.call_soon_threadsafe(self._fanout, event, targets)
        else:
            self._fanout(event, targets)
        return event

    @staticmethod
    def _fanout(event: Event, targets: list[asyncio.Queue[Event]]) -> None:
        for queue in targets:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                # Підписник не встигає читати. Втратити подію краще, ніж
                # заблокувати індексацію на повільному WebView.
                pass

    # ------------------------------------------------------------ підписка
    def replay(self, last_event_id: int | None) -> list[Event]:
        """Події, що сталися ПІСЛЯ вказаного id (для перепідключення)."""
        if last_event_id is None:
            return []
        with self._lock:
            return [e for e in self._buffer if e.id > last_event_id]

    def history(self, limit: int = 100) -> list[Event]:
        with self._lock:
            return list(self._buffer)[-limit:]

    async def stream(
        self,
        *,
        last_event_id: int | None = None,
        keepalive_s: float = 15.0,
        max_queue: int = 1024,
    ) -> AsyncIterator[str]:
        """Готові SSE-кадри: спершу пропущені, далі живі."""
        queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=max_queue)
        with self._lock:
            # Свіже підключення (без Last-Event-ID) НЕ отримує буфер: стан UI
            # бере з REST (`/api/health`, список документів), а перепрогравання
            # історії показало б сповіщення про вчорашню індексацію на кожному
            # відкритті вкладки. Буфер існує рівно для ВІДНОВЛЕННЯ.
            missed = [e for e in self._buffer if last_event_id is not None and e.id > last_event_id]
            self._subscribers.add(queue)
        if self._loop is None:
            self.attach_loop()
        try:
            # `retry` кажемо один раз: браузер сам перепідключиться через 2 с,
            # а не через дефолтні 3 с, і зробить це з Last-Event-ID.
            yield "retry: 2000\n\n"
            for event in missed:
                yield event.to_sse()
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=keepalive_s)
                except TimeoutError:
                    # Коментар-пінг: без нього проксі й сплячі вкладки рвуть
                    # з'єднання посеред індексації, і прогрес просто зникає.
                    yield ": ping\n\n"
                    continue
                yield event.to_sse()
        finally:
            with self._lock:
                self._subscribers.discard(queue)

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subscribers)


def parse_last_event_id(raw: str | None) -> int | None:
    """`Last-Event-ID` приходить від браузера рядком і може бути сміттям."""
    if not raw:
        return None
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return None
    return value if value >= 0 else None
