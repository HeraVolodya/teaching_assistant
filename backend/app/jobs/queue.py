"""Черга завдань: лізинг, heartbeat, кооперативне скасування, прогрес.

Обгортка над `JobRepo`, яка перетворює три таблиці на одну придатну до
використання абстракцію. Усе SQL лишається в `repositories.py`; тут — політика.

ТРИ ВЛАСТИВОСТІ, ЗАРАДИ ЯКИХ ЦЕЙ ФАЙЛ ІСНУЄ

1. **Лізинг + heartbeat кожні 2 с.** Воркер може впасти будь-коли (некоректний
   PDF валить pdfium, машину перезавантажили, викладач закрив кришку). Без
   лізингу завдання лишилось би в стані RUNNING назавжди й документ висів би
   «індексується» до кінця часів. Прострочений лізинг повертає завдання в
   чергу, а після `max_attempts` — у FAILED зі збереженою помилкою.

2. **Кооперативне скасування між СТОРІНКАМИ.** Скасування не може бути
   миттєвим: воркер більшість часу сидить усередині C-виклику Docling, який
   перервати неможливо. Тому точка перевірки — межа сторінки, і найгірша
   затримка скасування дорівнює вартості ОДНІЄЇ сторінки (0.4–0.7 с у
   швидкому режимі). Обіцяти менше означало б обіцяти вбивство процесу, а це
   втрата вже виконаної роботи.

3. **Посторінкове відновлення через `document_pages_state`.** 1000-сторінковий
   підручник — це 1000 рядків. Падіння на 940-й сторінці без цієї таблиці
   коштує 40 хвилин повторної обробки; з нею — 40 секунд.

ЧОМУ ПРОГРЕС ЗВАЖЕНИЙ, А НЕ ЛІНІЙНИЙ
Сторінки різнорідні: сканована таблична сторінка коштує приблизно в 20 разів
більше за порожню. Лінійний лічильник «сторінка 300 з 1000» на такому корпусі
бреше в рази, і саме він породжує прогрес-бари, що стоять на 90% годину. Проба
призначає кожній сторінці вагу `w = 1 + 3·скан + 2·таблиця + 1·формула +
0.5·рисунки`, а ETA рахується з КОВЗНОЇ МЕДІАНИ секунд на одиницю ваги —
медіани, а не середнього, бо одна сторінка з великою таблицею інакше зсуває
оцінку на хвилини.
"""

from __future__ import annotations

import contextlib
import statistics
import time
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from app.db.repositories import JobRepo
from app.domain import JobState, JobType

__all__ = [
    "HEARTBEAT_INTERVAL_S",
    "ClaimedJob",
    "JobCancelled",
    "JobQueue",
    "Lease",
    "ProgressTracker",
]

# Heartbeat рідше за лізинг у 30 разів: лізинг 60 с, удар серця 2 с. Проміжок
# такий великий саме тому, що між ударами воркер може сидіти в довгій сторінці.
HEARTBEAT_INTERVAL_S = 2.0


class JobCancelled(RuntimeError):
    """Викладач натиснув «Скасувати». Не помилка — це запитаний результат."""

    def __init__(self, job_id: str) -> None:
        super().__init__(f"Завдання {job_id} скасовано на запит користувача.")
        self.job_id = job_id


@dataclass(slots=True)
class ClaimedJob:
    """Завдання, взяте під лізинг."""

    id: str
    type: JobType
    document_id: str | None
    collection_id: str | None
    stage: str | None
    attempts: int
    weight_total: float
    weight_done: float
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> ClaimedJob:
        return cls(
            id=str(row["id"]),
            type=JobType(row["type"]),
            document_id=row.get("document_id"),
            collection_id=row.get("collection_id"),
            stage=row.get("stage"),
            attempts=int(row.get("attempts") or 0),
            weight_total=float(row.get("progress_weight_total") or 0.0),
            weight_done=float(row.get("progress_weight_done") or 0.0),
            raw=row,
        )


class ProgressTracker:
    """Зважений прогрес і ETA з ковзної медіани.

    ETA свідомо НЕ рахується з середнього: розподіл вартості сторінок у
    підручнику важкохвостий (одна сторінка з великою таблицею коштує як
    двадцять текстових), і середнє на ній стрибає. Медіана останніх N
    вимірювань дає стабільну оцінку, яка не смикається на кожній таблиці.
    """

    __slots__ = ("_last_at", "_samples", "_started", "done", "total")

    def __init__(self, total_weight: float, *, window: int = 24) -> None:
        self.total = max(float(total_weight), 1e-9)
        self.done = 0.0
        self._samples: deque[float] = deque(maxlen=max(3, window))
        self._started = time.monotonic()
        self._last_at = self._started

    def advance(self, weight: float) -> None:
        now = time.monotonic()
        elapsed = now - self._last_at
        self._last_at = now
        self.done = min(self.total, self.done + max(0.0, float(weight)))
        if weight > 0 and elapsed > 0:
            self._samples.append(elapsed / float(weight))

    def set_done(self, done_weight: float) -> None:
        """Абсолютна позиція. Потрібна, бо колбек парсера звітує саме її."""
        self.advance(max(0.0, float(done_weight) - self.done))

    @property
    def fraction(self) -> float:
        return min(1.0, self.done / self.total)

    @property
    def seconds_per_weight(self) -> float | None:
        if not self._samples:
            return None
        return float(statistics.median(self._samples))

    @property
    def eta_seconds(self) -> float | None:
        """None, поки нема жодного виміру: чесне «невідомо» краще за 0."""
        rate = self.seconds_per_weight
        if rate is None:
            return None
        remaining = max(0.0, self.total - self.done)
        return round(remaining * rate, 1)

    @property
    def elapsed_seconds(self) -> float:
        return time.monotonic() - self._started


class Lease:
    """Оренда одного завдання на час його виконання.

    Використання:

        with queue.lease(job) as lease:
            for page in pages:
                lease.checkpoint(stage="PARSE", weight=page.cost_weight)
                ...
    """

    __slots__ = ("_cancelled", "_last_beat", "_on_progress", "_stage", "job", "queue", "tracker")

    def __init__(
        self,
        queue: JobQueue,
        job: ClaimedJob,
        *,
        total_weight: float | None = None,
        on_progress: Any | None = None,
    ) -> None:
        self.queue = queue
        self.job = job
        self.tracker = ProgressTracker(
            total_weight if total_weight is not None else (job.weight_total or 1.0)
        )
        self._last_beat = 0.0
        self._cancelled = False
        self._stage: str | None = job.stage
        self._on_progress = on_progress

    # ------------------------------------------------------------- службове
    def set_total(self, total_weight: float) -> None:
        """Проба знає справжню суму ваг лише після свого проходу."""
        self.tracker = ProgressTracker(total_weight)
        self.queue.set_weight_total(self.job.id, total_weight)

    @property
    def stage(self) -> str | None:
        return self._stage

    def is_cancelled(self) -> bool:
        return self._cancelled

    # ------------------------------------------------------------ головне
    def checkpoint(
        self,
        *,
        stage: str | None = None,
        weight: float = 0.0,
        force: bool = False,
    ) -> None:
        """Точка кооперативної перевірки: heartbeat + скасування + прогрес.

        Кидає `JobCancelled`, якщо скасування запитано. Викликати МІЖ
        сторінками — усередині сторінки перервати нічого не можна.
        """
        if stage is not None and stage != self._stage:
            self._stage = stage
            force = True
        if weight:
            self.tracker.advance(weight)

        now = time.monotonic()
        if not force and (now - self._last_beat) < HEARTBEAT_INTERVAL_S:
            return
        self._last_beat = now

        alive = self.queue.heartbeat(self.job.id)
        self.queue.progress(self.job.id, stage=self._stage or "", weight_done=self.tracker.done)
        if self._on_progress is not None:
            self._on_progress(self)
        if not alive:
            self._cancelled = True
            raise JobCancelled(self.job.id)

    def touch(self, stage: str | None = None) -> None:
        """Той самий checkpoint, але без ваги — для довгих одноетапних кроків."""
        self.checkpoint(stage=stage, weight=0.0)

    def report(self, done_weight: float) -> None:
        """Абсолютний прогрес від колбека парсера (він звітує саме так)."""
        self.tracker.set_done(done_weight)
        # Колбек парсера НЕ має права кидати: Docling викликає його всередині
        # свого циклу вікон, і виняток звідти лишив би напівзібраний документ
        # без шансу на впорядковане завершення. Прапорець уже виставлено, і
        # `poll_cancel` зупинить цикл на наступній межі вікна.
        with contextlib.suppress(JobCancelled):
            self.checkpoint()

    def poll_cancel(self) -> bool:
        """Неруйнівна перевірка скасування — для колбека `ParseOptions.cancel`.

        `force=True` навмисно: цей колбек викликається раз на ВІКНО сторінок
        (25 сторінок — це 10–17 с роботи), тож один короткий SELECT нічого не
        коштує. Під загальним 2-секундним тротлінгом він, навпаки, майже
        завжди повертав би закешоване «не скасовано» — і кнопка «Скасувати»
        працювала б через раз, що гірше, ніж не працювала б зовсім.
        """
        try:
            self.checkpoint(force=True)
        except JobCancelled:
            return True
        return self._cancelled

    # ---------------------------------------------------------- контекст
    def __enter__(self) -> Lease:
        self.queue.heartbeat(self.job.id)
        self._last_beat = time.monotonic()
        return self

    def __exit__(self, exc_type: type[BaseException] | None, *_: object) -> bool:
        return False


class JobQueue:
    """Фасад черги. Кожна операція — власна коротка транзакція.

    Довгі транзакції тут заборонені: письменник у WAL один, і транзакція,
    відкрита на час обробки сторінки, заблокувала б чат.
    """

    def __init__(self, db: Any) -> None:
        self.db = db

    # ------------------------------------------------------------ постановка
    def enqueue(
        self,
        job_type: JobType,
        *,
        document_id: str | None = None,
        collection_id: str | None = None,
        priority: int = 100,
        weight_total: float = 0.0,
    ) -> str:
        with self.db.transaction() as con:
            return JobRepo(con).enqueue(
                job_type,
                document_id=document_id,
                collection_id=collection_id,
                priority=priority,
                weight_total=weight_total,
            )

    def enqueue_once(
        self,
        job_type: JobType,
        *,
        collection_id: str,
        priority: int = 200,
    ) -> str | None:
        """Поставити завдання рівня колекції, якщо такого ще немає в черзі.

        Збірка ANN-індексу — операція на всю колекцію. Ставити її після
        КОЖНОГО документа означало б перебудовувати граф двадцять разів
        замість одного при масовому завантаженні.
        """
        with self.db.transaction() as con:
            row = con.execute(
                "SELECT id FROM jobs WHERE type=? AND collection_id=?"
                " AND state IN ('QUEUED','LEASED','RUNNING') LIMIT 1",
                (job_type.value, collection_id),
            ).fetchone()
            if row is not None:
                return None
            return JobRepo(con).enqueue(
                job_type, collection_id=collection_id, priority=priority
            )

    # ------------------------------------------------------------ отримання
    def claim(self, types: Sequence[JobType] | None = None) -> ClaimedJob | None:
        with self.db.transaction() as con:
            row = JobRepo(con).claim(types)
        return ClaimedJob.from_row(row) if row else None

    def lease(
        self,
        job: ClaimedJob,
        *,
        total_weight: float | None = None,
        on_progress: Any | None = None,
    ) -> Lease:
        return Lease(self, job, total_weight=total_weight, on_progress=on_progress)

    # ------------------------------------------------------------ стан
    def heartbeat(self, job_id: str) -> bool:
        """False → скасовано, воркер мусить зупинитись."""
        with self.db.transaction() as con:
            return JobRepo(con).heartbeat(job_id)

    def progress(self, job_id: str, *, stage: str, weight_done: float) -> None:
        with self.db.transaction() as con:
            JobRepo(con).progress(job_id, stage=stage, weight_done=weight_done)

    def set_weight_total(self, job_id: str, total: float) -> None:
        """Сума ваг відома лише після проби — а завдання створюється до неї.

        Прямий UPDATE, бо в `JobRepo` немає сетера саме на це поле, а
        `repositories.py` — спільний фундамент, який цей модуль не редагує.
        """
        with self.db.transaction() as con:
            con.execute(
                "UPDATE jobs SET progress_weight_total=? WHERE id=?", (float(total), job_id)
            )

    def finish(
        self,
        job_id: str,
        state: JobState,
        *,
        error_code: str | None = None,
        error_detail: str | None = None,
    ) -> None:
        with self.db.transaction() as con:
            JobRepo(con).finish(
                job_id, state, error_code=error_code, error_detail=error_detail
            )

    def request_cancel(self, job_id: str) -> None:
        with self.db.transaction() as con:
            JobRepo(con).request_cancel(job_id)

    def cancel_document(self, document_id: str) -> int:
        """Скасувати всі активні завдання документа. Повертає їх кількість.

        Завдання в стані QUEUED знімаємо одразу: воркер їх ще не бачив, тож
        чекати на кооперативну перевірку немає на що.
        """
        with self.db.transaction() as con:
            repo = JobRepo(con)
            rows = con.execute(
                "SELECT id, state FROM jobs WHERE document_id=?"
                " AND state IN ('QUEUED','LEASED','RUNNING')",
                (document_id,),
            ).fetchall()
            for row in rows:
                repo.request_cancel(row["id"])
                if row["state"] == "QUEUED":
                    repo.finish(row["id"], JobState.CANCELLED, error_code="CANCELLED")
            return len(rows)

    def reclaim_expired(self) -> int:
        with self.db.transaction() as con:
            return JobRepo(con).reclaim_expired()

    # ------------------------------------------------------------ читання
    def active(self) -> list[dict[str, Any]]:
        with self.db.connection() as con:
            return JobRepo(con).active()

    def for_document(self, document_id: str) -> list[dict[str, Any]]:
        with self.db.connection() as con:
            return JobRepo(con).for_document(document_id)

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        with self.db.connection() as con:
            return [
                dict(r)
                for r in con.execute(
                    "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
                )
            ]

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self.db.connection() as con:
            row = con.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            return dict(row) if row else None

    # -------------------------------------------------- посторінковий стан
    def init_pages(self, document_id: str, page_numbers: Iterable[int]) -> None:
        with self.db.transaction() as con:
            JobRepo(con).init_pages(document_id, page_numbers)

    def mark_page(
        self,
        document_id: str,
        page_number: int,
        state: str,
        *,
        page_hash: str | None = None,
        error_code: str | None = None,
    ) -> None:
        with self.db.transaction() as con:
            JobRepo(con).mark_page(
                document_id, page_number, state, page_hash=page_hash, error_code=error_code
            )

    def mark_pages(self, document_id: str, page_numbers: Sequence[int], state: str) -> None:
        """Пачкою: 25 сторінок вікна — це 25 транзакцій, якщо робити поштучно."""
        if not page_numbers:
            return
        with self.db.transaction() as con:
            repo = JobRepo(con)
            for number in page_numbers:
                repo.mark_page(document_id, number, state)

    def pending_pages(self, document_id: str) -> list[int]:
        with self.db.connection() as con:
            return JobRepo(con).pending_pages(document_id)

    def done_pages(self, document_id: str) -> set[int]:
        with self.db.connection() as con:
            return {
                int(r[0])
                for r in con.execute(
                    "SELECT page_number FROM document_pages_state"
                    " WHERE document_id=? AND state='DONE'",
                    (document_id,),
                )
            }
