"""Міст «воркер-процес → SSE».

Воркер — окремий процес, тож він фізично не може покласти подію в шину подій
API. Єдине сховище, спільне для обох, — SQLite, і саме воно є джерелом істини
прогресу (таблиці `jobs` і `documents`). Спостерігач опитує їх раз на пів
секунди й перетворює зміни на ті самі типізовані події, які inline-воркер
публікує напряму. Для фронтенду різниці немає — і це навмисно: режим воркера
не має протікати в UI.

ЧОМУ ОПИТУВАННЯ, А НЕ ЗВОРОТНИЙ HTTP-ВИКЛИК ІЗ ВОРКЕРА
Зворотний виклик зробив би воркер залежним від живого API (а він переживає
його перезапуск), додав би HTTP-клієнт у процес, де його не має бути, і
загубив би події, що сталися під час обриву. Опитування таблиці, яка вже є
джерелом істини, не має жодної з цих вад, а піврічної затримки ніхто не
помітить: сторінка індексується 0.4–0.7 с.

ETA рахується ТУТ, бо в БД його немає: беремо швидкість набору ваги між двома
опитуваннями, згладжену експоненційно (α=0.3). Без згладжування ETA стрибав би
від 12 с до 400 с на кожній табличній сторінці.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any

from app.api.events import DOC_READY, JOB_FAILED, JOB_PROGRESS
from app.api.state import Services

__all__ = ["JobWatcher", "document_ready_payload"]

log = logging.getLogger("asistent.watcher")

SMOOTHING = 0.3


@dataclass(slots=True)
class _JobSnapshot:
    state: str
    stage: str
    weight_done: float
    weight_total: float
    at: float
    rate: float | None = None


class JobWatcher:
    """Фонова задача API, що озвучує роботу воркер-процесів."""

    def __init__(self, services: Services, *, interval: float | None = None) -> None:
        self.services = services
        self.interval = interval or services.settings.job_watch_interval_s
        self._jobs: dict[str, _JobSnapshot] = {}
        self._doc_status: dict[str, str] = {}
        self._task: asyncio.Task[None] | None = None
        self._stopping = False

    async def start(self) -> None:
        self._stopping = False
        self._prime()
        self._task = asyncio.create_task(self._loop(), name="job-watcher")

    async def stop(self) -> None:
        self._stopping = True
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._task = None

    # ------------------------------------------------------------ цикл
    async def _loop(self) -> None:
        try:
            while not self._stopping:
                await asyncio.sleep(self.interval)
                try:
                    await asyncio.to_thread(self.poll)
                except Exception:  # noqa: BLE001 — спостерігач не валить API
                    log.debug("Помилка опитування черги", exc_info=True)
        except asyncio.CancelledError:  # pragma: no cover
            raise

    def _prime(self) -> None:
        """Зняти початковий стан, щоб не сипати подіями про минуле.

        Без цього кожен перезапуск API виплюнув би `doc.ready` на всі вже
        готові документи, і UI показав би двадцять сповіщень про роботу,
        зроблену вчора.
        """
        for row in self._documents():
            self._doc_status[str(row["id"])] = str(row["status"])

    # ------------------------------------------------------------ опитування
    def poll(self) -> None:
        self._poll_jobs()
        self._poll_documents()

    def _poll_jobs(self) -> None:
        now = time.monotonic()
        seen: set[str] = set()
        for row in self.services.queue.recent(limit=40):
            job_id = str(row["id"])
            seen.add(job_id)
            state = str(row["state"])
            stage = str(row["stage"] or row["type"])
            done = float(row["progress_weight_done"] or 0.0)
            total = float(row["progress_weight_total"] or 0.0)
            prev = self._jobs.get(job_id)

            if prev is not None and prev.state == state and prev.stage == stage \
                    and abs(prev.weight_done - done) < 1e-9:
                continue

            rate = prev.rate if prev else None
            if prev is not None and done > prev.weight_done and now > prev.at:
                sample = (done - prev.weight_done) / (now - prev.at)
                rate = sample if rate is None else (1 - SMOOTHING) * rate + SMOOTHING * sample
            self._jobs[job_id] = _JobSnapshot(state, stage, done, total, now, rate)

            if state in ("QUEUED", "LEASED", "RUNNING"):
                eta = None
                if rate and rate > 0 and total > 0:
                    eta = round(max(0.0, total - done) / rate, 1)
                self.services.emit(JOB_PROGRESS, {
                    "jobId": job_id,
                    "docId": row["document_id"],
                    "collectionId": row["collection_id"],
                    "stage": stage,
                    "current": round(done, 3),
                    "total": round(total, 3),
                    "fraction": round(min(1.0, done / total), 4) if total else 0.0,
                    "etaSeconds": eta,
                })
            elif state == "FAILED" and (prev is None or prev.state != "FAILED"):
                self.services.emit(JOB_FAILED, {
                    "jobId": job_id,
                    "docId": row["document_id"],
                    "errorCode": row["error_code"] or "UNKNOWN",
                    "message": row["error_detail"] or "Обробку не завершено.",
                    "hint": "Відкрийте діагностику документа, щоб побачити деталі.",
                })
        # Прибираємо старі знімки, щоб словник не ріс без меж.
        for stale in [k for k in self._jobs if k not in seen]:
            self._jobs.pop(stale, None)

    def _documents(self) -> list[dict[str, Any]]:
        with self.services.db.connection() as con:
            return [
                dict(r)
                for r in con.execute(
                    "SELECT id, status, collection_id FROM documents"
                    " ORDER BY created_at DESC LIMIT 200"
                )
            ]

    def _poll_documents(self) -> None:
        for row in self._documents():
            doc_id = str(row["id"])
            status = str(row["status"])
            if self._doc_status.get(doc_id) == status:
                continue
            self._doc_status[doc_id] = status
            if status != "READY":
                continue
            self.services.emit(DOC_READY, document_ready_payload(self.services.db, doc_id))


def document_ready_payload(db: Any, document_id: str) -> dict[str, Any]:
    """`doc.ready` з реальними лічильниками таблиць і формул.

    Рахуємо в Python по одному документу, а не через `json_array_length`:
    складання SQLite у python-build-standalone не гарантує розширення JSON1,
    і тиха відмова тут перетворила б лічильники на нулі без жодного сліду.
    """
    tables = formulas = pictures = chunks = 0
    with db.connection() as con:
        for row in con.execute(
            "SELECT tables_json, formulas_json, pictures_json FROM chunks"
            " WHERE document_id=? AND level='L2'",
            (document_id,),
        ):
            chunks += 1
            tables += _count_json(row["tables_json"])
            formulas += _count_json(row["formulas_json"])
            pictures += _count_json(row["pictures_json"])
    return {
        "docId": document_id,
        "chunks": chunks,
        "tables": tables,
        "formulas": formulas,
        "pictures": pictures,
    }


def _count_json(raw: str | None) -> int:
    if not raw:
        return 0
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return 0
    return len(value) if isinstance(value, (list, dict)) else 0
