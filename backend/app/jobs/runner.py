"""Цикл обробки завдань: спільний для процесу-воркера і для inline-режиму.

Один клас `JobRunner` виконує рівно одну річ — бере завдання, виконує його,
чесно закриває. Він нічого не знає ні про процеси, ні про asyncio, тому та
сама логіка працює:

  * у процесі-воркері постачання (`app/worker.py`), де поруч живуть torch і
    docling;
  * у пулі потоків API-процесу (`worker_mode="inline"`) для розробки, CI і
    stub-режиму, де важких імпортів немає взагалі.

ЩО СТАЄТЬСЯ З ПОМИЛКАМИ
Завдання ніколи не «зникає». Три результати, і всі три видимі:
  * `JobCancelled`  → jobs.CANCELLED + documents.CANCELLED, подія не потрібна:
    скасування ініціював сам користувач і він уже його бачить;
  * `PipelineError` → jobs.FAILED з КОДОМ і УКРАЇНСЬКОЮ підказкою, подія
    `job.failed`, documents.FAILED. Код потрібен UI, підказка — викладачеві;
  * будь-що інше    → те саме, але з кодом `UNEXPECTED` і повним traceback у
    лог. Мовчазне падіння воркера — найгірший можливий стан цієї системи:
    документ вічно висить у «Обробляється».
"""

from __future__ import annotations

import logging
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

from app.config import Paths
from app.db.repositories import DocumentRepo
from app.domain import DocStatus, JobState, JobType
from app.jobs.pipeline import (
    IngestReport,
    PipelineError,
    build_collection_index_job,
    ingest_document,
)
from app.jobs.queue import ClaimedJob, JobCancelled, JobQueue, Lease
from app.settings import Settings

__all__ = ["JobRunner", "make_provider"]

log = logging.getLogger("asistent.worker")

EmitFn = Callable[[str, dict[str, Any]], None]


def make_provider(settings: Settings) -> Any:
    """Провайдер ембедингів для воркера.

    Два аргументи передаються ЯВНО, і обидва — не косметика. `stub` віддав би
    рішення змінній `ASISTENT_STUB`, а середовище воркер-процесу може
    відрізнятись від середовища API. `models_dir` без явної передачі
    резолвиться через `Paths.resolve()` БЕЗ нашого каталогу даних — тобто
    застосунок із власним `ASISTENT_DATA_DIR` шукав би ваги в типовому
    місці й не знаходив би їх.
    """
    from app.embeddings.provider import create_provider

    return create_provider(
        settings.embedding_model, stub=settings.stub, models_dir=settings.paths().models_dir
    )


class JobRunner:
    """Виконавець завдань. Не має власного циклу — цикл будує той, хто його
    використовує (процес-воркер або inline-задача API)."""

    def __init__(
        self,
        db: Any,
        settings: Settings,
        *,
        paths: Paths | None = None,
        provider: Any | None = None,
        emit: EmitFn | None = None,
    ) -> None:
        self.db = db
        self.settings = settings
        self.paths = paths or settings.paths()
        self.queue = JobQueue(db)
        self.emit: EmitFn = emit or (lambda _t, _d: None)
        self._provider = provider
        self.jobs_done = 0

    # ------------------------------------------------------------ ресурси
    @property
    def provider(self) -> Any:
        if self._provider is None:
            self._provider = make_provider(self.settings)
        return self._provider

    # ------------------------------------------------------------ головне
    def run_once(self) -> bool:
        """Взяти й виконати одне завдання. False → черга порожня."""
        job = self.queue.claim()
        if job is None:
            return False
        self.execute(job)
        self.jobs_done += 1
        return True

    def execute(self, job: ClaimedJob) -> None:
        lease = self.queue.lease(job, on_progress=self._on_progress)
        try:
            with lease:
                if job.type is JobType.INDEX and job.collection_id and not job.document_id:
                    self._run_index(job, lease)
                else:
                    self._run_document(job, lease)
            self.queue.finish(job.id, JobState.DONE)
        except JobCancelled:
            log.info("Завдання %s скасовано.", job.id)
            self.queue.finish(job.id, JobState.CANCELLED, error_code="CANCELLED")
            if job.document_id:
                self._set_document(job.document_id, DocStatus.CANCELLED, code="CANCELLED")
        except PipelineError as exc:
            log.warning("Завдання %s провалилось [%s]: %s", job.id, exc.code, exc)
            self._fail(job, exc.code, str(exc), exc.hint)
        except Exception as exc:
            detail = traceback.format_exc(limit=12)
            log.error("Завдання %s впало неочікувано: %s\n%s", job.id, exc, detail)
            self._fail(
                job,
                "UNEXPECTED",
                f"Неочікувана помилка обробки: {exc}",
                "Надішліть діагностичний пакет розробнику — у ньому є повний трейс.",
                detail=detail,
            )

    # ------------------------------------------------------------ гілки
    def _run_document(self, job: ClaimedJob, lease: Lease) -> IngestReport | None:
        if not job.document_id:
            raise PipelineError(
                "NO_DOCUMENT",
                f"Завдання {job.id} типу {job.type.value} не має документа.",
                "Це помилка постановки завдання; завдання знято.",
            )
        return ingest_document(
            self.db,
            self.queue,
            lease,
            document_id=job.document_id,
            paths=self.paths,
            provider=self.provider,
            device=self.settings.ingest_device,
            emit=self.emit,
        )

    def _run_index(self, job: ClaimedJob, lease: Lease) -> None:
        assert job.collection_id is not None
        result = build_collection_index_job(
            self.db, job.collection_id, index_dir=self.paths.index_dir, lease=lease
        )
        log.info("Індекс колекції %s зібрано: %s", job.collection_id, result)

    # ------------------------------------------------------------ службове
    def _on_progress(self, lease: Lease) -> None:
        tracker = lease.tracker
        self.emit(
            "job.progress",
            {
                "jobId": lease.job.id,
                "docId": lease.job.document_id,
                "collectionId": lease.job.collection_id,
                "stage": lease.stage or lease.job.type.value,
                "current": round(tracker.done, 3),
                "total": round(tracker.total, 3),
                "fraction": round(tracker.fraction, 4),
                "etaSeconds": tracker.eta_seconds,
            },
        )

    def _fail(
        self,
        job: ClaimedJob,
        code: str,
        message: str,
        hint: str = "",
        *,
        detail: str | None = None,
    ) -> None:
        self.queue.finish(
            job.id, JobState.FAILED, error_code=code, error_detail=detail or message
        )
        if job.document_id:
            self._set_document(job.document_id, DocStatus.FAILED, code=code, detail=message)
        self.emit(
            "job.failed",
            {
                "jobId": job.id,
                "docId": job.document_id,
                "errorCode": code,
                "message": message,
                "hint": hint,
            },
        )

    def _set_document(
        self, document_id: str, status: DocStatus, *, code: str | None = None,
        detail: str | None = None,
    ) -> None:
        try:
            with self.db.transaction() as con:
                DocumentRepo(con).set_status(
                    document_id, status, error_code=code, error_detail=detail
                )
        except Exception:
            log.exception("Не вдалося оновити статус документа %s", document_id)


def open_database(settings: Settings) -> Any:
    """З'єднання воркера з тією самою БД, що й в API.

    WAL дає рівно потрібну топологію: один письменник плюс N читачів — це і є
    «воркер індексації + потік запитів».
    """
    from app.db.database import Database

    paths = settings.paths()
    return Database(paths.db_path)


def index_dir_for(settings: Settings) -> Path:
    return settings.paths().index_dir
