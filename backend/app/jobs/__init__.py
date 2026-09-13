"""Черга завдань, конвеєр приймання і нагляд за воркерами.

Публічний інтерфейс модуля:
    JobQueue      — лізинг, heartbeat, кооперативне скасування, посторінковий стан
    Lease         — оренда завдання з `checkpoint()` між сторінками
    JobRunner     — виконання одного завдання (спільне для процесу й inline)
    ingest_document(...) -> IngestReport  — конвеєр PROBE→PARSE→CHUNK→INDEX
    create_supervisor(...) -> Supervisor  — процеси або inline-задача

Імпорти піднято сюди свідомо: жоден із цих модулів не тягне torch чи docling
на верхньому рівні (важкі залежності лінива всередині `app.ingestion`), тож
API-процес може імпортувати пакет, не платячи за старт.
"""

from __future__ import annotations

from app.jobs.pipeline import (
    IngestReport,
    PipelineError,
    build_collection_index_job,
    content_sha256,
    ingest_document,
    resolve_document_path,
    reuse_cached_document,
)
from app.jobs.queue import (
    HEARTBEAT_INTERVAL_S,
    ClaimedJob,
    JobCancelled,
    JobQueue,
    Lease,
    ProgressTracker,
)
from app.jobs.runner import JobRunner
from app.jobs.supervisor import create_supervisor

__all__ = [
    "HEARTBEAT_INTERVAL_S", "ClaimedJob", "JobCancelled", "JobQueue", "Lease",
    "ProgressTracker", "JobRunner", "IngestReport", "PipelineError",
    "ingest_document", "build_collection_index_job", "resolve_document_path",
    "reuse_cached_document", "content_sha256", "create_supervisor",
]
