"""Черга завдань: лізинг, heartbeat, скасування, прогрес, ETA, сторінки."""

from __future__ import annotations

import time

import pytest

from app.db.database import Database
from app.db.repositories import AssistantRepo, CollectionRepo, DocumentRepo, JobRepo
from app.domain import Assistant, Collection, DocStatus, Document, JobState, JobType, new_id
from app.jobs.queue import (
    HEARTBEAT_INTERVAL_S,
    JobCancelled,
    JobQueue,
    ProgressTracker,
)


def _fixture(tmp_path) -> tuple[Database, str, str]:
    db = Database(tmp_path / "assistant.db")
    with db.transaction() as con:
        assistant = Assistant(id=new_id(), name="А")
        AssistantRepo(con).create(assistant)
        collection = Collection(
            id=new_id(), assistant_id=assistant.id, name="Основна",
            embedding_model_id="m", embedding_model_key="k", dim=8,
        )
        CollectionRepo(con).create(collection)
        document = Document(
            id=new_id(), collection_id=collection.id, title="Підручник",
            original_name="p.pdf", stored_path="p.pdf", content_sha256="s",
            status=DocStatus.QUEUED,
        )
        DocumentRepo(con).create(document)
    return db, collection.id, document.id


# --------------------------------------------------------------- прогрес
def test_tracker_reports_fraction_and_eta() -> None:
    tracker = ProgressTracker(total_weight=100.0)
    assert tracker.eta_seconds is None          # без жодного виміру — чесне «невідомо»
    for _ in range(4):
        time.sleep(0.01)
        tracker.advance(10.0)
    assert tracker.done == pytest.approx(40.0)
    assert 0.39 < tracker.fraction < 0.41
    assert tracker.eta_seconds is not None and tracker.eta_seconds > 0


def test_eta_uses_median_not_mean() -> None:
    """Розподіл вартості сторінок важкохвостий: одна сторінка з великою
    таблицею коштує як двадцять текстових. Середнє на ній стрибає, медіана —
    ні, і саме тому прогрес-бар не показує «залишилось 7 хвилин» одразу
    після «залишилось 20 секунд»."""
    tracker = ProgressTracker(total_weight=1000.0, window=8)
    # Сім швидких сторінок і одна аномально повільна.
    for _ in range(7):
        tracker._samples.append(0.1)
    tracker._samples.append(20.0)
    tracker.done = 900.0
    rate = tracker.seconds_per_weight
    assert rate is not None and rate < 1.0      # медіана ≈ 0.1, середнє було б ≈ 2.6
    assert tracker.eta_seconds == pytest.approx(10.0, abs=1.0)


def test_set_done_is_absolute_not_delta() -> None:
    """Колбек парсера звітує АБСОЛЮТНУ вагу; якби ми трактували її як
    приріст, прогрес добіг би до 100% на четвертій сторінці."""
    tracker = ProgressTracker(total_weight=50.0)
    tracker.set_done(10.0)
    tracker.set_done(20.0)
    tracker.set_done(20.0)                      # повтор не має рухати прогрес
    assert tracker.done == pytest.approx(20.0)


def test_progress_never_exceeds_total() -> None:
    tracker = ProgressTracker(total_weight=10.0)
    tracker.advance(100.0)
    assert tracker.done == 10.0
    assert tracker.fraction == 1.0


# ---------------------------------------------------------------- черга
def test_enqueue_and_claim_roundtrip(tmp_path) -> None:
    db, collection_id, document_id = _fixture(tmp_path)
    queue = JobQueue(db)
    job_id = queue.enqueue(JobType.PARSE, document_id=document_id, collection_id=collection_id)

    claimed = queue.claim()
    assert claimed is not None
    assert claimed.id == job_id
    assert claimed.type is JobType.PARSE
    assert claimed.document_id == document_id
    # Друге взяття не дає нічого: завдання вже під лізингом.
    assert queue.claim() is None


def test_enqueue_once_deduplicates_collection_jobs(tmp_path) -> None:
    """Збірка ANN-графа — операція на ВСЮ колекцію. Ставити її після кожного
    з двадцяти підручників означало б двадцять збірок замість однієї."""
    db, collection_id, _ = _fixture(tmp_path)
    queue = JobQueue(db)
    first = queue.enqueue_once(JobType.INDEX, collection_id=collection_id)
    second = queue.enqueue_once(JobType.INDEX, collection_id=collection_id)
    assert first is not None
    assert second is None

    queue.finish(first, JobState.DONE)
    third = queue.enqueue_once(JobType.INDEX, collection_id=collection_id)
    assert third is not None            # після завершення — знову можна


def test_heartbeat_returns_false_after_cancel(tmp_path) -> None:
    db, collection_id, document_id = _fixture(tmp_path)
    queue = JobQueue(db)
    job_id = queue.enqueue(JobType.PARSE, document_id=document_id, collection_id=collection_id)
    job = queue.claim()
    assert job is not None
    assert queue.heartbeat(job_id) is True
    queue.request_cancel(job_id)
    assert queue.heartbeat(job_id) is False


def test_checkpoint_raises_job_cancelled(tmp_path) -> None:
    db, collection_id, document_id = _fixture(tmp_path)
    queue = JobQueue(db)
    queue.enqueue(JobType.PARSE, document_id=document_id, collection_id=collection_id)
    job = queue.claim()
    assert job is not None
    with queue.lease(job, total_weight=10.0) as lease:
        lease.checkpoint(stage="PARSE", weight=1.0, force=True)
        queue.request_cancel(job.id)
        with pytest.raises(JobCancelled):
            lease.checkpoint(stage="PARSE", weight=1.0, force=True)
        assert lease.is_cancelled()


def test_poll_cancel_does_not_raise(tmp_path) -> None:
    """Колбек `ParseOptions.cancel` викликається ВСЕРЕДИНІ циклу вікон
    Docling: виняток звідти лишив би напівзібраний документ без шансу на
    впорядковане завершення."""
    db, collection_id, document_id = _fixture(tmp_path)
    queue = JobQueue(db)
    queue.enqueue(JobType.PARSE, document_id=document_id, collection_id=collection_id)
    job = queue.claim()
    assert job is not None
    lease = queue.lease(job, total_weight=5.0)
    assert lease.poll_cancel() is False
    queue.request_cancel(job.id)
    assert lease.poll_cancel() is True          # без винятку


def test_heartbeat_is_throttled(tmp_path) -> None:
    """Удар серця раз на 2 с, а не на кожну сторінку: на 1000-сторінковому
    підручнику це різниця між 500 і 250 000 транзакціями."""
    db, collection_id, document_id = _fixture(tmp_path)
    queue = JobQueue(db)
    queue.enqueue(JobType.PARSE, document_id=document_id, collection_id=collection_id)
    job = queue.claim()
    assert job is not None

    beats = 0
    original = queue.heartbeat

    def counting(job_id: str) -> bool:
        nonlocal beats
        beats += 1
        return original(job_id)

    queue.heartbeat = counting  # type: ignore[method-assign]
    with queue.lease(job, total_weight=100.0) as lease:
        for _ in range(50):
            lease.checkpoint(stage="PARSE", weight=1.0)
    assert beats <= 2, f"очікували не більше двох ударів за {HEARTBEAT_INTERVAL_S} с, було {beats}"


def test_reclaim_expired_returns_job_to_queue(tmp_path) -> None:
    """Воркер упав посеред сторінки. Без повернення лізингу документ висів би
    «Обробляється» назавжди."""
    db, collection_id, document_id = _fixture(tmp_path)
    queue = JobQueue(db)
    job_id = queue.enqueue(JobType.PARSE, document_id=document_id, collection_id=collection_id)
    assert queue.claim() is not None

    with db.transaction() as con:
        con.execute(
            "UPDATE jobs SET lease_until=datetime('now','-1 hour') WHERE id=?", (job_id,)
        )
    assert queue.reclaim_expired() == 1
    again = queue.claim()
    assert again is not None and again.id == job_id
    assert again.attempts == 1


def test_reclaim_fails_job_after_max_attempts(tmp_path) -> None:
    db, collection_id, document_id = _fixture(tmp_path)
    queue = JobQueue(db)
    job_id = queue.enqueue(JobType.PARSE, document_id=document_id, collection_id=collection_id)
    with db.transaction() as con:
        con.execute(
            "UPDATE jobs SET state='RUNNING', attempts=3,"
            " lease_until=datetime('now','-1 hour') WHERE id=?", (job_id,)
        )
    queue.reclaim_expired()
    row = queue.get(job_id)
    assert row is not None
    assert row["state"] == "FAILED"
    assert row["error_code"] == "LEASE_EXPIRED"


def test_cancel_document_removes_queued_and_flags_running(tmp_path) -> None:
    db, collection_id, document_id = _fixture(tmp_path)
    queue = JobQueue(db)
    running = queue.enqueue(JobType.PARSE, document_id=document_id, collection_id=collection_id)
    queue.claim()
    queued = queue.enqueue(JobType.CHUNK, document_id=document_id, collection_id=collection_id)

    assert queue.cancel_document(document_id) == 2
    # QUEUED знімається одразу: воркер його ще не бачив, чекати нема на що.
    assert queue.get(queued)["state"] == "CANCELLED"
    # Активне лише позначається — зупиниться на межі сторінки.
    assert queue.get(running)["state"] in ("LEASED", "RUNNING")
    assert queue.get(running)["cancel_requested"] == 1


def test_set_weight_total_after_probe(tmp_path) -> None:
    """Сума ваг відома лише після проби, а завдання створюється до неї."""
    db, collection_id, document_id = _fixture(tmp_path)
    queue = JobQueue(db)
    queue.enqueue(JobType.PARSE, document_id=document_id, collection_id=collection_id)
    job = queue.claim()
    assert job is not None and job.weight_total == 0.0

    lease = queue.lease(job)
    lease.set_total(133.0)
    assert queue.get(job.id)["progress_weight_total"] == pytest.approx(133.0)
    assert lease.tracker.total == pytest.approx(133.0)


# ------------------------------------------------- посторінковий стан
def test_page_state_supports_resume(tmp_path) -> None:
    """1000-сторінковий підручник — 1000 рядків. Падіння на 940-й сторінці
    без цієї таблиці коштує 40 хвилин; із нею — 40 секунд."""
    db, _collection_id, document_id = _fixture(tmp_path)
    queue = JobQueue(db)
    queue.init_pages(document_id, range(1, 11))
    assert queue.pending_pages(document_id) == list(range(1, 11))

    queue.mark_pages(document_id, list(range(1, 8)), "DONE")
    assert queue.pending_pages(document_id) == [8, 9, 10]
    assert queue.done_pages(document_id) == set(range(1, 8))

    # Повторна ініціалізація не скидає вже зроблене.
    queue.init_pages(document_id, range(1, 11))
    assert queue.pending_pages(document_id) == [8, 9, 10]


def test_mark_page_records_error_code(tmp_path) -> None:
    db, _collection_id, document_id = _fixture(tmp_path)
    queue = JobQueue(db)
    queue.init_pages(document_id, [1, 2])
    queue.mark_page(document_id, 2, "FAILED", error_code="OCR_EMPTY")
    with db.connection() as con:
        row = con.execute(
            "SELECT state, error_code FROM document_pages_state"
            " WHERE document_id=? AND page_number=2", (document_id,)
        ).fetchone()
    assert row["state"] == "FAILED"
    assert row["error_code"] == "OCR_EMPTY"


def test_finish_records_error_detail(tmp_path) -> None:
    db, collection_id, document_id = _fixture(tmp_path)
    queue = JobQueue(db)
    job_id = queue.enqueue(JobType.PARSE, document_id=document_id, collection_id=collection_id)
    queue.finish(job_id, JobState.FAILED, error_code="PARSE_FAILED", error_detail="деталі")
    row = queue.get(job_id)
    assert row["state"] == "FAILED"
    assert row["error_code"] == "PARSE_FAILED"
    assert row["error_detail"] == "деталі"


def test_claim_respects_priority(tmp_path) -> None:
    """Переіндексація вручну (priority=50) має випереджати фонову чергу."""
    db, collection_id, document_id = _fixture(tmp_path)
    queue = JobQueue(db)
    queue.enqueue(JobType.PARSE, document_id=document_id, collection_id=collection_id,
                  priority=100)
    urgent = queue.enqueue(JobType.PARSE, document_id=document_id,
                           collection_id=collection_id, priority=10)
    job = queue.claim()
    assert job is not None and job.id == urgent


def test_claim_filters_by_type(tmp_path) -> None:
    db, collection_id, document_id = _fixture(tmp_path)
    queue = JobQueue(db)
    queue.enqueue(JobType.PARSE, document_id=document_id, collection_id=collection_id)
    index_job = queue.enqueue(JobType.INDEX, collection_id=collection_id)
    job = queue.claim([JobType.INDEX])
    assert job is not None and job.id == index_job


def test_cancelled_job_is_not_claimed(tmp_path) -> None:
    db, collection_id, document_id = _fixture(tmp_path)
    queue = JobQueue(db)
    job_id = queue.enqueue(JobType.PARSE, document_id=document_id, collection_id=collection_id)
    with db.transaction() as con:
        JobRepo(con).request_cancel(job_id)
    assert queue.claim() is None
