"""Конвеєр приймання й виконавець завдань (PROBE → PARSE → CHUNK → INDEX)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from app.db.database import Database
from app.db.repositories import (
    AssistantRepo,
    ChunkRepo,
    CollectionRepo,
    DocumentRepo,
    PageRepo,
)
from app.domain import (
    Assistant,
    ChunkLevel,
    Collection,
    DocStatus,
    Document,
    IngestMode,
    JobType,
    new_id,
)
from app.jobs.pipeline import (
    PipelineError,
    StageWeights,
    content_sha256,
    resolve_document_path,
)
from app.jobs.queue import JobCancelled, JobQueue
from app.jobs.runner import JobRunner
from app.settings import Settings
from tests.helpers_api import SAMPLE_TEXT, make_settings


class _Harness:
    """Мінімальний стенд: БД, колекція, документ на диску, воркер."""

    def __init__(self, tmp_path: Path, **overrides: Any) -> None:
        self.settings: Settings = make_settings(tmp_path, **overrides)
        self.paths = self.settings.paths()
        self.db = Database(self.paths.db_path)
        self.queue = JobQueue(self.db)
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.runner = JobRunner(
            self.db, self.settings, paths=self.paths,
            emit=lambda t, d: self.events.append((t, d)),
        )
        with self.db.transaction() as con:
            assistant = Assistant(id=new_id(), name="Артилерія")
            AssistantRepo(con).create(assistant)
            from app.embeddings import registry

            model = registry.get(None)
            self.collection = Collection(
                id=new_id(), assistant_id=assistant.id, name="Основна",
                embedding_model_id=model.id,
                embedding_model_key=registry.model_key(model), dim=model.dim,
            )
            CollectionRepo(con).create(self.collection)

    def add_document(self, text: str = SAMPLE_TEXT, *, name: str = "b.txt",
                     mode: IngestMode = IngestMode.FAST) -> Document:
        document_id = new_id()
        stored = f"{document_id}.txt"
        path = self.paths.documents_dir / stored
        path.write_text(text, encoding="utf-8")
        document = Document(
            id=document_id, collection_id=self.collection.id, title=Path(name).stem,
            original_name=name, stored_path=stored,
            content_sha256=content_sha256(path), ingest_mode=mode,
        )
        with self.db.transaction() as con:
            DocumentRepo(con).create(document)
        self.queue.enqueue(JobType.PARSE, document_id=document_id,
                           collection_id=self.collection.id)
        return document

    def drain(self, limit: int = 10) -> int:
        done = 0
        while done < limit and self.runner.run_once():
            done += 1
        return done

    def document(self, document_id: str) -> Document:
        with self.db.connection() as con:
            doc = DocumentRepo(con).get(document_id)
        assert doc is not None
        return doc

    def emitted(self, event_type: str) -> list[dict[str, Any]]:
        return [d for t, d in self.events if t == event_type]


# ------------------------------------------------------------- ваги етапів
def test_stage_weights_reserve_room_for_chunk_and_index() -> None:
    """Якби бюджет прогресу покривав лише парсинг, індикатор доходив би до
    100% і ще хвилину стояв на ембедингах — класичний «застряглий» бар."""
    weights = StageWeights(pages=100.0)
    assert weights.parse_end == 100.0
    assert weights.chunk_end == pytest.approx(108.0)
    assert weights.total == pytest.approx(133.0)
    assert weights.total > weights.parse_end


# --------------------------------------------------------------- конвеєр
def test_full_pipeline_makes_document_ready(tmp_path) -> None:
    harness = _Harness(tmp_path)
    document = harness.add_document()
    assert harness.drain() >= 1

    stored = harness.document(document.id)
    assert stored.status is DocStatus.READY
    assert stored.page_count >= 1
    assert stored.parse_profile_hash                      # кеш-ключ проставлено
    assert stored.quality_grade is not None

    with harness.db.connection() as con:
        repo = ChunkRepo(con)
        leaves = repo.count(harness.collection.id, ChunkLevel.LEAF)
        sections = repo.count(harness.collection.id, ChunkLevel.SECTION)
        cards = repo.count(harness.collection.id, ChunkLevel.DOCUMENT_CARD)
        embedded = con.execute(
            "SELECT count(*) FROM chunks WHERE document_id=? AND level='L2'"
            " AND embedding IS NOT NULL", (document.id,)
        ).fetchone()[0]
        fts = con.execute(
            "SELECT count(*) FROM chunk_fts JOIN chunks c ON c.id = chunk_fts.rowid"
            " WHERE c.document_id=?", (document.id,)
        ).fetchone()[0]
    assert cards == 1 and sections >= 1 and leaves >= 1
    # Індексуються ЛИШЕ листки: батьки зберігаються для auto-merge.
    assert embedded == leaves
    assert fts == leaves


def test_parent_links_survive_insert(tmp_path) -> None:
    """`parent_id` — INTEGER, і він існує лише ПІСЛЯ вставки. Якщо цей крок
    забути, auto-merge до L1-батька мовчки перестане працювати назавжди."""
    harness = _Harness(tmp_path)
    document = harness.add_document()
    harness.drain()
    with harness.db.connection() as con:
        rows = con.execute(
            "SELECT level, parent_id FROM chunks WHERE document_id=?", (document.id,)
        ).fetchall()
    leaves = [r for r in rows if r["level"] == "L2"]
    assert leaves and all(r["parent_id"] is not None for r in leaves)
    sections = [r for r in rows if r["level"] == "L1"]
    assert sections and all(r["parent_id"] is not None for r in sections)


def test_pages_are_persisted_with_labels(tmp_path) -> None:
    harness = _Harness(tmp_path)
    document = harness.add_document()
    harness.drain()
    with harness.db.connection() as con:
        pages = PageRepo(con).for_document(document.id)
    assert pages
    assert all(p.page_number >= 1 for p in pages)
    assert all(p.cost_weight > 0 for p in pages)


def test_doc_ready_event_carries_counts(tmp_path) -> None:
    harness = _Harness(tmp_path)
    document = harness.add_document()
    harness.drain()
    ready = harness.emitted("doc.ready")
    assert len(ready) == 1
    assert ready[0]["docId"] == document.id
    assert ready[0]["chunks"] >= 1


def test_progress_events_are_weighted_and_monotonic(tmp_path) -> None:
    harness = _Harness(tmp_path)
    harness.add_document()
    harness.drain()
    progress = harness.emitted("job.progress")
    assert progress, "конвеєр не звітував жодного прогресу"

    # Монотонність перевіряється В МЕЖАХ завдання: після документа йде окреме
    # завдання збірки індексу, і його прогрес починається з нуля законно.
    by_job: dict[str, list[float]] = {}
    for row in progress:
        by_job.setdefault(row["jobId"], []).append(row["fraction"])
    for job_id, fractions in by_job.items():
        assert fractions == sorted(fractions), f"прогрес завдання {job_id} відкотився назад"

    document_job = next(
        rows for rows in by_job.values() if len(rows) > 2
    )
    assert document_job[-1] == pytest.approx(1.0)
    stages = [p["stage"] for p in progress]
    for expected in ("PROBE", "PARSE", "CHUNK", "INDEX"):
        assert expected in stages, f"етап {expected} не з'явився у прогресі: {stages}"


def test_index_job_is_enqueued_for_the_collection(tmp_path) -> None:
    harness = _Harness(tmp_path)
    harness.add_document()
    harness.runner.run_once()                       # лише завдання документа
    active = [j for j in harness.queue.active() if j["type"] == "INDEX"]
    assert len(active) == 1
    assert active[0]["collection_id"] == harness.collection.id


def test_index_job_builds_the_ann_index(tmp_path) -> None:
    harness = _Harness(tmp_path)
    harness.add_document()
    harness.drain()
    with harness.db.connection() as con:
        collection = CollectionRepo(con).get(harness.collection.id)
    assert collection is not None
    assert collection.index_generation >= 1
    assert collection.dirty is False


# ------------------------------------------------------------- скасування
def test_cancellation_marks_document_cancelled(tmp_path) -> None:
    harness = _Harness(tmp_path)
    document = harness.add_document()
    job = harness.queue.claim()
    assert job is not None
    harness.queue.request_cancel(job.id)
    harness.runner.execute(job)

    assert harness.document(document.id).status is DocStatus.CANCELLED
    assert harness.queue.get(job.id)["state"] == "CANCELLED"
    # Скасування — це запитаний результат, а не збій: події про помилку немає.
    assert harness.emitted("job.failed") == []


# ----------------------------------------------------------------- помилки
def test_missing_file_fails_with_code_and_hint(tmp_path) -> None:
    harness = _Harness(tmp_path)
    document = harness.add_document()
    resolve_document_path(harness.paths, document).unlink()
    harness.drain()

    stored = harness.document(document.id)
    assert stored.status is DocStatus.FAILED
    assert stored.error_code == "FILE_MISSING"
    failed = harness.emitted("job.failed")
    assert failed and failed[0]["errorCode"] == "FILE_MISSING"
    # Підказка українською — це те, що викладач реально прочитає.
    assert "Завантажте файл ще раз" in failed[0]["hint"]


def test_empty_document_fails_with_no_text(tmp_path) -> None:
    """Скан без текстового шару, де OCR не спрацював, мусить давати ЗРОЗУМІЛУ
    помилку, а не порожній READY-документ, який нічого не знаходить."""
    harness = _Harness(tmp_path)
    document = harness.add_document(text="   \n\n  \n")
    harness.drain()
    stored = harness.document(document.id)
    assert stored.status is DocStatus.FAILED
    assert stored.error_code == "NO_TEXT"


def test_unexpected_error_is_reported_not_swallowed(tmp_path, monkeypatch) -> None:
    """Мовчазне падіння воркера — найгірший стан системи: документ вічно
    висить у «Обробляється»."""
    harness = _Harness(tmp_path)
    document = harness.add_document()

    def boom(*_a: Any, **_kw: Any) -> None:
        raise RuntimeError("несподівано")

    monkeypatch.setattr("app.jobs.runner.ingest_document", boom)
    harness.drain()
    stored = harness.document(document.id)
    assert stored.status is DocStatus.FAILED
    assert stored.error_code == "UNEXPECTED"
    assert harness.queue.get(harness.queue.for_document(document.id)[0]["id"])["state"] == "FAILED"


def test_pipeline_error_carries_ukrainian_message() -> None:
    error = PipelineError("X", "Повідомлення", "Підказка")
    assert error.code == "X"
    assert str(error) == "Повідомлення"
    assert error.hint == "Підказка"


# -------------------------------------------------------------------- кеш
def test_second_copy_reuses_parsed_artifacts(tmp_path) -> None:
    """Той самий підручник у другого асистента — миттєво.

    Це і є те, що робить багатоасистентність практичною: без кешу додавання
    спільного матеріалу до другої колекції коштувало б повторних годин CPU.
    """
    harness = _Harness(tmp_path)
    first = harness.add_document(name="перший.txt")
    harness.drain()
    assert harness.document(first.id).status is DocStatus.READY

    # Друга колекція (інший асистент), той самий файл байт-у-байт.
    with harness.db.transaction() as con:
        assistant = Assistant(id=new_id(), name="Другий")
        AssistantRepo(con).create(assistant)
        second_collection = Collection(
            id=new_id(), assistant_id=assistant.id, name="Основна",
            embedding_model_id=harness.collection.embedding_model_id,
            embedding_model_key=harness.collection.embedding_model_key,
            dim=harness.collection.dim,
        )
        CollectionRepo(con).create(second_collection)

    document_id = new_id()
    stored = f"{document_id}.txt"
    (harness.paths.documents_dir / stored).write_text(SAMPLE_TEXT, encoding="utf-8")
    duplicate = Document(
        id=document_id, collection_id=second_collection.id, title="перший",
        original_name="перший.txt", stored_path=stored,
        content_sha256=first.content_sha256,
    )
    with harness.db.transaction() as con:
        DocumentRepo(con).create(duplicate)
    harness.queue.enqueue(JobType.PARSE, document_id=document_id,
                          collection_id=second_collection.id)
    harness.drain()

    assert harness.document(document_id).status is DocStatus.READY
    ready = [e for e in harness.emitted("doc.ready") if e["docId"] == document_id]
    assert ready and ready[0]["reusedFrom"] == first.id

    with harness.db.connection() as con:
        original_leaves = ChunkRepo(con).count(harness.collection.id, ChunkLevel.LEAF)
        copied_leaves = ChunkRepo(con).count(second_collection.id, ChunkLevel.LEAF)
        # Вектори скопійовані, а не перераховані заново — але вони Є.
        embedded = con.execute(
            "SELECT count(*) FROM chunks WHERE collection_id=? AND level='L2'"
            " AND embedding IS NOT NULL", (second_collection.id,)
        ).fetchone()[0]
        # `chunk_uid` рахується від document_id, тож копії не колідують.
        uids = con.execute(
            "SELECT count(DISTINCT chunk_uid) FROM chunks WHERE document_id IN (?,?)",
            (first.id, document_id),
        ).fetchone()[0]
        total = con.execute(
            "SELECT count(*) FROM chunks WHERE document_id IN (?,?)",
            (first.id, document_id),
        ).fetchone()[0]
    assert copied_leaves == original_leaves
    assert embedded == copied_leaves
    assert uids == total


def test_content_sha256_is_streamed(tmp_path) -> None:
    """300-мегабайтний підручник не має жити в RAM."""
    import hashlib

    path = tmp_path / "big.bin"
    payload = b"\xff" * (3 * 1024 * 1024)
    path.write_bytes(payload)
    assert content_sha256(path, chunk=64 * 1024) == hashlib.sha256(payload).hexdigest()


def test_resolve_document_path_prefers_relative(tmp_path) -> None:
    """Каталог даних можна перенести на інший диск (закритий контур,
    встановлення з USB); абсолютний шлях у БД перетворив би це на масову
    втрату файлів."""
    settings = make_settings(tmp_path)
    paths = settings.paths()
    document = Document(
        id="d", collection_id="c", title="t", original_name="t.pdf",
        stored_path="d.pdf", content_sha256="s",
    )
    assert resolve_document_path(paths, document) == (paths.documents_dir / "d.pdf").resolve()

    absolute = tmp_path / "somewhere" / "t.pdf"
    document.stored_path = str(absolute)
    assert resolve_document_path(paths, document) == absolute


def test_deep_mode_records_honest_warning(tmp_path) -> None:
    """Рівні 1.5/2 потребують перемикання LM Studio на VLM. Поки цього немає,
    режим мусить сказати про це прямо, а не тихо працювати як FAST."""
    harness = _Harness(tmp_path)
    document = harness.add_document(mode=IngestMode.DEEP)
    harness.drain()
    assert harness.document(document.id).status is DocStatus.READY
    ready = harness.emitted("doc.ready")
    assert any("VLM" in w for w in ready[0]["warnings"])


def test_cancelled_lease_stops_between_pages(tmp_path) -> None:
    """Найгірша затримка скасування — одна сторінка, бо перевірка стоїть на
    межі сторінки, а не всередині C-виклику Docling."""
    harness = _Harness(tmp_path)
    harness.add_document()
    job = harness.queue.claim()
    assert job is not None
    lease = harness.queue.lease(job, total_weight=10.0)
    harness.queue.request_cancel(job.id)
    with pytest.raises(JobCancelled):
        lease.checkpoint(stage="PARSE", weight=1.0, force=True)
