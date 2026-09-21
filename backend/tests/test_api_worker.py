"""Воркер-процес, наглядач і міст «воркер → SSE»."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from app.api.state import build_services
from app.api.watcher import JobWatcher, document_ready_payload
from app.db.database import Database
from app.db.repositories import AssistantRepo, CollectionRepo, DocumentRepo
from app.domain import Assistant, Collection, DocStatus, Document, JobState, JobType, new_id
from app.jobs.queue import JobQueue
from app.jobs.supervisor import (
    InlineSupervisor,
    NullSupervisor,
    ProcessSupervisor,
    create_supervisor,
)
from app.worker import RECYCLE_EXIT_CODE, run_worker
from tests.helpers_api import SAMPLE_TEXT, make_settings

BACKEND_DIR = Path(__file__).resolve().parents[1]


# --------------------------------------------------- правило №1 контракту
def test_api_process_never_imports_torch_docling_or_rapidocr() -> None:
    """`import torch` коштує 1.5–4 с на Windows.

    В API це означає 6+ секунд до того, як вікно стане чутливим, і викладач
    вирішить, що застосунок зламався. Перевіряємо в ОКРЕМОМУ процесі: у
    поточному ці модулі міг підтягнути будь-який інший тест.
    """
    code = (
        "import sys; import app.main;"
        " bad=[m for m in ('torch','docling','rapidocr','sentence_transformers')"
        " if m in sys.modules];"
        " print(','.join(bad))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=BACKEND_DIR, capture_output=True,
        text=True, timeout=180, env={"PATH": "/usr/bin:/bin", "ASISTENT_STUB": "1",
                                     "HOME": str(BACKEND_DIR)},
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "", f"API-процес підтягнув: {result.stdout.strip()}"


def test_net_guard_is_installed_before_anything_else() -> None:
    """Гард ставиться до будь-якого імпорту, що створює HTTP-клієнти —
    інакше клієнт, збудований на імпорті, лишиться незалатаним."""
    code = (
        "import app.main, httpx;"
        " print(getattr(httpx.AsyncClient.send, '_asistent_guarded', False))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=BACKEND_DIR, capture_output=True,
        text=True, timeout=180, env={"PATH": "/usr/bin:/bin", "ASISTENT_STUB": "1",
                                     "HOME": str(BACKEND_DIR)},
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "True"


# ------------------------------------------------------------- цикл воркера
def _seed(tmp_path: Path) -> tuple[Any, str, str]:
    settings = make_settings(tmp_path)
    paths = settings.paths()
    db = Database(paths.db_path)
    with db.transaction() as con:
        assistant = Assistant(id=new_id(), name="А")
        AssistantRepo(con).create(assistant)
        from app.embeddings import registry

        model = registry.get(None)
        collection = Collection(
            id=new_id(), assistant_id=assistant.id, name="Основна",
            embedding_model_id=model.id, embedding_model_key=registry.model_key(model),
            dim=model.dim,
        )
        CollectionRepo(con).create(collection)
        document_id = new_id()
        (paths.documents_dir / f"{document_id}.txt").write_text(SAMPLE_TEXT, encoding="utf-8")
        DocumentRepo(con).create(Document(
            id=document_id, collection_id=collection.id, title="Балістика",
            original_name="b.txt", stored_path=f"{document_id}.txt",
            content_sha256="s" * 64,
        ))
    JobQueue(db).enqueue(JobType.PARSE, document_id=document_id, collection_id=collection.id)
    return settings, collection.id, document_id


def test_worker_once_drains_the_queue(tmp_path) -> None:
    settings, _collection_id, document_id = _seed(tmp_path)
    assert run_worker(settings, once=True, max_jobs=0) == 0

    db = Database(settings.paths().db_path)
    with db.connection() as con:
        document = DocumentRepo(con).get(document_id)
    assert document is not None and document.status is DocStatus.READY


def test_worker_recycles_after_n_jobs(tmp_path) -> None:
    """Кешувальний алокатор torch не повертає звільнені блоки ОС: воркер
    після сорока документів тримає ~3 ГБ. Переробка процесу — єдиний
    механізм, який повертає пам'ять; для потоків його не існує.
    """
    settings, _collection_id, _document_id = _seed(tmp_path)
    assert run_worker(settings, max_jobs=1) == RECYCLE_EXIT_CODE


def test_worker_exits_on_idle_timeout(tmp_path) -> None:
    settings = make_settings(tmp_path)
    settings.paths()
    assert run_worker(settings, idle_timeout=0.0, max_jobs=0) == 0


def test_worker_module_runs_as_a_subprocess(tmp_path) -> None:
    """`python -m app.worker --once` — саме так його запускає наглядач."""
    settings, _collection_id, document_id = _seed(tmp_path)
    result = subprocess.run(
        [sys.executable, "-X", "utf8", "-m", "app.worker", "--once",
         "--data-dir", str(settings.paths().data_dir), "--log-level", "WARNING"],
        cwd=BACKEND_DIR, capture_output=True, text=True, timeout=300,
        env={"PATH": "/usr/bin:/bin", "ASISTENT_STUB": "1",
             "ASISTENT_DATA_DIR": str(settings.paths().data_dir),
             "HOME": str(BACKEND_DIR)},
    )
    assert result.returncode == 0, result.stderr[-3000:]
    db = Database(settings.paths().db_path)
    with db.connection() as con:
        document = DocumentRepo(con).get(document_id)
    assert document is not None and document.status is DocStatus.READY


# ---------------------------------------------------------------- наглядач
def test_create_supervisor_picks_the_mode(tmp_path) -> None:
    db = Database(make_settings(tmp_path).paths().db_path)
    assert isinstance(
        create_supervisor(make_settings(tmp_path, worker_mode="off"), db=db,
                          emit=lambda *_: None), NullSupervisor)
    assert isinstance(
        create_supervisor(make_settings(tmp_path, worker_mode="inline"), db=db,
                          emit=lambda *_: None), InlineSupervisor)
    assert isinstance(
        create_supervisor(make_settings(tmp_path, worker_mode="process"), db=db,
                          emit=lambda *_: None), ProcessSupervisor)


def test_process_supervisor_command_carries_utf8_and_recycle_limit(tmp_path, monkeypatch) -> None:
    """`-X utf8` обов'язковий: кириличні імена файлів у межі subprocess
    інакше падають на cp1251 на Windows."""
    settings = make_settings(tmp_path, worker_mode="process", worker_recycle_after=7)
    captured: dict[str, Any] = {}

    class _Fake:
        pid = 1234

        def poll(self) -> int | None:
            return None

    def fake_popen(cmd, **kw):  # type: ignore[no-untyped-def]
        captured["cmd"] = cmd
        captured["env"] = kw.get("env", {})
        return _Fake()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    supervisor = ProcessSupervisor(settings)
    supervisor._spawn()

    assert "-X" in captured["cmd"] and "utf8" in captured["cmd"]
    assert captured["cmd"][-2:] == ["--max-jobs", "7"]
    assert "app.worker" in captured["cmd"]
    assert captured["env"]["ASISTENT_STUB"] == "1"
    assert captured["env"]["ASISTENT_DATA_DIR"] == str(settings.paths().data_dir)


async def test_null_supervisor_is_inert(tmp_path) -> None:
    supervisor = NullSupervisor()
    await supervisor.start()
    await supervisor.stop()
    assert supervisor.status()["running"] == 0


async def test_inline_supervisor_reports_completed_jobs(tmp_path) -> None:
    settings, _collection_id, document_id = _seed(tmp_path)
    settings = make_settings(tmp_path, worker_mode="inline")
    db = Database(settings.paths().db_path)
    supervisor = InlineSupervisor(settings, db=db, emit=lambda *_: None)
    await supervisor.start()
    try:
        import asyncio

        for _ in range(400):
            with db.connection() as con:
                document = DocumentRepo(con).get(document_id)
            if document is not None and document.status is DocStatus.READY:
                break
            await asyncio.sleep(0.01)
    finally:
        await supervisor.stop()
    assert document is not None and document.status is DocStatus.READY
    assert supervisor.status()["completed"] >= 1


# ------------------------------------------------- міст «воркер → SSE»
async def test_watcher_turns_db_rows_into_events(tmp_path) -> None:
    """Воркер — окремий процес і не може писати в шину API. Єдине спільне
    сховище — SQLite, тож спостерігач читає його й озвучує ті самі типізовані
    події, що їх inline-воркер публікує напряму."""
    settings = make_settings(tmp_path, worker_mode="process")
    services = build_services(settings)
    queue = services.queue
    with services.db.transaction() as con:
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
        )
        DocumentRepo(con).create(document)

    watcher = JobWatcher(services, interval=0.01)
    watcher._prime()

    job_id = queue.enqueue(JobType.PARSE, document_id=document.id,
                           collection_id=collection.id, weight_total=100.0)
    queue.claim()
    queue.progress(job_id, stage="PARSE", weight_done=40.0)
    watcher.poll()

    progress = [e for e in services.events.history() if e.type == "job.progress"]
    assert progress and progress[-1].data["stage"] == "PARSE"
    assert progress[-1].data["fraction"] == pytest.approx(0.4)

    # Провал стає подією рівно один раз.
    queue.finish(job_id, JobState.FAILED, error_code="PARSE_FAILED", error_detail="деталі")
    watcher.poll()
    watcher.poll()
    failures = [e for e in services.events.history() if e.type == "job.failed"]
    assert len(failures) == 1
    assert failures[0].data["errorCode"] == "PARSE_FAILED"

    # Готовий документ теж озвучується один раз.
    with services.db.transaction() as con:
        DocumentRepo(con).set_status(document.id, DocStatus.READY)
    watcher.poll()
    watcher.poll()
    ready = [e for e in services.events.history() if e.type == "doc.ready"]
    assert len(ready) == 1
    assert ready[0].data["docId"] == document.id
    services.close()


def test_watcher_priming_prevents_replaying_yesterdays_work(tmp_path) -> None:
    """Без початкового знімка кожен перезапуск API виплюнув би `doc.ready` на
    всі вже готові документи — двадцять сповіщень про вчорашню роботу."""
    settings = make_settings(tmp_path)
    services = build_services(settings)
    with services.db.transaction() as con:
        assistant = Assistant(id=new_id(), name="А")
        AssistantRepo(con).create(assistant)
        collection = Collection(
            id=new_id(), assistant_id=assistant.id, name="Основна",
            embedding_model_id="m", embedding_model_key="k", dim=8,
        )
        CollectionRepo(con).create(collection)
        DocumentRepo(con).create(Document(
            id=new_id(), collection_id=collection.id, title="Готовий",
            original_name="p.pdf", stored_path="p.pdf", content_sha256="s",
            status=DocStatus.READY,
        ))

    watcher = JobWatcher(services, interval=0.01)
    watcher._prime()
    watcher.poll()
    assert [e for e in services.events.history() if e.type == "doc.ready"] == []
    services.close()


def test_document_ready_payload_counts_tables_in_python(tmp_path) -> None:
    """Лічильники рахуються в Python, а не через `json_array_length`:
    складання SQLite у python-build-standalone не гарантує розширення JSON1,
    і тиха відмова перетворила б лічильники на нулі без жодного сліду."""
    settings = make_settings(tmp_path)
    services = build_services(settings)
    with services.db.transaction() as con:
        assistant = Assistant(id=new_id(), name="А")
        AssistantRepo(con).create(assistant)
        collection = Collection(
            id=new_id(), assistant_id=assistant.id, name="Основна",
            embedding_model_id="m", embedding_model_key="k", dim=8,
        )
        CollectionRepo(con).create(collection)
        document = Document(
            id=new_id(), collection_id=collection.id, title="Т",
            original_name="p.pdf", stored_path="p.pdf", content_sha256="s",
        )
        DocumentRepo(con).create(document)
        con.execute(
            "INSERT INTO chunks (chunk_uid,collection_id,document_id,level,ordinal,"
            "embed_text,rerank_text,display_text,tables_json,formulas_json,pictures_json)"
            " VALUES (?,?,?,'L2',0,'e','r','d','[1,2,3]','[9]','{\"a\":1}')",
            (new_id(), collection.id, document.id),
        )
    payload = document_ready_payload(services.db, document.id)
    assert payload == {"docId": document.id, "chunks": 1, "tables": 3,
                       "formulas": 1, "pictures": 1}
    services.close()
