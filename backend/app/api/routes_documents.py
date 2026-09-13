"""Документи: завантаження, статус, сторінки, переіндексація, віддача файлу.

ЗАВАНТАЖЕННЯ ПИШЕТЬСЯ НА ДИСК ПОТОКОВО.
300-мегабайтний підручник не має проходити через RAM: `UploadFile.read()`
без аргументу зробив би саме це, і на 8-гігабайтній машині, де LM Studio вже
тримає 9 ГБ, воно закінчилось би MemoryError. Пишемо блоками, паралельно
рахуючи sha256 — тобто за один прохід замість двох.

ФАЙЛ ЗБЕРІГАЄТЬСЯ ПІД UUID, А НЕ ПІД ОРИГІНАЛЬНОЮ НАЗВОЮ.
«Методичні рекомендації щодо організації бойової підготовки артилерійських
підрозділів.pdf» у теці `%LOCALAPPDATA%\\Asistent\\docs` — це вже 150+
символів, і разом із глибиною шляху воно тривіально пробиває MAX_PATH 260 на
Windows. Оригінальна назва живе метаданими в SQLite, де їй нічого не загрожує.

ВІДДАЧА ФАЙЛУ — З ПІДТРИМКОЮ HTTP RANGE.
PDF.js читає файл частинами: спершу хвіст із таблицею xref, потім потрібні
сторінки. Без Range він змушений завантажити всі 300 МБ, перш ніж показати
першу сторінку — тобто клік по цитаті означав би хвилину очікування.
"""

from __future__ import annotations

import hashlib
import logging
import mimetypes
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import Response, StreamingResponse

from app.api.routes_assistants import services_of
from app.api.schemas import (
    DocumentOut,
    PageOut,
    ReingestIn,
    document_out,
    page_out,
)
from app.api.state import Services
from app.db.repositories import ChunkRepo, CollectionRepo, DocumentRepo, JobRepo, PageRepo
from app.domain import DocStatus, Document, IngestMode, JobType, new_id
from app.jobs.pipeline import resolve_document_path

__all__ = ["router"]

log = logging.getLogger("asistent.api.documents")

router = APIRouter(tags=["documents"])

# Формати, які конвеєр справді вміє прийняти. DOCX свідомо відсутній:
# `msword_backend` Docling НІКОЛИ не створює `ProvenanceItem`, тобто цитата на
# сторінку для DOCX неможлива в принципі, а тихо прийняти файл і потім не
# вміти на нього послатися — гірше, ніж відмовити одразу з поясненням.
ALLOWED_SUFFIXES = {".pdf", ".txt", ".md", ".markdown"}
_RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")
_UNSAFE_NAME_RE = re.compile(r"[\x00-\x1f/\\]")


def _get_document(services: Services, document_id: str) -> Document:
    with services.db.connection() as con:
        document = DocumentRepo(con).get(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Документ не знайдено.")
    return document


def _job_summary(services: Services, document_id: str) -> dict[str, Any] | None:
    """Найсвіжіше завдання документа — щоб UI знав, що показувати в рядку."""
    jobs = services.queue.for_document(document_id)
    if not jobs:
        return None
    job = jobs[-1]
    total = float(job["progress_weight_total"] or 0.0)
    done = float(job["progress_weight_done"] or 0.0)
    return {
        "id": job["id"],
        "type": job["type"],
        "state": job["state"],
        "stage": job["stage"],
        "attempts": job["attempts"],
        "fraction": round(min(1.0, done / total), 4) if total else 0.0,
        "errorCode": job["error_code"],
    }


# ------------------------------------------------------------------ список
@router.get("/collections/{collection_id}/documents", response_model=list[DocumentOut])
def list_documents(request: Request, collection_id: str) -> list[DocumentOut]:
    services = services_of(request)
    with services.db.connection() as con:
        if CollectionRepo(con).get(collection_id) is None:
            raise HTTPException(status_code=404, detail="Колекцію не знайдено.")
        documents = DocumentRepo(con).for_collection(collection_id)
        counts = {
            row["document_id"]: row["c"]
            for row in con.execute(
                "SELECT document_id, count(*) AS c FROM chunks"
                " WHERE collection_id=? AND level='L2' GROUP BY document_id",
                (collection_id,),
            )
        }
    out: list[DocumentOut] = []
    for document in documents:
        path = resolve_document_path(services.paths, document)
        size = path.stat().st_size if path.is_file() else None
        out.append(
            document_out(
                document,
                size_bytes=size,
                chunks=int(counts.get(document.id, 0)),
                job=_job_summary(services, document.id),
            )
        )
    return out


# ------------------------------------------------------------- завантаження
@router.post("/collections/{collection_id}/documents",
             response_model=list[DocumentOut], status_code=201)
async def upload_documents(
    request: Request,
    collection_id: str,
    files: list[UploadFile],
    doc_type: str = Form("textbook", alias="docType"),
    language: str = Form("uk"),
    year: int | None = Form(None),
    author: str | None = Form(None),
    mode: str = Form("", alias="ingestMode"),
) -> list[DocumentOut]:
    services = services_of(request)
    with services.db.connection() as con:
        if CollectionRepo(con).get(collection_id) is None:
            raise HTTPException(status_code=404, detail="Колекцію не знайдено.")

    ingest_mode = _parse_mode(mode) if mode else services.settings.default_ingest_mode
    created: list[DocumentOut] = []
    for upload in files:
        document = await _store_upload(
            services,
            upload,
            collection_id=collection_id,
            doc_type=doc_type,
            language=language,
            year=year,
            author=author,
            ingest_mode=ingest_mode,
        )
        created.append(document_out(document, job=_job_summary(services, document.id)))
    return created


async def _store_upload(
    services: Services,
    upload: UploadFile,
    *,
    collection_id: str,
    doc_type: str,
    language: str,
    year: int | None,
    author: str | None,
    ingest_mode: IngestMode,
) -> Document:
    original = _UNSAFE_NAME_RE.sub("_", (upload.filename or "документ").strip()) or "документ"
    suffix = Path(original).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Формат {suffix or '(без розширення)'} не підтримується. "
                f"Підтримуються: {', '.join(sorted(ALLOWED_SUFFIXES))}. "
                "DOCX слід спершу конвертувати у PDF: інакше цитата на конкретну "
                "сторінку неможлива в принципі."
            ),
        )

    document_id = new_id()
    stored_name = f"{document_id}{suffix}"
    target = services.paths.documents_dir / stored_name
    digest = hashlib.sha256()
    written = 0
    limit = services.settings.max_upload_bytes

    try:
        with target.open("wb") as fh:
            while block := await upload.read(services.settings.upload_chunk_bytes):
                written += len(block)
                if written > limit:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Файл більший за дозволені {limit // (1024 ** 2)} МБ.",
                    )
                digest.update(block)
                fh.write(block)
    except HTTPException:
        target.unlink(missing_ok=True)
        raise
    except OSError as exc:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=507, detail=f"Не вдалося зберегти файл: {exc}") from exc
    finally:
        await upload.close()

    if written == 0:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Файл порожній.")

    document = Document(
        id=document_id,
        collection_id=collection_id,
        title=Path(original).stem or original,
        original_name=original,
        # ВІДНОСНИЙ шлях: каталог даних можна перенести на інший диск, і
        # абсолютний шлях у БД перетворив би це на масову втрату файлів.
        stored_path=stored_name,
        content_sha256=digest.hexdigest(),
        doc_type=doc_type or "textbook",
        language=language or "uk",
        year=year,
        author=author,
        ingest_mode=ingest_mode,
        status=DocStatus.QUEUED,
    )
    with services.db.transaction() as con:
        DocumentRepo(con).create(document)
    services.queue.enqueue(
        JobType.PARSE, document_id=document.id, collection_id=collection_id
    )
    return document


def _parse_mode(raw: str) -> IngestMode:
    try:
        return IngestMode(raw.upper())
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Невідомий режим індексації {raw!r}. Доступні: FAST, DEEP.",
        ) from exc


# ------------------------------------------------------------- керування
@router.delete("/documents/{document_id}", status_code=204)
def delete_document(request: Request, document_id: str) -> None:
    services = services_of(request)
    document = _get_document(services, document_id)
    services.queue.cancel_document(document_id)
    with services.db.transaction() as con:
        ChunkRepo(con).delete_for_document(document_id)
        DocumentRepo(con).delete(document_id)
        # Колекція «брудна»: у графі HNSW лишились ключі видалених чанків.
        # Пошук від цього не зламається (`by_ids` просто не знайде рядка), але
        # перебудувати індекс треба.
        CollectionRepo(con).mark_dirty(document.collection_id, True)
    path = resolve_document_path(services.paths, document)
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:  # pragma: no cover — файл зайнятий читачем на Windows
        log.warning("Не вдалося видалити файл %s: %s", path, exc)
    services.queue.enqueue_once(JobType.INDEX, collection_id=document.collection_id)


@router.post("/documents/{document_id}/reingest", response_model=DocumentOut)
def reingest_document(
    request: Request, document_id: str, payload: ReingestIn | None = None
) -> DocumentOut:
    """Переіндексувати документ у режимі FAST або DEEP.

    Старі чанки видаляються ДО постановки завдання, а не всередині конвеєра:
    інакше між скасуванням і новим проходом у пошуку співіснували б дві
    версії документа, і одна відповідь могла б процитувати обидві.
    """
    services = services_of(request)
    document = _get_document(services, document_id)
    mode = _parse_mode((payload.mode if payload else "FAST"))

    services.queue.cancel_document(document_id)
    with services.db.transaction() as con:
        ChunkRepo(con).delete_for_document(document_id)
        con.execute("DELETE FROM document_pages_state WHERE document_id=?", (document_id,))
        con.execute(
            "UPDATE documents SET ingest_mode=?, status=?, error_code=NULL,"
            " error_detail=NULL, parse_profile_hash=NULL WHERE id=?",
            (mode.value, DocStatus.QUEUED.value, document_id),
        )
        CollectionRepo(con).mark_dirty(document.collection_id, True)
    services.queue.enqueue(
        JobType.PARSE, document_id=document_id, collection_id=document.collection_id,
        priority=50,
    )
    return document_out(
        _get_document(services, document_id), job=_job_summary(services, document_id)
    )


@router.post("/documents/{document_id}/cancel")
def cancel_document(request: Request, document_id: str) -> dict[str, Any]:
    """Кооперативне скасування.

    Воркер перевіряє прапорець МІЖ сторінками, тож найгірша затримка — одна
    сторінка (0.4–0.7 с у швидкому режимі). Обіцяти миттєвість означало б
    обіцяти вбивство процесу, тобто втрату вже виконаної роботи.
    """
    services = services_of(request)
    _get_document(services, document_id)
    cancelled = services.queue.cancel_document(document_id)
    return {
        "cancelled": cancelled,
        "detail": (
            "Скасування запитано. Обробка зупиниться на межі поточної сторінки."
            if cancelled else "Активних завдань для цього документа немає."
        ),
    }


# --------------------------------------------------------------- сторінки
@router.get("/documents/{document_id}/pages", response_model=list[PageOut])
def document_pages(request: Request, document_id: str) -> list[PageOut]:
    services = services_of(request)
    _get_document(services, document_id)
    with services.db.connection() as con:
        pages = PageRepo(con).for_document(document_id)
    return [page_out(p) for p in pages]


@router.get("/documents/{document_id}/jobs")
def document_jobs(request: Request, document_id: str) -> list[dict[str, Any]]:
    services = services_of(request)
    _get_document(services, document_id)
    with services.db.connection() as con:
        jobs = JobRepo(con).for_document(document_id)
    return jobs


# ------------------------------------------------------------ віддача файлу
@router.get("/documents/{document_id}/file")
def document_file(
    request: Request,
    document_id: str,
    download: bool = Query(default=False),
) -> Response:
    services = services_of(request)
    document = _get_document(services, document_id)
    path = resolve_document_path(services.paths, document)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Файл документа відсутній на диску.")

    size = path.stat().st_size
    media_type = mimetypes.guess_type(document.original_name)[0] or "application/octet-stream"
    disposition = "attachment" if download else "inline"
    # RFC 5987: кирилична назва в `filename=` ламає заголовок; `filename*` —
    # єдиний спосіб віддати «Настанова.pdf» так, щоб браузер її прочитав.
    quoted = _rfc5987(document.original_name)
    headers = {
        "Accept-Ranges": "bytes",
        "Content-Disposition": f"{disposition}; filename*=UTF-8''{quoted}",
        "Cache-Control": "private, max-age=3600",
    }

    span = _parse_range(request.headers.get("range"), size)
    if span is None:
        headers["Content-Length"] = str(size)
        return StreamingResponse(
            _iter_file(path, 0, size - 1), media_type=media_type, headers=headers
        )
    start, end = span
    if start >= size:
        headers["Content-Range"] = f"bytes */{size}"
        return Response(status_code=416, headers=headers)
    headers["Content-Range"] = f"bytes {start}-{end}/{size}"
    headers["Content-Length"] = str(end - start + 1)
    return StreamingResponse(
        _iter_file(path, start, end), status_code=206, media_type=media_type, headers=headers
    )


def _rfc5987(value: str) -> str:
    from urllib.parse import quote

    return quote(value, safe="")


def _parse_range(header: str | None, size: int) -> tuple[int, int] | None:
    """Розібрати `Range: bytes=…`. None → віддати файл цілком.

    Свідомо підтримується лише ОДИН діапазон: multipart/byteranges вимагає
    складання multipart-відповіді, а PDF.js його ніколи не просить.
    """
    if not header or size == 0:
        return None
    match = _RANGE_RE.fullmatch(header.strip())
    if match is None:
        return None
    raw_start, raw_end = match.group(1), match.group(2)
    if not raw_start and not raw_end:
        return None
    if not raw_start:                      # bytes=-500 — останні 500 байт
        length = min(int(raw_end), size)
        return size - length, size - 1
    start = int(raw_start)
    end = int(raw_end) if raw_end else size - 1
    return start, min(end, size - 1)


def _iter_file(path: Path, start: int, end: int, block: int = 256 * 1024) -> Iterator[bytes]:
    remaining = end - start + 1
    with path.open("rb") as fh:
        fh.seek(start)
        while remaining > 0:
            data = fh.read(min(block, remaining))
            if not data:
                break
            remaining -= len(data)
            yield data
