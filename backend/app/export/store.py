"""Таблиця `message_documents` і збирання файлу з повідомлення з історії.

Окремо від `db/repositories.py`: фундамент контракту не переписується, а цей
модуль торкається рівно однієї своєї таблиці (міграція 0002) і читає
повідомлення та фрагменти.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import cast

from app.db.repositories import ChunkRepo
from app.domain import RetrievedChunk
from app.export.intent import FORMATS, DocumentFormat, detect_document_request
from app.export.outline import parse_markdown
from app.export.render import RenderedFile, SourceRef, document_filename, render_document

__all__ = [
    "FormatRequired",
    "MessageNotFound",
    "StoredDocument",
    "load_document",
    "previous_topic",
    "render_message",
    "save_document",
    "session_documents",
    "sources_from_evidence",
    "sources_from_map",
]


class MessageNotFound(LookupError):
    """Немає такої відповіді асистента або вона порожня."""


class FormatRequired(ValueError):
    """Відповідь без замовленого файлу, а формат не вказано."""


@dataclass(frozen=True, slots=True)
class StoredDocument:
    message_id: str
    format: DocumentFormat
    title: str
    stats: dict[str, int] = field(default_factory=dict)

    def payload(self) -> dict[str, object]:
        """Те, що бачить UI: картка файлу під відповіддю."""
        return {
            "format": self.format,
            "title": self.title,
            "filename": document_filename(self.title, self.format),
            "stats": dict(self.stats),
        }


def save_document(con: sqlite3.Connection, message_id: str, fmt: DocumentFormat,
                  title: str, stats: dict[str, int]) -> StoredDocument:
    con.execute(
        "INSERT OR REPLACE INTO message_documents (message_id, format, title, stats_json)"
        " VALUES (?, ?, ?, ?)",
        (message_id, fmt, title, json.dumps(stats, ensure_ascii=False)),
    )
    return StoredDocument(message_id, fmt, title, dict(stats))


def load_document(con: sqlite3.Connection, message_id: str) -> StoredDocument | None:
    row = con.execute(
        "SELECT * FROM message_documents WHERE message_id=?", (message_id,)
    ).fetchone()
    return _stored(row) if row is not None else None


def session_documents(con: sqlite3.Connection, session_id: str) -> dict[str, StoredDocument]:
    rows = con.execute(
        "SELECT d.* FROM message_documents d JOIN chat_messages m ON m.id = d.message_id"
        " WHERE m.session_id=?",
        (session_id,),
    ).fetchall()
    return {row["message_id"]: _stored(row) for row in rows}


def _stored(row: sqlite3.Row) -> StoredDocument:
    try:
        stats = {str(k): int(v) for k, v in json.loads(row["stats_json"] or "{}").items()}
    except (ValueError, TypeError, AttributeError):
        stats = {}
    return StoredDocument(row["message_id"], row["format"], row["title"] or "", stats)


# --------------------------------------------------------------- історія
def previous_topic(con: sqlite3.Connection, session_id: str, before_message_id: str) -> str:
    """Тема для «зроби з цього презентацію»: попереднє звичайне питання.

    Прохання про файл без теми пропускаємо («зроби з цього таблицю» одразу
    після «зроби з цього презентацію» стосується того самого питання), а
    прохання з темою віддає саме тему. Порядок — за rowid: `created_at` має
    секундну точність, і два ходи в межах секунди переплутали б місцями.
    """
    rows = con.execute(
        "SELECT content FROM chat_messages WHERE session_id=? AND role='user'"
        " AND rowid < (SELECT rowid FROM chat_messages WHERE id=?)"
        " ORDER BY rowid DESC LIMIT 8",
        (session_id, before_message_id),
    ).fetchall()
    # Перший рядок — поточне питання: його вставлено перед відповіддю.
    for row in rows[1:]:
        text = (row["content"] or "").strip()
        request = detect_document_request(text)
        if request is None:
            return text
        if request.topic:
            return request.topic
    return ""


def _question_before(con: sqlite3.Connection, session_id: str, message_id: str) -> str:
    row = con.execute(
        "SELECT content FROM chat_messages WHERE session_id=? AND role='user'"
        " AND rowid < (SELECT rowid FROM chat_messages WHERE id=?)"
        " ORDER BY rowid DESC LIMIT 1",
        (session_id, message_id),
    ).fetchone()
    return (row["content"] or "").strip() if row is not None else ""


# ---------------------------------------------------------------- джерела
def sources_from_evidence(evidence: Sequence[RetrievedChunk]) -> dict[int, SourceRef]:
    return {
        rc.ordinal_in_prompt: SourceRef(
            rc.ordinal_in_prompt, rc.document_title or "Без назви", rc.chunk.citation_label()
        )
        for rc in evidence
        if rc.ordinal_in_prompt
    }


def sources_from_map(con: sqlite3.Connection, mapping: dict[str, str]) -> dict[int, SourceRef]:
    """Мапа повідомлення «[n] → chunk_uid» → назви документів і сторінки.

    Фрагмент, якого вже немає (документ видалено чи переіндексовано), просто
    випадає зі списку джерел: маркер у тексті лишається, але без підпису.
    """
    chunks = ChunkRepo(con)
    titles: dict[str, str] = {}
    out: dict[int, SourceRef] = {}
    for key, uid in mapping.items():
        try:
            ordinal = int(key)
        except (TypeError, ValueError):
            continue
        chunk = chunks.by_uid(uid)
        if chunk is None:
            continue
        if chunk.document_id not in titles:
            titles[chunk.document_id] = chunks.document_title(chunk.document_id) or "Без назви"
        out[ordinal] = SourceRef(ordinal, titles[chunk.document_id], chunk.citation_label())
    return out


# ------------------------------------------------------------- збирання
def render_message(con: sqlite3.Connection, message_id: str, fmt: str | None = None, *,
                   today: date | None = None) -> RenderedFile:
    """Зібрати файл із відповіді асистента.

    Без `fmt` — той формат, що просили в чаті. З `fmt` — будь-яку відповідь
    можна забрати у Word (чи PowerPoint, чи Excel) і без прохання наперед.
    """
    row = con.execute(
        "SELECT id, session_id, role, content, citation_map_json FROM chat_messages WHERE id=?",
        (message_id,),
    ).fetchone()
    if row is None or row["role"] != "assistant" or not (row["content"] or "").strip():
        raise MessageNotFound(message_id)
    stored = load_document(con, message_id)
    requested = fmt or (stored.format if stored is not None else None)
    if requested not in FORMATS:
        raise FormatRequired(str(fmt))
    chosen = cast(DocumentFormat, requested)
    try:
        mapping = json.loads(row["citation_map_json"] or "{}")
    except ValueError:
        mapping = {}
    fallback = (stored.title if stored is not None else "") or _question_before(
        con, row["session_id"], message_id
    )
    outline = parse_markdown(row["content"], fallback_title=fallback)
    return render_document(outline, chosen, sources_from_map(con, mapping), today=today)
