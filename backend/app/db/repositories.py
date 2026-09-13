"""Доступ до даних. Єдине місце, що знає SQL.

Вектори зберігаються як float16 little-endian BLOB у `chunks.embedding` —
це канонічна копія. USearch-індекс перебудовується з них за 1–3 хв на
100 тис. чанків, тому він кеш, а не джерело істини.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from app.domain import (
    Assistant, AssistantConfig, BBox, Chapter, Chunk, ChunkLevel, Collection,
    DocStatus, Document, IngestMode, JobState, JobType, OcrModeName, PageClass,
    PageInfo, QualityGrade, new_id, utcnow,
)

__all__ = [
    "AssistantRepo", "CollectionRepo", "DocumentRepo", "PageRepo", "ChapterRepo",
    "ChunkRepo", "JobRepo", "ChatRepo", "TelemetryRepo", "vector_to_blob", "blob_to_vector",
]


def vector_to_blob(vec: "np.ndarray") -> bytes:
    return np.asarray(vec, dtype="<f2").tobytes()


def blob_to_vector(blob: bytes, dim: int) -> "np.ndarray":
    return np.frombuffer(blob, dtype="<f2", count=dim).astype(np.float32)


def _j(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


class _Repo:
    def __init__(self, con: sqlite3.Connection) -> None:
        self.con = con


# --------------------------------------------------------------- асистенти
class AssistantRepo(_Repo):
    def create(self, a: Assistant) -> Assistant:
        self.con.execute(
            "INSERT INTO assistants (id,name,description,colour,emoji,config_json,config_version)"
            " VALUES (?,?,?,?,?,?,?)",
            (a.id, a.name, a.description, a.colour, a.emoji, a.config.to_json(), a.config_version),
        )
        return a

    def update(self, a: Assistant) -> None:
        self.con.execute(
            "UPDATE assistants SET name=?, description=?, colour=?, emoji=?, config_json=?,"
            " config_version=config_version+1, updated_at=? WHERE id=?",
            (a.name, a.description, a.colour, a.emoji, a.config.to_json(), utcnow(), a.id),
        )

    def get(self, assistant_id: str) -> Assistant | None:
        row = self.con.execute("SELECT * FROM assistants WHERE id=?", (assistant_id,)).fetchone()
        return self._row(row) if row else None

    def list(self) -> list[Assistant]:
        rows = self.con.execute("SELECT * FROM assistants ORDER BY updated_at DESC").fetchall()
        return [self._row(r) for r in rows]

    def delete(self, assistant_id: str) -> None:
        self.con.execute("DELETE FROM assistants WHERE id=?", (assistant_id,))

    @staticmethod
    def _row(r: sqlite3.Row) -> Assistant:
        return Assistant(
            id=r["id"], name=r["name"], description=r["description"], colour=r["colour"],
            emoji=r["emoji"], config=AssistantConfig.from_json(r["config_json"]),
            config_version=r["config_version"],
        )


# --------------------------------------------------------------- колекції
class CollectionRepo(_Repo):
    def create(self, c: Collection) -> Collection:
        self.con.execute(
            "INSERT INTO collections (id,assistant_id,name,embedding_model_id,embedding_model_key,"
            "dim,metric,build_id,index_generation,dirty) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (c.id, c.assistant_id, c.name, c.embedding_model_id, c.embedding_model_key,
             c.dim, c.metric, c.build_id, c.index_generation, int(c.dirty)),
        )
        return c

    def get(self, collection_id: str) -> Collection | None:
        row = self.con.execute("SELECT * FROM collections WHERE id=?", (collection_id,)).fetchone()
        return self._row(row) if row else None

    def for_assistant(self, assistant_id: str) -> list[Collection]:
        rows = self.con.execute(
            "SELECT * FROM collections WHERE assistant_id=? ORDER BY created_at", (assistant_id,)
        ).fetchall()
        return [self._row(r) for r in rows]

    def mark_dirty(self, collection_id: str, dirty: bool = True) -> None:
        self.con.execute("UPDATE collections SET dirty=? WHERE id=?", (int(dirty), collection_id))

    def bump_index_generation(self, collection_id: str, build_id: str) -> int:
        """Читачі перевідкривають індекс, коли generation змінилась.

        Викликати В ТІЙ САМІЙ транзакції, що й os.replace нового файлу індексу.
        """
        self.con.execute(
            "UPDATE collections SET build_id=?, index_generation=index_generation+1, dirty=0"
            " WHERE id=?", (build_id, collection_id))
        return self.con.execute(
            "SELECT index_generation FROM collections WHERE id=?", (collection_id,)).fetchone()[0]

    @staticmethod
    def _row(r: sqlite3.Row) -> Collection:
        return Collection(
            id=r["id"], assistant_id=r["assistant_id"], name=r["name"],
            embedding_model_id=r["embedding_model_id"], embedding_model_key=r["embedding_model_key"],
            dim=r["dim"], metric=r["metric"], build_id=r["build_id"],
            index_generation=r["index_generation"], dirty=bool(r["dirty"]),
        )


# --------------------------------------------------------------- документи
class DocumentRepo(_Repo):
    def create(self, d: Document) -> Document:
        self.con.execute(
            "INSERT INTO documents (id,collection_id,title,original_name,stored_path,content_sha256,"
            "parse_profile_hash,doc_type,language,year,author,page_count,ingest_mode,status)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (d.id, d.collection_id, d.title, d.original_name, d.stored_path, d.content_sha256,
             d.parse_profile_hash, d.doc_type, d.language, d.year, d.author, d.page_count,
             d.ingest_mode.value, d.status.value),
        )
        return d

    def get(self, document_id: str) -> Document | None:
        row = self.con.execute("SELECT * FROM documents WHERE id=?", (document_id,)).fetchone()
        return self._row(row) if row else None

    def for_collection(self, collection_id: str) -> list[Document]:
        rows = self.con.execute(
            "SELECT * FROM documents WHERE collection_id=? ORDER BY created_at DESC",
            (collection_id,)).fetchall()
        return [self._row(r) for r in rows]

    def find_cached(self, content_sha256: str, parse_profile_hash: str) -> Document | None:
        """Той самий підручник, той самий профіль парсингу → перевикористати артефакти.

        Робить додавання документа до другого асистента миттєвим, що важливо
        саме через вимогу кількох асистентів над спільним матеріалом.
        """
        row = self.con.execute(
            "SELECT * FROM documents WHERE content_sha256=? AND parse_profile_hash=?"
            " AND status='READY' LIMIT 1", (content_sha256, parse_profile_hash)).fetchone()
        return self._row(row) if row else None

    def set_status(self, document_id: str, status: DocStatus, *,
                   error_code: str | None = None, error_detail: str | None = None) -> None:
        self.con.execute(
            "UPDATE documents SET status=?, error_code=?, error_detail=?,"
            " ingested_at=CASE WHEN ?='READY' THEN ? ELSE ingested_at END WHERE id=?",
            (status.value, error_code, error_detail, status.value, utcnow(), document_id))

    def set_page_count(self, document_id: str, count: int) -> None:
        self.con.execute("UPDATE documents SET page_count=? WHERE id=?", (count, document_id))

    def set_quality(self, document_id: str, grade: QualityGrade) -> None:
        self.con.execute("UPDATE documents SET quality_grade=? WHERE id=?", (grade.value, document_id))

    def set_parse_profile(self, document_id: str, profile_hash: str) -> None:
        self.con.execute("UPDATE documents SET parse_profile_hash=? WHERE id=?",
                         (profile_hash, document_id))

    def delete(self, document_id: str) -> None:
        self.con.execute("DELETE FROM documents WHERE id=?", (document_id,))

    @staticmethod
    def _row(r: sqlite3.Row) -> Document:
        return Document(
            id=r["id"], collection_id=r["collection_id"], title=r["title"],
            original_name=r["original_name"], stored_path=r["stored_path"],
            content_sha256=r["content_sha256"], parse_profile_hash=r["parse_profile_hash"],
            doc_type=r["doc_type"], language=r["language"], year=r["year"], author=r["author"],
            page_count=r["page_count"], ingest_mode=IngestMode(r["ingest_mode"]),
            status=DocStatus(r["status"]), error_code=r["error_code"], error_detail=r["error_detail"],
            quality_grade=QualityGrade(r["quality_grade"]) if r["quality_grade"] else None,
        )


# ---------------------------------------------------------------- сторінки
class PageRepo(_Repo):
    def upsert_many(self, document_id: str, pages: Iterable[PageInfo]) -> None:
        self.con.executemany(
            "INSERT INTO pages (document_id,page_number,page_label,char_from,char_to,width,height,"
            "page_class,ocr_mode,cost_weight,lexicon_hit_rate,cyrillic_ratio,mojibake_ratio,"
            "parse_score,layout_score,table_score,ocr_score)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
            " ON CONFLICT(document_id,page_number) DO UPDATE SET"
            " page_label=excluded.page_label, char_from=excluded.char_from,"
            " char_to=excluded.char_to, width=excluded.width, height=excluded.height,"
            " page_class=excluded.page_class, ocr_mode=excluded.ocr_mode,"
            " cost_weight=excluded.cost_weight, lexicon_hit_rate=excluded.lexicon_hit_rate,"
            " cyrillic_ratio=excluded.cyrillic_ratio, mojibake_ratio=excluded.mojibake_ratio,"
            " parse_score=excluded.parse_score, layout_score=excluded.layout_score,"
            " table_score=excluded.table_score, ocr_score=excluded.ocr_score",
            [(document_id, p.page_number, p.page_label, p.char_from, p.char_to, p.width, p.height,
              p.page_class.value, p.ocr_mode.value, p.cost_weight, p.lexicon_hit_rate,
              p.cyrillic_ratio, p.mojibake_ratio, p.parse_score, p.layout_score,
              p.table_score, p.ocr_score) for p in pages],
        )

    def for_document(self, document_id: str) -> list[PageInfo]:
        rows = self.con.execute(
            "SELECT * FROM pages WHERE document_id=? ORDER BY page_number", (document_id,)).fetchall()
        return [self._row(r) for r in rows]

    def label_map(self, document_id: str) -> dict[int, str]:
        rows = self.con.execute(
            "SELECT page_number, page_label FROM pages WHERE document_id=? AND page_label IS NOT NULL",
            (document_id,)).fetchall()
        return {r["page_number"]: r["page_label"] for r in rows}

    @staticmethod
    def _row(r: sqlite3.Row) -> PageInfo:
        return PageInfo(
            page_number=r["page_number"], page_label=r["page_label"],
            char_from=r["char_from"] or 0, char_to=r["char_to"] or 0,
            width=r["width"] or 0.0, height=r["height"] or 0.0,
            page_class=PageClass(r["page_class"] or "DIGITAL_CLEAN"),
            ocr_mode=OcrModeName(r["ocr_mode"] or "DEFAULT"),
            cost_weight=r["cost_weight"], lexicon_hit_rate=r["lexicon_hit_rate"],
            cyrillic_ratio=r["cyrillic_ratio"], mojibake_ratio=r["mojibake_ratio"],
            parse_score=r["parse_score"], layout_score=r["layout_score"],
            table_score=r["table_score"], ocr_score=r["ocr_score"],
        )


# ----------------------------------------------------------------- розділи
class ChapterRepo(_Repo):
    def replace_for_document(self, document_id: str, chapters: Sequence[Chapter]) -> dict[str, int]:
        """Вкладені множини: «усе під Розділом 2» стає range scan по індексу,
        а не LIKE по header_path."""
        self.con.execute("DELETE FROM chapters WHERE document_id=?", (document_id,))
        mapping: dict[str, int] = {}
        for ch in chapters:
            cur = self.con.execute(
                "INSERT INTO chapters (document_id,parent_id,level,title,lft,rgt)"
                " VALUES (?,?,?,?,?,?)",
                (document_id, ch.parent_id, ch.level, ch.title, ch.lft, ch.rgt))
            mapping[ch.title] = cur.lastrowid
        return mapping

    def descendants(self, document_id: str, chapter_id: int) -> list[int]:
        row = self.con.execute("SELECT lft, rgt FROM chapters WHERE id=?", (chapter_id,)).fetchone()
        if not row:
            return []
        rows = self.con.execute(
            "SELECT id FROM chapters WHERE document_id=? AND lft BETWEEN ? AND ?",
            (document_id, row["lft"], row["rgt"])).fetchall()
        return [r["id"] for r in rows]


# ------------------------------------------------------------------ чанки
class ChunkRepo(_Repo):
    def insert_many(self, chunks: Sequence[Chunk]) -> list[int]:
        ids: list[int] = []
        for c in chunks:
            cur = self.con.execute(
                "INSERT INTO chunks (chunk_uid,collection_id,document_id,parent_id,level,ordinal,"
                "header_path,chapter_id,language,page_from,page_to,page_label_from,page_label_to,"
                "char_from,char_to,bbox_json,siblings_index,siblings_count,embed_text,rerank_text,"
                "display_text,context_note,pictures_json,tables_json,formulas_json,simhash)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (c.chunk_uid, c.collection_id, c.document_id, c.parent_id, c.level.value, c.ordinal,
                 c.header_path, c.chapter_id, c.language, c.page_from, c.page_to,
                 c.page_label_from, c.page_label_to, c.char_from, c.char_to,
                 _j([b.as_dict() for b in c.bboxes]), c.siblings_index, c.siblings_count,
                 c.embed_text, c.rerank_text, c.display_text, c.context_note,
                 _j(c.pictures), _j(c.tables), _j(c.formulas), c.simhash))
            c.id = cur.lastrowid
            ids.append(c.id)
            if c.is_indexable:
                self.con.execute(
                    "INSERT INTO chunk_facets (chunk_id,collection_id,document_id,doc_type,language,"
                    "year,chapter_id) SELECT ?,?,?,d.doc_type,?,d.year,? FROM documents d WHERE d.id=?",
                    (c.id, c.collection_id, c.document_id, c.language, c.chapter_id, c.document_id))
        return ids

    def index_fts(self, chunk_id: int, lemmas: str, forms: str, codes: str) -> None:
        self.con.execute("DELETE FROM chunk_fts WHERE rowid=?", (chunk_id,))
        self.con.execute(
            "INSERT INTO chunk_fts (rowid,lemmas,forms,codes) VALUES (?,?,?,?)",
            (chunk_id, lemmas, forms, codes))
        if codes.strip():
            self.con.execute("DELETE FROM code_fts WHERE rowid=?", (chunk_id,))
            self.con.execute("INSERT INTO code_fts (rowid,codes) VALUES (?,?)", (chunk_id, codes))

    def set_embedding(self, chunk_id: int, vector: "np.ndarray", model_key: str) -> None:
        self.con.execute(
            "UPDATE chunks SET embedding=?, embedding_model_key=? WHERE id=?",
            (vector_to_blob(vector), model_key, chunk_id))

    def set_embeddings(self, items: Sequence[tuple[int, "np.ndarray"]], model_key: str) -> None:
        self.con.executemany(
            "UPDATE chunks SET embedding=?, embedding_model_key=? WHERE id=?",
            [(vector_to_blob(v), model_key, cid) for cid, v in items])

    def get(self, chunk_id: int) -> Chunk | None:
        row = self.con.execute("SELECT * FROM chunks WHERE id=?", (chunk_id,)).fetchone()
        return self._row(row) if row else None

    def by_uid(self, chunk_uid: str) -> Chunk | None:
        row = self.con.execute("SELECT * FROM chunks WHERE chunk_uid=?", (chunk_uid,)).fetchone()
        return self._row(row) if row else None

    def by_ids(self, chunk_ids: Sequence[int]) -> list[Chunk]:
        if not chunk_ids:
            return []
        marks = ",".join("?" * len(chunk_ids))
        rows = self.con.execute(f"SELECT * FROM chunks WHERE id IN ({marks})", tuple(chunk_ids)).fetchall()
        by_id = {r["id"]: self._row(r) for r in rows}
        return [by_id[i] for i in chunk_ids if i in by_id]

    def parent_of(self, chunk_id: int) -> Chunk | None:
        row = self.con.execute(
            "SELECT p.* FROM chunks c JOIN chunks p ON p.id = c.parent_id WHERE c.id=?",
            (chunk_id,)).fetchone()
        return self._row(row) if row else None

    def neighbours(self, chunk: Chunk, radius: int = 1) -> list[Chunk]:
        """Сусіди в порядку читання — для auto-merge розширення контексту."""
        rows = self.con.execute(
            "SELECT * FROM chunks WHERE collection_id=? AND document_id=? AND level=?"
            " AND ordinal BETWEEN ? AND ? ORDER BY ordinal",
            (chunk.collection_id, chunk.document_id, chunk.level.value,
             chunk.ordinal - radius, chunk.ordinal + radius)).fetchall()
        return [self._row(r) for r in rows]

    def indexable(self, collection_id: str, model_key: str) -> list[tuple[int, bytes]]:
        rows = self.con.execute(
            "SELECT id, embedding FROM chunks WHERE collection_id=? AND level='L2'"
            " AND embedding IS NOT NULL AND embedding_model_key=? ORDER BY id",
            (collection_id, model_key)).fetchall()
        return [(r["id"], r["embedding"]) for r in rows]

    def pending_embedding(self, collection_id: str, model_key: str, limit: int = 512) -> list[Chunk]:
        rows = self.con.execute(
            "SELECT * FROM chunks WHERE collection_id=? AND level='L2'"
            " AND (embedding IS NULL OR embedding_model_key IS NOT ?) ORDER BY id LIMIT ?",
            (collection_id, model_key, limit)).fetchall()
        return [self._row(r) for r in rows]

    def count(self, collection_id: str, level: ChunkLevel | None = None) -> int:
        if level is None:
            return self.con.execute(
                "SELECT count(*) FROM chunks WHERE collection_id=?", (collection_id,)).fetchone()[0]
        return self.con.execute(
            "SELECT count(*) FROM chunks WHERE collection_id=? AND level=?",
            (collection_id, level.value)).fetchone()[0]

    def max_id(self, collection_id: str) -> int:
        row = self.con.execute(
            "SELECT max(id) FROM chunks WHERE collection_id=?", (collection_id,)).fetchone()
        return int(row[0] or 0)

    def delete_for_document(self, document_id: str) -> None:
        ids = [r[0] for r in self.con.execute(
            "SELECT id FROM chunks WHERE document_id=?", (document_id,))]
        for cid in ids:
            self.con.execute("DELETE FROM chunk_fts WHERE rowid=?", (cid,))
            self.con.execute("DELETE FROM code_fts WHERE rowid=?", (cid,))
        self.con.execute("DELETE FROM chunks WHERE document_id=?", (document_id,))

    def document_title(self, document_id: str) -> str:
        row = self.con.execute("SELECT title FROM documents WHERE id=?", (document_id,)).fetchone()
        return row["title"] if row else ""

    @staticmethod
    def _row(r: sqlite3.Row) -> Chunk:
        return Chunk(
            id=r["id"], chunk_uid=r["chunk_uid"], collection_id=r["collection_id"],
            document_id=r["document_id"], parent_id=r["parent_id"],
            level=ChunkLevel(r["level"]), ordinal=r["ordinal"], header_path=r["header_path"],
            chapter_id=r["chapter_id"], language=r["language"],
            page_from=r["page_from"], page_to=r["page_to"],
            page_label_from=r["page_label_from"], page_label_to=r["page_label_to"],
            char_from=r["char_from"], char_to=r["char_to"],
            bboxes=[BBox.from_dict(d) for d in json.loads(r["bbox_json"] or "[]")],
            siblings_index=r["siblings_index"], siblings_count=r["siblings_count"],
            embed_text=r["embed_text"], rerank_text=r["rerank_text"],
            display_text=r["display_text"], context_note=r["context_note"],
            pictures=json.loads(r["pictures_json"] or "{}"),
            tables=json.loads(r["tables_json"] or "[]"),
            formulas=json.loads(r["formulas_json"] or "[]"),
            simhash=r["simhash"],
        )


# ------------------------------------------------------------ черга завдань
class JobRepo(_Repo):
    """Лізинг + heartbeat роблять чергу стійкою до падіння.

    Відновлюваність живе в `document_pages_state`, а не тут: 1000-сторінковий
    підручник — 1000 рядків, і після падіння на 940-й сторінці ми відновлюємось
    із неї, а не з початку.
    """

    LEASE_SECONDS = 60

    def enqueue(self, job_type: JobType, *, document_id: str | None = None,
                collection_id: str | None = None, priority: int = 100,
                weight_total: float = 0.0) -> str:
        job_id = new_id()
        self.con.execute(
            "INSERT INTO jobs (id,type,document_id,collection_id,priority,progress_weight_total)"
            " VALUES (?,?,?,?,?,?)",
            (job_id, job_type.value, document_id, collection_id, priority, weight_total))
        return job_id

    def claim(self, types: Sequence[JobType] | None = None) -> dict[str, Any] | None:
        """Атомарно взяти одне завдання під лізинг."""
        self.reclaim_expired()
        clause, params = "", []
        if types:
            clause = " AND type IN (" + ",".join("?" * len(types)) + ")"
            params = [t.value for t in types]
        row = self.con.execute(
            "SELECT * FROM jobs WHERE state='QUEUED' AND cancel_requested=0" + clause +
            " ORDER BY priority, created_at LIMIT 1", tuple(params)).fetchone()
        if not row:
            return None
        cur = self.con.execute(
            "UPDATE jobs SET state='LEASED', started_at=COALESCE(started_at,?),"
            " lease_until=datetime('now', ?), heartbeat_at=? WHERE id=? AND state='QUEUED'",
            (utcnow(), f"+{self.LEASE_SECONDS} seconds", utcnow(), row["id"]))
        if cur.rowcount == 0:
            return None  # хтось випередив
        return dict(row)

    def heartbeat(self, job_id: str) -> bool:
        """Продовжити лізинг. False → завдання скасовано, воркер має зупинитись."""
        self.con.execute(
            "UPDATE jobs SET heartbeat_at=?, lease_until=datetime('now', ?), state='RUNNING'"
            " WHERE id=? AND state IN ('LEASED','RUNNING')",
            (utcnow(), f"+{self.LEASE_SECONDS} seconds", job_id))
        row = self.con.execute("SELECT cancel_requested FROM jobs WHERE id=?", (job_id,)).fetchone()
        return bool(row) and not row["cancel_requested"]

    def progress(self, job_id: str, *, stage: str, weight_done: float) -> None:
        self.con.execute("UPDATE jobs SET stage=?, progress_weight_done=? WHERE id=?",
                         (stage, weight_done, job_id))

    def finish(self, job_id: str, state: JobState, *,
               error_code: str | None = None, error_detail: str | None = None) -> None:
        self.con.execute(
            "UPDATE jobs SET state=?, finished_at=?, error_code=?, error_detail=?,"
            " lease_until=NULL WHERE id=?",
            (state.value, utcnow(), error_code, error_detail, job_id))

    def request_cancel(self, job_id: str) -> None:
        self.con.execute("UPDATE jobs SET cancel_requested=1 WHERE id=?", (job_id,))

    def reclaim_expired(self) -> int:
        """Лізинг закінчився → воркер помер. Повернути в чергу, збільшити attempts.

        Понад max_attempts → FAILED із збереженою останньою помилкою.
        """
        cur = self.con.execute(
            "UPDATE jobs SET state='QUEUED', attempts=attempts+1, lease_until=NULL"
            " WHERE state IN ('LEASED','RUNNING') AND lease_until IS NOT NULL"
            " AND lease_until < datetime('now') AND attempts+1 <= max_attempts")
        reclaimed = cur.rowcount
        self.con.execute(
            "UPDATE jobs SET state='FAILED', finished_at=?,"
            " error_code=COALESCE(error_code,'LEASE_EXPIRED'), lease_until=NULL"
            " WHERE state IN ('LEASED','RUNNING') AND lease_until IS NOT NULL"
            " AND lease_until < datetime('now') AND attempts+1 > max_attempts", (utcnow(),))
        return reclaimed

    def for_document(self, document_id: str) -> list[dict[str, Any]]:
        return [dict(r) for r in self.con.execute(
            "SELECT * FROM jobs WHERE document_id=? ORDER BY created_at", (document_id,))]

    def active(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self.con.execute(
            "SELECT * FROM jobs WHERE state IN ('QUEUED','LEASED','RUNNING')"
            " ORDER BY priority, created_at")]

    # ---- посторінковий стан: саме тут живе відновлюваність ----
    def init_pages(self, document_id: str, page_numbers: Iterable[int]) -> None:
        self.con.executemany(
            "INSERT INTO document_pages_state (document_id,page_number,state) VALUES (?,?,'PENDING')"
            " ON CONFLICT(document_id,page_number) DO NOTHING",
            [(document_id, p) for p in page_numbers])

    def mark_page(self, document_id: str, page_number: int, state: str,
                  *, page_hash: str | None = None, error_code: str | None = None) -> None:
        self.con.execute(
            "INSERT INTO document_pages_state (document_id,page_number,state,page_hash,error_code,updated_at)"
            " VALUES (?,?,?,?,?,?) ON CONFLICT(document_id,page_number) DO UPDATE SET"
            " state=excluded.state, page_hash=excluded.page_hash,"
            " error_code=excluded.error_code, updated_at=excluded.updated_at",
            (document_id, page_number, state, page_hash, error_code, utcnow()))

    def pending_pages(self, document_id: str) -> list[int]:
        return [r[0] for r in self.con.execute(
            "SELECT page_number FROM document_pages_state WHERE document_id=? AND state!='DONE'"
            " ORDER BY page_number", (document_id,))]


# -------------------------------------------------------------------- чат
class ChatRepo(_Repo):
    def create_session(self, assistant_id: str, title: str = "") -> str:
        sid = new_id()
        self.con.execute("INSERT INTO chat_sessions (id,assistant_id,title) VALUES (?,?,?)",
                         (sid, assistant_id, title))
        return sid

    def sessions(self, assistant_id: str) -> list[dict[str, Any]]:
        return [dict(r) for r in self.con.execute(
            "SELECT * FROM chat_sessions WHERE assistant_id=? ORDER BY updated_at DESC",
            (assistant_id,))]

    def add_message(self, session_id: str, role: str, content: str, *,
                    citation_map: dict[str, str] | None = None,
                    retrieval_debug: str | None = None, abstained: bool = False,
                    model_id: str | None = None, ttft_ms: int | None = None,
                    tokens_out: int | None = None) -> str:
        mid = new_id()
        self.con.execute(
            "INSERT INTO chat_messages (id,session_id,role,content,citation_map_json,"
            "retrieval_debug_json,abstained,model_id,ttft_ms,tokens_out)"
            " VALUES (?,?,?,?,?,?,?,?,?,?)",
            (mid, session_id, role, content, _j(citation_map or {}), retrieval_debug,
             int(abstained), model_id, ttft_ms, tokens_out))
        self.con.execute("UPDATE chat_sessions SET updated_at=? WHERE id=?", (utcnow(), session_id))
        return mid

    def messages(self, session_id: str) -> list[dict[str, Any]]:
        return [dict(r) for r in self.con.execute(
            "SELECT * FROM chat_messages WHERE session_id=? ORDER BY created_at", (session_id,))]

    def cumulative_citation_map(self, session_id: str) -> dict[str, str]:
        """Накопичена мапа [n] → chunk_uid по всіх ходах.

        Правило NeoLens: відповідь на уточнювальне питання може цитувати чанк,
        знайдений два ходи тому, тож мапа зберігається НА КОЖНЕ повідомлення
        й зливається в порядку часу.
        """
        merged: dict[str, str] = {}
        for r in self.con.execute(
            "SELECT citation_map_json FROM chat_messages WHERE session_id=? AND role='assistant'"
            " ORDER BY created_at", (session_id,)):
            merged.update(json.loads(r["citation_map_json"] or "{}"))
        return merged

    def clear_messages(self, session_id: str) -> None:
        self.con.execute("DELETE FROM chat_messages WHERE session_id=?", (session_id,))


# -------------------------------------------------------------- телеметрія
class TelemetryRepo(_Repo):
    """ЛИШЕ локальна. Жодного мережевого виклику ніколи."""

    def event(self, name: str, *, duration_ms: int | None = None, ok: bool = True,
              error_code: str | None = None, meta: dict[str, Any] | None = None) -> None:
        self.con.execute(
            "INSERT INTO events (name,duration_ms,ok,error_code,meta_json) VALUES (?,?,?,?,?)",
            (name, duration_ms, int(ok), error_code, _j(meta or {})))

    def unresolved_citation(self, message_id: str, emitted: str, available: Sequence[str]) -> None:
        """Безкоштовна телеметрія галюцинацій: що модель написала і що їй було доступно."""
        self.con.execute(
            "INSERT INTO unresolved_citations (message_id,emitted,available) VALUES (?,?,?)",
            (message_id, emitted, _j(list(available))))

    def feedback(self, message_id: str | None, verdict: str, *,
                 chunk_uid: str | None = None, note: str = "") -> None:
        self.con.execute(
            "INSERT INTO feedback (id,message_id,chunk_uid,verdict,note) VALUES (?,?,?,?,?)",
            (new_id(), message_id, chunk_uid, verdict, note))

    def summary(self) -> dict[str, Any]:
        docs = self.con.execute(
            "SELECT status, count(*) c FROM documents GROUP BY status").fetchall()
        cites = self.con.execute("SELECT count(*) FROM unresolved_citations").fetchone()[0]
        return {
            "documents_by_status": {r["status"]: r["c"] for r in docs},
            "unresolved_citations": cites,
        }
