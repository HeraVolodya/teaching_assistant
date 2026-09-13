"""Конвеєр приймання документа: PROBE → PARSE → CHUNK → INDEX.

    PROBE            тріаж рівня 0: сторінки, класи, ваги вартості, план OCR
    PARSE            Docling посторінковими вікнами (кооперативне скасування)
    ENRICH_FIGURES   лише DEEP: опис рисунків локальною VLM         (див. нижче)
    VLM_REPAIR       лише сторінки, що провалилися                  (див. нижче)
    CHUNK            L0/L1/L2 + дерево розділів
    INDEX            ембединги + FTS5; збірка ANN — окреме завдання колекції

ЧОМУ ЗБІРКА ANN — ОКРЕМЕ ЗАВДАННЯ
Граф HNSW будується на ВСЮ колекцію. Перебудовувати його після кожного з
двадцяти завантажених підручників означало б зробити двадцять збірок замість
однієї. Документ стає READY одразу після ембедингів і FTS, а індекс наздоганяє
окремим завданням: доти `CollectionIndex` сам деградує до точного пошуку з
SQLite — повільніше, але завжди правильно, тож «індекс застарів» ніколи не
перетворюється на помилку в чаті.

ЧОМУ ЦЕЙ МОДУЛЬ НЕ ІМПОРТУЄ DOCLING НА ВЕРХНЬОМУ РІВНІ
`app.ingestion` тримає всі важкі імпорти лінивими всередині функцій, тож сам
факт імпорту конвеєра нічого не коштує. Це те, що дозволяє запускати конвеєр
у режимі `inline` (розробка, CI, stub) без torch у процесі — і водночас не
заважає воркеру в постачанні тягнути повний Docling.

РІВНІ 1.5 і 2 (ENRICH_FIGURES, VLM_REPAIR) СВІДОМО НЕ ВИКОНУЮТЬСЯ ТУТ.
Вони потребують перемикання LM Studio з чат-моделі на VLM
(`unload(chat) → load(vlm) → … → unload(vlm) → load(chat)`), бо на 8–12 ГБ
VRAM вони не співіснують. Конвеєр натомість ЧЕСНО рахує, скільки сторінок
потребують ремонту (`PageInfo.needs_vlm_repair`), пише це в телеметрію й
повертає у звіті — щоб цифра була видима, а не мовчазна.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.config import Paths
from app.db.repositories import (
    ChunkRepo,
    CollectionRepo,
    DocumentRepo,
    PageRepo,
    TelemetryRepo,
)
from app.domain import (
    Chunk,
    ChunkLevel,
    DocStatus,
    Document,
    IngestMode,
    JobType,
    PageInfo,
    QualityGrade,
)
from app.jobs.queue import JobCancelled, JobQueue, Lease

__all__ = [
    "IngestReport",
    "StageWeights",
    "PipelineError",
    "resolve_document_path",
    "ingest_document",
    "build_collection_index_job",
    "reuse_cached_document",
    "content_sha256",
]

log = logging.getLogger("asistent.pipeline")

# Розподіл ваги між етапами відносно суми ваг сторінок. Парсинг домінує
# на порядок; чанкування майже безкоштовне; ембединги коштують помітно, бо
# на CPU це реальні матричні множення, а не читання файлу.
STAGE_PARSE = 1.0
STAGE_CHUNK = 0.08
STAGE_INDEX = 0.25


@dataclass(frozen=True, slots=True)
class StageWeights:
    pages: float

    @property
    def total(self) -> float:
        return self.pages * (STAGE_PARSE + STAGE_CHUNK + STAGE_INDEX)

    @property
    def parse_end(self) -> float:
        return self.pages * STAGE_PARSE

    @property
    def chunk_end(self) -> float:
        return self.pages * (STAGE_PARSE + STAGE_CHUNK)


class PipelineError(RuntimeError):
    """Помилка з кодом і українською підказкою — рівно те, що йде в UI."""

    def __init__(self, code: str, message: str, hint: str = "") -> None:
        super().__init__(message)
        self.code = code
        self.hint = hint


@dataclass(slots=True)
class IngestReport:
    document_id: str
    pages: int = 0
    chunks: int = 0
    leaves: int = 0
    tables: int = 0
    formulas: int = 0
    pictures: int = 0
    truncated: int = 0
    needs_repair_pages: int = 0
    quality: QualityGrade | None = None
    backend: str = ""
    reused_from: str | None = None
    warnings: list[str] = field(default_factory=list)
    duration_ms: int = 0

    def as_event(self) -> dict[str, Any]:
        """Payload події `doc.ready`."""
        return {
            "docId": self.document_id,
            "chunks": self.leaves,
            "tables": self.tables,
            "formulas": self.formulas,
            "pictures": self.pictures,
            "pages": self.pages,
            "quality": self.quality.value if self.quality else None,
            "reusedFrom": self.reused_from,
            "warnings": self.warnings[:8],
        }


EmitFn = Callable[[str, dict[str, Any]], None]


def _noop(_type: str, _data: dict[str, Any]) -> None:
    return None


# --------------------------------------------------------------------- шляхи
def resolve_document_path(paths: Paths, document: Document) -> Path:
    """Абсолютний шлях до збереженого оригіналу.

    `stored_path` зберігається ВІДНОСНИМ до каталогу документів: користувач
    може перенести каталог даних на інший диск (закритий контур, встановлення
    з USB), і абсолютний шлях у БД перетворив би це на масову втрату файлів.
    """
    stored = Path(document.stored_path)
    if stored.is_absolute():
        return stored
    return (paths.documents_dir / stored).resolve()


# ------------------------------------------------------------------- конвеєр
def ingest_document(
    db: Any,
    queue: JobQueue,
    lease: Lease,
    *,
    document_id: str,
    paths: Paths,
    provider: Any,
    ingest_mode: IngestMode | None = None,
    device: str = "cpu",
    emit: EmitFn | None = None,
) -> IngestReport:
    """Повний конвеєр одного документа. Кидає `JobCancelled` при скасуванні."""
    emit = emit or _noop
    started = time.perf_counter()

    with db.connection() as con:
        document = DocumentRepo(con).get(document_id)
    if document is None:
        raise PipelineError(
            "DOC_NOT_FOUND",
            f"Документ {document_id} зник із бази під час обробки.",
            "Найімовірніше, його видалили. Завантажте файл ще раз.",
        )

    path = resolve_document_path(paths, document)
    if not path.is_file():
        raise PipelineError(
            "FILE_MISSING",
            f"Файл документа не знайдено: {path}.",
            "Перевірте, чи не видалено каталог даних. Завантажте файл ще раз.",
        )

    mode = ingest_mode or document.ingest_mode
    report = IngestReport(document_id=document_id)

    # ---------------------------------------------------------- 0. PROBE
    _set_status(db, document_id, DocStatus.PROBING)
    lease.checkpoint(stage="PROBE", force=True)
    probe = _probe(path)
    report.pages = probe.page_count
    report.warnings.extend(probe.warnings)

    weights = StageWeights(pages=max(probe.total_weight, 1.0))
    lease.set_total(weights.total)
    lease.checkpoint(stage="PROBE", force=True)

    with db.transaction() as con:
        DocumentRepo(con).set_page_count(document_id, probe.page_count)
        PageRepo(con).upsert_many(document_id, [p.to_page_info() for p in probe.pages])
    queue.init_pages(document_id, [p.page_number for p in probe.pages])

    # ------------------------------------------- 0.5 кеш за хешем контенту
    opts = _parse_options(document, mode, device, lease)
    profile_hash = _profile_hash(opts, path)
    cached = _find_cached(db, document, profile_hash)
    if cached is not None:
        reused = reuse_cached_document(db, source=cached, target=document)
        if reused:
            report.reused_from = cached.id
            report.leaves = reused
            _finish_document(db, document_id, probe, report, profile_hash)
            report.duration_ms = int((time.perf_counter() - started) * 1000)
            queue.mark_pages(document_id, [p.page_number for p in probe.pages], "DONE")
            emit("doc.ready", report.as_event())
            return report

    # ---------------------------------------------------------- 1. PARSE
    _set_status(db, document_id, DocStatus.PARSING)
    lease.checkpoint(stage="PARSE", force=True)
    parsed = _parse(path, opts, lease)
    report.backend = parsed.backend
    report.warnings.extend(parsed.warnings)
    if lease.is_cancelled():
        raise JobCancelled(lease.job.id)

    page_infos = _page_infos(probe, parsed)
    with db.transaction() as con:
        PageRepo(con).upsert_many(document_id, page_infos)
    queue.mark_pages(document_id, [p.page_number for p in page_infos], "DONE")
    report.needs_repair_pages = sum(1 for p in page_infos if p.needs_vlm_repair())
    lease.tracker.set_done(weights.parse_end)
    lease.checkpoint(stage="PARSE", force=True)

    # ------------------------------- 1.5/2. ENRICH_FIGURES / VLM_REPAIR
    if mode is IngestMode.DEEP:
        _set_status(db, document_id, DocStatus.ENRICHING)
        lease.checkpoint(stage="ENRICH_FIGURES", force=True)
        report.warnings.append(
            "Поглиблений режим: опис рисунків і VLM-ремонт сторінок ще не "
            "під'єднані — потрібне перемикання LM Studio на VLM-модель."
        )
    if report.needs_repair_pages:
        _telemetry(
            db,
            "vlm_repair_needed",
            meta={"document_id": document_id, "pages": report.needs_repair_pages},
        )

    # ---------------------------------------------------------- 3. CHUNK
    _set_status(db, document_id, DocStatus.CHUNKING)
    lease.checkpoint(stage="CHUNK", force=True)
    tree = _chunk(parsed, document)
    _persist_chunks(db, document, parsed, tree)
    report.chunks = len(tree.chunks)
    leaves = tree.leaves()
    report.leaves = len(leaves)
    report.tables = sum(len(c.tables) for c in tree.chunks if c.level is ChunkLevel.LEAF)
    report.formulas = sum(len(c.formulas) for c in tree.chunks if c.level is ChunkLevel.LEAF)
    report.pictures = sum(
        len(c.pictures) for c in tree.chunks if c.level is ChunkLevel.LEAF
    )
    lease.tracker.set_done(weights.chunk_end)
    lease.checkpoint(stage="CHUNK", force=True)

    if not leaves:
        raise PipelineError(
            "NO_TEXT",
            "З документа не вдалося витягти жодного фрагмента тексту.",
            "Найімовірніше, це скан без текстового шару, а OCR не спрацював. "
            "Спробуйте режим «Поглиблено» або перевірте якість сканування.",
        )

    # ---------------------------------------------------------- 4. INDEX
    _set_status(db, document_id, DocStatus.INDEXING)
    lease.checkpoint(stage="INDEX", force=True)
    _index_chunks(db, leaves, provider=provider, lease=lease, weights=weights)

    _finish_document(db, document_id, probe, report, profile_hash)
    with db.transaction() as con:
        CollectionRepo(con).mark_dirty(document.collection_id, True)
    queue.enqueue_once(JobType.INDEX, collection_id=document.collection_id)

    lease.tracker.set_done(weights.total)
    lease.checkpoint(stage="READY", force=True)
    report.duration_ms = int((time.perf_counter() - started) * 1000)
    _telemetry(
        db,
        "document_ingested",
        duration_ms=report.duration_ms,
        meta={
            "document_id": document_id,
            "pages": report.pages,
            "leaves": report.leaves,
            "backend": report.backend,
            "mode": mode.value,
        },
    )
    emit("doc.ready", report.as_event())
    return report


# ---------------------------------------------------------------- складові
def _probe(path: Path) -> Any:
    from app.ingestion.probe import probe_pdf, probe_text

    if path.suffix.lower() == ".pdf":
        return probe_pdf(path)
    return probe_text(
        path.read_text(encoding="utf-8", errors="replace"), source_path=str(path)
    )


def _parse_options(document: Document, mode: IngestMode, device: str, lease: Lease) -> Any:
    from app.ingestion.docling_pipeline import ParseOptions

    # `for_mode`, а не `ParseOptions(...)`: склад конвеєра мусить визначатися
    # режимом в ОДНОМУ місці. Коли режим задавався лише полем `ingest_mode`,
    # дороге VLM-збагачення лишалось увімкненим і в «Швидко».
    return ParseOptions.for_mode(
        mode,
        # GPU за замовчуванням НЕ береться: він уже зайнятий LM Studio, і
        # конкуренція дає CUDA OOM саме на завантаженні чат-моделі — тобто
        # викладач переживає це як «ШІ перестав працювати».
        device=device,  # type: ignore[arg-type]
        title=document.title,
        language=document.language,
        cancel=lease.poll_cancel,
        progress=lambda done, _total: lease.report(done),
    )


def _profile_hash(opts: Any, path: Path) -> str:
    from app.ingestion.docling_pipeline import build_parse_profile, parse_profile_hash

    backend = "docling" if path.suffix.lower() == ".pdf" else "text"
    return parse_profile_hash(build_parse_profile(opts, backend=backend))


def _parse(path: Path, opts: Any, lease: Lease) -> Any:
    from app.ingestion.docling_pipeline import parse_document

    try:
        return parse_document(path, opts)
    except ValueError as exc:
        raise PipelineError("UNSUPPORTED_FORMAT", str(exc),
                            "Конвертуйте файл у PDF і завантажте ще раз.") from exc
    except JobCancelled:
        raise
    except MemoryError as exc:
        raise PipelineError(
            "OUT_OF_MEMORY",
            "Не вистачило пам'яті на обробку документа.",
            "Закрийте LM Studio на час індексації або розділіть файл на частини.",
        ) from exc
    except Exception as exc:  # noqa: BLE001 — воркер не має права падати мовчки
        raise PipelineError(
            "PARSE_FAILED",
            f"Не вдалося опрацювати документ: {exc}",
            "Перевірте, чи файл не пошкоджений. Діагностичний пакет містить деталі.",
        ) from exc


def _page_infos(probe: Any, parsed: Any) -> list[PageInfo]:
    """Метрики проби + мітки й оцінки Docling в один рядок таблиці `pages`."""
    by_number = {p.page_number: p for p in probe.pages}
    out: list[PageInfo] = []
    for page in parsed.pages:
        base = by_number.get(page.page_number)
        info = base.to_page_info(page.page_label) if base is not None else PageInfo(
            page_number=page.page_number, page_label=page.page_label
        )
        info.width = page.width or info.width
        info.height = page.height or info.height
        info.char_from = page.char_from or info.char_from
        info.char_to = page.char_to or info.char_to
        info.page_class = page.page_class
        info.ocr_mode = page.ocr_mode
        info.cost_weight = page.cost_weight or info.cost_weight
        # ConfidenceReport від Docling є не завжди (текстовий фолбек його не
        # дає); перезаписувати виміряне None означало б стерти дані проби.
        for name in ("parse_score", "layout_score", "table_score", "ocr_score",
                     "lexicon_hit_rate", "cyrillic_ratio", "mojibake_ratio"):
            value = getattr(page, name, None)
            if value is not None:
                setattr(info, name, value)
        out.append(info)
    return out or [p.to_page_info() for p in probe.pages]


def _chunk(parsed: Any, document: Document) -> Any:
    from app.ingestion.chunker import ChunkConfig, chunk_document_tree

    cfg = ChunkConfig(
        language=document.language,
        document_id=document.id,
        collection_id=document.collection_id,
        doc_type=document.doc_type,
    )
    return chunk_document_tree(
        parsed, cfg, document_id=document.id, collection_id=document.collection_id
    )


def _persist_chunks(db: Any, document: Document, parsed: Any, tree: Any) -> None:
    """Розділи, чанки, зв'язки батьківства — однією транзакцією.

    `parent_id` чанка — це INTEGER, і він існує лише ПІСЛЯ вставки. Тому
    чанкер віддає зв'язки за стабільними `chunk_uid`, ми вставляємо, а потім
    проставляємо справжні id окремим UPDATE. Порядок у `tree.chunks`
    (L0 → усі L1 → усі L2) гарантує, що батько вже вставлений.
    """
    from app.ingestion.chunker import build_chapters

    chapters = build_chapters(parsed, document.id)
    with db.transaction() as con:
        ChunkRepo(con).delete_for_document(document.id)
        chapter_ids = _insert_chapters(con, document.id, chapters)
        _attach_chapters(tree.chunks, chapters, chapter_ids)
        ChunkRepo(con).insert_many(tree.chunks)
        tree.apply_parent_ids()
        # Прямий UPDATE: у `ChunkRepo` немає сетера батька, а `repositories.py`
        # — спільний фундамент, який цей модуль не редагує.
        con.executemany(
            "UPDATE chunks SET parent_id=? WHERE id=?",
            [(c.parent_id, c.id) for c in tree.chunks if c.parent_id and c.id],
        )


def _insert_chapters(con: Any, document_id: str, chapters: Sequence[Any]) -> list[int]:
    """Вставити дерево розділів, ПЕРЕКЛАВШИ індекси списку в справжні id.

    Тут зшиваються два чужі модулі, які розійшлися в контракті одного поля:
    `chunker.build_chapters()` повертає `Chapter.parent_id` як ІНДЕКС у
    списку (справжніх id ще не існує), а `ChapterRepo.replace_for_document()`
    вставляє це поле ДОСЛІВНО як зовнішній ключ. Разом вони дають
    `FOREIGN KEY constraint failed` на будь-якому документі з підзаголовком —
    тобто на кожному підручнику. Завдання приймання — єдине місце, яке бачить
    обидві сторони, тож переклад робиться тут; жоден із двох модулів при цьому
    не редагується.
    """
    con.execute("DELETE FROM chapters WHERE document_id=?", (document_id,))
    ids: list[int] = []
    for chapter in chapters:
        # Порядок гарантовано пре-ордерний: батько додається до списку раніше
        # за дитину, тож його справжній id уже відомий.
        parent = ids[chapter.parent_id] if chapter.parent_id is not None else None
        cur = con.execute(
            "INSERT INTO chapters (document_id,parent_id,level,title,lft,rgt)"
            " VALUES (?,?,?,?,?,?)",
            (document_id, parent, chapter.level, chapter.title, chapter.lft, chapter.rgt),
        )
        ids.append(int(cur.lastrowid))
    return ids


def _attach_chapters(
    chunks: Sequence[Chunk], chapters: Sequence[Any], chapter_ids: Sequence[int]
) -> None:
    """Проставити `chunk.chapter_id` за останнім сегментом `header_path`.

    Без цього фасет `chapter_id` у `chunk_facets` завжди NULL, і фільтр «шукай
    лише в Розділі 2» — той самий range scan по вкладених множинах, заради
    якого дерево й будувалося, — не має за що зачепитися.
    """
    if not chapters:
        return
    by_title: dict[str, int] = {}
    for chapter, chapter_id in zip(chapters, chapter_ids, strict=True):
        # Останній однойменний заголовок виграє: у підручниках «Загальні
        # положення» трапляється в кожному розділі, і прив'язка до першого
        # була б просто хибною.
        by_title[chapter.title.strip()] = chapter_id
    for chunk in chunks:
        tail = chunk.header_path.strip("/").split("//")[-1].strip()
        if tail:
            chunk.chapter_id = by_title.get(tail)


def _index_chunks(
    db: Any,
    leaves: Sequence[Chunk],
    *,
    provider: Any,
    lease: Lease,
    weights: StageWeights,
    batch: int = 64,
) -> None:
    """Ембединги + FTS5 пачками.

    Пачки навмисно невеликі: письменник у WAL один, і транзакція на всі 5000
    чанків підручника заблокувала б чат на хвилини. Плюс кожна пачка — це
    точка кооперативного скасування.
    """
    from app.ingestion.lemmatize import Lemmatizer, SqliteLemmaCache

    total = len(leaves)
    if not total:
        return
    span = max(0.0, weights.total - weights.chunk_end)

    for start in range(0, total, batch):
        window = list(leaves[start : start + batch])
        vectors = provider.embed_documents([c.embed_text for c in window])
        with db.transaction() as con:
            repo = ChunkRepo(con)
            repo.set_embeddings(
                [(c.id, v) for c, v in zip(window, vectors, strict=True) if c.id is not None],
                provider.model_key,
            )
            # Кеш словоформ живе в БД саме тому, що аналізатор має викликатись
            # раз на УНІКАЛЬНУ словоформу за всю історію бази, а не раз на
            # документ: 10^6 викликів замість 10^8.
            lemmatizer = Lemmatizer(cache=SqliteLemmaCache(con))
            for chunk in window:
                if chunk.id is None:
                    continue
                stream = lemmatizer.analyze(chunk.display_text, chunk.language)
                repo.index_fts(chunk.id, *stream.as_fts_columns())
        done = weights.chunk_end + span * min(1.0, (start + len(window)) / total)
        lease.tracker.set_done(done)
        lease.checkpoint(stage="INDEX")


# ------------------------------------------------------------------ кеш
def _find_cached(db: Any, document: Document, profile_hash: str) -> Document | None:
    with db.connection() as con:
        cached = DocumentRepo(con).find_cached(document.content_sha256, profile_hash)
    if cached is None or cached.id == document.id:
        return None
    return cached


def reuse_cached_document(db: Any, *, source: Document, target: Document) -> int:
    """Скопіювати сторінки, чанки, вектори й FTS з уже опрацьованого документа.

    Той самий підручник, той самий профіль парсингу → результат ідентичний
    байт-у-байт, тож повторний парсинг — це години марного CPU. Саме це робить
    додавання спільного матеріалу до ДРУГОГО асистента миттєвим, а
    багатоасистентність — практичною.

    Копіюються рядки, а не посилання: колекції ізольовані фізично (один файл
    індексу на колекцію), і спільний рядок чанка зруйнував би цю ізоляцію.
    `chunk_uid` рахується від `document_id`, тож нові uid не колідують зі
    старими за побудовою.
    """
    with db.transaction() as con:
        rows = con.execute(
            "SELECT * FROM chunks WHERE document_id=? ORDER BY id", (source.id,)
        ).fetchall()
        if not rows:
            return 0

        pages = con.execute(
            "SELECT * FROM pages WHERE document_id=? ORDER BY page_number", (source.id,)
        ).fetchall()
        page_repo = PageRepo(con)
        page_repo.upsert_many(
            target.id,
            [
                PageInfo(
                    page_number=p["page_number"], page_label=p["page_label"],
                    char_from=p["char_from"] or 0, char_to=p["char_to"] or 0,
                    width=p["width"] or 0.0, height=p["height"] or 0.0,
                    cost_weight=p["cost_weight"], lexicon_hit_rate=p["lexicon_hit_rate"],
                    cyrillic_ratio=p["cyrillic_ratio"], mojibake_ratio=p["mojibake_ratio"],
                    parse_score=p["parse_score"], layout_score=p["layout_score"],
                    table_score=p["table_score"], ocr_score=p["ocr_score"],
                )
                for p in pages
            ],
        )

        ChunkRepo(con).delete_for_document(target.id)
        old_to_new: dict[int, int] = {}
        leaves = 0
        for row in rows:
            data = dict(row)
            old_id = int(data.pop("id"))
            old_parent = data.pop("parent_id", None)
            data["document_id"] = target.id
            data["collection_id"] = target.collection_id
            data["parent_id"] = old_to_new.get(int(old_parent)) if old_parent else None
            data["chunk_uid"] = _rekey_uid(target.id, data["ordinal"], data["display_text"])
            data.pop("created_at", None)
            columns = ",".join(data)
            marks = ",".join("?" * len(data))
            cur = con.execute(
                f"INSERT INTO chunks ({columns}) VALUES ({marks})", tuple(data.values())
            )
            new_id = int(cur.lastrowid)
            old_to_new[old_id] = new_id
            if data["level"] == ChunkLevel.LEAF.value:
                leaves += 1
                con.execute(
                    "INSERT INTO chunk_facets (chunk_id,collection_id,document_id,doc_type,"
                    "language,year,chapter_id) SELECT ?,?,?,d.doc_type,?,d.year,?"
                    " FROM documents d WHERE d.id=?",
                    (new_id, target.collection_id, target.id, data["language"],
                     data["chapter_id"], target.id),
                )
                fts = con.execute(
                    "SELECT lemmas, forms, codes FROM chunk_fts WHERE rowid=?", (old_id,)
                ).fetchone()
                if fts is not None:
                    ChunkRepo(con).index_fts(new_id, fts[0] or "", fts[1] or "", fts[2] or "")
        return leaves


def _rekey_uid(document_id: str, ordinal: int, text: str) -> str:
    from app.domain import chunk_uid_for

    return chunk_uid_for(document_id, int(ordinal), text or "")


# ------------------------------------------------------------- завершення
def _finish_document(
    db: Any, document_id: str, probe: Any, report: IngestReport, profile_hash: str
) -> None:
    grade = _quality(probe)
    report.quality = grade
    with db.transaction() as con:
        repo = DocumentRepo(con)
        repo.set_parse_profile(document_id, profile_hash)
        repo.set_quality(document_id, grade)
        repo.set_status(document_id, DocStatus.READY)


def _quality(probe: Any) -> QualityGrade:
    """Оцінка якості документа — за НАЙГІРШИМИ сторінками, не за середнім.

    Середнє по тисячі сторінок ховає ті тридцять, що розпізналися на сміття,
    а цитата з них буде саме такою. Беремо 10-й перцентиль: він чутливий до
    поганого хвоста і не смикається від однієї порожньої сторінки.
    """
    scores: list[float] = []
    for page in getattr(probe, "pages", []):
        hit = getattr(page, "lexicon_hit_rate", None)
        if hit is not None:
            scores.append(float(hit))
        elif getattr(page, "chars_per_page", 0) > 40:
            scores.append(0.85)
    if not scores:
        return QualityGrade.FAIR
    scores.sort()
    index = max(0, int(len(scores) * 0.10) - 1) if len(scores) >= 10 else 0
    return QualityGrade.from_score(scores[index])


def _set_status(db: Any, document_id: str, status: DocStatus, **kw: Any) -> None:
    with db.transaction() as con:
        DocumentRepo(con).set_status(document_id, status, **kw)


def _telemetry(db: Any, name: str, **kw: Any) -> None:
    try:
        with db.transaction() as con:
            TelemetryRepo(con).event(name, **kw)
    except Exception:  # noqa: BLE001 — телеметрія не має права валити конвеєр
        log.debug("Не вдалося записати подію телеметрії %s", name, exc_info=True)


# ------------------------------------------------- завдання рівня колекції
def build_collection_index_job(
    db: Any, collection_id: str, *, index_dir: Path | str, lease: Lease | None = None
) -> dict[str, Any]:
    """Зібрати ANN-індекс колекції. Викликається окремим завданням INDEX."""
    from app.retrieval.vector_index import build_collection_index

    if lease is not None:
        lease.checkpoint(stage="INDEX", force=True)
    report = build_collection_index(db, collection_id, index_dir=index_dir)
    return {
        "collectionId": collection_id,
        "buildId": report.build_id,
        "count": report.count,
        "backend": str(report.backend),
        "generation": report.index_generation,
    }


def content_sha256(path: Path, *, chunk: int = 1024 * 1024) -> str:
    """Хеш вмісту потоково: 300-мегабайтний підручник не має жити в RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()
