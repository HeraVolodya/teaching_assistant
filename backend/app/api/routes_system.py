"""Здоров'я, майстер першого запуску, SSE-канал, діагностика, playground.

ЗДОРОВ'Я НЕ МАЄ ПРАВА БУТИ ПОВІЛЬНИМ.
Оболонка опитує `/api/health`, щоб зрозуміти, чи піднявся sidecar, і робить це
щосекунди. Тому перевірка LM Studio кешується на дві секунди, а перевірка
цілісності БД не робиться взагалі (`PRAGMA integrity_check` на 2-гігабайтній
базі — це хвилини); цілісність живе в діагностичному пакеті, який запускають
свідомо.

МАЙСТЕР ПЕРШОГО ЗАПУСКУ — ЦЕ ТРИ ФАКТИ І ОДНА КНОПКА.
Факти: чи є LM Studio, скільки VRAM, які моделі вже лежать на диску. Кнопка:
`POST /api/setup/models/download`, яка проксіює завантаження в LM Studio й
віддає прогрес у той самий SSE-канал. Це найбільший UX-виграш v1 API —
викладач ніколи не відкриває вкладку Discover і не вибирає квантизацію.

ДІАГНОСТИЧНИЙ ПАКЕТ НІКОЛИ НЕ МІСТИТЬ ВМІСТУ ДОКУМЕНТІВ.
Назви — лише за явним увімкненням, вимкненим за замовчуванням. Це військова
академія: перелік завантажених матеріалів сам по собі є інформацією.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import platform
import shutil
import subprocess
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import Response, StreamingResponse

from app.api.events import MODEL_DOWNLOAD, parse_last_event_id
from app.api.routes_assistants import services_of
from app.api.schemas import DownloadIn
from app.api.state import Services
from app.db.repositories import TelemetryRepo

__all__ = ["APP_VERSION", "router"]

log = logging.getLogger("asistent.api.system")

APP_VERSION = "0.1.0"

router = APIRouter(tags=["system"])

# Власники фонових задач завантаження моделей. `asyncio` тримає на задачу лише
# слабке посилання, тож без цієї множини збирач сміття може прибрати її посеред
# роботи — і завантаження обірветься тихо, без винятку й без запису в лог.
_DOWNLOAD_TASKS: set[asyncio.Task[None]] = set()


# ---------------------------------------------------------------- здоров'я
@router.get("/health")
async def health(request: Request) -> dict[str, Any]:
    services = services_of(request)
    db_ok, db_detail, schema = _database_state(services)
    llm = await services.llm_health()
    supervisor = services.supervisor.status() if services.supervisor else {"mode": "off"}
    jobs = services.queue.active()
    return {
        "ok": db_ok,
        "version": APP_VERSION,
        "stub": services.settings.stub,
        "uptimeSeconds": services.uptime_seconds,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "database": {"ok": db_ok, "detail": db_detail, "schemaVersion": schema,
                     "path": str(services.paths.db_path)},
        "lmstudio": llm,
        "worker": supervisor,
        "jobs": {"active": len(jobs),
                 "queued": sum(1 for j in jobs if j["state"] == "QUEUED")},
        "problems": [p.as_dict() for p in services.problems],
        "eventSubscribers": services.events.subscriber_count,
    }


def _database_state(services: Services) -> tuple[bool, str, int]:
    try:
        with services.db.connection() as con:
            row = con.execute("SELECT max(version) FROM schema_version").fetchone()
        return True, "", int(row[0] or 0)
    except Exception as exc:
        return False, str(exc), 0


# ------------------------------------------------------ майстер запуску
@router.get("/setup/status")
async def setup_status(request: Request) -> dict[str, Any]:
    """Усе, що потрібно майстру, одним викликом."""
    services = services_of(request)
    hardware, plan = await asyncio.to_thread(_hardware_and_plan)
    llm = await services.llm_health()
    models = _model_files(services)
    return {
        "hardware": hardware,
        "recommendation": plan,
        "lmstudio": {
            **llm,
            "cliAvailable": shutil.which("lms") is not None,
            "configuredUrl": services.settings.lmstudio_url,
        },
        "models": models,
        "stub": services.settings.stub,
        "dataDir": str(services.paths.data_dir),
        # `chatReady`, а не `ok`: сервер може відповідати, маючи порожню пам'ять,
        # і тоді кожен запит генерації падає з «No models loaded», тоді як
        # майстер першого запуску показує зелений стан і «готово».
        "ready": bool(
            services.settings.stub
            or (llm.get("chatReady") and models["embeddings"]["present"])
        ),
        "problems": [p.as_dict() for p in services.problems],
    }


def _hardware_and_plan() -> tuple[dict[str, Any], dict[str, Any]]:
    """Проба заліза й вибір моделі. Ніколи не кидає: екзотична машина не має
    права завалити майстер першого запуску."""
    try:
        from app.backends.hardware_probe import probe_hardware
        from app.backends.model_registry import plan_generation

        hw = probe_hardware()
        gpu = hw.best_gpu
        hardware = {
            "platform": hw.platform,
            "machine": hw.machine,
            "cpuCount": hw.cpu_count,
            "ramBytes": hw.ram_bytes,
            "vramBytes": hw.vram_bytes,
            "isAppleSilicon": hw.is_apple_silicon,
            "unifiedMemory": hw.unified_memory,
            "gpu": {"name": gpu.name, "totalBytes": gpu.total_bytes} if gpu else None,
            "describe": hw.describe(),
        }
        try:
            plan = plan_generation(hw)
            recommendation: dict[str, Any] = {
                "modelKey": plan.spec.key,
                "title": plan.spec.title,
                "quant": plan.spec.quant,
                "contextLength": plan.n_ctx,
                "gpuOffload": plan.gpu_offload,
                "fileBytes": plan.spec.file_bytes,
                "archVerified": getattr(plan.spec, "arch_verified", True),
            }
        except Exception as exc:
            recommendation = {"error": str(exc)}
        return hardware, recommendation
    except Exception as exc:
        return {"error": str(exc)}, {"error": str(exc)}


def _model_files(services: Services) -> dict[str, Any]:
    """Чи лежать ваги на диску. Перевіряємо файл, а не завантажуємо модель:
    майстер мусить відповідати миттєво."""
    from app.embeddings import registry as embed_registry
    from app.embeddings.provider import default_model_dir as embed_dir
    from app.rerank import reranker as rerank_module

    embed_model = embed_registry.get(services.settings.embedding_model)
    embed_path = embed_dir(embed_model, services.paths.models_dir)
    rerank_model = rerank_module.get(services.settings.rerank_model)
    rerank_path = rerank_module.default_model_dir(rerank_model, services.paths.models_dir)
    return {
        "embeddings": {
            "id": embed_model.id,
            "dim": embed_model.dim,
            "dir": str(embed_path),
            # Рантайм вимагає І токенайзер, І ваги: onnx_backend кидає
            # ModelFilesMissing на відсутньому .onnx. Перевірка лише
            # tokenizer.json давала «present: true» і «ready: true», а перша
            # ж індексація падала — у закритому контурі це найдорожчий клас
            # помилки, бо дозавантажити ваги нема звідки.
            "present": _model_files_present(embed_path, embed_model.onnx_file),
        },
        "rerank": {
            "id": rerank_model.id,
            "dir": str(rerank_path),
            "present": _model_files_present(rerank_path, rerank_model.onnx_file),
        },
        "docling": {
            "dir": str(services.paths.models_dir / "docling"),
            "present": (services.paths.models_dir / "docling").is_dir(),
        },
    }


@router.post("/setup/lmstudio/start")
async def start_lmstudio(request: Request) -> dict[str, Any]:
    """`lms server start`.

    Свідомо НЕ редагуємо конфіг LM Studio й не чіпаємо гардрейли: реєстр
    ризиків прямо забороняє змінювати чужі налаштування за користувача.
    Робимо рівно те, що зробив би він сам у терміналі.
    """
    services = services_of(request)
    binary = shutil.which("lms")
    if binary is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "CLI `lms` не знайдено в PATH. Запустіть LM Studio вручну й "
                "увімкніть локальний сервер: Developer → Start Server."
            ),
        )
    try:
        proc = await asyncio.to_thread(
            subprocess.run, [binary, "server", "start"],
            capture_output=True, text=True, timeout=60,
        )
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(status_code=504, detail="`lms server start` не відповів за 60 с.") from exc
    services.reset_backend()
    health_payload = await services.llm_health()
    return {
        "ok": proc.returncode == 0 and bool(health_payload.get("ok")),
        "returncode": proc.returncode,
        "stdout": (proc.stdout or "").strip()[-2000:],
        "stderr": (proc.stderr or "").strip()[-2000:],
        "lmstudio": health_payload,
    }


@router.post("/setup/models/download")
async def download_model(request: Request, payload: DownloadIn) -> dict[str, Any]:
    """Проксі до `POST /api/v1/models/download` + прогрес у SSE-канал."""
    services = services_of(request)
    backend = await services.backend()
    downloader = getattr(backend, "download_model", None)
    if backend is None or downloader is None:
        raise HTTPException(
            status_code=503,
            detail="LM Studio недоступний — завантажити модель нічим.",
        )
    # LM Studio приймає лише ПОВНИЙ шлях репозиторію ("mlx-community/gemma-4-12b-it-4bit"),
    # а не наше внутрішнє гасло ("gemma-4-12b-mlx-4bit"): на короткий ключ він
    # відповідає `Invalid model name format`. Тому перекладаємо ключ реєстру в
    # репозиторій, а вже готовий шлях пропускаємо як є.
    model_ref = _resolve_model_ref(payload.model)
    try:
        download_id = await downloader(model_ref)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"LM Studio відхилив запит: {exc}") from exc

    # Посилання тримаємо НАВМИСНО: `asyncio` зберігає лише слабке посилання на
    # задачу, тож без власника збирач сміття може прибрати її посеред
    # завантаження моделі — і воно тихо обірветься без жодної помилки.
    task = asyncio.create_task(_watch_download(services, backend, download_id, model_ref))
    _DOWNLOAD_TASKS.add(task)
    task.add_done_callback(_DOWNLOAD_TASKS.discard)
    return {"downloadId": download_id, "model": model_ref}


def _resolve_model_ref(model: str) -> str:
    """Ключ реєстру → шлях репозиторію. Готовий шлях лишається без змін."""
    if "/" in model:
        return model
    from app.backends import model_registry

    try:
        spec = model_registry.get(model)
    except KeyError:
        return model
    if spec.engine == "mlx" and spec.mlx_repo:
        return spec.mlx_repo
    return spec.gguf_repo or model


async def _watch_download(
    services: Services, backend: Any, download_id: str, model: str
) -> None:
    """Опитувати статус і озвучувати його в канал подій.

    Прогрес завантаження 7-гігабайтної моделі без індикатора виглядає як
    зависання — а це перше, що бачить викладач при першому запуску.
    """
    status_fn = getattr(backend, "download_status", None)
    if status_fn is None:
        return
    try:
        while True:
            state = await status_fn(download_id)
            services.emit(MODEL_DOWNLOAD, {
                "downloadId": download_id,
                "model": model,
                "percent": state.get("percent") or state.get("progress"),
                "state": state.get("status") or state.get("state"),
                "raw": state,
            })
            if str(state.get("status") or state.get("state") or "").lower() in {
                "completed", "done", "failed", "cancelled", "error"
            }:
                return
            await asyncio.sleep(1.0)
    except asyncio.CancelledError:  # pragma: no cover
        raise
    except Exception as exc:
        services.emit(MODEL_DOWNLOAD, {
            "downloadId": download_id, "model": model,
            "state": "error", "message": str(exc),
        })


# ------------------------------------------------------------------ SSE
@router.get("/events")
async def events(
    request: Request,
    last_event_id: str | None = Query(default=None, alias="lastEventId"),
) -> StreamingResponse:
    """Єдиний мультиплексований канал подій.

    `Last-Event-ID` браузер надсилає заголовком автоматично при
    перепідключенні; параметр запиту потрібен для `EventSource`-полифілів і
    для ручної діагностики через curl.
    """
    services = services_of(request)
    resume = parse_last_event_id(request.headers.get("last-event-id") or last_event_id)
    stream = services.events.stream(
        last_event_id=resume, keepalive_s=services.settings.event_keepalive_s
    )
    return StreamingResponse(
        stream,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.get("/events/history")
def events_history(request: Request, limit: int = 100) -> list[dict[str, Any]]:
    """Кільцевий буфер як звичайний JSON — для діагностики й тестів."""
    services = services_of(request)
    return [
        {"id": e.id, "type": e.type, "data": e.data}
        for e in services.events.history(limit=limit)
    ]


# --------------------------------------------------------------- діагностика
@router.get("/diagnostics/bundle")
def diagnostics_bundle(
    request: Request,
    include_titles: bool = Query(default=False, alias="includeTitles"),
    integrity: bool = Query(default=False),
) -> Response:
    """ZIP для підтримки. Вмісту документів у ньому немає ніколи."""
    services = services_of(request)
    # ОБИДВІ умови, а не «або». Через пріоритет операторів попередній вираз
    # `a and b or a` згортався до просто `a`: налаштування
    # `diagnostics_include_titles`, задокументоване як «вимкнено за
    # замовчуванням: військова академія, самі назви матеріалів є інформацією»,
    # не мало жодного впливу, і `?includeTitles=true` вивантажувало назви
    # документів у пакет підтримки в обхід політики.
    allow_titles = bool(include_titles and services.settings.diagnostics_include_titles)
    buffer = io.BytesIO()
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")

    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("readme.txt", _README)
        zf.writestr("settings.json", _dump(services.settings.describe()))
        zf.writestr("environment.json", _dump(_environment(services)))
        zf.writestr("jobs.json", _dump(services.queue.recent(limit=200)))
        zf.writestr("documents.json", _dump(_documents_summary(services, allow_titles)))
        zf.writestr("telemetry.json", _dump(_telemetry_summary(services)))
        zf.writestr("events.json", _dump(
            [{"id": e.id, "type": e.type, "data": e.data} for e in services.events.history(200)]
        ))
        if integrity:
            # Свідомо за окремим прапорцем: на 2-гігабайтній базі це хвилини.
            zf.writestr("integrity.txt", services.db.integrity_check())
        for log_file in sorted(services.paths.logs_dir.glob("*.log"))[-3:]:
            try:
                zf.writestr(f"logs/{log_file.name}", _tail(log_file))
            except OSError as exc:  # pragma: no cover
                zf.writestr(f"logs/{log_file.name}.error", str(exc))

    data = buffer.getvalue()
    return Response(
        content=data,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="asistent-diagnostics-{stamp}.zip"',
            "Content-Length": str(len(data)),
        },
    )


_README = (
    "Діагностичний пакет застосунку «Асістент».\n\n"
    "Що тут Є: налаштування, стан черги завдань, лічильники документів,\n"
    "локальна телеметрія, останні події SSE, хвости журналів.\n\n"
    "Чого тут НЕМАЄ і не буде НІКОЛИ: вмісту навчальних матеріалів, тексту\n"
    "чанків, векторів. Назви документів включаються лише за явним\n"
    "увімкненням і за замовчуванням вимкнені.\n"
)


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def _environment(services: Services) -> dict[str, Any]:
    return {
        "version": APP_VERSION,
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "dataDir": str(services.paths.data_dir),
        "uptimeSeconds": services.uptime_seconds,
        "problems": [p.as_dict() for p in services.problems],
        "worker": services.supervisor.status() if services.supervisor else None,
    }


def _documents_summary(services: Services, include_titles: bool) -> list[dict[str, Any]]:
    with services.db.connection() as con:
        rows = con.execute(
            "SELECT id, collection_id, status, error_code, page_count, quality_grade,"
            " ingest_mode, language, doc_type, title FROM documents ORDER BY created_at"
        ).fetchall()
        counts = {
            r["document_id"]: r["c"]
            for r in con.execute(
                "SELECT document_id, count(*) AS c FROM chunks GROUP BY document_id"
            )
        }
    out: list[dict[str, Any]] = []
    for row in rows:
        # `dict(row)`, А НЕ `for k in row`. `sqlite3.Row` при ітерації віддає
        # ЗНАЧЕННЯ, а не ключі, тож `row[k]` шукав би колонку за її ж вмістом і
        # падав з `IndexError: No item with that key`. У вихідному коді стояло
        # `row.keys()`; `.keys()` прибрало автовиправлення ruff SIM118 (коміт
        # d040fca) — для словника воно слушне, для Row хибне. Форма через
        # `.items()` цього не переживе повторно: SIM118 до неї не застосовна.
        item = {k: v for k, v in dict(row).items() if k != "title"}
        item["chunks"] = counts.get(row["id"], 0)
        if include_titles:
            item["title"] = row["title"]
        out.append(item)
    return out


def _telemetry_summary(services: Services) -> dict[str, Any]:
    with services.db.connection() as con:
        summary = TelemetryRepo(con).summary()
        events_rows = [
            dict(r)
            for r in con.execute(
                "SELECT name, count(*) AS n, avg(duration_ms) AS avg_ms,"
                " sum(CASE WHEN ok=0 THEN 1 ELSE 0 END) AS failures"
                " FROM events GROUP BY name ORDER BY n DESC LIMIT 50"
            )
        ]
        feedback_rows = [
            dict(r)
            for r in con.execute(
                "SELECT verdict, count(*) AS n FROM feedback GROUP BY verdict"
            )
        ]
    return {"summary": summary, "events": events_rows, "feedback": feedback_rows}


def _tail(path: Any, limit: int = 200_000) -> str:
    size = path.stat().st_size
    with path.open("rb") as fh:
        if size > limit:
            fh.seek(size - limit)
        return fh.read().decode("utf-8", errors="replace")


# ------------------------------------------------------------- playground
@router.get("/dev/retrieval-playground")
def retrieval_playground(
    request: Request,
    collection_id: str = Query(alias="collectionId"),
    q: str = Query(min_length=1),
    top_k: int = Query(default=10, alias="topK", ge=1, le=100),
    rerank: bool = Query(default=True),
) -> dict[str, Any]:
    """Топ-k ДО і ПІСЛЯ реранкінгу на одному екрані.

    Це інструмент калібрування параметра №1 плану (`final_top_k` /
    `max_per_document` / `relative_score_floor`): суперечність між top-2 UNLP
    і диверсифікацією джерел розсуджується вимірюванням на власному корпусі,
    а не здогадом, і мірятиме її саме цей екран.
    """
    services = services_of(request)
    retriever = services.retriever
    if retriever is None:
        raise HTTPException(status_code=503, detail="Модель ембедингів недоступна.")

    from app.domain import AssistantConfig

    config = AssistantConfig(final_top_k=top_k, rerank_top_k=max(top_k, 20))
    saved = retriever.reranker
    if not rerank:
        retriever.reranker = None
    try:
        chunks, debug = retriever.retrieve(q, config, collection_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    finally:
        retriever.reranker = saved

    fused = [row for row in debug.candidates if row.get("kind") == "candidate"]
    return {
        "query": q,
        "collectionId": collection_id,
        "reranked": rerank and saved is not None,
        "before": [
            {
                "chunkUid": row.get("chunk_uid"),
                "documentTitle": row.get("document_title"),
                "pages": row.get("pages"),
                "rrf": row.get("rrf_score"),
                "dense": row.get("dense_score"),
                "sparse": row.get("sparse_score"),
                "ngram": row.get("ngram_score"),
                "rerank": row.get("rerank"),
                "selected": row.get("selected"),
            }
            for row in fused[: top_k * 4]
        ],
        "after": [
            {
                "ordinal": i,
                "chunkUid": rc.chunk.chunk_uid,
                "documentTitle": rc.document_title,
                "pages": rc.chunk.citation_label(),
                "level": rc.chunk.level.value,
                "score": rc.score,
                "rerank": rc.rerank_score,
                "fused": rc.fused_score,
                "text": rc.chunk.display_text[:400],
            }
            for i, rc in enumerate(chunks, start=1)
        ],
        "debug": {
            "denseCount": debug.dense_count,
            "sparseCount": debug.sparse_count,
            "ngramCount": debug.ngram_count,
            "fusedCount": debug.fused_count,
            "rerankedCount": debug.reranked_count,
            "finalCount": debug.final_count,
            "distinctDocuments": debug.distinct_documents,
            "abstained": debug.abstained,
            "confidence": debug.abstain_confidence,
            "latencyMs": debug.latency_ms,
        },
    }

def _model_files_present(model_dir: Path, onnx_file: str) -> bool:
    """Модель придатна лише коли на місці І токенайзер, І ваги."""
    if not (model_dir / "tokenizer.json").is_file():
        return False
    weights = model_dir / onnx_file
    # Великі експорти кладуть вагу поруч як `<name>.onnx_data`; сам .onnx тоді
    # маленький, але без сусіда сесія не піднімається.
    return weights.is_file()
