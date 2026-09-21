"""Чат: сесії, повідомлення, SSE-стрім відповіді, екран «Чому ця відповідь».

ЧОМУ ВІДПОВІДЬ СТРІМИТЬСЯ ДВІЧІ
Кадри йдуть і в тіло `POST /api/chat`, і в спільний канал `GET /api/events`.
Це не дублювання з недогляду: перше потрібне тому, що клієнт, який поставив
питання, мусить отримати токени на тому самому запиті (інакше він змушений
корелювати два з'єднання, і гонка між ними видима як «відповідь почалась
раніше, ніж з'явилось повідомлення»); друге — тому, що друге вікно, вкладка
історії чи перепідключений WebView мусять бачити той самий хід.

ПОРЯДОК КРОКІВ ЖОРСТКИЙ
  1. зберегти питання користувача — до будь-чого іншого, бо саме воно має
     пережити падіння моделі;
  2. створити ПОРОЖНЄ повідомлення асистента й одразу віддати його id —
     без стабільного `messageId` кадри `chat.token` нема до чого приліпити;
  3. пошук (у потоці: він синхронний і CPU-bound, а event loop мусить далі
     обслуговувати SSE інших вкладок);
  4. генерація;
  5. дописати повідомлення — у `finally`, тобто навіть при обриві: викладач
     бачив половину відповіді, і ця половина мусить лишитись в історії.

УТРИМАННЯ ВИРІШУЄТЬСЯ ДО МОДЕЛІ
Якщо ретривер підняв `debug.abstained`, генератор не викликає LLM узагалі.
Дати моделі докази, які код уже визнав недостатніми, — це дати їй шанс
вигадати. Поріг живе в `AssistantConfig`, тобто в коді, а не в промпті
(контракт, правило 8).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.api.events import (
    CHAT_CITATIONS,
    CHAT_DEBUG,
    CHAT_DONE,
    CHAT_ERROR,
    CHAT_TOKEN,
    sse_frame,
)
from app.api.routes_assistants import services_of
from app.api.schemas import (
    ChatIn,
    FeedbackIn,
    MessageOut,
    SessionIn,
    SessionOut,
    citation_out,
)
from app.api.state import Services
from app.db.repositories import (
    AssistantRepo,
    ChatRepo,
    ChunkRepo,
    CollectionRepo,
    TelemetryRepo,
)
from app.domain import Assistant, Citation, RetrievalDebug, RetrievedChunk

__all__ = ["router"]

log = logging.getLogger("asistent.api.chat")

router = APIRouter(tags=["chat"])


# --------------------------------------------------------------- телеметрія
class _Telemetry:
    """Адаптер `TelemetryRepo` під качиний контракт генератора.

    Генерація не імпортує репозиторії (щоб лишатись тестованою без БД) і
    вимагає лише двох методів. Кожен виклик — власна коротка транзакція:
    телеметрія ніколи не має тримати письменника WAL під час стрімінгу.
    """

    def __init__(self, db: Any) -> None:
        self.db = db

    def event(self, name: str, **kw: Any) -> None:
        try:
            with self.db.transaction() as con:
                TelemetryRepo(con).event(name, **kw)
        except Exception:
            log.debug("Не вдалося записати подію %s", name, exc_info=True)

    def unresolved_citation(self, message_id: str, emitted: str, available: Any) -> None:
        try:
            with self.db.transaction() as con:
                TelemetryRepo(con).unresolved_citation(message_id, emitted, list(available))
        except Exception:
            log.debug("Не вдалося записати нерозв'язану цитату", exc_info=True)


# ------------------------------------------------------------------ сесії
@router.get("/sessions", response_model=list[SessionOut])
def list_sessions(request: Request, assistant_id: str) -> list[SessionOut]:
    services = services_of(request)
    with services.db.connection() as con:
        if AssistantRepo(con).get(assistant_id) is None:
            raise HTTPException(status_code=404, detail="Асистента не знайдено.")
        rows = ChatRepo(con).sessions(assistant_id)
        counts = {
            r["session_id"]: r["c"]
            for r in con.execute(
                "SELECT session_id, count(*) AS c FROM chat_messages GROUP BY session_id"
            )
        }
    return [
        SessionOut(
            id=r["id"], assistant_id=r["assistant_id"], title=r["title"],
            created_at=r["created_at"], updated_at=r["updated_at"],
            messages=int(counts.get(r["id"], 0)),
        )
        for r in rows
    ]


@router.post("/sessions", response_model=SessionOut, status_code=201)
def create_session(request: Request, payload: SessionIn) -> SessionOut:
    services = services_of(request)
    with services.db.transaction() as con:
        if AssistantRepo(con).get(payload.assistant_id) is None:
            raise HTTPException(status_code=404, detail="Асистента не знайдено.")
        session_id = ChatRepo(con).create_session(payload.assistant_id, payload.title)
        row = con.execute("SELECT * FROM chat_sessions WHERE id=?", (session_id,)).fetchone()
    return SessionOut(
        id=row["id"], assistant_id=row["assistant_id"], title=row["title"],
        created_at=row["created_at"], updated_at=row["updated_at"], messages=0,
    )


@router.get("/sessions/{session_id}/messages", response_model=list[MessageOut])
def session_messages(request: Request, session_id: str) -> list[MessageOut]:
    services = services_of(request)
    _session_row(services, session_id)
    with services.db.connection() as con:
        rows = ChatRepo(con).messages(session_id)
    return [_message_out(r) for r in rows]


@router.delete("/sessions/{session_id}/messages", status_code=204)
def clear_messages(request: Request, session_id: str) -> None:
    """Очистити розмову, лишивши її саму.

    Відрізняється від видалення сесії навмисно: «очистити» — це продовжити
    працювати в тій самій вкладці з чистого аркуша, і посилання на неї в
    історії має пережити операцію.
    """
    services = services_of(request)
    _session_row(services, session_id)
    with services.db.transaction() as con:
        ChatRepo(con).clear_messages(session_id)
        # Назву теж скидаємо. Вона походить від ПЕРШОГО питання (`_maybe_title`),
        # тож після очищення порожня розмова несла б у списку історії заголовок
        # питання, якого в ній уже немає. `updated_at` рухаємо разом: список
        # сортується за ним, і без цього щойно очищена розмова лишалась би на
        # своєму старому місці.
        con.execute(
            "UPDATE chat_sessions SET title='', updated_at=CURRENT_TIMESTAMP WHERE id=?",
            (session_id,),
        )


@router.delete("/sessions/{session_id}", status_code=204)
def delete_session(request: Request, session_id: str) -> None:
    """Видалити розмову разом з усіма повідомленнями.

    Повідомлення прибирає каскад (`chat_messages.session_id` →
    `ON DELETE CASCADE`), а за ними — відгуки (`feedback.message_id`). Окремо
    видаляти їх не треба й не можна: між двома DELETE відгук лишився б
    прив'язаним до неіснуючого повідомлення.

    `retrieval_debug_json` кожної відповіді — це найважчий стовпець у чатах
    (повний розклад пошуку з таймінгами), тож саме видалення старих розмов
    реально зменшує базу, а не просто прибирає рядки з очей.
    """
    services = services_of(request)
    _session_row(services, session_id)
    with services.db.transaction() as con:
        con.execute("DELETE FROM chat_sessions WHERE id=?", (session_id,))


def _session_row(services: Services, session_id: str) -> dict[str, Any]:
    with services.db.connection() as con:
        row = con.execute("SELECT * FROM chat_sessions WHERE id=?", (session_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Сесію не знайдено.")
    return dict(row)


def _message_out(row: dict[str, Any]) -> MessageOut:
    mapping = json.loads(row["citation_map_json"] or "{}")
    return MessageOut(
        id=row["id"], session_id=row["session_id"], role=row["role"],
        content=row["content"], abstained=bool(row["abstained"]),
        model_id=row["model_id"], ttft_ms=row["ttft_ms"], tokens_out=row["tokens_out"],
        created_at=row["created_at"],
        citations=[
            {"ordinal": int(k), "chunkUid": v}
            for k, v in sorted(mapping.items(), key=lambda kv: int(kv[0]))
        ],
    )


# -------------------------------------------------------------------- чат
@router.post("/chat")
async def chat(request: Request, payload: ChatIn) -> StreamingResponse:
    services = services_of(request)
    session = _session_row(services, payload.session_id)
    with services.db.connection() as con:
        assistant = AssistantRepo(con).get(session["assistant_id"])
    if assistant is None:
        raise HTTPException(status_code=404, detail="Асистента не знайдено.")

    question = payload.message.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Питання порожнє.")

    with services.db.transaction() as con:
        repo = ChatRepo(con)
        repo.add_message(session["id"], "user", question)
        # Порожнє повідомлення асистента створюється ДО генерації: без
        # стабільного id кадри chat.token нема до чого приліпити, а UI не
        # може показати «друкує…» у правильному місці історії.
        message_id = repo.add_message(session["id"], "assistant", "")

    stream = _chat_stream(
        request, services, assistant, session_id=session["id"],
        message_id=message_id, question=question, payload=payload,
    )
    return StreamingResponse(
        stream,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",   # інакше проксі буферизує стрім у ніщо
            "Connection": "keep-alive",
            "X-Message-Id": message_id,
        },
    )


async def _chat_stream(
    request: Request,
    services: Services,
    assistant: Assistant,
    *,
    session_id: str,
    message_id: str,
    question: str,
    payload: ChatIn,
) -> AsyncIterator[str]:
    from app.generation.generator import CancelToken, Generator
    from app.generation.prompt_builder import citation_map

    started = time.perf_counter()
    config = assistant.config
    text_parts: list[str] = []
    citations: list[Citation] = []
    evidence: list[RetrievedChunk] = []
    debug: RetrievalDebug | None = None
    abstained = False
    ttft_ms: int | None = None
    tokens_out = 0
    model_id: str | None = None

    def frame(event: str, data: dict[str, Any]) -> str:
        """Кадр іде і в тіло відповіді, і в спільний канал подій."""
        published = services.events.publish(event, data)
        return sse_frame(event, data, event_id=published.id)

    try:
        # --- пошук --------------------------------------------------------
        retriever = services.retriever
        if retriever is None:
            yield frame(CHAT_ERROR, {
                "messageId": message_id,
                "errorCode": "EMBEDDER_MISSING",
                "message": "Модель ембедингів недоступна — пошук неможливий.",
                "hint": "Розпакуйте моделі або запустіть із ASISTENT_STUB=1.",
            })
            return

        evidence, debug = await asyncio.to_thread(
            _retrieve, services, assistant, question, payload
        )
        abstained = bool(debug.abstained)
        yield frame(CHAT_DEBUG, {
            "messageId": message_id,
            "stage": "retrieval",
            "found": debug.final_count,
            "documents": debug.distinct_documents,
            "abstained": abstained,
            "confidence": debug.abstain_confidence,
            "latencyMs": debug.latency_ms,
        })

        # --- генерація ----------------------------------------------------
        backend = await services.backend()
        if backend is None and not abstained:
            yield frame(CHAT_ERROR, {
                "messageId": message_id,
                "errorCode": "LMSTUDIO_UNAVAILABLE",
                "message": "LM Studio недоступний — відповідь згенерувати нічим.",
                "hint": "Запустіть LM Studio і локальний сервер, потім повторіть питання.",
            })
            return

        model_id = services.settings.lmstudio_model
        generator = Generator(
            backend,
            config=config,
            model_id=model_id,
            persona=config.instructions,
            context_tokens=services.settings.context_tokens,
            telemetry=_Telemetry(services.db),
        )
        cancel = CancelToken()

        async for chunk in generator.answer(
            question, evidence, debug=debug, message_id=message_id,
            detailed=payload.detailed, cancel=cancel,
        ):
            if await request.is_disconnected():
                # Вкладку закрили. Скасовуємо генерацію: локальна модель
                # інакше догенерує 600 токенів у нікуди, зайнявши GPU.
                cancel.cancel()
            if chunk.kind == "token":
                tokens_out += 1
                text_parts.append(chunk.text)
                yield frame(CHAT_TOKEN, {"messageId": message_id, "delta": chunk.text})
            elif chunk.kind == "citations":
                citations = list(chunk.payload.get("citations") or [])
                yield frame(CHAT_CITATIONS, {
                    "messageId": message_id,
                    "citations": [citation_out(c) for c in citations],
                    "merged": [citation_out(c) for c in (chunk.payload.get("merged") or [])],
                    "unresolved": list(chunk.payload.get("unresolved") or []),
                })
            elif chunk.kind == "debug":
                yield frame(CHAT_DEBUG, {"messageId": message_id, **(chunk.payload or {})})
            elif chunk.kind == "error":
                yield frame(CHAT_ERROR, {
                    "messageId": message_id,
                    "errorCode": str(chunk.payload),
                    "message": chunk.text,
                    "hint": "",
                })
            elif chunk.kind == "done":
                result = chunk.payload
                if result is not None:
                    if result.text:
                        text_parts = [result.text]
                    abstained = bool(result.abstained)
                    ttft_ms = result.ttft_ms
                    tokens_out = result.tokens_out or tokens_out
                    model_id = result.model_id or model_id
    except Exception as exc:
        log.exception("Помилка під час відповіді")
        yield frame(CHAT_ERROR, {
            "messageId": message_id,
            "errorCode": "CHAT_FAILED",
            "message": f"Не вдалося сформувати відповідь: {exc}",
            "hint": "Деталі — у діагностичному пакеті.",
        })
    finally:
        answer = "".join(text_parts)
        _persist_answer(
            services,
            message_id=message_id,
            content=answer,
            citation_map=citation_map(evidence) if evidence else {},
            debug=debug,
            abstained=abstained,
            model_id=model_id,
            ttft_ms=ttft_ms,
            tokens_out=tokens_out,
        )
        _maybe_title(services, session_id, question)

    yield frame(CHAT_DONE, {
        "messageId": message_id,
        "abstained": abstained,
        "tokensOut": tokens_out,
        "ttftMs": ttft_ms,
        "citations": len(citations),
        "elapsedMs": int((time.perf_counter() - started) * 1000),
    })


def _retrieve(
    services: Services, assistant: Assistant, question: str, payload: ChatIn
) -> tuple[list[RetrievedChunk], RetrievalDebug]:
    """Пошук по всіх колекціях асистента.

    Колекції ізольовані ФІЗИЧНО (один файл індексу на колекцію), тож
    об'єднання робиться тут, після пошуку, а не фільтром усередині одного
    індексу: так асистент може мати кілька колекцій, не втрачаючи властивості,
    заради якої ізоляція існує — матеріали чужого асистента не потрапляють у
    відповідь ніколи.
    """
    from app.retrieval.hybrid import MetadataFilter

    retriever = services.retriever
    assert retriever is not None
    config = assistant.config
    flt = MetadataFilter(
        document_ids=tuple(payload.document_ids),
        doc_types=tuple(payload.doc_types),
        languages=tuple(payload.languages),
        year_from=payload.year_from,
        year_to=payload.year_to,
    )

    with services.db.connection() as con:
        collections = CollectionRepo(con).for_assistant(assistant.id)
    if not collections:
        empty = RetrievalDebug(query=question, abstained=True, abstain_confidence=0.0)
        return [], empty

    merged: list[RetrievedChunk] = []
    debugs: list[RetrievalDebug] = []
    for collection in collections:
        try:
            chunks, debug = retriever.retrieve(
                question, config, collection.id, filters=flt, language="uk"
            )
        except KeyError:
            continue
        merged.extend(chunks)
        debugs.append(debug)

    if not debugs:
        return [], RetrievalDebug(query=question, abstained=True, abstain_confidence=0.0)
    if len(debugs) == 1:
        return merged, debugs[0]

    merged.sort(key=lambda rc: -rc.score)
    merged = merged[: config.final_top_k]
    combined = RetrievalDebug(
        query=question,
        dense_count=sum(d.dense_count for d in debugs),
        sparse_count=sum(d.sparse_count for d in debugs),
        ngram_count=sum(d.ngram_count for d in debugs),
        fused_count=sum(d.fused_count for d in debugs),
        reranked_count=sum(d.reranked_count for d in debugs),
        final_count=len(merged),
        distinct_documents=len({rc.chunk.document_id for rc in merged}),
        # Утримуємось лише якщо ЖОДНА колекція не дала опори: одна колекція з
        # доказами робить відповідь можливою, і мовчати в цьому разі означало б
        # відмовитись від матеріалів, які викладач сам завантажив.
        abstained=all(d.abstained for d in debugs),
        abstain_confidence=max((d.abstain_confidence or 0.0) for d in debugs),
        latency_ms={f"c{i}": sum(d.latency_ms.values()) for i, d in enumerate(debugs)},
        candidates=[row for d in debugs for row in d.candidates],
    )
    return merged, combined


def _persist_answer(
    services: Services,
    *,
    message_id: str,
    content: str,
    citation_map: dict[str, str],
    debug: RetrievalDebug | None,
    abstained: bool,
    model_id: str | None,
    ttft_ms: int | None,
    tokens_out: int,
) -> None:
    """Дописати повідомлення асистента.

    Прямий UPDATE, бо `ChatRepo` вміє лише вставляти: рядок уже створено на
    початку ходу, щоб `messageId` існував до першого токена.
    """
    try:
        with services.db.transaction() as con:
            con.execute(
                "UPDATE chat_messages SET content=?, citation_map_json=?,"
                " retrieval_debug_json=?, abstained=?, model_id=?, ttft_ms=?, tokens_out=?"
                " WHERE id=?",
                (
                    content,
                    json.dumps(citation_map, ensure_ascii=False),
                    debug.to_json() if debug is not None else None,
                    int(abstained),
                    model_id,
                    ttft_ms,
                    tokens_out,
                    message_id,
                ),
            )
    except Exception:
        log.exception("Не вдалося зберегти відповідь %s", message_id)


def _maybe_title(services: Services, session_id: str, question: str) -> None:
    """Назва сесії — перше питання. Без виклику LLM: назва в списку не варта
    ані секунди GPU, ані ризику, що модель вигадає щось стороннє."""
    try:
        with services.db.transaction() as con:
            row = con.execute(
                "SELECT title FROM chat_sessions WHERE id=?", (session_id,)
            ).fetchone()
            if row is not None and not (row["title"] or "").strip():
                title = question.strip().replace("\n", " ")[:80]
                con.execute(
                    "UPDATE chat_sessions SET title=? WHERE id=?", (title, session_id)
                )
    except Exception:
        log.debug("Не вдалося оновити назву сесії", exc_info=True)


# --------------------------------------------------- «Чому ця відповідь»
@router.get("/messages/{message_id}/why")
def message_why(request: Request, message_id: str) -> dict[str, Any]:
    """Фрагменти, на яких стоїть відповідь, плюс повний `RetrievalDebug`.

    Це не режим розробника: саме цей екран є джерелом зворотного зв'язку для
    таблиці `feedback`, тобто безкоштовним набором даних для оцінювання НДР.
    """
    services = services_of(request)
    with services.db.connection() as con:
        row = con.execute("SELECT * FROM chat_messages WHERE id=?", (message_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Повідомлення не знайдено.")
        mapping = json.loads(row["citation_map_json"] or "{}")
        chunks = ChunkRepo(con)
        fragments = []
        for ordinal, chunk_uid in sorted(mapping.items(), key=lambda kv: int(kv[0])):
            chunk = chunks.by_uid(chunk_uid)
            if chunk is None:
                fragments.append({"ordinal": int(ordinal), "chunkUid": chunk_uid,
                                  "missing": True})
                continue
            fragments.append({
                "ordinal": int(ordinal),
                "chunkUid": chunk_uid,
                "documentId": chunk.document_id,
                "documentTitle": chunks.document_title(chunk.document_id),
                "headerPath": chunk.header_path,
                "pageFrom": chunk.page_from,
                "pageTo": chunk.page_to,
                "pageLabel": chunk.citation_label(),
                "level": chunk.level.value,
                "language": chunk.language,
                "text": chunk.display_text,
                "bboxes": [b.as_dict() for b in chunk.bboxes],
                "missing": False,
            })
        unresolved = [
            dict(r)
            for r in con.execute(
                "SELECT emitted, available, created_at FROM unresolved_citations"
                " WHERE message_id=? ORDER BY id",
                (message_id,),
            )
        ]
    return {
        "messageId": message_id,
        "abstained": bool(row["abstained"]),
        "modelId": row["model_id"],
        "ttftMs": row["ttft_ms"],
        "tokensOut": row["tokens_out"],
        "fragments": fragments,
        "unresolved": unresolved,
        "debug": json.loads(row["retrieval_debug_json"]) if row["retrieval_debug_json"] else None,
    }


@router.post("/feedback", status_code=201)
def feedback(request: Request, payload: FeedbackIn) -> dict[str, Any]:
    services = services_of(request)
    with services.db.transaction() as con:
        TelemetryRepo(con).feedback(
            payload.message_id, payload.verdict,
            chunk_uid=payload.chunk_uid, note=payload.note,
        )
    return {"ok": True}
