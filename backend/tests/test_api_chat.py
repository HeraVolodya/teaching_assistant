"""Стрім чату: дзеркалення в канал подій, збереження, фільтри, галюцинації."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from app.backends.base import BackendHealth, ChatParams, LoadConfig, Messages, ModelInfo
from tests.helpers_api import (
    api_client,
    create_assistant,
    make_settings,
    read_sse,
    upload_text,
    wait_for_status,
)


def _frames(body: str, kind: str) -> list[dict[str, Any]]:
    return [json.loads(data) for event, data in read_sse(body) if event == kind]


class _ScriptedBackend:
    """Бекенд, що видає рівно заданий текст — для перевірки пост-обробки."""

    name = "scripted"

    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[Messages] = []

    async def health(self) -> BackendHealth:
        return BackendHealth(ok=True, api_version="stub")

    async def list_models(self) -> list[ModelInfo]:
        return []

    async def load_model(self, cfg: LoadConfig) -> ModelInfo:
        return ModelInfo(key="scripted", loaded=True)

    async def chat_stream(
        self, messages: Messages, *, params: ChatParams | None = None,
        model: str | None = None, cancel: Any = None, warm: bool = True,
    ) -> AsyncIterator[str]:
        self.calls.append(list(messages))
        for piece in self.text.split(" "):
            yield piece + " "


async def _ready_assistant(client: Any, **config: Any) -> tuple[dict, str, dict]:
    assistant = await create_assistant(client, "Артилерія", **config)
    collection_id = assistant["collections"][0]["id"]
    document = await upload_text(client, collection_id)
    await wait_for_status(client, collection_id, document["id"])
    session = (await client.post(
        "/api/sessions", json={"assistantId": assistant["id"]}
    )).json()
    return assistant, collection_id, session


# ------------------------------------------------------------- дзеркалення
async def test_chat_frames_also_land_on_the_shared_channel(tmp_path) -> None:
    """Токени йдуть і в тіло POST /api/chat, і в спільний GET /api/events.

    Перше — щоб клієнт, який поставив питання, не мусив корелювати два
    з'єднання; друге — щоб друге вікно й перепідключений WebView бачили той
    самий хід.
    """
    settings = make_settings(tmp_path, worker_mode="inline")
    async with api_client(settings) as (client, _application):
        _assistant, _collection_id, session = await _ready_assistant(client)
        response = await client.post("/api/chat", json={
            "sessionId": session["id"], "message": "Що таке деривація снаряда?",
        })
        message_id = response.headers["x-message-id"]

        history = (await client.get("/api/events/history", params={"limit": 500})).json()

    channel_tokens = [
        e["data"]["delta"] for e in history
        if e["type"] == "chat.token" and e["data"]["messageId"] == message_id
    ]
    body_tokens = [t["delta"] for t in _frames(response.text, "chat.token")]
    assert channel_tokens == body_tokens
    assert any(e["type"] == "chat.citations" for e in history)
    assert any(e["type"] == "chat.done" for e in history)


async def test_tokens_concatenate_without_separators(tmp_path) -> None:
    """Пробіл лишається на початку токена, як у SentencePiece. Якщо десь
    зробити `" ".join(deltas)` замість `"".join`, текст подвоїть пробіли —
    ця перевірка ловить саме це."""
    settings = make_settings(tmp_path, worker_mode="inline")
    async with api_client(settings) as (client, _app):
        _assistant, _collection_id, session = await _ready_assistant(client)
        response = await client.post("/api/chat", json={
            "sessionId": session["id"], "message": "Що таке деривація снаряда?",
        })
        message_id = response.headers["x-message-id"]
        messages = (await client.get(f"/api/sessions/{session['id']}/messages")).json()

    streamed = "".join(t["delta"] for t in _frames(response.text, "chat.token"))
    stored = next(m for m in messages if m["id"] == message_id)["content"]
    assert stored == streamed
    assert "  " not in stored


# ------------------------------------------------------- цитати й галюцинації
async def test_invented_marker_is_stripped_and_logged(tmp_path) -> None:
    """Маркер [42], якого немає серед доказів, ВИДАЛЯЄТЬСЯ з тексту й
    потрапляє в `unresolved_citations`. Лишити посилання, що нікуди не веде,
    означає підірвати те єдине, на чому тримається довіра до системи."""
    settings = make_settings(tmp_path, worker_mode="inline")
    async with api_client(settings) as (client, application):
        _assistant, _collection_id, session = await _ready_assistant(client)
        application.state.services._backend = _ScriptedBackend(
            "Деривація [1] залежить від обертання [42] снаряда."
        )
        response = await client.post("/api/chat", json={
            "sessionId": session["id"], "message": "Що таке деривація снаряда?",
        })
        message_id = response.headers["x-message-id"]
        why = (await client.get(f"/api/messages/{message_id}/why")).json()

    answer = "".join(t["delta"] for t in _frames(response.text, "chat.token"))
    citations = _frames(response.text, "chat.citations")[-1]
    assert "[42]" in answer                       # у СИРОМУ потоці ще є
    assert "[42]" in citations["unresolved"]
    assert why["unresolved"], "нерозв'язаний маркер не потрапив у телеметрію"
    available = json.loads(why["unresolved"][0]["available"])
    assert available, "телеметрія має нести і список ДОСТУПНИХ маркерів"
    # У збереженому тексті вигаданого маркера вже немає.
    messages_payload = why["fragments"]
    assert messages_payload
    assert all(f["missing"] is False for f in messages_payload)


async def test_stored_answer_is_the_cleaned_text(tmp_path) -> None:
    settings = make_settings(tmp_path, worker_mode="inline")
    async with api_client(settings) as (client, application):
        _assistant, _collection_id, session = await _ready_assistant(client)
        application.state.services._backend = _ScriptedBackend(
            "Відповідь [1] і вигадка [77]."
        )
        response = await client.post("/api/chat", json={
            "sessionId": session["id"], "message": "Що таке деривація снаряда?",
        })
        message_id = response.headers["x-message-id"]
        messages = (await client.get(f"/api/sessions/{session['id']}/messages")).json()

    stored = next(m for m in messages if m["id"] == message_id)["content"]
    assert "[77]" not in stored
    assert "[1]" in stored


# ------------------------------------------------------------------ фільтри
async def test_metadata_filter_narrows_the_search(tmp_path) -> None:
    """Стаття автора називає фільтрацію за метаданими ОБОВ'ЯЗКОВОЮ."""
    settings = make_settings(tmp_path, worker_mode="inline")
    async with api_client(settings) as (client, _app):
        assistant = await create_assistant(client)
        collection_id = assistant["collections"][0]["id"]
        first = await upload_text(client, collection_id, name="перший.txt")
        second = await upload_text(
            client, collection_id, name="другий.txt",
            text="# Інше\n\nДеривація снаряда згадується і тут, у другому "
                 "документі, з тими самими словами про відхилення снаряда.\n",
        )
        await wait_for_status(client, collection_id, first["id"])
        await wait_for_status(client, collection_id, second["id"])

        session = (await client.post(
            "/api/sessions", json={"assistantId": assistant["id"]}
        )).json()
        response = await client.post("/api/chat", json={
            "sessionId": session["id"],
            "message": "Що таке деривація снаряда?",
            "documentIds": [second["id"]],
        })

    citations = _frames(response.text, "chat.citations")[-1]["citations"]
    assert citations
    assert {c["documentId"] for c in citations} == {second["id"]}


async def test_empty_filter_result_abstains_instead_of_failing(tmp_path) -> None:
    """Фільтр, що не залишив нічого, — це порожній результат, а не 500."""
    settings = make_settings(tmp_path, worker_mode="inline")
    async with api_client(settings) as (client, _app):
        assistant = await create_assistant(client)
        collection_id = assistant["collections"][0]["id"]
        document = await upload_text(client, collection_id)
        await wait_for_status(client, collection_id, document["id"])
        session = (await client.post(
            "/api/sessions", json={"assistantId": assistant["id"]}
        )).json()
        response = await client.post("/api/chat", json={
            "sessionId": session["id"],
            "message": "Що таке деривація снаряда?",
            "yearFrom": 3000,
        })
    assert response.status_code == 200
    done = _frames(response.text, "chat.done")
    assert done and done[-1]["abstained"] is True


# ----------------------------------------------------------------- помилки
async def test_chat_on_unknown_session_is_404(tmp_path) -> None:
    async with api_client(make_settings(tmp_path)) as (client, _app):
        response = await client.post("/api/chat", json={
            "sessionId": "немає", "message": "Питання",
        })
    assert response.status_code == 404


async def test_backend_failure_becomes_a_chat_error_frame(tmp_path) -> None:
    """LM Studio впав — викладач мусить побачити зрозуміле пояснення в чаті,
    а не обірваний стрім."""
    settings = make_settings(tmp_path, worker_mode="inline")

    class _Broken(_ScriptedBackend):
        async def chat_stream(self, messages, **kw):  # type: ignore[no-untyped-def]
            from app.backends.base import ModelNotLoaded

            raise ModelNotLoaded("gemma")
            yield ""    # pragma: no cover — робить функцію генератором

    async with api_client(settings) as (client, application):
        _assistant, _collection_id, session = await _ready_assistant(client)
        application.state.services._backend = _Broken("")
        response = await client.post("/api/chat", json={
            "sessionId": session["id"], "message": "Що таке деривація снаряда?",
        })

    assert response.status_code == 200
    errors = _frames(response.text, "chat.error")
    assert errors
    assert "ModelNotLoaded" in errors[0]["errorCode"]
    assert _frames(response.text, "chat.done")


async def test_lmstudio_absent_is_reported_not_hidden(tmp_path) -> None:
    """LM Studio не запущено. Пошук працює, генерувати нічим — і саме це має
    бути написано в чаті, разом із тим, що робити."""
    settings = make_settings(tmp_path, worker_mode="inline")
    async with api_client(settings) as (client, application):
        _assistant, _collection_id, session = await _ready_assistant(client)
        # `False` — це стан «шукали й не знайшли»: повторних спроб не буде,
        # рівно як після невдалого автовиявлення.
        application.state.services._backend = False

        response = await client.post("/api/chat", json={
            "sessionId": session["id"], "message": "Що таке деривація снаряда?",
        })

    errors = _frames(response.text, "chat.error")
    assert errors and errors[0]["errorCode"] == "LMSTUDIO_UNAVAILABLE"
    assert "Developer" in errors[0]["hint"] or "локальний сервер" in errors[0]["hint"]


async def test_models_are_looked_up_in_the_configured_data_dir(tmp_path) -> None:
    """Ваги шукаються в каталозі з налаштувань, а не в типовому.

    Без явного `models_dir` застосунок із власним `ASISTENT_DATA_DIR` шукав
    би моделі поруч із кодом і не знаходив би їх — на машині викладача це
    виглядало б як «моделі встановлені, але не працюють».
    """
    settings = make_settings(tmp_path, stub=False, use_reranker=False)
    async with api_client(settings) as (_client, application):
        services = application.state.services
        assert services.provider is None      # ваг немає — це очікувано
        problem = next(p for p in services.problems if p.code == "EMBEDDER_MISSING")
    assert str(settings.paths().models_dir) in problem.message


async def test_why_reports_missing_chunk_instead_of_crashing(tmp_path) -> None:
    """Чанк міг зникнути після переіндексації. Екран «Чому ця відповідь» має
    чесно сказати про це, а не впасти на None."""
    settings = make_settings(tmp_path, worker_mode="inline")
    async with api_client(settings) as (client, application):
        _assistant, _collection_id, session = await _ready_assistant(client)
        response = await client.post("/api/chat", json={
            "sessionId": session["id"], "message": "Що таке деривація снаряда?",
        })
        message_id = response.headers["x-message-id"]

        with application.state.services.db.transaction() as con:
            con.execute(
                "UPDATE chat_messages SET citation_map_json=? WHERE id=?",
                ('{"1": "0000000000000000000000000000dead"}', message_id),
            )
        why = (await client.get(f"/api/messages/{message_id}/why")).json()

    assert why["fragments"] == [
        {"ordinal": 1, "chunkUid": "0000000000000000000000000000dead", "missing": True}
    ]


# ------------------------------------------------ прогрес і розігрів моделі
async def test_rerank_progress_streams_in_the_body_but_not_on_the_bus(tmp_path) -> None:
    """Реранкінг 50 кандидатів на CPU — десятки секунд; без кадрів прогресу
    екран мовчить і читається як зависання. Кадри ефемерні: у спільній шині
    вони витісняли б із кільцевого буфера події, що мають значення."""
    settings = make_settings(tmp_path, worker_mode="inline")
    async with api_client(settings) as (client, _app):
        _assistant, _collection_id, session = await _ready_assistant(client)
        response = await client.post("/api/chat", json={
            "sessionId": session["id"], "message": "Що таке деривація снаряда?",
        })
        message_id = response.headers["x-message-id"]
        history = (await client.get("/api/events/history", params={"limit": 500})).json()

    progress = _frames(response.text, "chat.progress")
    assert progress, "реранкінг пройшов без жодного кадру прогресу"
    assert {p["messageId"] for p in progress} == {message_id}
    assert {p["stage"] for p in progress} == {"rerank"}
    assert progress[-1]["done"] == progress[-1]["total"] > 0
    # Прогрес — до результату пошуку, а не впереміш із токенами.
    events = [event for event, _ in read_sse(response.text)]
    assert events.index("chat.debug") > max(
        i for i, e in enumerate(events) if e == "chat.progress")
    assert not any(e["type"] == "chat.progress" for e in history)
    assert _frames(response.text, "chat.done")


class _WarmableBackend(_ScriptedBackend):
    """Бекенд, що вміє піднімати модель заздалегідь, і рахує, скільки разів."""

    def __init__(self, text: str, *, delay: float = 0.0) -> None:
        super().__init__(text)
        self.delay = delay
        self.warmups = 0

    async def ensure_loaded(self, model: str | None = None) -> str | None:
        import asyncio

        self.warmups += 1
        await asyncio.sleep(self.delay)
        return "scripted"


async def test_opening_a_chat_starts_loading_the_model(tmp_path) -> None:
    """Історію UI читає саме тоді, коли викладач відкриває чат: поки питання
    набирається, 15–22 с JIT-завантаження LM Studio минають."""
    import asyncio

    settings = make_settings(tmp_path, worker_mode="inline")
    async with api_client(settings) as (client, application):
        _assistant, _collection_id, session = await _ready_assistant(client)
        backend = _WarmableBackend("Відповідь [1].")
        application.state.services._backend = backend

        await client.get(f"/api/sessions/{session['id']}/messages")
        for _ in range(100):
            if backend.warmups:
                break
            await asyncio.sleep(0.01)

    assert backend.warmups == 1


async def test_chat_loads_the_model_in_parallel_with_search_once(tmp_path) -> None:
    settings = make_settings(tmp_path, worker_mode="inline")
    async with api_client(settings) as (client, application):
        _assistant, _collection_id, session = await _ready_assistant(client)
        backend = _WarmableBackend("Деривація — це відхилення [1].", delay=0.05)
        application.state.services._backend = backend

        response = await client.post("/api/chat", json={
            "sessionId": session["id"], "message": "Що таке деривація снаряда?",
        })

    assert backend.warmups == 1
    assert backend.calls, "генерація так і не почалась"
    assert "".join(t["delta"] for t in _frames(response.text, "chat.token"))


async def test_concurrent_warmups_share_one_request(tmp_path) -> None:
    """Відкриття чату і питання, поставлене одразу, не мають слати LM Studio
    два запити на завантаження."""
    import asyncio

    from app.api.state import build_services

    services = build_services(make_settings(tmp_path))
    backend = _WarmableBackend("так", delay=0.05)
    services._backend = backend
    try:
        await asyncio.gather(services.warm_llm(), services.warm_llm(), services.warm_llm())
    finally:
        services.close()
    assert backend.warmups == 1


def test_model_warmup_builds_one_retriever_under_concurrency(tmp_path) -> None:
    """Фоновий розігрів і перше питання одночасно: без замка вийшли б дві копії
    ембедера й реранкера (~2 ГБ RAM)."""
    from concurrent.futures import ThreadPoolExecutor

    from app.api.state import build_services

    services = build_services(make_settings(tmp_path))
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            retrievers = list(pool.map(lambda _: services.retriever, range(4)))
        services.warm_models()
        assert len({id(r) for r in retrievers}) == 1
        assert services.retriever is retrievers[0]
    finally:
        services.close()


# ------------------------------------------------------ вибір реранкера
def test_llm_reranker_is_the_default_and_loads_no_onnx_weights(tmp_path) -> None:
    """CPU-крос-енкодер (1.2 ГБ ваг, ~80 с на 50 кандидатів) лише запасний: до
    першої відмови LM Studio він навіть не створюється."""
    from app.api.state import build_services
    from app.rerank.llm_reranker import LlmListwiseReranker

    services = build_services(make_settings(tmp_path, stub=False))
    try:
        reranker = services.reranker
        assert isinstance(reranker, LlmListwiseReranker)
        assert reranker._fallback is None
        assert not any(p.code == "RERANKER_MISSING" for p in services.problems)
    finally:
        services.close()


def test_onnx_reranker_can_be_chosen_explicitly(tmp_path) -> None:
    from app.api.state import build_services
    from app.rerank.reranker import StubReranker

    services = build_services(make_settings(tmp_path, rerank_with="onnx"))
    try:
        assert isinstance(services.reranker, StubReranker)
    finally:
        services.close()


def test_model_warmup_never_calls_the_llm_reranker(tmp_path) -> None:
    """Розігрів на старті не має піднімати мовну модель у LM Studio — а запит
    LLM-реранкера саме це й зробив би."""
    from app.api.state import build_services

    class _NoCalls:
        name = "llm"

        def score(self, query, docs):
            raise AssertionError("LLM-реранкер не мав викликатися під час розігріву")

    services = build_services(make_settings(tmp_path))
    services._reranker = _NoCalls()
    try:
        services.warm_models()
        assert services.retriever.reranker is services._reranker
    finally:
        services.close()
