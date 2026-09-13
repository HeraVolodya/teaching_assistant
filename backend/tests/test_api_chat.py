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
    async with api_client(settings) as (client, application):
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
        application.state.services._backend = _ScriptedBackend(   # noqa: SLF001
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
        application.state.services._backend = _ScriptedBackend(   # noqa: SLF001
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
        application.state.services._backend = _Broken("")   # noqa: SLF001
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
        application.state.services._backend = False        # noqa: SLF001

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
