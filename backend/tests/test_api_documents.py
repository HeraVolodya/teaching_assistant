"""Прохання «зроби презентацію / документ / таблицю» наскрізно через HTTP.

Усе в stub-режимі: заглушка LLM віддає Markdown потрібної форми з тих самих
фрагментів, тож перевіряється НАШ шлях — розпізнавання, пошук за темою,
стрім змісту в чат, кадр `chat.file`, історія й збирання файлу на завантаженні.
"""

from __future__ import annotations

import io
import json
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import unquote

from docx import Document
from openpyxl import load_workbook
from pptx import Presentation

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


async def _ready(client: Any) -> dict[str, Any]:
    assistant = await create_assistant(client, "Артилерія")
    collection_id = assistant["collections"][0]["id"]
    document = await upload_text(client, collection_id)
    await wait_for_status(client, collection_id, document["id"])
    return (await client.post("/api/sessions", json={"assistantId": assistant["id"]})).json()


async def _ask(client: Any, session: dict[str, Any], message: str, **extra: Any) -> Any:
    return await client.post("/api/chat", json={"sessionId": session["id"], "message": message, **extra})


class _ScriptedBackend:
    """Модель, що віддає рівно заданий Markdown."""

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

    async def chat_stream(self, messages: Messages, *, params: ChatParams | None = None,
                          model: str | None = None, cancel: Any = None,
                          warm: bool = True) -> AsyncIterator[str]:
        self.calls.append(list(messages))
        for piece in self.text.split(" "):
            yield piece + " "


# --------------------------------------------------------------- наскрізно
async def test_presentation_request_streams_content_and_offers_a_file(tmp_path) -> None:
    settings = make_settings(tmp_path, worker_mode="inline")
    async with api_client(settings) as (client, _app):
        session = await _ready(client)
        response = await _ask(client, session, "Зроби презентацію на 3 слайди про деривацію снаряда")
        message_id = response.headers["x-message-id"]
        file_response = await client.get(f"/api/messages/{message_id}/file")
        why = (await client.get(f"/api/messages/{message_id}/why")).json()
        history = (await client.get(f"/api/sessions/{session['id']}/messages")).json()
        channel = (await client.get("/api/events/history", params={"limit": 500})).json()

    events = [event for event, _ in read_sse(response.text)]
    assert events[-1] == "chat.done"
    # Файл оголошується ПІСЛЯ змісту й цитат: кнопка не може випередити текст.
    assert events.index("chat.file") > events.index("chat.citations")
    assert _frames(response.text, "chat.debug")[0]["document"] == "pptx"

    document = _frames(response.text, "chat.file")[0]["document"]
    assert document["format"] == "pptx"
    assert document["filename"].endswith(".pptx")
    assert document["stats"]["slides"] >= 1
    assert any(e["type"] == "chat.file" for e in channel), "друге вікно мусить побачити файл"

    # Шукали ТЕМУ, а не команду «зроби презентацію на 3 слайди».
    assert why["debug"]["query"] == "деривацію снаряда"

    # Зміст стрімився в чат як звичайна відповідь — із посиланнями.
    streamed = "".join(t["delta"] for t in _frames(response.text, "chat.token"))
    assert streamed.startswith("# ") and "[1]" in streamed

    assert file_response.status_code == 200
    assert file_response.headers["content-type"].startswith(
        "application/vnd.openxmlformats-officedocument.presentationml.presentation")
    disposition = file_response.headers["content-disposition"]
    assert disposition.startswith("attachment;")
    assert unquote(disposition.split("filename*=UTF-8''", 1)[1]) == document["filename"]
    prs = Presentation(io.BytesIO(file_response.content))
    assert len(prs.slides) >= 3                      # титул + зміст + джерела

    stored = next(m for m in history if m["id"] == message_id)
    assert stored["document"] == document


async def test_from_this_uses_the_previous_question(tmp_path) -> None:
    """«Зроби з цього таблицю» — тема попереднього питання, а не «з цього»."""
    settings = make_settings(tmp_path, worker_mode="inline")
    async with api_client(settings) as (client, application):
        session = await _ready(client)
        await _ask(client, session, "Що таке деривація снаряда?")
        response = await _ask(client, session, "Зроби з цього таблицю в Excel")
        message_id = response.headers["x-message-id"]
        why = (await client.get(f"/api/messages/{message_id}/why")).json()
        file_response = await client.get(f"/api/messages/{message_id}/file")
        prompt = application.state.services._backend.calls[-1][-1]["content"]

    assert why["debug"]["query"] == "Що таке деривація снаряда?"
    # Тему бачить і модель, а не лише пошук: на живих даних без цього вийшла
    # таблиця «про все, що знайшлося».
    assert "Питання: Зроби з цього таблицю в Excel (тема: Що таке деривація снаряда?)" in prompt
    assert _frames(response.text, "chat.file")[0]["document"]["format"] == "xlsx"
    book = load_workbook(io.BytesIO(file_response.content))
    assert book.sheetnames == ["Таблиця", "Джерела"]
    assert book["Таблиця"].max_row >= 4


async def test_request_without_any_topic_asks_for_one(tmp_path) -> None:
    settings = make_settings(tmp_path, worker_mode="inline")
    async with api_client(settings) as (client, _app):
        session = await _ready(client)
        response = await _ask(client, session, "Зроби з цього презентацію")
        history = (await client.get(f"/api/sessions/{session['id']}/messages")).json()

    assert not _frames(response.text, "chat.file")
    assert _frames(response.text, "chat.done")
    answer = history[-1]
    assert answer["role"] == "assistant" and "Уточніть" in answer["content"]
    assert answer["document"] is None


async def test_no_file_without_grounds_in_the_materials(tmp_path) -> None:
    """Пороги утримання діють і на файли: без опори в матеріалах — чесна
    відмова, а не презентація з повітря."""
    settings = make_settings(tmp_path, worker_mode="inline")
    async with api_client(settings) as (client, _app):
        session = await _ready(client)
        response = await _ask(client, session, "Зроби презентацію про деривацію снаряда",
                              yearFrom=3000)
    assert _frames(response.text, "chat.done")[-1]["abstained"] is True
    assert not _frames(response.text, "chat.file")


async def test_model_markdown_is_cleaned_and_becomes_the_file(tmp_path) -> None:
    """Вигаданий маркер [42] зникає і з тексту в історії, і з файлу."""
    settings = make_settings(tmp_path, worker_mode="inline")
    async with api_client(settings) as (client, application):
        session = await _ready(client)
        application.state.services._backend = _ScriptedBackend(
            "# Деривація снаряда\n\n## Що це\n- Відхилення снаряда від площини стрільби [1].\n"
            "- Залежить від обертання [42].\n\n## Як враховують\n- Поправкою за таблицями [1]."
        )
        response = await _ask(client, session, "Створи документ про деривацію снаряда")
        message_id = response.headers["x-message-id"]
        history = (await client.get(f"/api/sessions/{session['id']}/messages")).json()
        file_response = await client.get(f"/api/messages/{message_id}/file")

    stored = next(m for m in history if m["id"] == message_id)
    assert "[42]" not in stored["content"]
    assert stored["document"]["title"] == "Деривація снаряда"
    assert stored["document"]["filename"] == "Деривація снаряда.docx"
    doc = Document(io.BytesIO(file_response.content))
    texts = [p.text for p in doc.paragraphs]
    assert texts[0] == "Деривація снаряда"
    assert "Залежить від обертання." in texts
    assert any(t.startswith("[1] ") for t in texts), "список джерел із назвою документа"


async def test_any_answer_can_be_taken_as_word(tmp_path) -> None:
    settings = make_settings(tmp_path, worker_mode="inline")
    async with api_client(settings) as (client, _app):
        session = await _ready(client)
        response = await _ask(client, session, "Що таке деривація снаряда?")
        message_id = response.headers["x-message-id"]
        as_word = await client.get(f"/api/messages/{message_id}/file", params={"format": "docx"})
        no_format = await client.get(f"/api/messages/{message_id}/file")
        bad_format = await client.get(f"/api/messages/{message_id}/file", params={"format": "pdf"})
        missing = await client.get("/api/messages/немає/file", params={"format": "docx"})
        history = (await client.get(f"/api/sessions/{session['id']}/messages")).json()

    assert not _frames(response.text, "chat.file")
    assert next(m for m in history if m["id"] == message_id)["document"] is None
    assert as_word.status_code == 200
    # Назва — з питання, на яке відповідали.
    assert "filename*=UTF-8''" in as_word.headers["content-disposition"]
    assert unquote(as_word.headers["content-disposition"].split("''", 1)[1]) == \
        "Що таке деривація снаряда.docx"
    assert Document(io.BytesIO(as_word.content)).paragraphs[0].text == "Що таке деривація снаряда?"
    assert (no_format.status_code, bad_format.status_code, missing.status_code) == (400, 400, 404)


async def test_clearing_the_conversation_removes_its_files(tmp_path) -> None:
    """Байтів на диску немає, а запис про файл іде каскадом за повідомленням."""
    settings = make_settings(tmp_path, worker_mode="inline")
    async with api_client(settings) as (client, application):
        session = await _ready(client)
        response = await _ask(client, session, "Склади таблицю про деривацію снаряда")
        message_id = response.headers["x-message-id"]
        db = application.state.services.db
        with db.connection() as con:
            before = con.execute("SELECT count(*) FROM message_documents").fetchone()[0]
        await client.delete(f"/api/sessions/{session['id']}/messages")
        with db.connection() as con:
            after = con.execute("SELECT count(*) FROM message_documents").fetchone()[0]
        gone = await client.get(f"/api/messages/{message_id}/file")

    assert (before, after) == (1, 0)
    assert gone.status_code == 404


async def test_teachers_real_two_messages(tmp_path) -> None:
    """Дослівно перші живі прохання викладача (23.11.2026). Тоді перше дало
    звичайну відповідь без файлу («ПРИЗЕНТАЦІЮ» не впізналося), а друге шукало
    тему «ФАЙЛ ПРИЗЕНТАЦІЇ» замість попереднього питання."""
    settings = make_settings(tmp_path, worker_mode="inline")
    async with api_client(settings) as (client, _app):
        session = await _ready(client)
        first = await _ask(client, session, "ЗРОБИ МЕНІ ПРИЗЕНТАЦІЮ ПРО ДЕРИВАЦІЮ СНАРЯДА")
        second = await _ask(client, session, "ТЕПЕР ЗРОБИ МЕНІ ФАЙЛ ПРИЗЕНТАЦІЇ PPTX")
        why = (await client.get(f"/api/messages/{second.headers['x-message-id']}/why")).json()

    assert _frames(first.text, "chat.file")[0]["document"]["format"] == "pptx"
    assert _frames(second.text, "chat.file")[0]["document"]["format"] == "pptx"
    assert why["debug"]["query"] == "ДЕРИВАЦІЮ СНАРЯДА"


async def test_preamble_is_not_stored_and_not_in_the_file(tmp_path) -> None:
    settings = make_settings(tmp_path, worker_mode="inline")
    async with api_client(settings) as (client, application):
        session = await _ready(client)
        application.state.services._backend = _ScriptedBackend(
            "Я не можу створити файл формату PPTX, оскільки я генерую лише текст. Однак, "
            "ось структура презентації:\n\n# Деривація снаряда\n\n## Суть\n- Відхилення снаряда [1]."
        )
        response = await _ask(client, session, "Зроби презентацію про деривацію снаряда")
        message_id = response.headers["x-message-id"]
        history = (await client.get(f"/api/sessions/{session['id']}/messages")).json()
        file_response = await client.get(f"/api/messages/{message_id}/file")

    stored = next(m for m in history if m["id"] == message_id)
    assert stored["content"].startswith("# Деривація снаряда")
    assert stored["document"]["title"] == "Деривація снаряда"
    prs = Presentation(io.BytesIO(file_response.content))
    texts = " ".join(sh.text_frame.text for s in prs.slides for sh in s.shapes if sh.has_text_frame)
    assert "не можу" not in texts
