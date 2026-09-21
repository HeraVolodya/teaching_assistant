"""Наскрізний тест продукту.

Створити асистента → завантажити .txt → пройти конвеєр → поставити питання →
отримати відповідь із цитатою на конкретну сторінку конкретного файлу.

Це і є визначення «працює» для цієї системи. Усе в stub-режимі: жодної
завантаженої моделі, детерміновані вектори, ехо-реранкер, українська
відповідь LLM. Якщо цей файл зелений, застосунок проходить наскрізь на машині
без жодного мегабайта ваг — саме та властивість, заради якої stub-режим є
вимогою, а не опцією.
"""

from __future__ import annotations

import json

import pytest

from tests.helpers_api import (
    api_client,
    create_assistant,
    make_settings,
    read_sse,
    upload_text,
    wait_for_status,
)


def _frames(body: str, kind: str) -> list[dict]:
    return [json.loads(data) for event, data in read_sse(body) if event == kind]


async def test_end_to_end_question_returns_cited_answer(tmp_path) -> None:
    settings = make_settings(tmp_path, worker_mode="inline")
    async with api_client(settings) as (client, _application):
        # --- 1. асистент -------------------------------------------------
        assistant = await create_assistant(
            client, "Артилерія", instructions="Ти асистент кафедри артилерії."
        )
        collection_id = assistant["collections"][0]["id"]

        # --- 2. документ -------------------------------------------------
        document = await upload_text(client, collection_id, name="балістика.txt")
        ready = await wait_for_status(client, collection_id, document["id"])
        assert ready["status"] == "READY", ready
        assert ready["chunks"] >= 1
        assert ready["pageCount"] >= 1
        assert ready["qualityGrade"] is not None

        # Подія doc.ready дійшла до єдиного каналу.
        history = (await client.get("/api/events/history", params={"limit": 200})).json()
        ready_events = [e for e in history if e["type"] == "doc.ready"]
        assert ready_events and ready_events[-1]["data"]["docId"] == document["id"]

        # Сторінки з мітками й якістю доступні для перегляду.
        pages = (await client.get(f"/api/documents/{document['id']}/pages")).json()
        assert pages and pages[0]["pageNumber"] == 1

        # --- 3. сесія ----------------------------------------------------
        session = (await client.post(
            "/api/sessions", json={"assistantId": assistant["id"]}
        )).json()

        # --- 4. питання --------------------------------------------------
        response = await client.post("/api/chat", json={
            "sessionId": session["id"],
            "message": "Що таке деривація снаряда і як обчислюється поправка на деривацію?",
        })
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        message_id = response.headers["x-message-id"]
        body = response.text

        tokens = _frames(body, "chat.token")
        assert tokens, "жодного токена відповіді"
        answer = "".join(t["delta"] for t in tokens)
        assert "деривац" in answer.lower()

        # --- 5. цитата ---------------------------------------------------
        citations = _frames(body, "chat.citations")
        assert citations, "кадр chat.citations не надійшов"
        cited = citations[-1]["citations"]
        assert cited, f"відповідь без жодної цитати: {answer!r}"
        first = cited[0]
        assert first["documentId"] == document["id"]
        assert first["documentTitle"] == "балістика"
        # Головна вимога продукту: покликання на КОНКРЕТНУ сторінку.
        assert first["pageLabel"].startswith("с.")
        assert first["pageFrom"] is not None
        assert first["chunkUid"]
        # Маркери [n] у тексті мусять збігатися з виданими цитатами.
        assert f"[{first['ordinal']}]" in answer

        done = _frames(body, "chat.done")
        assert done and done[-1]["abstained"] is False

        # --- 6. історія збережена ----------------------------------------
        messages = (await client.get(f"/api/sessions/{session['id']}/messages")).json()
        assert [m["role"] for m in messages] == ["user", "assistant"]
        assert messages[1]["content"] == answer
        assert messages[1]["citations"], "мапа [n] → chunk_uid не збережена"
        # Назва сесії — перше питання, без жодного виклику LLM.
        sessions = (await client.get(
            "/api/sessions", params={"assistant_id": assistant["id"]}
        )).json()
        assert sessions[0]["title"].startswith("Що таке деривація")

        # --- 7. «Чому ця відповідь» --------------------------------------
        why = (await client.get(f"/api/messages/{message_id}/why")).json()
        assert why["fragments"], "екран «Чому ця відповідь» порожній"
        fragment = why["fragments"][0]
        assert fragment["documentTitle"] == "балістика"
        assert fragment["text"]
        assert fragment["missing"] is False
        assert why["debug"]["final_count"] >= 1
        assert why["debug"]["distinct_documents"] >= 1

        # --- 8. зворотний зв'язок ----------------------------------------
        feedback = await client.post("/api/feedback", json={
            "messageId": message_id, "verdict": "up",
            "chunkUid": fragment["chunkUid"], "note": "точна цитата",
        })
        assert feedback.status_code == 201


async def test_abstains_when_the_corpus_has_no_answer(tmp_path) -> None:
    """Питання, відповіді на яке в матеріалах свідомо немає.

    Утримання виконується В КОДІ (поріг реранкера), а не проханням у промпті:
    промптом неможливо надійно змусити 12B-модель відмовитись. Перевіряємо
    саме результат — модель не вигадала відповідь і не видала цитат.
    """
    settings = make_settings(tmp_path, worker_mode="inline")
    async with api_client(settings) as (client, _app):
        assistant = await create_assistant(client, "Артилерія", confidence_required="high")
        collection_id = assistant["collections"][0]["id"]
        document = await upload_text(client, collection_id)
        await wait_for_status(client, collection_id, document["id"])

        session = (await client.post(
            "/api/sessions", json={"assistantId": assistant["id"]}
        )).json()
        response = await client.post("/api/chat", json={
            "sessionId": session["id"],
            "message": "Яка ставка податку на додану вартість у Португалії?",
        })
        body = response.text

    done = _frames(body, "chat.done")
    assert done and done[-1]["abstained"] is True
    answer = "".join(t["delta"] for t in _frames(body, "chat.token"))
    assert "недостатньо інформації" in answer
    assert _frames(body, "chat.citations")[-1]["citations"] == []


async def test_two_assistants_do_not_leak_into_each_other(tmp_path) -> None:
    """Ізоляція асистентів фізична: одна колекція = один файл індексу.

    Другий асистент із дослівно тією ж темою не має з'явитися у відповіді
    першого — це вимога багатоасистентності, і вона мусить триматися на
    рівні API, а не лише всередині ретривера.
    """
    settings = make_settings(tmp_path, worker_mode="inline")
    async with api_client(settings) as (client, _app):
        first = await create_assistant(client, "Перший")
        second = await create_assistant(client, "Другий")

        doc_a = await upload_text(
            client, first["collections"][0]["id"], name="перший.txt"
        )
        doc_b = await upload_text(
            client, second["collections"][0]["id"], name="другий.txt",
            text="# Тема\n\nДеривація снаряда — це відхилення снаряда від площини "
                 "стрільби внаслідок обертання снаряда навколо власної осі.\n",
        )
        await wait_for_status(client, first["collections"][0]["id"], doc_a["id"])
        await wait_for_status(client, second["collections"][0]["id"], doc_b["id"])

        session = (await client.post(
            "/api/sessions", json={"assistantId": first["id"]}
        )).json()
        response = await client.post("/api/chat", json={
            "sessionId": session["id"],
            "message": "Що таке деривація снаряда?",
        })
        citations = _frames(response.text, "chat.citations")[-1]["citations"]

    assert citations
    documents = {c["documentId"] for c in citations}
    assert doc_b["id"] not in documents, "матеріал чужого асистента протік у відповідь"
    assert documents == {doc_a["id"]}


async def test_retrieval_playground_shows_before_and_after(tmp_path) -> None:
    """Інструмент калібрування параметра №1 плану: top-k до і після реранкінгу."""
    settings = make_settings(tmp_path, worker_mode="inline")
    async with api_client(settings) as (client, _app):
        assistant = await create_assistant(client)
        collection_id = assistant["collections"][0]["id"]
        document = await upload_text(client, collection_id)
        await wait_for_status(client, collection_id, document["id"])

        body = (await client.get("/api/dev/retrieval-playground", params={
            "collectionId": collection_id, "q": "поправка на деривацію", "topK": 5,
        })).json()

    assert body["after"], "playground нічого не знайшов"
    assert body["before"], "немає кандидатів до реранкінгу"
    assert body["reranked"] is True
    top = body["after"][0]
    assert top["ordinal"] == 1
    assert top["rerank"] is not None
    assert top["text"]
    assert body["debug"]["denseCount"] > 0


async def test_chat_survives_missing_embedder(tmp_path, monkeypatch) -> None:
    """Машина без розпакованих ваг мусить дати ЗРОЗУМІЛУ помилку в чаті,
    а не 500 і порожній екран."""
    settings = make_settings(tmp_path)
    async with api_client(settings) as (client, application):
        assistant = await create_assistant(client)
        session = (await client.post(
            "/api/sessions", json={"assistantId": assistant["id"]}
        )).json()
        # Імітуємо стан «ваг немає»: провайдер не створився.
        application.state.services._provider = False
        application.state.services._retriever = None

        response = await client.post("/api/chat", json={
            "sessionId": session["id"], "message": "Що таке деривація?",
        })
    assert response.status_code == 200
    errors = _frames(response.text, "chat.error")
    assert errors and errors[0]["errorCode"] == "EMBEDDER_MISSING"
    assert "ASISTENT_STUB=1" in errors[0]["hint"]


async def test_empty_question_is_rejected(tmp_path) -> None:
    settings = make_settings(tmp_path)
    async with api_client(settings) as (client, _app):
        assistant = await create_assistant(client)
        session = (await client.post(
            "/api/sessions", json={"assistantId": assistant["id"]}
        )).json()
        response = await client.post("/api/chat", json={
            "sessionId": session["id"], "message": "   ",
        })
    assert response.status_code in (400, 422)


async def test_clear_messages_keeps_the_session(tmp_path) -> None:
    settings = make_settings(tmp_path, worker_mode="inline")
    async with api_client(settings) as (client, _app):
        assistant = await create_assistant(client)
        collection_id = assistant["collections"][0]["id"]
        document = await upload_text(client, collection_id)
        await wait_for_status(client, collection_id, document["id"])
        session = (await client.post(
            "/api/sessions", json={"assistantId": assistant["id"]}
        )).json()
        await client.post("/api/chat", json={
            "sessionId": session["id"], "message": "Що таке деривація снаряда?",
        })
        assert (await client.get(f"/api/sessions/{session['id']}/messages")).json()

        assert (await client.delete(
            f"/api/sessions/{session['id']}/messages"
        )).status_code == 204
        assert (await client.get(f"/api/sessions/{session['id']}/messages")).json() == []
        assert (await client.get(
            "/api/sessions", params={"assistant_id": assistant["id"]}
        )).json()


@pytest.mark.parametrize("detailed", [False, True])
async def test_detailed_flag_is_accepted(tmp_path, detailed: bool) -> None:
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
            "detailed": detailed,
        })
    assert _frames(response.text, "chat.token")
