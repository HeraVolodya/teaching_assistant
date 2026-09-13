"""HTTP-ендпоїнти: асистенти, документи, файл із Range, діагностика, setup."""

from __future__ import annotations

import io
import json
import zipfile

import pytest

from tests.helpers_api import (
    SAMPLE_TEXT,
    api_client,
    create_assistant,
    make_settings,
    upload_text,
)


# ---------------------------------------------------------------- здоров'я
async def test_health_reports_stub_database_and_worker(tmp_path) -> None:
    async with api_client(make_settings(tmp_path)) as (client, _app):
        body = (await client.get("/api/health")).json()
    assert body["ok"] is True
    assert body["stub"] is True
    assert body["database"]["ok"] is True
    assert body["database"]["schemaVersion"] >= 1
    assert body["lmstudio"]["apiVersion"] == "stub"
    assert body["worker"]["mode"] == "off"


async def test_health_does_not_run_integrity_check(tmp_path) -> None:
    """`PRAGMA integrity_check` на 2-гігабайтній базі — це хвилини, а health
    опитується щосекунди. Цілісність живе в діагностичному пакеті."""
    async with api_client(make_settings(tmp_path)) as (client, _app):
        body = (await client.get("/api/health")).json()
    assert "integrity" not in json.dumps(body)


async def test_setup_status_answers_without_models(tmp_path) -> None:
    """Машина без розпакованих ваг мусить піднятися й пояснити, чого бракує,
    а не відмовитись стартувати."""
    async with api_client(make_settings(tmp_path)) as (client, _app):
        body = (await client.get("/api/setup/status")).json()
    assert "hardware" in body and "recommendation" in body
    assert body["models"]["embeddings"]["present"] is False
    assert body["stub"] is True
    assert body["ready"] is True          # у stub-режимі працює все


async def test_lmstudio_start_without_cli_is_a_clear_503(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("app.api.routes_system.shutil.which", lambda _n: None)
    async with api_client(make_settings(tmp_path)) as (client, _app):
        response = await client.post("/api/setup/lmstudio/start")
    assert response.status_code == 503
    assert "Developer" in response.json()["detail"]


# --------------------------------------------------------------- асистенти
async def test_assistant_crud_and_default_collection(tmp_path) -> None:
    async with api_client(make_settings(tmp_path)) as (client, _app):
        created = await create_assistant(client, "Артилерія")
        assert created["name"] == "Артилерія"
        # Колекція створюється РАЗОМ з асистентом: інакше падіння між двома
        # викликами лишає асистента, у який неможливо покласти документ.
        assert len(created["collections"]) == 1
        assert created["collections"][0]["dim"] > 0

        listed = (await client.get("/api/assistants")).json()
        assert [a["id"] for a in listed] == [created["id"]]

        updated = (await client.put(
            f"/api/assistants/{created['id']}",
            json={"name": "Балістика", "config": {"confidenceRequired": "high"}},
        )).json()
        assert updated["name"] == "Балістика"
        assert updated["configVersion"] == created["configVersion"] + 1

        assert (await client.delete(f"/api/assistants/{created['id']}")).status_code == 204
        assert (await client.get(f"/api/assistants/{created['id']}")).status_code == 404


async def test_put_without_config_keeps_search_settings(tmp_path) -> None:
    """PUT без `config` не має скидати налаштування пошуку до дефолтів — це
    найдорожча несподіванка для викладача, який півдня калібрував асистента."""
    async with api_client(make_settings(tmp_path)) as (client, _app):
        created = await create_assistant(client, "А", final_top_k=9, max_per_document=3)
        assert created["config"]["final_top_k"] == 9

        updated = (await client.put(
            f"/api/assistants/{created['id']}", json={"name": "Б"}
        )).json()
    assert updated["config"]["final_top_k"] == 9
    assert updated["config"]["max_per_document"] == 3


async def test_unknown_config_fields_are_ignored(tmp_path) -> None:
    """Старіший бекенд не має ламатися об нове поле фронтенду."""
    async with api_client(make_settings(tmp_path)) as (client, _app):
        created = await create_assistant(client, "А", final_top_k=4, майбутнє_поле=42)
    assert created["config"]["final_top_k"] == 4
    assert "майбутнє_поле" not in created["config"]


async def test_prompt_preview_shows_thresholds_next_to_the_prompt(tmp_path) -> None:
    """Промптом неможливо надійно змусити 12B-модель відмовитись; порогом
    реранкера — можна. Прев'ю мусить показувати обидві половини разом."""
    async with api_client(make_settings(tmp_path)) as (client, _app):
        created = await create_assistant(
            client, "А", instructions="Ти асистент кафедри артилерії.",
            confidence_required="high",
        )
        preview = (await client.get(
            f"/api/assistants/{created['id']}/prompt-preview"
        )).json()
    assert preview["confidenceThreshold"] == pytest.approx(0.55)
    assert preview["minSupportingChunks"] == 2
    assert "кафедри артилерії" in preview["persona"]
    assert preview["rulesTokens"] > 0
    assert "ДЖЕРЕЛА" in preview["sample"]


# --------------------------------------------------------------- документи
async def test_upload_rejects_docx_with_an_explanation(tmp_path) -> None:
    """DOCX не має номерів сторінок узагалі: `msword_backend` ніколи не
    створює `ProvenanceItem`. Тихо прийняти файл і потім не вміти на нього
    послатись — гірше, ніж відмовити одразу."""
    async with api_client(make_settings(tmp_path)) as (client, _app):
        assistant = await create_assistant(client)
        collection_id = assistant["collections"][0]["id"]
        response = await client.post(
            f"/api/collections/{collection_id}/documents",
            files={"files": ("методичка.docx", b"PK\x03\x04", "application/octet-stream")},
        )
    assert response.status_code == 415
    assert "PDF" in response.json()["detail"]


async def test_upload_rejects_empty_file(tmp_path) -> None:
    async with api_client(make_settings(tmp_path)) as (client, _app):
        assistant = await create_assistant(client)
        collection_id = assistant["collections"][0]["id"]
        response = await client.post(
            f"/api/collections/{collection_id}/documents",
            files={"files": ("порожній.txt", b"", "text/plain")},
        )
    assert response.status_code == 400


async def test_upload_enforces_size_limit(tmp_path) -> None:
    settings = make_settings(tmp_path, max_upload_bytes=1024, upload_chunk_bytes=256)
    async with api_client(settings) as (client, application):
        assistant = await create_assistant(client)
        collection_id = assistant["collections"][0]["id"]
        response = await client.post(
            f"/api/collections/{collection_id}/documents",
            files={"files": ("велике.txt", b"x" * 5000, "text/plain")},
        )
        assert response.status_code == 413
        # Недописаний файл прибирається: інакше кожна відхилена спроба
        # лишала б сміття в каталозі даних.
        docs_dir = application.state.services.paths.documents_dir
        assert list(docs_dir.iterdir()) == []


async def test_upload_stores_under_uuid_not_original_name(tmp_path) -> None:
    """Кирилична назва плюс глибокий шлях тривіально пробивають MAX_PATH 260
    на Windows. Оригінальна назва живе метаданими в SQLite."""
    async with api_client(make_settings(tmp_path)) as (client, application):
        assistant = await create_assistant(client)
        collection_id = assistant["collections"][0]["id"]
        document = await upload_text(
            client, collection_id,
            name="Методичні рекомендації щодо стрільби артилерії.txt",
        )
        docs_dir = application.state.services.paths.documents_dir
        stored = list(docs_dir.iterdir())
    assert len(stored) == 1
    assert stored[0].name == f"{document['id']}.txt"
    assert document["originalName"].startswith("Методичні")


async def test_documents_list_carries_status_and_job(tmp_path) -> None:
    async with api_client(make_settings(tmp_path)) as (client, _app):
        assistant = await create_assistant(client)
        collection_id = assistant["collections"][0]["id"]
        document = await upload_text(client, collection_id)

        listed = (await client.get(f"/api/collections/{collection_id}/documents")).json()
    assert len(listed) == 1
    assert listed[0]["id"] == document["id"]
    assert listed[0]["status"] == "QUEUED"
    assert listed[0]["sizeBytes"] == len(SAMPLE_TEXT.encode("utf-8"))
    assert listed[0]["job"]["type"] == "PARSE"


async def test_reingest_switches_mode_and_requeues(tmp_path) -> None:
    async with api_client(make_settings(tmp_path)) as (client, application):
        assistant = await create_assistant(client)
        collection_id = assistant["collections"][0]["id"]
        document = await upload_text(client, collection_id)

        response = await client.post(
            f"/api/documents/{document['id']}/reingest", json={"mode": "DEEP"}
        )
        assert response.status_code == 200
        assert response.json()["ingestMode"] == "DEEP"
        assert response.json()["status"] == "QUEUED"

        jobs = application.state.services.queue.for_document(document["id"])
        assert [j["state"] for j in jobs].count("CANCELLED") == 1
        assert any(j["state"] == "QUEUED" and j["priority"] == 50 for j in jobs)


async def test_reingest_rejects_unknown_mode(tmp_path) -> None:
    async with api_client(make_settings(tmp_path)) as (client, _app):
        assistant = await create_assistant(client)
        collection_id = assistant["collections"][0]["id"]
        document = await upload_text(client, collection_id)
        response = await client.post(
            f"/api/documents/{document['id']}/reingest", json={"mode": "МАГІЯ"}
        )
    assert response.status_code == 422


async def test_cancel_reports_page_boundary_latency(tmp_path) -> None:
    """Скасування кооперативне: воркер перевіряє прапорець МІЖ сторінками."""
    async with api_client(make_settings(tmp_path)) as (client, _app):
        assistant = await create_assistant(client)
        collection_id = assistant["collections"][0]["id"]
        document = await upload_text(client, collection_id)
        body = (await client.post(f"/api/documents/{document['id']}/cancel")).json()
    assert body["cancelled"] == 1
    assert "сторінки" in body["detail"]


async def test_delete_document_removes_file_and_chunks(tmp_path) -> None:
    async with api_client(make_settings(tmp_path)) as (client, application):
        assistant = await create_assistant(client)
        collection_id = assistant["collections"][0]["id"]
        document = await upload_text(client, collection_id)
        docs_dir = application.state.services.paths.documents_dir
        assert list(docs_dir.iterdir())

        assert (await client.delete(f"/api/documents/{document['id']}")).status_code == 204
        assert list(docs_dir.iterdir()) == []
        assert (await client.get(f"/api/documents/{document['id']}/pages")).status_code == 404


# ------------------------------------------------------- віддача файлу
async def test_file_endpoint_supports_range(tmp_path) -> None:
    """PDF.js читає файл частинами: спершу хвіст із xref, потім потрібні
    сторінки. Без Range клік по цитаті означав би завантаження всіх 300 МБ."""
    payload = "0123456789" * 40
    async with api_client(make_settings(tmp_path)) as (client, _app):
        assistant = await create_assistant(client)
        collection_id = assistant["collections"][0]["id"]
        document = await upload_text(client, collection_id, text=payload)

        whole = await client.get(f"/api/documents/{document['id']}/file")
        assert whole.status_code == 200
        assert whole.headers["accept-ranges"] == "bytes"
        assert whole.text == payload

        part = await client.get(
            f"/api/documents/{document['id']}/file", headers={"Range": "bytes=10-19"}
        )
        assert part.status_code == 206
        assert part.headers["content-range"] == f"bytes 10-19/{len(payload)}"
        assert part.text == payload[10:20]

        tail = await client.get(
            f"/api/documents/{document['id']}/file", headers={"Range": "bytes=-5"}
        )
        assert tail.status_code == 206
        assert tail.text == payload[-5:]

        open_ended = await client.get(
            f"/api/documents/{document['id']}/file", headers={"Range": "bytes=395-"}
        )
        assert open_ended.status_code == 206
        assert open_ended.text == payload[395:]

        beyond = await client.get(
            f"/api/documents/{document['id']}/file", headers={"Range": "bytes=9999-"}
        )
        assert beyond.status_code == 416
        assert beyond.headers["content-range"] == f"bytes */{len(payload)}"


async def test_file_name_is_rfc5987_encoded(tmp_path) -> None:
    """Кирилична назва в голому `filename=` ламає заголовок; браузер
    показує «Настанова.pdf» лише через `filename*`."""
    async with api_client(make_settings(tmp_path)) as (client, _app):
        assistant = await create_assistant(client)
        collection_id = assistant["collections"][0]["id"]
        document = await upload_text(client, collection_id, name="Настанова.txt")
        response = await client.get(
            f"/api/documents/{document['id']}/file", params={"download": "true"}
        )
    disposition = response.headers["content-disposition"]
    assert disposition.startswith("attachment; filename*=UTF-8''")
    assert "%D0%9D" in disposition          # «Н» у percent-encoding


async def test_file_missing_on_disk_is_404(tmp_path) -> None:
    async with api_client(make_settings(tmp_path)) as (client, application):
        assistant = await create_assistant(client)
        collection_id = assistant["collections"][0]["id"]
        document = await upload_text(client, collection_id)
        for path in application.state.services.paths.documents_dir.iterdir():
            path.unlink()
        response = await client.get(f"/api/documents/{document['id']}/file")
    assert response.status_code == 404


# ------------------------------------------------------------- діагностика
async def test_diagnostics_bundle_hides_titles_by_default(tmp_path) -> None:
    """Це військова академія: перелік завантажених матеріалів сам по собі є
    інформацією. Вміст документів не потрапляє в пакет ніколи."""
    async with api_client(make_settings(tmp_path)) as (client, _app):
        assistant = await create_assistant(client)
        collection_id = assistant["collections"][0]["id"]
        await upload_text(client, collection_id, name="Таємна методичка.txt")

        response = await client.get("/api/diagnostics/bundle")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/zip"
        archive = zipfile.ZipFile(io.BytesIO(response.content))
        names = set(archive.namelist())
        assert {"readme.txt", "settings.json", "documents.json", "jobs.json"} <= names
        documents = json.loads(archive.read("documents.json"))
        assert documents and "title" not in documents[0]
        assert "Таємна" not in response.content.decode("utf-8", errors="ignore")
        assert "Деривація" not in response.content.decode("utf-8", errors="ignore")

        # ?includeTitles=true САМ ПО СОБІ не має права обійти політику.
        # Раніше вираз `include_titles and setting or include_titles`
        # згортався до просто `include_titles`, тому налаштування
        # `diagnostics_include_titles` («вимкнено за замовчуванням: військова
        # академія, самі назви матеріалів є інформацією») не мало жодного
        # впливу, і запит вивантажував назви в пакет підтримки.
        with_titles = await client.get(
            "/api/diagnostics/bundle", params={"includeTitles": "true"}
        )
        archive = zipfile.ZipFile(io.BytesIO(with_titles.content))
        documents = json.loads(archive.read("documents.json"))
        assert "title" not in documents[0], (
            "запит не має обходити політику: потрібні І дозвіл у налаштуваннях, І параметр"
        )


async def test_titles_appear_only_when_policy_allows_and_caller_asks(tmp_path) -> None:
    """Обидві умови разом — і лише разом."""
    settings = make_settings(tmp_path)
    object.__setattr__(settings, "diagnostics_include_titles", True) \
        if getattr(type(settings), "model_config", None) is None else None
    settings = settings.model_copy(update={"diagnostics_include_titles": True})

    async with api_client(settings) as (client, _app):
        assistant = await create_assistant(client)
        await upload_text(client, assistant["collections"][0]["id"],
                          name="Таємна методичка.txt")

        without = await client.get("/api/diagnostics/bundle")
        docs = json.loads(zipfile.ZipFile(io.BytesIO(without.content)).read("documents.json"))
        assert "title" not in docs[0], "без параметра назв не видаємо навіть за дозволу"

        with_titles = await client.get(
            "/api/diagnostics/bundle", params={"includeTitles": "true"}
        )
        docs = json.loads(zipfile.ZipFile(io.BytesIO(with_titles.content)).read("documents.json"))
        assert docs[0]["title"] == "Таємна методичка"


# --------------------------------------------------------------- 404-и
async def test_unknown_ids_are_404_not_500(tmp_path) -> None:
    async with api_client(make_settings(tmp_path)) as (client, _app):
        assert (await client.get("/api/assistants/невідомий")).status_code == 404
        assert (await client.get("/api/collections/невідома/documents")).status_code == 404
        assert (await client.get("/api/documents/невідомий/pages")).status_code == 404
        assert (await client.get("/api/documents/невідомий/file")).status_code == 404
        assert (await client.get("/api/sessions/невідома/messages")).status_code == 404
        assert (await client.get("/api/messages/невідоме/why")).status_code == 404
        assert (await client.get(
            "/api/sessions", params={"assistant_id": "невідомий"}
        )).status_code == 404
