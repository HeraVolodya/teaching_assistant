"""Спільна оснастка для тестів HTTP-шару, черги й воркера.

Клієнт піднімається через `httpx.ASGITransport` — тобто без сокета, без порту
й без гонки за 8765. Життєвий цикл (`lifespan`) проганяється явно: саме в
ньому створюються БД, шина подій і наглядач за воркерами, тож тест без нього
перевіряв би не той застосунок, що працює в постачанні.

Усе крутиться в stub-режимі: жодної завантаженої моделі, детерміновані
вектори, ехо-реранкер, українська відповідь LLM із маркерами [n]. Це вимога
контракту, а не зручність тестів.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx

from app.settings import Settings

__all__ = [
    "SAMPLE_TEXT",
    "make_settings",
    "api_client",
    "upload_text",
    "wait_for_status",
    "create_assistant",
    "read_sse",
    "sse_probe",
]

# Маленький український корпус із заголовками: чанкер будує з нього
# L0 (картка) + L1 (секції) + L2 (листки), тож конвеєр перевіряється цілком,
# а не на виродженому одноабзацному вході.
SAMPLE_TEXT = """# Основи стрільби артилерії

## Деривація снаряда

Деривація снаряда — це відхилення снаряда від площини стрільби внаслідок
обертання снаряда навколо власної осі. Величина деривації зростає з
дальністю стрільби та залежить від крутизни нарізів каналу ствола.

Поправка на деривацію обчислюється за таблицями стрільби окремо для кожної
гармати та кожного заряду. Без урахування поправки на деривацію відхилення
на граничних дальностях сягає десятків метрів.

## Гаубиця Д-30

Гаубиця Д-30 калібру 122 мм має максимальну дальність стрільби 15300 метрів
осколково-фугасним снарядом. Бойова маса гармати становить 3200 кілограмів,
розрахунок складається з шести осіб.

Таблиці стрільби для гаубиці Д-30 оформлені згідно з ДСТУ 3008:2015 і
містять поправки на деривацію, температуру заряду та зношення ствола.
"""


def make_settings(tmp_path: Path, **overrides: Any) -> Settings:
    """Налаштування тесту: свій каталог даних, заглушки, воркер вимкнено."""
    values: dict[str, Any] = {
        "stub": True,
        "data_dir": tmp_path / "data",
        "worker_mode": "off",
        "worker_poll_interval_s": 0.01,
        "job_watch_interval_s": 0.02,
        "event_keepalive_s": 0.2,
        "use_reranker": True,
    }
    values.update(overrides)
    return Settings(**values)


@asynccontextmanager
async def api_client(settings: Settings) -> AsyncIterator[tuple[httpx.AsyncClient, Any]]:
    """Піднятий застосунок плюс клієнт до нього."""
    from app.main import create_app

    application = create_app(settings)
    async with application.router.lifespan_context(application):
        transport = httpx.ASGITransport(app=application)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://127.0.0.1", timeout=30.0
        ) as client:
            yield client, application


async def create_assistant(client: httpx.AsyncClient, name: str = "Артилерія",
                           **config: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"name": name, "emoji": "🎯"}
    if config:
        payload["config"] = config
    response = await client.post("/api/assistants", json=payload)
    response.raise_for_status()
    return response.json()


async def upload_text(
    client: httpx.AsyncClient,
    collection_id: str,
    *,
    name: str = "balistyka.txt",
    text: str = SAMPLE_TEXT,
    **form: Any,
) -> dict[str, Any]:
    response = await client.post(
        f"/api/collections/{collection_id}/documents",
        files={"files": (name, text.encode("utf-8"), "text/plain")},
        data={k: str(v) for k, v in form.items()},
    )
    response.raise_for_status()
    return response.json()[0]


async def wait_for_status(
    client: httpx.AsyncClient,
    collection_id: str,
    document_id: str,
    *,
    target: str = "READY",
    timeout: float = 20.0,
) -> dict[str, Any]:
    """Дочекатись статусу документа, опитуючи публічний API.

    Опитуємо саме API, а не БД: тест мусить перевіряти те, що бачить UI.
    """
    deadline = asyncio.get_running_loop().time() + timeout
    last: dict[str, Any] = {}
    while asyncio.get_running_loop().time() < deadline:
        response = await client.get(f"/api/collections/{collection_id}/documents")
        response.raise_for_status()
        for item in response.json():
            if item["id"] == document_id:
                last = item
                if item["status"] in (target, "FAILED", "CANCELLED"):
                    return item
        await asyncio.sleep(0.02)
    raise AssertionError(
        f"Документ {document_id} не досяг статусу {target} за {timeout} с; останній стан: {last}"
    )


async def sse_probe(
    application: Any,
    path: str,
    *,
    headers: dict[str, str] | None = None,
    max_frames: int = 3,
    timeout: float = 10.0,
) -> tuple[int, dict[str, str], str]:
    """Прочитати кілька кадрів НЕСКІНЧЕННОГО SSE-каналу просто з ASGI.

    Через `httpx.ASGITransport` це неможливо в принципі: він БУФЕРИЗУЄ —
    `handle_async_request` спершу доганяє `await app(...)` до кінця й лише
    потім віддає `Response`. Для `/api/events`, який за призначенням не
    закінчується ніколи, це вічне очікування. Тому канал читається на рівні
    ASGI: так тест бачить справжній ендпоїнт разом із заголовками й обробкою
    `Last-Event-ID`, а не якийсь спрощений сурогат.
    """
    query = ""
    if "?" in path:
        path, query = path.split("?", 1)
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": query.encode(),
        "root_path": "",
        "headers": [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()],
        "server": ("127.0.0.1", 8765),
        "client": ("127.0.0.1", 5000),
    }
    status = 0
    out_headers: dict[str, str] = {}
    chunks: list[str] = []
    enough = asyncio.Event()
    disconnected = asyncio.Event()

    async def receive() -> dict[str, Any]:
        await disconnected.wait()
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        nonlocal status
        if message["type"] == "http.response.start":
            status = int(message["status"])
            out_headers.update(
                {k.decode(): v.decode() for k, v in message.get("headers", [])}
            )
        elif message["type"] == "http.response.body":
            body = message.get("body", b"")
            if body:
                chunks.append(body.decode("utf-8"))
            if "".join(chunks).count("\n\n") >= max_frames or not message.get("more_body", True):
                enough.set()

    task = asyncio.create_task(application(scope, receive, send))
    try:
        await asyncio.wait_for(enough.wait(), timeout=timeout)
    finally:
        disconnected.set()
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
    return status, out_headers, "".join(chunks)


def read_sse(body: str) -> list[tuple[str, str]]:
    """Розібрати SSE-тіло на пари (подія, JSON-рядок)."""
    out: list[tuple[str, str]] = []
    for block in body.split("\n\n"):
        event = data = ""
        for line in block.splitlines():
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: "):
                data = line[6:]
        if event:
            out.append((event, data))
    return out
