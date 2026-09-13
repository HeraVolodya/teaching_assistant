"""Шина подій і єдиний SSE-канал."""

from __future__ import annotations

import asyncio
import json

import pytest

from app.api.events import (
    CHAT_TOKEN,
    DOC_READY,
    EVENT_TYPES,
    JOB_PROGRESS,
    EventBus,
    parse_last_event_id,
    sse_frame,
)
from tests.helpers_api import api_client, make_settings, read_sse, sse_probe


# ------------------------------------------------------------------ кадри
def test_frame_keeps_ukrainian_readable() -> None:
    """`ensure_ascii=False` — не косметика: інакше кожна українська відповідь
    роздувається втричі в \\uXXXX і стрімінг видимо гальмує."""
    frame = sse_frame(CHAT_TOKEN, {"delta": "деривація"}, event_id=7)
    assert "деривація" in frame
    assert "\\u" not in frame
    assert frame.startswith("id: 7\nevent: chat.token\ndata: ")
    assert frame.endswith("\n\n")


def test_frame_never_breaks_on_newlines() -> None:
    """Багаторядкового `data:` не буває за побудовою: json.dumps екранує \\n.

    Якби кадр містив сирий перевід рядка, клієнт прочитав би половину події,
    а другу половину — як нову подію без типу.
    """
    frame = sse_frame(CHAT_TOKEN, {"delta": "рядок\nдругий"})
    body = frame.split("data: ", 1)[1]
    assert body.count("\n") == 2  # рівно термінатор кадру
    assert json.loads(body.strip())["delta"] == "рядок\nдругий"


def test_dataclasses_serialise_through_fallback() -> None:
    from app.domain import Citation

    citation = Citation(ordinal=1, chunk_uid="a" * 32, document_id="d",
                        document_title="Підручник", page_from=5, page_to=5,
                        page_label="с. 5", quote="цитата")
    frame = sse_frame("chat.citations", {"citations": [citation]})
    payload = json.loads(frame.split("data: ", 1)[1])
    assert payload["citations"][0]["page_label"] == "с. 5"


# -------------------------------------------------------------------- шина
def test_ids_are_monotonic() -> None:
    bus = EventBus()
    ids = [bus.publish(JOB_PROGRESS, {"n": i}).id for i in range(5)]
    assert ids == [1, 2, 3, 4, 5]


def test_replay_returns_only_events_after_the_id() -> None:
    bus = EventBus()
    for i in range(5):
        bus.publish(JOB_PROGRESS, {"n": i})
    replayed = bus.replay(2)
    assert [e.data["n"] for e in replayed] == [2, 3, 4]


def test_replay_of_none_is_empty_not_everything() -> None:
    """Свіже підключення без Last-Event-ID НЕ має отримати весь буфер:
    інакше кожне відкриття вкладки перепрогравало б історію індексації."""
    bus = EventBus()
    bus.publish(DOC_READY, {"docId": "d"})
    assert bus.replay(None) == []


def test_buffer_is_bounded() -> None:
    bus = EventBus(buffer=16)
    for i in range(100):
        bus.publish(JOB_PROGRESS, {"n": i})
    assert len(bus.history(limit=1000)) == 16


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("12", 12), ("0", 0), (" 5 ", 5), ("", None), (None, None), ("abc", None), ("-3", None)],
)
def test_last_event_id_parsing_survives_garbage(raw: str | None, expected: int | None) -> None:
    assert parse_last_event_id(raw) == expected


async def test_stream_delivers_live_events() -> None:
    bus = EventBus()
    bus.attach_loop()
    stream = bus.stream(keepalive_s=5.0)
    first = await stream.__anext__()
    assert first.startswith("retry:")

    bus.publish(DOC_READY, {"docId": "abc"})
    frame = await asyncio.wait_for(stream.__anext__(), timeout=2.0)
    assert "doc.ready" in frame and "abc" in frame
    await stream.aclose()


async def test_stream_replays_missed_then_goes_live() -> None:
    bus = EventBus()
    bus.attach_loop()
    bus.publish(JOB_PROGRESS, {"n": 0})
    bus.publish(JOB_PROGRESS, {"n": 1})

    stream = bus.stream(last_event_id=1, keepalive_s=5.0)
    await stream.__anext__()                       # retry
    missed = await stream.__anext__()
    assert '"n": 1' in missed                      # подія з id=2

    bus.publish(JOB_PROGRESS, {"n": 2})
    live = await asyncio.wait_for(stream.__anext__(), timeout=2.0)
    assert '"n": 2' in live
    await stream.aclose()


async def test_keepalive_comment_is_sent_when_idle() -> None:
    """Без пінга проксі й приспані вкладки рвуть з'єднання посеред
    індексації, і прогрес просто зникає."""
    bus = EventBus()
    bus.attach_loop()
    stream = bus.stream(keepalive_s=0.05)
    await stream.__anext__()
    ping = await asyncio.wait_for(stream.__anext__(), timeout=2.0)
    assert ping == ": ping\n\n"
    await stream.aclose()


async def test_publish_from_another_thread_reaches_subscriber() -> None:
    """У inline-режимі конвеєр публікує з пулу потоків.

    `asyncio.Queue.put_nowait` з чужого потоку — це тиха втрата події; тест
    ловить саме регресію на `call_soon_threadsafe`.
    """
    bus = EventBus()
    bus.attach_loop()
    stream = bus.stream(keepalive_s=5.0)
    await stream.__anext__()

    await asyncio.to_thread(bus.publish, DOC_READY, {"docId": "з-потоку"})
    frame = await asyncio.wait_for(stream.__anext__(), timeout=2.0)
    assert "з-потоку" in frame
    await stream.aclose()


async def test_slow_subscriber_is_dropped_not_blocking() -> None:
    """Повільний WebView не має права зупинити індексацію."""
    bus = EventBus()
    bus.attach_loop()
    stream = bus.stream(keepalive_s=5.0, max_queue=2)
    await stream.__anext__()
    for i in range(50):
        bus.publish(JOB_PROGRESS, {"n": i})       # не має ні впасти, ні зависнути
    assert bus.subscriber_count == 1
    await stream.aclose()


def test_unsubscribe_on_close() -> None:
    async def scenario() -> int:
        bus = EventBus()
        bus.attach_loop()
        stream = bus.stream(keepalive_s=5.0)
        await stream.__anext__()
        assert bus.subscriber_count == 1
        await stream.aclose()
        return bus.subscriber_count

    assert asyncio.run(scenario()) == 0


def test_event_type_catalogue_covers_the_contract() -> None:
    """П'ять типів із завдання мусять існувати під точними іменами:
    фронтенд підписується на рядки, а не на константи Python."""
    for required in ("job.progress", "job.failed", "doc.ready",
                     "chat.token", "chat.citations"):
        assert required in EVENT_TYPES


# ------------------------------------------------------------------- HTTP
async def test_events_endpoint_streams_and_resumes(tmp_path) -> None:
    settings = make_settings(tmp_path)
    async with api_client(settings) as (client, application):
        services = application.state.services
        services.emit("doc.ready", {"docId": "перший"})
        services.emit("doc.ready", {"docId": "другий"})

        history = (await client.get("/api/events/history")).json()
        assert [e["data"]["docId"] for e in history] == ["перший", "другий"]

        # Перепідключення з Last-Event-ID віддає ЛИШЕ пропущене.
        status, headers, body = await sse_probe(
            application, "/api/events", headers={"Last-Event-ID": "1"}, max_frames=2
        )
        assert status == 200
        assert headers["content-type"].startswith("text/event-stream")
        # Без цього заголовка проксі буферизує стрім у ніщо, і викладач бачить
        # порожній екран замість відповіді, що друкується.
        assert headers["x-accel-buffering"] == "no"

        docs = [json.loads(d)["docId"] for e, d in read_sse(body) if e == "doc.ready"]
        assert docs == ["другий"]


async def test_events_endpoint_without_resume_starts_empty(tmp_path) -> None:
    """Нове підключення не перепрогравання історії: інакше кожне відкриття
    вкладки сипало б сповіщеннями про вчорашню індексацію."""
    settings = make_settings(tmp_path)
    async with api_client(settings) as (_client, application):
        application.state.services.emit("doc.ready", {"docId": "старий"})
        _status, _headers, body = await sse_probe(
            application, "/api/events", max_frames=1
        )
        assert "старий" not in body
        assert body.startswith("retry:")
