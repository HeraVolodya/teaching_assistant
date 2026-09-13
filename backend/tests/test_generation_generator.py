"""Generator.answer: стрімінг, утримання, map-reduce, скасування.

Усе — у stub-режимі. Тест «map-reduce не губить джерела» перевіряє рівно ту
регресію, через яку відкинуто CompactAndRefine: на малих моделях refine
вироджується в повторення відповіді №1 із тихим ігноруванням джерел 2..n.
"""

from __future__ import annotations

import pytest

from app.backends.base import LlmUnavailable, Messages
from app.backends.stub_backend import StubBackend
from app.domain import AnswerChunk, AssistantConfig, GenerationResult, RetrievalDebug
from app.generation.generator import CancelToken, Generator
from tests.helpers_llm import evidence_two_documents, make_retrieved


class FakeTelemetry:
    def __init__(self) -> None:
        self.events: list[tuple] = []
        self.unresolved: list[tuple] = []

    def event(self, name, *, duration_ms=None, ok=True, error_code=None, meta=None):
        self.events.append((name, ok, error_code, meta or {}))

    def unresolved_citation(self, message_id, emitted, available):
        self.unresolved.append((message_id, emitted, list(available)))


class BrokenBackend:
    """Бекенд, який завжди падає — перевірка, що помилка доходить до UI
    українською, а не роняє стрім."""

    name = "broken"

    async def health(self): ...
    async def list_models(self): ...
    async def load_model(self, cfg): ...

    async def chat_stream(self, messages: Messages, **kwargs):
        raise LlmUnavailable("з'єднання відхилено")
        yield ""    # pragma: no cover — робить функцію генератором


async def collect(gen) -> list[AnswerChunk]:
    return [ac async for ac in gen]


def text_of(chunks: list[AnswerChunk]) -> str:
    return "".join(c.text for c in chunks if c.kind == "token")


def result_of(chunks: list[AnswerChunk]) -> GenerationResult:
    done = [c for c in chunks if c.kind == "done"]
    assert len(done) == 1, "рівно один термінальний AnswerChunk"
    return done[0].payload


def citations_frame(chunks: list[AnswerChunk]) -> dict:
    frames = [c for c in chunks if c.kind == "citations"]
    assert frames
    return frames[-1].payload


# ------------------------------------------------------------ прямий прохід
async def test_direct_answer_streams_cites_and_finishes() -> None:
    gen = Generator(StubBackend(), config=AssistantConfig())
    out = await collect(gen.answer("Як враховують деривацію?", evidence_two_documents()))

    assert [c.kind for c in out].count("done") == 1
    assert out[-1].kind == "done"
    text = text_of(out)
    assert text and "[1]" in text

    res = result_of(out)
    assert not res.abstained
    assert res.text.startswith(text.strip()[:20])
    assert res.tokens_out and res.tokens_out > 3
    assert res.ttft_ms is not None
    assert len(res.citations) >= 3


async def test_answer_combines_several_documents() -> None:
    """Пряма вимога НДР: поєднувати декілька джерел і посилатися на них."""
    gen = Generator(StubBackend())
    out = await collect(gen.answer("Питання?", evidence_two_documents()))
    res = result_of(out)
    assert len({c.document_id for c in res.citations}) >= 3


async def test_citations_frame_carries_merged_view() -> None:
    """Пігулки [n] — по кожному маркеру; підвал «Джерела» — зі злитими
    діапазонами сторінок."""
    gen = Generator(StubBackend())
    out = await collect(gen.answer("Питання?", evidence_two_documents()))
    frame = citations_frame(out)
    assert set(frame) == {"citations", "merged", "unresolved"}
    # doc-1 дав два суміжні діапазони (147-148 і 149-150) → один зливок.
    doc1 = [c for c in frame["merged"] if c.document_id == "doc-1"]
    assert len(doc1) == 1
    assert (doc1[0].page_from, doc1[0].page_to) == (147, 150)


# ------------------------------------------------------------------ відмова
async def test_abstain_never_calls_the_model() -> None:
    """Якщо ретривер утримався, кликати LLM означає дати їй шанс вигадати."""
    backend = StubBackend()
    gen = Generator(backend)
    debug = RetrievalDebug(query="щось", abstained=True, abstain_confidence=0.02)
    out = await collect(gen.answer("Питання?", evidence_two_documents(), debug=debug))

    assert backend.calls == []
    res = result_of(out)
    assert res.abstained
    assert "недостатньо інформації" in res.text
    assert res.citations == []


async def test_abstain_shows_nearest_matches() -> None:
    gen = Generator(StubBackend())
    debug = RetrievalDebug(query="q", abstained=True)
    out = await collect(gen.answer("Питання?", evidence_two_documents(), debug=debug))
    assert "Балістика та стрільба" in text_of(out)
    assert citations_frame(out)["merged"]


async def test_empty_evidence_abstains_without_model_call() -> None:
    backend = StubBackend()
    out = await collect(Generator(backend).answer("Питання?", []))
    assert backend.calls == []
    assert result_of(out).abstained


# --------------------------------------------------------------- map-reduce
async def test_map_reduce_keeps_every_document() -> None:
    """Ключовий тест модуля: жоден документ не зникає у зведенні."""
    backend = StubBackend()
    gen = Generator(backend)
    evidence = evidence_two_documents()
    out = await collect(gen.answer("Питання?", evidence, force_map_reduce=True))

    stages = [c.payload["stage"] for c in out if c.kind == "debug"]
    assert stages.count("map") == 3        # три різні документи
    assert stages[-1] == "reduce"
    # Три map-виклики + один reduce.
    assert len(backend.calls) == 4

    res = result_of(out)
    assert len({c.document_id for c in res.citations}) == 3, \
        "map-reduce втратив документ — це саме та регресія refine"


async def test_map_reduce_triggers_automatically_on_overflow() -> None:
    """Стратегія переповнення: менше діставати → обрізати вікно → map-reduce."""
    evidence = [
        make_retrieved(document_id=f"doc-{i}", ordinal=i, text="слово " * 3000,
                       rerank_score=0.9 - i * 0.01)
        for i in range(4)
    ]
    gen = Generator(StubBackend(), context_tokens=4096)
    out = await collect(gen.answer("Питання?", evidence))
    stages = [c.payload.get("stage") for c in out if c.kind == "debug"]
    assert "map" in stages and "reduce" in stages


async def test_map_reduce_skips_documents_with_no_information() -> None:
    """«Немає відомостей.» — корисний сигнал покриття, а не сміття."""
    class Silent(StubBackend):
        async def chat_stream(self, messages, **kwargs):
            user = messages[-1]["content"]
            if "документ «Методичні рекомендації з підготовки»" in user:
                yield "Немає відомостей."
                return
            async for d in super().chat_stream(messages, **kwargs):
                yield d

    telemetry = FakeTelemetry()
    gen = Generator(Silent(), telemetry=telemetry)
    out = await collect(gen.answer("Питання?", evidence_two_documents(),
                                   force_map_reduce=True))
    assert any(name == "map_no_information" for name, *_ in telemetry.events)
    res = result_of(out)
    assert "doc-3" not in {c.document_id for c in res.citations}
    assert len({c.document_id for c in res.citations}) == 2


async def test_map_reduce_abstains_when_no_document_covers_the_question() -> None:
    class AlwaysEmpty(StubBackend):
        async def chat_stream(self, messages, **kwargs):
            yield "Немає відомостей."

    gen = Generator(AlwaysEmpty())
    out = await collect(gen.answer("Питання?", evidence_two_documents(),
                                   force_map_reduce=True))
    assert result_of(out).abstained


# ----------------------------------------------------- галюциновані маркери
async def test_hallucinated_marker_is_stripped_and_logged() -> None:
    class Hallucinating(StubBackend):
        async def chat_stream(self, messages, **kwargs):
            for token in ("Твердження", " [1]", " і", " вигадка", " [42].", ""):
                if token:
                    yield token

    telemetry = FakeTelemetry()
    gen = Generator(Hallucinating(), telemetry=telemetry)
    out = await collect(gen.answer("Питання?", evidence_two_documents(),
                                   message_id="msg-7"))
    res = result_of(out)
    assert "[42]" not in res.text
    assert res.unresolved == ["[42]"]
    assert telemetry.unresolved == [("msg-7", "[42]", ["1", "2", "3", "4"])]
    assert citations_frame(out)["unresolved"] == ["[42]"]


# ------------------------------------------------------------- скасування
async def test_cancel_stops_generation_mid_stream() -> None:
    cancel = CancelToken()
    gen = Generator(StubBackend())
    out: list[AnswerChunk] = []
    async for ac in gen.answer("Питання?", evidence_two_documents(), cancel=cancel):
        out.append(ac)
        if len([c for c in out if c.kind == "token"]) == 3:
            cancel.cancel()
    assert out[-1].kind == "done"
    assert len([c for c in out if c.kind == "token"]) <= 4


# ---------------------------------------------------------------- помилки
async def test_backend_failure_surfaces_as_ukrainian_error_frame() -> None:
    telemetry = FakeTelemetry()
    gen = Generator(BrokenBackend(), telemetry=telemetry)
    out = await collect(gen.answer("Питання?", evidence_two_documents()))
    errors = [c for c in out if c.kind == "error"]
    assert errors and "LM Studio" in errors[0].text
    assert out[-1].kind == "done"
    assert any(name == "generation_failed" for name, *_ in telemetry.events)


# --------------------------------------------------------------- телеметрія
async def test_generation_event_records_strategy_and_documents() -> None:
    telemetry = FakeTelemetry()
    gen = Generator(StubBackend(), telemetry=telemetry)
    await collect(gen.answer("Питання?", evidence_two_documents()))
    meta = next(m for name, _ok, _e, m in telemetry.events if name == "generation")
    assert meta["strategy"] == "direct"
    assert meta["documents"] >= 3
    assert meta["unresolved"] == 0


@pytest.mark.parametrize("confidence", ["low", "medium", "high"])
async def test_min_supporting_chunks_is_respected(confidence: str) -> None:
    """Поріг виконується в КОДІ, а не в промпті."""
    cfg = AssistantConfig(confidence_required=confidence)   # type: ignore[arg-type]
    evidence = [make_retrieved(document_id=f"d{i}", ordinal=i, text="слово " * 5000)
                for i in range(4)]
    gen = Generator(StubBackend(), config=cfg, context_tokens=4096)
    out = await collect(gen.answer("Питання?", evidence))
    assert result_of(out) is not None
