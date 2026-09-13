"""Заглушка LLM — несуча конструкція CI, а не декорація.

Якщо ці тести падають, увесь конвеєр генерації стає неперевірюваним без
1.5 ГБ ваг і GPU.
"""

from __future__ import annotations

import asyncio

import pytest

from app.backends import get_backend, is_stub_enabled
from app.backends.base import ChatParams, LoadConfig, ModelNotLoaded
from app.backends.stub_backend import StubBackend, compose_stub_answer, tokenize_for_stream
from app.generation.prompt_builder import build_prompt
from tests.helpers_llm import evidence_two_documents


@pytest.fixture
def prompt():
    return build_prompt("Як враховують деривацію?", evidence_two_documents())


def test_stub_flag_reading(monkeypatch) -> None:
    monkeypatch.setenv("ASISTENT_STUB", "1")
    assert is_stub_enabled()
    monkeypatch.setenv("ASISTENT_STUB", "0")
    assert not is_stub_enabled()
    monkeypatch.delenv("ASISTENT_STUB")
    assert not is_stub_enabled()


async def test_get_backend_returns_stub_when_flag_set(monkeypatch) -> None:
    """Прапорець має пріоритет над усім: інакше випадково запущений на
    машині розробника LM Studio зробив би тести недетермінованими."""
    monkeypatch.setenv("ASISTENT_STUB", "1")
    backend = await get_backend()
    assert isinstance(backend, StubBackend)


async def test_stub_answer_is_ukrainian_and_cites_every_source(prompt) -> None:
    backend = StubBackend()
    text = "".join([d async for d in backend.chat_stream(prompt)])

    assert "[1]" in text and "[2]" in text and "[3]" in text and "[4]" in text
    # Українська, а не «lorem ipsum»: заглушка має бути показуваною в UI.
    assert any(ch in text for ch in "іїєґ")
    assert "деривац" in text.lower()


async def test_stub_is_deterministic(prompt) -> None:
    a = "".join([d async for d in StubBackend().chat_stream(prompt)])
    b = "".join([d async for d in StubBackend().chat_stream(prompt)])
    assert a == b


def test_tokens_concatenate_without_separator() -> None:
    """Пробіл живе на початку токена, як у SentencePiece. Тест ловить
    класичну помилку `" ".join(deltas)` у SSE-шарі."""
    text = "Деривація зростає з дальністю."
    assert "".join(tokenize_for_stream(text)) == text


async def test_stub_respects_max_tokens() -> None:
    backend = StubBackend()
    prompt = build_prompt("питання", evidence_two_documents())
    parts = [d async for d in backend.chat_stream(prompt, params=ChatParams(max_tokens=5))]
    assert len(parts) == 5


async def test_stub_stops_on_cancel(prompt) -> None:
    backend = StubBackend()
    cancel = asyncio.Event()
    got = []
    async for delta in backend.chat_stream(prompt, cancel=cancel):
        got.append(delta)
        cancel.set()
    assert len(got) == 1


async def test_stub_can_simulate_unloaded_model(prompt) -> None:
    backend = StubBackend(fail_first=1)
    with pytest.raises(ModelNotLoaded):
        async for _ in backend.chat_stream(prompt):
            pass


def test_stub_abstains_without_sources() -> None:
    text = compose_stub_answer([{"role": "user", "content": "Питання: щось\n\nВідповідь:"}])
    assert "недостатньо інформації" in text


def test_stub_preserves_markers_in_reduce_step() -> None:
    """Крок REDUCE не має блоку ДЖЕРЕЛА — «джерелами» є часткові висновки.
    Втрата маркерів тут і є та регресія, заради якої існує map-reduce."""
    reduce_prompt = [{"role": "user", "content": (
        "Питання: як враховують деривацію?\n\n"
        "ЧАСТКОВИЙ ВИСНОВОК 1 (документ «А»):\nПерший висновок [1] [2].\n\n"
        "ЧАСТКОВИЙ ВИСНОВОК 2 (документ «Б»):\nДругий висновок [3].\n\n"
        "Зведена відповідь:")}]
    text = compose_stub_answer(reduce_prompt)
    assert "[1]" in text and "[2]" in text and "[3]" in text


def test_both_backends_satisfy_the_protocol() -> None:
    """Заглушка і LM Studio мусять бути взаємозамінні — на цьому тримається
    і CI без GPU, і запасний варіант із llama-server."""
    from app.backends import LlmBackend, LmStudioClient

    assert isinstance(StubBackend(), LlmBackend)
    assert isinstance(LmStudioClient("http://127.0.0.1:1234"), LlmBackend)


def test_llm_layer_never_pulls_torch_into_the_api_process() -> None:
    """`import torch` коштує 1.5-4 с на Windows; в API-процесі це 6+ секунд
    до чутливості, і викладач вважає, що застосунок зламався."""
    import subprocess
    import sys

    code = ("import app.backends, app.generation, sys;"
            "print(any(m in sys.modules for m in ('torch', 'docling', 'rapidocr')))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         check=True)
    assert out.stdout.strip() == "False"


async def test_stub_health_and_load_are_no_ops() -> None:
    backend = StubBackend()
    health = await backend.health()
    assert health.ok and health.api_version == "stub"
    info = await backend.load_model(LoadConfig(model="x", context_length=4096))
    assert info.loaded_context == 4096
