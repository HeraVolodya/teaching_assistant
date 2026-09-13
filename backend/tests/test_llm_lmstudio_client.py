"""Клієнт LM Studio: виявлення, драбина контексту, стрімінг, скасування.

Реального LM Studio тут немає і бути не може — CI без GPU. Замість нього
httpx.MockTransport, що говорить тією самою OpenAI-сумісною формою. Це і є
прихована перевага архітектури з LM Studio: межа LLM тривіально мокається.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from app.backends.base import (
    LlmTimeout,
    LoadConfig,
    ModelLoadRefused,
    ModelNotLoaded,
    stream_with_idle_timeout,
)
from app.backends.lmstudio_client import (
    LmStudioClient,
    _iter_sse_deltas,
    candidate_base_urls,
    read_configured_port,
)

BASE = "http://127.0.0.1:1234"


def sse(*deltas: str) -> bytes:
    lines = []
    for d in deltas:
        payload = {"choices": [{"delta": {"content": d}, "index": 0}]}
        lines.append(f"data: {json.dumps(payload, ensure_ascii=False)}\n\n")
    lines.append("data: [DONE]\n\n")
    return "".join(lines).encode()


def v1_models_payload(loaded: bool = True) -> dict:
    return {"data": [{
        "model_key": "mamaylm-12b",
        "display_name": "MamayLM 12B",
        "type": "llm",
        "state": "loaded" if loaded else "not-loaded",
        "max_context_length": 32768,
        "quantization": "Q4_K_M",
        "engine": "llama.cpp",
    }]}


# ---------------------------------------------------------------- виявлення
def test_read_configured_port(tmp_path) -> None:
    cfg = tmp_path / "http-server-config.json"
    cfg.write_text(json.dumps({"port": 4321}), encoding="utf-8")
    assert read_configured_port(cfg) == 4321


def test_missing_or_broken_config_is_not_fatal(tmp_path) -> None:
    assert read_configured_port(tmp_path / "nope.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text("{не json", encoding="utf-8")
    assert read_configured_port(bad) is None


def test_candidates_prefer_config_then_default_ports(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("ASISTENT_LMSTUDIO_PORT", raising=False)
    cfg = tmp_path / "http-server-config.json"
    cfg.write_text(json.dumps({"port": 4321}), encoding="utf-8")
    urls = candidate_base_urls(cfg)
    assert urls[0] == "http://127.0.0.1:4321"
    assert urls[1:] == ["http://127.0.0.1:1234", "http://127.0.0.1:1235"]
    # Тільки петля назад: мережевий гард усе одно заблокує будь-що інше.
    assert all(u.startswith("http://127.0.0.1:") for u in urls)


# ------------------------------------------------------------------ health
async def test_health_prefers_v1_because_only_v1_reports_load_state() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/models"
        return httpx.Response(200, json=v1_models_payload())

    async with LmStudioClient(BASE, transport=httpx.MockTransport(handler)) as c:
        health = await c.health()
    assert health.ok and health.api_version == "v1"
    assert health.loaded_models[0].max_context == 32768
    assert health.loaded_models[0].quantization == "Q4_K_M"


async def test_health_falls_back_to_v0_then_openai() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if request.url.path == "/api/v1/models":
            return httpx.Response(404)
        if request.url.path == "/api/v0/models":
            return httpx.Response(404)
        return httpx.Response(200, json={"data": [{"id": "some-model"}]})

    async with LmStudioClient(BASE, transport=httpx.MockTransport(handler)) as c:
        health = await c.health()
    assert seen == ["/api/v1/models", "/api/v0/models", "/v1/models"]
    assert health.ok and health.api_version == "openai"
    # Деградована поверхня має чесно казати, що стан завантаження невідомий.
    assert "недоступний" in health.detail


async def test_health_false_when_nothing_answers() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    async with LmStudioClient(BASE, transport=httpx.MockTransport(handler)) as c:
        health = await c.health()
    assert not health.ok and health.api_version == "none"


# -------------------------------------------------------- драбина контексту
async def test_context_ladder_walks_down_on_guardrail_error() -> None:
    """Оцінка LM Studio буває завищена вдвічі, а «Load anyway» через REST
    немає — тож єдиний правильний хід це просити менший контекст."""
    tried: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        tried.append(body["context_length"])
        if body["context_length"] > 8192:
            return httpx.Response(400, json={
                "error": "modelLoadingGuardrails: estimated 22.9 GB exceeds available"})
        return httpx.Response(200, json={"model": {"model_key": body["model"],
                                                   "state": "loaded",
                                                   "loaded_context_length": body["context_length"]}})

    async with LmStudioClient(BASE, transport=httpx.MockTransport(handler)) as c:
        info, applied = await c.load_with_context_ladder(
            LoadConfig(model="mamaylm-12b", context_length=32768))

    assert tried == [32768, 16384, 8192]
    assert applied.context_length == 8192
    assert info.loaded_context == 8192


async def test_ladder_never_goes_above_requested_context() -> None:
    tried: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        tried.append(body["context_length"])
        return httpx.Response(200, json={"model": {"model_key": body["model"]}})

    async with LmStudioClient(BASE, transport=httpx.MockTransport(handler)) as c:
        await c.load_with_context_ladder(LoadConfig(model="m", context_length=8192))
    assert tried == [8192]


async def test_non_guardrail_failure_exhausts_ladder_and_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="model file not found")

    async with LmStudioClient(BASE, transport=httpx.MockTransport(handler)) as c:
        with pytest.raises(ModelLoadRefused):
            await c.load_with_context_ladder(LoadConfig(model="ghost", context_length=8192))


# ---------------------------------------------------------------- стрімінг
async def test_chat_stream_parses_sse_and_never_requests_logprobs() -> None:
    """logprobs у LM Studio не реалізовано (повертає NULL) — не просимо їх
    узагалі, щоб ніхто не будував на них оцінку впевненості."""
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, content=sse("Деривація", " — це", " відхилення."))

    async with LmStudioClient(BASE, transport=httpx.MockTransport(handler)) as c:
        parts = [d async for d in c.chat_stream(
            [{"role": "user", "content": "?"}], model="test/model"
        )]

    assert "".join(parts) == "Деривація — це відхилення."
    assert "logprobs" not in captured
    assert captured["stream"] is True
    # НІКОЛИ truncateMiddle: воно тихо викидає докази з середини промпту.
    assert captured["context_overflow_policy"] == "stopAtLimit"
    assert captured["stop"] == ["\n\nПитання:", "\n\nДЖЕРЕЛА", "\nUser:", "<end_of_turn>"]


async def test_cancel_stops_stream_and_marks_request_dead() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=sse(*[f"т{i}" for i in range(50)]))

    cancel = asyncio.Event()
    got: list[str] = []
    client = LmStudioClient(BASE, transport=httpx.MockTransport(handler))
    async for delta in client.chat_stream(
            [{"role": "user", "content": "?"}], model="test/model", cancel=cancel
        ):
        got.append(delta)
        if len(got) == 3:
            cancel.set()
    await client.aclose()

    assert len(got) < 50
    # Баг #1203: сервер може ще слати токени — запит позначено мертвим.
    assert client._dead_requests


async def test_reissues_exact_load_config_after_mid_answer_unload() -> None:
    """Auto-Evict вивантажив модель посеред відповіді → перевидати ТОЧНО ту
    конфігурацію, що застосована при старті, і повторити РІВНО один раз."""
    state = {"loaded": True, "chat_calls": 0, "load_payloads": []}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/models/load":
            body = json.loads(request.content)
            state["load_payloads"].append(body)
            state["loaded"] = True
            return httpx.Response(200, json={"model": {"model_key": body["model"]}})
        state["chat_calls"] += 1
        if state["chat_calls"] == 1:
            state["loaded"] = False
            return httpx.Response(400, json={"error": "Model not loaded"})
        return httpx.Response(200, content=sse("Відповідь."))

    cfg = LoadConfig(model="mamaylm-12b", context_length=8192, gpu_offload=0.61)
    async with LmStudioClient(BASE, transport=httpx.MockTransport(handler)) as c:
        await c.load_model(cfg)
        text = "".join([d async for d in c.chat_stream([{"role": "user", "content": "?"}])])

    assert text == "Відповідь."
    assert state["chat_calls"] == 2
    # Байт-у-байт та сама конфігурація, а не «приблизно така сама».
    assert state["load_payloads"][0] == state["load_payloads"][1]
    assert state["load_payloads"][1]["context_length"] == 8192
    assert state["load_payloads"][1]["gpu_offload"] == 0.61


async def test_unload_error_after_tokens_emitted_is_not_retried() -> None:
    """Повтор після вже відданих токенів склеїв би дві половини різних
    відповідей — це гірше за чесну помилку."""
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/models/load":
            return httpx.Response(200, json={"model": {"model_key": "m"}})
        return httpx.Response(400, json={"error": "model not loaded"})

    async with LmStudioClient(BASE, transport=httpx.MockTransport(handler)) as c:
        await c.load_model(LoadConfig(model="m", context_length=8192))
        with pytest.raises(ModelNotLoaded):
            async for _ in c.chat_stream([{"role": "user", "content": "?"}]):
                pass


# ---------------------------------------------- тайм-аут простою, не глобальний
async def test_idle_timeout_fires_between_tokens_not_on_total_duration() -> None:
    """Довга заземлена відповідь легітимно триває хвилини. Помилкою є не
    тривалість, а МОВЧАННЯ."""
    async def slow_but_steady():
        for i in range(6):
            await asyncio.sleep(0.02)
            yield f"t{i}"

    out = [x async for x in stream_with_idle_timeout(
        slow_but_steady(), first_timeout=0.2, idle_timeout=0.05)]
    assert len(out) == 6      # сумарно 0.12 с > idle_timeout, і це НЕ помилка

    async def stalls():
        yield "перший"
        await asyncio.sleep(0.5)
        yield "запізнілий"

    with pytest.raises(LlmTimeout) as exc:
        async for _ in stream_with_idle_timeout(stalls(), first_timeout=0.2,
                                                idle_timeout=0.05):
            pass
    assert "простій між токенами" in str(exc.value)


async def test_ttft_timeout_is_separate_from_idle_timeout() -> None:
    async def never_starts():
        await asyncio.sleep(0.5)
        yield "пізно"

    with pytest.raises(LlmTimeout) as exc:
        async for _ in stream_with_idle_timeout(never_starts(), first_timeout=0.05,
                                                idle_timeout=5.0):
            pass
    assert "перший токен" in str(exc.value)


# -------------------------------------------------------------- розбір SSE
@pytest.mark.parametrize("line", ["", "data: [DONE]", "не sse", "data: {зламано"])
def test_sse_noise_is_ignored_silently(line: str) -> None:
    assert _iter_sse_deltas(line) == []


def test_sse_delta_extracted() -> None:
    payload = json.dumps({"choices": [{"delta": {"content": "слово"}}]})
    assert _iter_sse_deltas(f"data: {payload}") == ["слово"]


# ---------------------------------------------------------------- вибір моделі
# Раніше `chat_stream` без явної моделі підставляв рядок "local-model", і
# LM Studio відповідав «No models loaded» — повідомлення, що вказує на порожню
# пам'ять, тоді як справжня причина була в тому, що ми не назвали модель.
# Два різні збої з однаковим текстом — найдорожчий вид діагностики.

def _models_response(*models: dict) -> httpx.Response:
    return httpx.Response(200, json={"models": list(models)})


def _model(key: str, *, loaded: bool, ctx: int = 8192, kind: str = "llm") -> dict:
    return {
        "key": key, "type": kind, "display_name": key, "format": "gguf",
        "max_context_length": 131072,
        "loaded_instances": ([{"identifier": "i", "context_length": ctx}] if loaded else []),
    }


async def test_resolves_the_loaded_model_and_never_sends_a_placeholder() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if "/models" in request.url.path:
            return _models_response(
                _model("org/small", loaded=True, ctx=4096),
                _model("org/big", loaded=True, ctx=32768),
                _model("org/cold", loaded=False),
            )
        captured.update(json.loads(request.content))
        return httpx.Response(200, content=sse("так"))

    async with LmStudioClient(BASE, transport=httpx.MockTransport(handler)) as c:
        [d async for d in c.chat_stream([{"role": "user", "content": "?"}])]

    assert captured["model"] != "local-model"
    # Серед піднятих обираємо ту, що дає найбільший корисний контекст.
    assert captured["model"] == "org/big"


async def test_jit_is_allowed_when_exactly_one_model_exists() -> None:
    """Один однозначний кандидат — довіряємо JIT-завантаженню LM Studio.

    Це не вигадування: ідентифікатор справжній, а якщо пам'яті не хватить,
    гардрейл відповість зрозумілою помилкою про ресурси.
    """
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if "/models" in request.url.path:
            return _models_response(_model("org/only", loaded=False))
        captured.update(json.loads(request.content))
        return httpx.Response(200, content=sse("так"))

    async with LmStudioClient(BASE, transport=httpx.MockTransport(handler)) as c:
        [d async for d in c.chat_stream([{"role": "user", "content": "?"}])]

    assert captured["model"] == "org/only"


async def test_no_loaded_model_raises_an_actionable_error() -> None:
    """Кілька моделей на диску, жодна в пам'яті → підказка, а не «No models loaded»."""
    from app.backends.base import NoChatModelReady

    def handler(request: httpx.Request) -> httpx.Response:
        if "/models" in request.url.path:
            return _models_response(
                _model("org/a", loaded=False), _model("org/b", loaded=False)
            )
        raise AssertionError("генерація не мала навіть починатися")

    async with LmStudioClient(BASE, transport=httpx.MockTransport(handler)) as c:
        with pytest.raises(NoChatModelReady) as err:
            [d async for d in c.chat_stream([{"role": "user", "content": "?"}])]

    text = str(err.value)
    assert "org/a" in text, "повідомлення мусить перелічити наявні на диску моделі"
    assert "lms load" in text or "Developer" in text, "мусить казати, ЩО зробити"


def test_loaded_instances_is_the_only_readiness_signal() -> None:
    """Список моделей — це те, що на ДИСКУ. У пам'яті — лише loaded_instances.

    Перевірка здоров'я бачила «сервер відповідає, дві моделі у списку» й
    показувала зелений стан, тоді як кожен запит генерації падав.
    """
    from app.backends.lmstudio_client import _parse_model

    cold = _parse_model(_model("org/x", loaded=False))
    assert cold.loaded is False
    assert cold.loaded_context is None

    warm = _parse_model(_model("org/x", loaded=True, ctx=16384))
    assert warm.loaded is True
    assert warm.loaded_context == 16384


def test_quantization_object_is_flattened_to_its_name() -> None:
    """v1 віддає quantization як ОБ'ЄКТ; str() давав «{'name': 'MXFP4', …}»
    у полі, яке показується користувачу."""
    from app.backends.lmstudio_client import _parse_model

    info = _parse_model({
        "key": "openai/gpt-oss-20b", "type": "llm", "format": "mlx",
        "quantization": {"name": "MXFP4", "bits_per_weight": 4},
        "loaded_instances": [],
    })
    assert info.quantization == "MXFP4"
    assert info.engine == "mlx"


def test_find_matches_lmstudio_key_against_repository_id() -> None:
    """Реєстр несе ID репозиторію, LM Studio — власний короткий ключ.

    Регресія: драбина рекомендує `mlx-community/gemma-4-12B-it-4bit`, а LM
    Studio показує ту саму модель як `gemma-4-12b-it`. `find()` порівнював
    рядки точно, тож завжди повертав None: застосунок бачив модель у списку
    й водночас вважав, що її немає, і пропонував «завантажте модель».
    Розбіжність подвійна — префікс організації, регістр і хвіст квантизації.
    """
    from app.backends.base import BackendHealth, ModelInfo

    health = BackendHealth(
        ok=True,
        api_version="v1",
        models=(
            ModelInfo(key="gemma-4-12b-it", kind="llm"),
            ModelInfo(key="openai/gpt-oss-20b", kind="llm"),
        ),
    )

    found = health.find("mlx-community/gemma-4-12B-it-4bit")
    assert found is not None and found.key == "gemma-4-12b-it"
    # Точний збіг лишається пріоритетним і працює як раніше.
    assert health.find("openai/gpt-oss-20b").key == "openai/gpt-oss-20b"
    # Нормалізація не має зливати РІЗНІ моделі в одну.
    assert health.find("google/gemma-4-31b-it-qat-GGUF") is None
    assert health.find("") is None


async def test_warm_ttft_timeout_is_retried_once_with_the_cold_budget() -> None:
    """JIT-завантаження не вкладається в теплий TTFT — повторити з холодним.

    Регресія з живої машини: `justInTimeModelLoading` у LM Studio вивантажує
    модель за власним TTL, і наступний запит мовчки чекає, поки 12B підніметься
    заново (~30 с — рівно межа TTFT_WARM_S). Сервер запит уже ПРИЙНЯВ, тож
    `ModelNotLoaded` не приходить, і гілка повтору не спрацьовувала: перше
    питання після паузи падало завжди, а повторене те саме — проходило.
    """
    calls: list[bool] = []

    async def fake_once(messages, *, params, model, cancel, warm):
        calls.append(warm)
        if len(calls) == 1:
            raise LlmTimeout(30.0, stage="перший токен")
        yield "Відповідь."

    async with LmStudioClient(BASE, transport=httpx.MockTransport(
        lambda r: httpx.Response(200, json={})
    )) as c:
        c._chat_stream_once = fake_once
        text = "".join([d async for d in c.chat_stream([{"role": "user", "content": "?"}])])

    assert text == "Відповідь."
    # Саме два виклики: теплий, потім холодний. Не три і не нескінченно.
    assert calls == [True, False]


async def test_ttft_timeout_after_tokens_emitted_is_not_retried() -> None:
    """Та сама межа, що й для вивантаження: повтор після відданих токенів
    склеїв би дві половини різних відповідей."""
    calls: list[bool] = []

    async def fake_once(messages, *, params, model, cancel, warm):
        calls.append(warm)
        yield "Почало"
        raise LlmTimeout(60.0, stage="між токенами")

    async with LmStudioClient(BASE, transport=httpx.MockTransport(
        lambda r: httpx.Response(200, json={})
    )) as c:
        c._chat_stream_once = fake_once
        with pytest.raises(LlmTimeout):
            async for _ in c.chat_stream([{"role": "user", "content": "?"}]):
                pass

    assert calls == [True]
