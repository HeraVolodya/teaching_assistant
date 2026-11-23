"""LLM-реранкер (listwise через LM Studio): уривки, розбір оцінок, запасний шлях.

Реального LM Studio тут немає — httpx.MockTransport говорить тією самою формою.
Регресія, заради якої модуль існує (живі дані, «що таке арт підрозділ?»): CPU-
крос-енкодер оцінював 50 кандидатів ~80 с і дав визначенню «Гарматна обслуга —
найменший вогневий підрозділ» 0.15, а загальним абзацам вступу — 0.86.
"""

from __future__ import annotations

import json

import httpx
import pytest

from app.rerank.llm_reranker import (
    GRADE_SCORES,
    LlmListwiseReranker,
    build_snippet,
    parse_grades,
)

BASE = "http://127.0.0.1:1234"
QUERY = "що таке артилерійський підрозділ?"
GLOSSARY = (
    "Б\n"
    "Балістична підготовка стрільби - визначення відхилень балістичних умов стрільби від "
    "табличних, що передбачає розподіл боєприпасів між підрозділами і гарматами та їх "
    "сортування щодо балістичних характеристик і урахування під час підготовки стрільби. "
    "Батарея - вогневий і тактичний підрозділ артилерії. Складається із двох-трьох "
    "вогневих взводів."
)
INTRO = (
    "Вступ\n"
    "Глибоке засвоєння матеріалу є необхідною умовою підготовки офіцерів артилерійських "
    "підрозділів. Підручник розкриває зміст навчальної програми."
)


# ------------------------------------------------------------------ уривки
def test_snippet_keeps_the_definition_sentence() -> None:
    """Прототип різав уривок з початку, і визначення «Батарея — …» не вміщалось
    після довгого речення про балістичну підготовку — модель його не бачила."""
    snippet = build_snippet(GLOSSARY, QUERY, limit=160)
    assert "Батарея - вогневий і тактичний підрозділ артилерії." in snippet
    assert len(snippet) <= 160


def test_snippet_shows_the_section_title() -> None:
    """Назва розділу — сигнал «це вступ / висновки / словник», без якого
    загальні абзаци легко сплутати з відповіддю."""
    assert build_snippet(INTRO, QUERY).startswith("(розділ: Вступ) ")


def test_long_first_line_is_text_not_a_title() -> None:
    """Порожній шлях заголовків → перший рядок `rerank_text` — уже тіло фрагмента."""
    text = ("Вогневий взвод - вогневий підрозділ, що входить до складу батареї, "
            "і має дві або більше гармат.\nДругий рядок.")
    snippet = build_snippet(text, "що таке вогневий взвод?")
    assert not snippet.startswith("(розділ:")
    assert "Вогневий взвод - вогневий підрозділ" in snippet


def test_snippet_without_any_match_falls_back_to_the_beginning() -> None:
    text = "Розділ\nПерше речення. Друге речення."
    expected = "(розділ: Розділ) Перше речення. Друге речення."
    assert build_snippet(text, "щось зовсім інше", limit=80) == expected
    # Ліміт діє на весь уривок разом із назвою розділу.
    assert len(build_snippet(text, "щось зовсім інше", limit=30)) == 30


# ------------------------------------------------------------ розбір оцінок
def test_grades_are_parsed_in_model_order() -> None:
    assert parse_grades("12:3, 4:2, 7:1", 20) == [(11, 3), (3, 2), (6, 1)]


def test_zero_means_nothing_helps() -> None:
    assert parse_grades("0", 20) == []


def test_unparsable_answer_is_reported_as_none() -> None:
    assert parse_grades("не можу визначити", 20) is None


def test_numbers_without_grades_are_kept_as_useful() -> None:
    assert parse_grades("5, 2", 10) == [(4, 2), (1, 2)]


def test_out_of_range_duplicates_and_zero_grades_are_dropped() -> None:
    assert parse_grades("3:3, 3:2, 99:3, 4:0, 1:2", 10) == [(2, 3), (0, 2)]


def test_listing_is_capped() -> None:
    assert len(parse_grades(", ".join(f"{i}:2" for i in range(1, 30)), 40, limit=8)) == 8


# ----------------------------------------------------------- наскрізно з HTTP
def _models(*, loaded: bool = True, context: int = 8192, reasoning: bool = True) -> dict:
    model = {
        "key": "google/gemma-4-e4b", "type": "llm",
        "loaded_instances": [{"id": "x", "context_length": context}] if loaded else [],
    }
    if reasoning:
        model["capabilities"] = {"reasoning": {"allowed_options": ["off", "on"], "default": "on"}}
    return {"models": [model]}


def _reranker(answer: str, sent: list[dict], **models_kw) -> LlmListwiseReranker:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json=_models(**models_kw))
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": answer}}]})

    return LlmListwiseReranker(base_url=BASE, transport=httpx.MockTransport(handler))


def test_scores_follow_the_model_grades_and_order() -> None:
    sent: list[dict] = []
    rr = _reranker("2:3, 1:1", sent)
    scores = rr.score(QUERY, [INTRO, GLOSSARY, "Орієнтир - місцевий предмет."])
    assert scores[1] == pytest.approx(GRADE_SCORES[3])
    assert scores[0] == pytest.approx(GRADE_SCORES[1] - 0.005)
    assert scores[2] == 0.0
    assert rr.last_mode == "llm"


def test_request_is_deterministic_and_skips_reasoning() -> None:
    """Без міркувань: інакше «кілька номерів» розтягувались би на сотні
    прихованих токенів (див. lmstudio_client.py)."""
    sent: list[dict] = []
    _reranker("1:3", sent).score(QUERY, [GLOSSARY])
    body = sent[0]
    assert body["model"] == "google/gemma-4-e4b"
    assert body["temperature"] == 0.0
    assert body["reasoning_effort"] == "none"
    prompt = body["messages"][0]["content"]
    assert "[1] (розділ: Б)" in prompt and "Батарея - вогневий" in prompt


def test_reasoning_parameter_is_not_sent_to_models_without_it() -> None:
    sent: list[dict] = []
    _reranker("1:3", sent, reasoning=False).score(QUERY, [GLOSSARY])
    assert "reasoning_effort" not in sent[0]


def test_small_context_lists_only_what_fits() -> None:
    """Вікно моделі менше за 50 уривків — оцінюємо перші (за RRF), що вміщаються,
    а не ловимо `stopAtLimit` посеред переліку."""
    sent: list[dict] = []
    scores = _reranker("1:3", sent, context=1400).score(QUERY, [GLOSSARY] * 50)
    prompt = sent[0]["messages"][0]["content"]
    assert "[1] " in prompt and "[50] " not in prompt
    assert len(scores) == 50


class _Fallback:
    def __init__(self) -> None:
        self.calls = 0

    def score(self, query: str, docs: list[str]) -> list[float]:
        self.calls += 1
        return [0.42] * len(docs)


def test_lmstudio_down_falls_back_to_the_cross_encoder() -> None:
    fallback = _Fallback()

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    rr = LlmListwiseReranker(base_url=BASE, transport=httpx.MockTransport(handler),
                             fallback=lambda: fallback)
    assert rr.score(QUERY, [INTRO, GLOSSARY]) == [0.42, 0.42]
    assert rr.last_mode == "fallback" and fallback.calls == 1


def test_unparsable_answer_falls_back_too() -> None:
    fallback = _Fallback()
    sent: list[dict] = []
    rr = _reranker("не знаю", sent)
    rr._fallback_factory = lambda: fallback
    assert rr.score(QUERY, [INTRO]) == [0.42]


def test_no_fallback_means_a_clear_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    rr = LlmListwiseReranker(base_url=BASE, transport=httpx.MockTransport(handler))
    with pytest.raises(RuntimeError, match="LM Studio"):
        rr.score(QUERY, [INTRO])


def test_rejected_reasoning_parameter_is_retried_without_it() -> None:
    sent: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json=_models())
        body = json.loads(request.content)
        sent.append(body)
        if "reasoning_effort" in body:
            return httpx.Response(400, json={"error": "does not support reasoning configuration"})
        return httpx.Response(200, json={"choices": [{"message": {"content": "1:2"}}]})

    rr = LlmListwiseReranker(base_url=BASE, transport=httpx.MockTransport(handler))
    assert rr.score(QUERY, [GLOSSARY]) == [pytest.approx(GRADE_SCORES[2])]
    assert ["reasoning_effort" in s for s in sent] == [True, False]


def test_progress_reports_completion() -> None:
    sent: list[dict] = []
    seen: list[tuple[int, int]] = []
    _reranker("1:3", sent).score(QUERY, [GLOSSARY, INTRO], on_progress=lambda d, t: seen.append((d, t)))
    assert seen == [(2, 2)]
