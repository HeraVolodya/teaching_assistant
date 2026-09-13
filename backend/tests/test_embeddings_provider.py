"""Ембедер: контракт провайдера й поведінка заглушки.

Усі тести працюють у stub-режимі — без жодного мегабайта ваг. Це не
компроміс, а вимога: заглушка успадковує ту саму `BaseEmbeddingProvider`, тому
шаблон префікса, нормалізація й облік обрізань перевіряються тим самим кодом,
який працює в постачанні.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.embeddings import registry
from app.embeddings.provider import create_provider, l2_normalize, stub_enabled
from app.embeddings.registry import UK_RETRIEVAL_TASK
from app.embeddings.stub_backend import StubEmbeddingProvider

QUESTION = "Яка максимальна дальність стрільби гаубиці Д-30?"
RELEVANT = (
    "122-мм гаубиця Д-30 має максимальну дальність стрільби осколково-фугасним снарядом "
    "15 300 м; активно-реактивним — до 21 900 м."
)
UNRELATED = (
    "Порядок ведення журналу обліку особового складу навчального взводу під час "
    "проведення планових занять з фізичної підготовки."
)


@pytest.fixture()
def provider() -> StubEmbeddingProvider:
    return create_provider(stub=True)  # type: ignore[return-value]


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ------------------------------------------------------------------- фабрика
def test_stub_flag_is_read_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASISTENT_STUB", "1")
    assert stub_enabled()
    assert isinstance(create_provider(), StubEmbeddingProvider)
    monkeypatch.setenv("ASISTENT_STUB", "0")
    assert not stub_enabled()


def test_explicit_stub_flag_overrides_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ASISTENT_STUB", raising=False)
    assert isinstance(create_provider(stub=True), StubEmbeddingProvider)


def test_unknown_model_raises_with_ukrainian_message() -> None:
    with pytest.raises(KeyError, match="Невідома embedding-модель"):
        create_provider("не-існує", stub=True)


# ------------------------------------------------------------------ контракт
def test_shapes_and_dtype_match_registry(provider: StubEmbeddingProvider) -> None:
    vectors = provider.embed_documents([RELEVANT, UNRELATED])
    assert vectors.shape == (2, registry.get().dim)
    assert vectors.dtype == np.float32


def test_empty_input_returns_empty_matrix_not_error(provider: StubEmbeddingProvider) -> None:
    assert provider.embed_queries([]).shape == (0, provider.dim)


def test_every_vector_is_unit_length(provider: StubEmbeddingProvider) -> None:
    vectors = provider.embed_documents([RELEVANT, UNRELATED, "", "   ", "Д-30"])
    norms = np.linalg.norm(vectors, axis=1)
    assert np.all(np.abs(norms - 1.0) < 1e-3), norms


def test_empty_text_is_not_a_zero_vector(provider: StubEmbeddingProvider) -> None:
    """Нульовий вектор отруює HNSW-граф при перебудові індексу."""
    vector = provider.embed_documents([""])[0]
    assert np.isfinite(vector).all()
    assert np.linalg.norm(vector) > 0.5


def test_l2_normalize_survives_a_zero_row() -> None:
    out = l2_normalize(np.zeros((1, 4), dtype=np.float32))
    assert np.isfinite(out).all()


# -------------------------------------------------------------- детермінізм
def test_same_text_gives_identical_vectors_across_instances() -> None:
    a = create_provider(stub=True).embed_documents([RELEVANT])
    b = create_provider(stub=True).embed_documents([RELEVANT])
    assert np.array_equal(a, b)


def test_vectors_do_not_depend_on_batch_composition(provider: StubEmbeddingProvider) -> None:
    alone = provider.embed_documents([RELEVANT])[0]
    together = provider.embed_documents([UNRELATED, RELEVANT, ""])[1]
    assert np.allclose(alone, together, atol=1e-6)


# ----------------------------------------------------- осмисленість схожості
def test_similar_texts_are_closer_than_unrelated_ones(provider: StubEmbeddingProvider) -> None:
    """Без цього тести пошуку перевіряли б лотерею, а не пошук."""
    query = provider.embed_queries([QUESTION])[0]
    relevant, unrelated = provider.embed_documents([RELEVANT, UNRELATED])
    assert _cos(query, relevant) > _cos(query, unrelated)


def test_morphological_variants_stay_close(provider: StubEmbeddingProvider) -> None:
    """Українська словозміна не має розривати схожість — заглушка тримає 3-грами."""
    a, b, c = provider.embed_documents(["гаубиця Д-30", "гаубиці Д-30", "радіостанція Р-159"])
    assert _cos(a, b) > _cos(a, c)


# ------------------------------------------------------------------ префікс
def test_query_template_is_applied_byte_for_byte(provider: StubEmbeddingProvider) -> None:
    """У картці Qwen `Query:` БЕЗ пробілу. «Виправлення» пробілу — тихий регрес."""
    formatted = provider.format_text(QUESTION, side="query")
    assert formatted == f"Instruct: {UK_RETRIEVAL_TASK}\nQuery:{QUESTION}"
    assert "Query: " not in formatted


def test_document_side_has_no_prefix_for_qwen(provider: StubEmbeddingProvider) -> None:
    assert provider.format_text(RELEVANT, side="document") == RELEVANT


def test_e5_applies_prefixes_on_both_sides() -> None:
    e5 = create_provider("multilingual-e5-large", stub=True)
    assert e5.format_text("питання", side="query") == "query: питання"
    assert e5.format_text("уривок", side="document") == "passage: уривок"


def test_granite_has_no_templates_at_all() -> None:
    granite = create_provider("granite-97m", stub=True)
    assert granite.format_text("текст", side="query") == "текст"
    assert granite.dim == 384


def test_missing_prefix_degrades_the_query_vector(provider: StubEmbeddingProvider) -> None:
    """Заглушка мусить відтворювати «вектор поза розподілом моделі».

    Інакше перевірка проводки префікса (selftest, §4.3) у stub-режимі
    проходила б вхолосту.
    """
    doc = provider.embed_documents([RELEVANT])[0]
    with_prefix = provider.encode([QUESTION], side="query")[0]
    without_prefix = provider.encode([QUESTION], side="query", apply_template=False)[0]
    assert _cos(with_prefix, doc) > _cos(without_prefix, doc)


# ------------------------------------------------------------ облік і бюджет
def test_input_ceiling_respects_max_seq_minus_eight(provider: StubEmbeddingProvider) -> None:
    assert provider.max_input_tokens <= registry.get().max_seq - 8


def test_small_context_model_lowers_the_ceiling() -> None:
    e5 = create_provider("multilingual-e5-large", stub=True)
    assert e5.max_input_tokens <= registry.get("multilingual-e5-large").max_seq - 8


def test_truncation_is_counted_not_silent() -> None:
    e5 = create_provider("multilingual-e5-large", stub=True)
    e5.stats.reset()
    e5.embed_documents(["Дуже довгий уривок. " * 2000])
    assert e5.stats.truncated == 1
    assert e5.stats.max_tokens_seen <= e5.max_input_tokens


def test_stats_accumulate_across_calls(provider: StubEmbeddingProvider) -> None:
    provider.stats.reset()
    provider.embed_documents([RELEVANT, UNRELATED])
    provider.embed_queries([QUESTION])
    assert provider.stats.texts == 3
    assert provider.stats.tokens > 0


# --------------------------------------------------------------- model_key
def test_model_key_is_stable_and_model_specific() -> None:
    default = create_provider(stub=True)
    assert default.model_key == create_provider("default", stub=True).model_key
    assert default.model_key != create_provider("granite-97m", stub=True).model_key


def test_model_key_changes_when_the_prefix_contract_changes() -> None:
    """Ключ версіювання мусить ловити зміну шаблону — інакше старі вектори
    тихо співіснували б із новими запитами."""
    import dataclasses

    model = registry.get()
    twisted = dataclasses.replace(model, query_template="Query: {query}")
    assert registry.model_key(model) != registry.model_key(twisted)
