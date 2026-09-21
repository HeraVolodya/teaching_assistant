"""Реранкер: контракт, реєстр, заглушка й лікування «12-38 секунд».

Ваг у CI немає, тому ONNX-шлях перевіряється двома способами: чистими функціями
(побайтовий шаблон пари Qwen3, бакетування пар за довжиною — саме воно й було
відсутнім множником у попередніх вимірах автора) і наскрізним прогоном
РЕАЛЬНОГО коду скорингу на підроблених токенайзері та сесії.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.rerank import reranker as rr

QUERY = "Яка максимальна дальність стрільби гаубиці Д-30?"
RELEVANT = (
    "122-мм гаубиця Д-30: максимальна дальність стрільби осколково-фугасним снарядом "
    "15 300 м, активно-реактивним — 21 900 м."
)
PARTIAL = "Гаубиця Д-30 буксирується автомобілем; маса в бойовому положенні 3 200 кг."
UNRELATED = "Порядок ведення журналу обліку особового складу навчального взводу."


@pytest.fixture()
def stub() -> rr.Reranker:
    return rr.create_reranker(stub=True)


# -------------------------------------------------------------------- реєстр
def test_default_is_qwen3_reranker_the_only_one_with_ukrainian_evidence() -> None:
    model = rr.get()
    assert model.id == "onnx-community/Qwen3-Reranker-0.6B-ONNX"
    assert model.scoring is rr.RerankScoring.YES_NO_LOGITS
    assert model.licence == "Apache-2.0"


def test_aliases_resolve() -> None:
    assert rr.get("default") is rr.get()
    assert rr.get("gte").id == "Alibaba-NLP/gte-multilingual-reranker-base"
    assert rr.get("gte").onnx_file.endswith("model_O3.onnx")


def test_every_registered_model_is_permissively_licensed() -> None:
    """Державна НДР із розповсюджуваним бінарником: лише Apache-2.0 / MIT."""
    assert {m.licence for m in rr.REGISTRY.values()} <= {"Apache-2.0", "MIT"}


def test_ms_marco_family_is_absent_on_purpose() -> None:
    """Підтверджений колапс на українській — ранги 11-20 у вимірах автора."""
    assert not [m for m in rr.REGISTRY.values() if "ms-marco" in m.id.lower()]


def test_unknown_model_raises_in_ukrainian() -> None:
    with pytest.raises(KeyError, match="Невідома модель реранкера"):
        rr.get("не-існує")


def test_model_env_overrides_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASISTENT_RERANK_MODEL", "gte")
    assert rr.get().id == "Alibaba-NLP/gte-multilingual-reranker-base"


def test_model_key_separates_models_and_survives_restarts() -> None:
    assert rr.model_key(rr.get()) == rr.model_key(rr.get("default"))
    assert rr.model_key(rr.get()) != rr.model_key(rr.get("gte"))


# ------------------------------------------------------------------- фабрика
def test_factory_follows_the_stub_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASISTENT_STUB", "1")
    assert isinstance(rr.create_reranker(), rr.StubReranker)
    monkeypatch.setenv("ASISTENT_STUB", "0")
    monkeypatch.setenv("ASISTENT_RERANK_MODEL_DIR", "/не/існує")
    with pytest.raises(FileNotFoundError, match=r"tokenizer\.json"):
        rr.create_reranker()


def test_missing_weights_message_points_at_the_stub_mode(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError) as info:
        rr.OnnxReranker(rr.get(), model_dir=tmp_path)
    assert "ASISTENT_STUB=1" in str(info.value)
    assert "не завантажує моделі з мережі" in str(info.value)


def test_stub_satisfies_the_protocol(stub: rr.Reranker) -> None:
    assert isinstance(stub, rr.Reranker)


# ------------------------------------------------------------------- заглушка
def test_scores_are_returned_in_input_order_and_bounded(stub: rr.Reranker) -> None:
    scores = stub.score(QUERY, [UNRELATED, RELEVANT, PARTIAL])
    assert len(scores) == 3
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_empty_document_list_is_not_an_error(stub: rr.Reranker) -> None:
    assert stub.score(QUERY, []) == []


def test_stub_ranks_the_relevant_passage_first(stub: rr.Reranker) -> None:
    """Заглушка мусить давати ПРАВИЛЬНИЙ порядок, інакше тести пошуку,
    утримання й генерації нижче за течією перевіряють нуль."""
    scores = stub.score(QUERY, [UNRELATED, PARTIAL, RELEVANT])
    assert scores.index(max(scores)) == 2
    assert scores[1] > scores[0]


def test_stub_is_deterministic_across_instances() -> None:
    docs = [RELEVANT, PARTIAL, UNRELATED]
    first = rr.create_reranker(stub=True).score(QUERY, docs)
    assert first == rr.create_reranker(stub=True).score(QUERY, docs)


def test_stub_breaks_ties_deterministically(stub: rr.Reranker) -> None:
    """Однакові скори зробили б порядок залежним від порядку кандидатів."""
    scores = stub.score(QUERY, ["зовсім інше", "теж інше"])
    assert scores[0] != scores[1]


def test_stub_handles_ukrainian_morphology(stub: rr.Reranker) -> None:
    scores = stub.score("дальність стрільби гаубиці", ["дальність стрільби гаубиць", UNRELATED])
    assert scores[0] > scores[1]


def test_stub_model_key_is_marked_as_a_stub(stub: rr.Reranker) -> None:
    """Калібрування утримання, зняте на заглушці, не сміє застосуватись до
    справжньої моделі — ключ мусить це показувати."""
    assert stub.model_key.startswith("stub:")


# --------------------------------------------------------- шаблон пари Qwen3
def test_qwen_pair_is_byte_exact() -> None:
    text = rr.build_qwen_pair("питання", "уривок", "інструкція")
    assert text.startswith("<|im_start|>system\nJudge whether the Document meets")
    assert "<Instruct>: інструкція\n<Query>: питання\n<Document>: уривок" in text
    # Порожній блок <think> обов'язковий: без нього логіти yes/no читаються
    # не з тієї позиції.
    assert text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_qwen_prompt_forces_a_binary_answer() -> None:
    assert 'answer can only be "yes" or "no"' in rr.QWEN_PREFIX


# --------------------------------------------------- бакетування пар за довжиною
def test_pairs_are_bucketed_by_length() -> None:
    """Паддінг КОЖНОЇ пари до max_length і був множником 10-27 разів."""
    lengths = [2000, 40, 1900, 45, 1950, 42]
    for batch in rr.plan_pair_batches(lengths, token_budget=4000, max_rows=16):
        inside = [lengths[i] for i in batch]
        assert inside == sorted(inside)
        assert len(batch) * max(inside) <= 4000


def test_pair_batching_covers_every_pair_once() -> None:
    lengths = [10, 3000, 12, 11, 3100]
    flat = [i for b in rr.plan_pair_batches(lengths, token_budget=2048, max_rows=8) for i in b]
    assert sorted(flat) == list(range(len(lengths)))


def test_pair_batch_row_cap_is_respected() -> None:
    lengths = [4] * 100
    assert all(len(b) <= 8 for b in rr.plan_pair_batches(lengths, token_budget=10**6, max_rows=8))


def test_pair_that_exceeds_the_window_is_cut_in_the_middle() -> None:
    """Хвіст пари нести не можна: у Qwen-форматі там обгортка assistant/<think>,
    без якої логіти yes/no читаються не з тієї позиції."""
    provider, _ = _onnx_reranker(max_pair_tokens=40)
    ids = provider._encode_pair("запит", "д" * 5000)
    assert len(ids) == 40
    tail = provider._tok.encode(rr.QWEN_SUFFIX, add_special_tokens=False).ids[-5:]
    assert list(ids[-5:]) == list(tail)


def test_naive_padding_to_max_length_is_measurably_worse() -> None:
    """Кількісне підтвердження діагнозу FLOP-моделі."""
    lengths = [2000, 40, 1900, 45, 1950, 42, 1980, 41]
    bucketed = sum(len(b) * max(lengths[i] for i in b)
                   for b in rr.plan_pair_batches(lengths, token_budget=4096, max_rows=8))
    padded_to_max = len(lengths) * 8192      # так міряли раніше: кожна пара до max_length
    assert bucketed * 4 < padded_to_max


# ------------------------------------------- наскрізний ONNX-шлях без ваг
# Підроблені токенайзер і сесія прогоняють РЕАЛЬНИЙ код скорингу: побудову
# пари, паддінг, вибір позиції останнього токена й softmax по yes/no.
YES_ID, NO_ID, PAD_ID = 1, 2, 0


class _FakeEncoding:
    def __init__(self, ids: list[int]) -> None:
        self.ids = ids


class _FakeTokenizer:
    """Один символ — один токен; `yes`/`no` мають зарезервовані id."""

    def encode(self, text: str, pair: str | None = None, add_special_tokens: bool = True) -> _FakeEncoding:
        if text == "yes":
            return _FakeEncoding([YES_ID])
        if text == "no":
            return _FakeEncoding([NO_ID])
        ids = [(ord(c) % 500) + 3 for c in text]
        if pair is not None:
            ids += [(ord(c) % 500) + 3 for c in pair]
        return _FakeEncoding(ids)

    def token_to_id(self, token: str) -> int | None:
        return PAD_ID if token in {"<pad>", "[PAD]"} else None

    def no_truncation(self) -> None: ...

    def no_padding(self) -> None: ...


class _Spec:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeSession:
    """Логіт `yes` пропорційний ОСТАННЬОМУ реальному токену рядка.

    Якщо код прочитає позицію паддінга замість останнього реального токена,
    скор одразу зміниться — саме це тести нижче й ловлять.
    """

    def __init__(self, scoring: rr.RerankScoring) -> None:
        self.scoring = scoring
        self.seen: list[dict] = []

    def get_inputs(self) -> list[_Spec]:
        return [_Spec("input_ids"), _Spec("attention_mask")]

    def get_outputs(self) -> list[_Spec]:
        return [_Spec("logits")]

    def get_providers(self) -> list[str]:
        return ["CPUExecutionProvider"]

    def run(self, _outputs, feeds):
        import numpy as np

        self.seen.append({k: v.copy() for k, v in feeds.items()})
        ids, mask = feeds["input_ids"], feeds["attention_mask"]
        if self.scoring is rr.RerankScoring.SEQUENCE_CLASSIFICATION:
            last = np.clip(mask.sum(axis=1) - 1, 0, ids.shape[1] - 1)
            value = ids[np.arange(ids.shape[0]), last] / 100.0
            return [value[:, None].astype("float32")]
        logits = np.zeros((ids.shape[0], ids.shape[1], 8), dtype="float32")
        logits[:, :, YES_ID] = ids / 100.0
        return [logits]


def _onnx_reranker(*, model_name: str = "default", max_pair_tokens: int = 512):
    from pathlib import Path as _Path

    model = rr.get(model_name)
    session = _FakeSession(model.scoring)
    return rr.OnnxReranker(
        model,
        model_dir=_Path("/не/використовується"),
        tokenizer=_FakeTokenizer(),
        session=session,
        max_pair_tokens=max_pair_tokens,
        token_budget=4096,
    ), session


def test_yes_no_scoring_reads_the_last_real_token() -> None:
    provider, session = _onnx_reranker()
    scores = provider.score(QUERY, [RELEVANT, "коротко"])
    assert len(scores) == 2
    assert all(0.0 <= s <= 1.0 for s in scores)
    # Паддінг маскується, тож коротка пара в батчі має той самий скор, що й окремо.
    alone, _ = _onnx_reranker()
    assert alone.score(QUERY, ["коротко"])[0] == pytest.approx(scores[1], abs=1e-6)
    assert session.seen


def test_padding_positions_are_masked_out() -> None:
    provider, session = _onnx_reranker()
    provider.score(QUERY, [RELEVANT, "а"])
    for feed in session.seen:
        assert ((feed["input_ids"] == PAD_ID) == (feed["attention_mask"] == 0)).all()


def test_scores_come_back_in_input_order() -> None:
    provider, _ = _onnx_reranker()
    docs = [RELEVANT, "а", PARTIAL, "бб"]
    batched = provider.score(QUERY, docs)
    for i, doc in enumerate(docs):
        alone, _ = _onnx_reranker()
        assert alone.score(QUERY, [doc])[0] == pytest.approx(batched[i], abs=1e-6)


def test_sequence_classification_head_uses_sigmoid() -> None:
    provider, _ = _onnx_reranker(model_name="gte")
    scores = provider.score(QUERY, [RELEVANT, UNRELATED])
    assert all(0.0 < s < 1.0 for s in scores)


def test_onnx_reranker_reports_the_model_key_for_calibration() -> None:
    provider, _ = _onnx_reranker()
    assert provider.model_key == rr.model_key(rr.get())
    assert not provider.model_key.startswith("stub:")
