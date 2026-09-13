"""ONNX-бекенд ембедера: чисті частини, які можна перевірити без 1.5 ГБ ваг.

Ваги в CI немає, тому тут перевіряються рівно ті місця, де живуть задокументовані
пастки: батчування за бюджетом токенів, вибір execution provider, pooling
(зокрема last-token, який без EOS мовчки бере не той токен) і зрозумілість
помилки про відсутні файли моделі.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.embeddings.onnx_backend import (
    ModelFilesMissing,
    OnnxEmbeddingProvider,
    TokenizerContractError,
    plan_batches,
    select_providers,
)
from app.embeddings.registry import Pooling, get


# ------------------------------------------------------------------ батчування
def test_batches_respect_the_token_budget_not_the_row_count() -> None:
    """Батч із 64 рядків по 1500 токенів і по 20 токенів різняться в 75 разів."""
    lengths = [1500] * 8 + [20] * 8
    batches = plan_batches(lengths, token_budget=3000, max_rows=64)
    for batch in batches:
        widest = max(lengths[i] for i in batch)
        assert len(batch) * widest <= 3000 or len(batch) == 1


def test_batches_are_sorted_by_length_to_cut_padding() -> None:
    lengths = [1000, 10, 900, 12, 950, 11]
    batches = plan_batches(lengths, token_budget=4000, max_rows=8)
    for batch in batches:
        inside = [lengths[i] for i in batch]
        assert inside == sorted(inside)
        # Вартість батча з паддінгом лишається в межах бюджету — саме це й
        # обмежує пік пам'яті сесії.
        assert len(batch) * max(inside) <= 4000


def test_sorting_reduces_total_padding_against_naive_order() -> None:
    lengths = [1000, 10, 900, 12, 950, 11, 980, 9]
    planned = sum(
        len(b) * max(lengths[i] for i in b)
        for b in plan_batches(lengths, token_budget=4000, max_rows=2)
    )
    naive = sum(
        len(chunk) * max(chunk)
        for chunk in (lengths[i:i + 2] for i in range(0, len(lengths), 2))
    )
    assert planned < naive


def test_every_index_appears_exactly_once() -> None:
    lengths = [7, 3, 9, 1, 4, 4, 12]
    flat = [i for b in plan_batches(lengths, token_budget=16, max_rows=4) for i in b]
    assert sorted(flat) == list(range(len(lengths)))


def test_row_longer_than_budget_gets_its_own_batch() -> None:
    batches = plan_batches([5, 100000, 5], token_budget=64, max_rows=8)
    assert [100000] in [[100000 if i == 1 else 5 for i in b] for b in batches]
    assert all(len(b) == 1 for b in batches if 1 in b)


def test_empty_input_plans_nothing() -> None:
    assert plan_batches([], token_budget=100) == []


# --------------------------------------------------------- execution provider
def test_cuda_wins_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ASISTENT_EMBED_DEVICE", raising=False)
    monkeypatch.delenv("ASISTENT_EMBED_PROVIDERS", raising=False)
    chosen = select_providers(["CUDAExecutionProvider", "CPUExecutionProvider"])
    assert chosen[0] == "CUDAExecutionProvider"
    assert chosen[-1] == "CPUExecutionProvider"


def test_cpu_is_always_the_last_resort(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ASISTENT_EMBED_DEVICE", raising=False)
    monkeypatch.delenv("ASISTENT_EMBED_PROVIDERS", raising=False)
    assert select_providers(["CPUExecutionProvider"]) == ["CPUExecutionProvider"]


def test_device_env_can_force_cpu_because_lm_studio_owns_the_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """План §0: індексація на зайнятій карті дає CUDA OOM у LM Studio."""
    monkeypatch.setenv("ASISTENT_EMBED_DEVICE", "cpu")
    monkeypatch.delenv("ASISTENT_EMBED_PROVIDERS", raising=False)
    assert select_providers(["CUDAExecutionProvider", "CPUExecutionProvider"]) == ["CPUExecutionProvider"]


def test_explicit_provider_list_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASISTENT_EMBED_PROVIDERS", "CUDAExecutionProvider")
    monkeypatch.setenv("ASISTENT_EMBED_DEVICE", "cpu")
    assert select_providers(["CUDAExecutionProvider", "CPUExecutionProvider"]) == [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]


# --------------------------------------------------------------------- pooling
def _hidden() -> tuple[np.ndarray, np.ndarray]:
    hidden = np.array(
        [[[1.0, 0.0], [0.0, 1.0], [5.0, 5.0]],      # третій токен — паддінг
         [[2.0, 0.0], [0.0, 2.0], [0.0, 4.0]]],
        dtype=np.float32,
    )
    mask = np.array([[1, 1, 0], [1, 1, 1]], dtype=np.int64)
    return hidden, mask


def test_cls_pooling_takes_the_first_token() -> None:
    hidden, mask = _hidden()
    out = OnnxEmbeddingProvider._pool(hidden, mask, Pooling.CLS)
    assert np.allclose(out, [[1.0, 0.0], [2.0, 0.0]])


def test_mean_pooling_ignores_padding() -> None:
    hidden, mask = _hidden()
    out = OnnxEmbeddingProvider._pool(hidden, mask, Pooling.MEAN)
    assert np.allclose(out[0], [0.5, 0.5])          # паддінговий токен не врахований
    assert np.allclose(out[1], [2 / 3, 2.0])


def test_last_token_pooling_takes_the_last_real_token_not_the_padding() -> None:
    """Саме тому EOS обов'язковий: без нього тут читається останній токен тексту."""
    hidden, mask = _hidden()
    out = OnnxEmbeddingProvider._pool(hidden, mask, Pooling.LAST_TOKEN)
    assert np.allclose(out[0], [0.0, 1.0])
    assert np.allclose(out[1], [0.0, 4.0])


# ----------------------------------------------------------- відсутні артефакти
def test_missing_model_directory_gives_an_actionable_ukrainian_error(tmp_path: Path) -> None:
    with pytest.raises(ModelFilesMissing) as info:
        OnnxEmbeddingProvider(get(), model_dir=tmp_path / "немає")
    message = str(info.value)
    assert "tokenizer.json" in message
    assert "закритому контурі" in message
    assert "ASISTENT_STUB=1" in message


def test_registry_default_still_demands_eos() -> None:
    """Регресійний захист: якщо хтось прибере requires_eos, last-token pooling
    почне мовчки брати не той токен."""
    model = get()
    assert model.pooling is Pooling.LAST_TOKEN
    assert model.requires_eos


# ------------------------------------------------- наскрізний шлях без ваг
# Підроблені токенайзер і сесія дозволяють прогнати РЕАЛЬНИЙ код кодування —
# EOS, обрізання, батчування, pooling, MRL-зріз — без жодного мегабайта ваг.
EOS_ID = 999
PAD_ID = 0


class _FakeEncoding:
    def __init__(self, ids: list[int]) -> None:
        self.ids = ids


class _FakeTokenizer:
    """Один символ — один токен. EOS токенайзер НЕ додає сам (як у Qwen)."""

    def __init__(self, *, appends_eos: bool = False) -> None:
        self.appends_eos = appends_eos

    def encode(self, text: str, pair: str | None = None, add_special_tokens: bool = True) -> _FakeEncoding:
        ids = [(ord(c) % 900) + 1 for c in text]
        if self.appends_eos and add_special_tokens:
            ids.append(EOS_ID)
        return _FakeEncoding(ids)

    def token_to_id(self, token: str) -> int | None:
        return {"<|endoftext|>": EOS_ID, "<pad>": PAD_ID}.get(token)

    def no_truncation(self) -> None: ...

    def no_padding(self) -> None: ...


class _Spec:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeSession:
    """Прихований стан токена = префіксна сума id-шок, розтягнута на 16 вимірів.

    Сенс саме такий: значення ОСТАННЬОГО токена залежить від усієї послідовності,
    тож pooling не того токена одразу дає інший вектор — що тест і ловить.
    """

    hidden_size = 16

    def __init__(self) -> None:
        self.seen: list[dict[str, np.ndarray]] = []

    def get_inputs(self) -> list[_Spec]:
        return [_Spec("input_ids"), _Spec("attention_mask")]

    def get_outputs(self) -> list[_Spec]:
        return [_Spec("last_hidden_state")]

    def get_providers(self) -> list[str]:
        return ["CPUExecutionProvider"]

    def run(self, _outputs, feeds: dict[str, np.ndarray]):  # noqa: ANN001
        self.seen.append({k: v.copy() for k, v in feeds.items()})
        ids = feeds["input_ids"].astype(np.float64)
        prefix = np.cumsum(ids * feeds["attention_mask"], axis=1)
        scale = np.arange(1, self.hidden_size + 1, dtype=np.float64)
        return [(np.sin(prefix[:, :, None] / scale)).astype(np.float32)]


def _provider(*, appends_eos: bool = False, max_input_tokens: int = 32, token_budget: int = 64):
    import dataclasses

    model = dataclasses.replace(get(), dim=8, max_seq=max_input_tokens + 8)
    session = _FakeSession()
    return OnnxEmbeddingProvider(
        model,
        model_dir=Path("/не/використовується"),
        tokenizer=_FakeTokenizer(appends_eos=appends_eos),
        session=session,
        max_input_tokens=max_input_tokens,
        token_budget=token_budget,
    ), session


def test_eos_is_appended_when_the_tokenizer_does_not_do_it() -> None:
    """Головна тиха помилка: без EOS last-token pooling бере не той токен."""
    provider, session = _provider(appends_eos=False)
    provider.embed_documents(["абв"])
    ids = session.seen[0]["input_ids"][0]
    assert ids[-1] == EOS_ID
    assert not provider.tokenizer_appends_eos


def test_eos_is_not_duplicated_when_the_tokenizer_already_adds_it() -> None:
    provider, session = _provider(appends_eos=True)
    provider.embed_documents(["абв"])
    ids = list(session.seen[0]["input_ids"][0])
    assert ids[-1] == EOS_ID
    assert ids.count(EOS_ID) == 1
    assert provider.tokenizer_appends_eos


def test_truncation_keeps_the_eos_and_is_counted() -> None:
    provider, session = _provider(max_input_tokens=16)
    provider.stats.reset()
    provider.embed_documents(["я" * 200])
    ids = session.seen[0]["input_ids"][0]
    assert len(ids) == 16
    assert ids[-1] == EOS_ID          # EOS переживає обрізання
    assert provider.stats.truncated == 1


def test_outputs_keep_input_order_despite_length_sorting() -> None:
    """Батчі йдуть за довжиною, а матриця мусить лишитись у порядку входу."""
    provider, _ = _provider()
    texts = ["коротко", "трохи довший текст", "а", "найдовший з усіх текстів тут"]
    together = provider.embed_documents(texts)
    for i, text in enumerate(texts):
        assert np.allclose(together[i], provider.embed_documents([text])[0], atol=1e-6)


def test_batches_are_actually_split_by_the_token_budget() -> None:
    provider, session = _provider(token_budget=32)
    provider.embed_documents(["я" * 20, "б" * 20, "в" * 20])
    assert len(session.seen) > 1


def test_vectors_are_unit_length_and_mrl_truncated() -> None:
    provider, _ = _provider()
    vectors = provider.embed_documents(["перший текст", "другий текст"])
    assert vectors.shape == (2, 8)     # 16 прихованих вимірів зрізано до dim=8
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_padding_is_masked_out() -> None:
    provider, session = _provider()
    provider.embed_documents(["а", "довший рядок для паддінгу"])
    for feed in session.seen:
        ids, mask = feed["input_ids"], feed["attention_mask"]
        assert ((ids == PAD_ID) == (mask == 0)).all()


def test_count_tokens_includes_the_eos() -> None:
    provider, _ = _provider()
    assert provider.count_tokens("абв") == 4


def test_missing_eos_token_is_a_fatal_construction_error() -> None:
    """`requires_eos` без EOS у токенайзері — не попередження, а відмова."""
    import dataclasses

    class _NoEos(_FakeTokenizer):
        def token_to_id(self, token: str) -> int | None:
            return PAD_ID if token == "<pad>" else None

    with pytest.raises(TokenizerContractError, match="last-token pooling"):
        OnnxEmbeddingProvider(
            dataclasses.replace(get(), dim=8, max_seq=64),
            model_dir=Path("/не/використовується"),
            tokenizer=_NoEos(),
            session=_FakeSession(),
        )
