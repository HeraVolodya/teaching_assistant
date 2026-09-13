"""Утримання від відповіді: `u_lin`, фолбек і калібрування по колекціях.

Центральний тест файлу — `test_constant_threshold_fails_across_languages`:
він відтворює саме ту ситуацію, заради якої механізм і збудовано (релевантна
англійська пара 0.9953, релевантна китайська 0.2093 на тому самому реранкері),
і показує, що форма розподілу скорів переживає зсув шкали, а константний
поріг — ні.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.domain import AssistantConfig
from app.rerank.abstention import (
    MIN_CALIBRATION_QUERIES,
    N_FEATURES,
    AbstentionCalibration,
    CalibrationSample,
    CalibrationStore,
    calibrate_threshold,
    decide,
    features_from_scores,
    fit_ridge,
    fit_u_lin,
    ndcg_at_k,
)


# --------------------------------------------------------------------- ознаки
def test_features_are_sorted_ascending_and_take_the_best_candidates() -> None:
    x = features_from_scores([0.1, 0.9, 0.5, 0.7], n_features=3)
    assert list(x) == [0.5, 0.7, 0.9]


def test_features_pad_in_front_with_the_worst_observed_score() -> None:
    """Нулі зсунули б розподіл ознак: «кандидатів мало» ≠ «є кандидати зі скором 0»."""
    x = features_from_scores([0.4, 0.6], n_features=5)
    assert list(x) == [0.4, 0.4, 0.4, 0.4, 0.6]


def test_features_of_an_empty_candidate_list_are_zeros() -> None:
    assert list(features_from_scores([], n_features=4)) == [0.0, 0.0, 0.0, 0.0]


def test_default_feature_width_matches_ndcg_at_10() -> None:
    assert N_FEATURES == 10


# ----------------------------------------------------------------------- nDCG
def test_ndcg_is_one_for_a_perfect_ranking() -> None:
    assert ndcg_at_k([1, 1, 0, 0], k=4) == pytest.approx(1.0)


def test_ndcg_punishes_a_reversed_ranking() -> None:
    assert ndcg_at_k([0, 0, 1, 1], k=4) < ndcg_at_k([1, 1, 0, 0], k=4)


def test_ndcg_of_nothing_relevant_is_zero() -> None:
    assert ndcg_at_k([0, 0, 0]) == 0.0


# -------------------------------------------------------------------- регресія
def test_ridge_recovers_a_linear_relationship() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 4))
    true_w = np.array([0.3, -0.2, 0.5, 0.1])
    y = x @ true_w + 0.7
    w, b = fit_ridge(x, y, lam=0.1)
    assert np.allclose(w, true_w, atol=0.02)
    assert b == pytest.approx(0.7, abs=0.02)


def test_ridge_intercept_is_not_shrunk_towards_zero() -> None:
    """Штрафований інтерсепт систематично занижував би передбачений nDCG,
    тобто асистент мовчав би частіше, ніж треба."""
    x = np.zeros((50, 3))
    y = np.full(50, 0.8)
    _, b = fit_ridge(x, y, lam=100.0)
    assert b == pytest.approx(0.8)


def test_ridge_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="Невідповідні форми"):
        fit_ridge(np.zeros((3, 2)), np.zeros(5))


def test_threshold_maximises_youden_j_not_accuracy() -> None:
    preds = [0.1, 0.15, 0.2, 0.8, 0.85, 0.9]
    targets = [0.0, 0.0, 0.1, 0.9, 0.95, 1.0]
    tau = calibrate_threshold(preds, targets, min_useful_ndcg=0.3)
    assert 0.2 < tau <= 0.8


def test_threshold_falls_back_when_one_class_is_missing() -> None:
    assert calibrate_threshold([0.5, 0.6], [0.9, 0.95], min_useful_ndcg=0.3) == 0.3


# --------------------------------------------------------------- навчання u_lin
def _samples(n: int, *, seed: int = 3) -> list[CalibrationSample]:
    """Синтетичний золотий набір: половина запитів покрита корпусом, половина ні.

    Покриті запити мають розрив між топ-кандидатом і рештою; непокриті — рівне
    посереднє плато. Саме цю ФОРМУ й вивчає u_lin.
    """
    rng = np.random.default_rng(seed)
    out: list[CalibrationSample] = []
    for i in range(n):
        covered = i % 2 == 0
        base = rng.uniform(0.15, 0.30)
        if covered:
            scores = [base + rng.uniform(0, 0.05) for _ in range(9)] + [base + 0.45]
            relevances = [1.0, 1.0] + [0.0] * 8
        else:
            scores = [base + rng.uniform(0, 0.04) for _ in range(10)]
            relevances = [0.0] * 10
        rng.shuffle(scores)
        out.append(CalibrationSample(scores=scores, relevances=relevances))
    return out


def test_u_lin_learns_the_shape_of_the_score_distribution() -> None:
    calibration = fit_u_lin(_samples(60), collection_id="c1", reranker_key="k1")
    assert calibration.ready
    assert calibration.n_samples == 60
    assert calibration.train_rmse < 0.2

    covered = [0.2] * 9 + [0.68]
    uncovered = [0.21] * 10
    assert calibration.predict(covered) > calibration.predict(uncovered)


def test_u_lin_predictions_stay_inside_the_ndcg_range() -> None:
    calibration = fit_u_lin(_samples(60), collection_id="c1", reranker_key="k1")
    assert 0.0 <= calibration.predict([9.0] * 10) <= 1.0
    assert 0.0 <= calibration.predict([-9.0] * 10) <= 1.0


def test_u_lin_accepts_plain_tuples_from_the_gold_set() -> None:
    pairs = [(s.scores, s.relevances) for s in _samples(40)]
    assert fit_u_lin(pairs, collection_id="c1", reranker_key="k1").ready


def test_empty_calibration_set_is_refused() -> None:
    with pytest.raises(ValueError, match="Порожній калібрувальний набір"):
        fit_u_lin([], collection_id="c1", reranker_key="k1")


def test_calibration_below_the_payback_point_is_not_ready() -> None:
    """~38 розмічених запитів — точка окупності; нижче регресія вчить шум."""
    small = fit_u_lin(_samples(MIN_CALIBRATION_QUERIES - 1), collection_id="c1", reranker_key="k1")
    assert not small.ready
    big = fit_u_lin(_samples(MIN_CALIBRATION_QUERIES), collection_id="c1", reranker_key="k1")
    assert big.ready


# --------------------------------------------------------------------- рішення
def test_no_candidates_always_abstains() -> None:
    decision = decide([], AssistantConfig())
    assert decision.abstain and decision.mode == "empty"


def test_fallback_uses_the_assistant_config_threshold() -> None:
    high = AssistantConfig(confidence_required="high")     # поріг 0.55, ≥2 фрагменти
    low = AssistantConfig(confidence_required="low")       # поріг 0.15, ≥1 фрагмент
    scores = [0.60, 0.20, 0.18]
    assert decide(scores, high).abstain                    # опорний лише один
    assert not decide(scores, low).abstain
    assert decide(scores, high).mode == "threshold"


def test_fallback_explains_itself_in_ukrainian() -> None:
    decision = decide([0.05], AssistantConfig())
    assert decision.abstain
    assert "Калібрування утримання ще немає" in decision.reason
    assert str(MIN_CALIBRATION_QUERIES) in decision.reason


def test_calibrated_decision_uses_u_lin_and_reports_it() -> None:
    calibration = fit_u_lin(_samples(60), collection_id="c1", reranker_key="k1")
    covered = decide([0.2] * 9 + [0.68], AssistantConfig(), calibration)
    uncovered = decide([0.21] * 10, AssistantConfig(), calibration)
    assert covered.mode == "u_lin" and not covered.abstain
    assert uncovered.mode == "u_lin" and uncovered.abstain
    assert "nDCG@10" in uncovered.reason


def test_unready_calibration_falls_back_instead_of_guessing() -> None:
    calibration = fit_u_lin(_samples(10), collection_id="c1", reranker_key="k1")
    assert decide([0.9, 0.8], AssistantConfig(), calibration).mode == "threshold"


def test_constant_threshold_fails_across_languages_where_u_lin_does_not() -> None:
    """Ядро механізму (план, §7).

    Той самий реранкер: релевантна англійська пара 0.9953, релевантна
    українська 0.2093. Константний поріг, налаштований на англійській,
    утримується від усього українського; u_lin дивиться на форму розподілу,
    яку крос-мовний зсув шкали не змінює.
    """
    calibration = fit_u_lin(_samples(60), collection_id="uk", reranker_key="k1")
    english_like = [0.30] * 9 + [0.9953]
    ukrainian_like = [0.02] * 9 + [0.2093]

    config = AssistantConfig(confidence_required="medium")   # поріг 0.35
    assert not decide(english_like, config).abstain
    assert decide(ukrainian_like, config).abstain            # хибне утримання

    # Форма однакова (розрив топ-кандидата), тому u_lin відповідає в обох.
    assert not decide(english_like, config, calibration).abstain
    assert not decide(ukrainian_like, config, calibration).abstain


def test_decision_is_serialisable_for_retrieval_debug() -> None:
    payload = decide([0.4, 0.2], AssistantConfig()).as_dict()
    assert json.dumps(payload, ensure_ascii=False)
    assert payload["details"]["top_score"] == 0.4


# --------------------------------------------------------------------- сховище
def test_calibration_round_trips_through_the_store(tmp_path: Path) -> None:
    store = CalibrationStore(tmp_path)
    original = fit_u_lin(_samples(60), collection_id="col-1", reranker_key="k1")
    store.save(original)
    loaded = store.load("col-1", reranker_key="k1")
    assert loaded is not None
    assert loaded.weights == pytest.approx(original.weights)
    assert loaded.tau == pytest.approx(original.tau)
    assert loaded.ready


def test_calibrations_are_isolated_per_collection(tmp_path: Path) -> None:
    """Багатоасистентність дає калібрування по колекціях безкоштовно."""
    store = CalibrationStore(tmp_path)
    store.save(fit_u_lin(_samples(60, seed=1), collection_id="col-1", reranker_key="k1"))
    store.save(fit_u_lin(_samples(60, seed=2), collection_id="col-2", reranker_key="k1"))
    assert store.load("col-1") is not None
    assert store.load("col-3") is None
    assert store.load("col-1").weights != store.load("col-2").weights


def test_calibration_from_a_different_reranker_is_rejected(tmp_path: Path) -> None:
    """Скори різних реранкерів живуть у різних шкалах — чуже калібрування
    гірше за відсутнє."""
    store = CalibrationStore(tmp_path)
    store.save(fit_u_lin(_samples(60), collection_id="col-1", reranker_key="qwen"))
    assert store.load("col-1", reranker_key="gte") is None
    assert store.load("col-1", reranker_key="qwen") is not None


def test_corrupt_calibration_file_degrades_to_the_fallback(tmp_path: Path) -> None:
    store = CalibrationStore(tmp_path)
    store.path_for("col-1").write_text("{не json", encoding="utf-8")
    assert store.load("col-1") is None


def test_collection_id_cannot_escape_the_directory(tmp_path: Path) -> None:
    store = CalibrationStore(tmp_path)
    assert store.path_for("../../etc/passwd").parent == tmp_path


def test_delete_removes_the_calibration(tmp_path: Path) -> None:
    store = CalibrationStore(tmp_path)
    store.save(fit_u_lin(_samples(60), collection_id="col-1", reranker_key="k1"))
    assert store.delete("col-1")
    assert not store.delete("col-1")


def test_store_directory_comes_from_the_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASISTENT_CALIBRATION_DIR", str(tmp_path / "калібрування"))
    assert CalibrationStore().directory == tmp_path / "калібрування"


def test_calibration_json_is_human_readable() -> None:
    calibration = fit_u_lin(_samples(40), collection_id="col-1", reranker_key="k1")
    data = json.loads(calibration.to_json())
    assert data["collection_id"] == "col-1"
    assert data["n_features"] == N_FEATURES
    assert AbstentionCalibration.from_json(calibration.to_json()).tau == pytest.approx(calibration.tau)
