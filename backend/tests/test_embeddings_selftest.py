"""Самотест ембедера — тест на запобіжник (план, §4).

Особливість: тут перевіряється не «чи працює модель», а «чи впаде самотест,
коли модель зламається». Тому половина тестів навмисно ЛАМАЄ провайдера й
вимагає, щоб звіт це помітив.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.embeddings.provider import create_provider
from app.embeddings.selftest import (
    NORM_TOLERANCE,
    REFERENCE_COS_MIN,
    SelfTestReport,
    run_selftest,
)


@pytest.fixture()
def reference(tmp_path: Path) -> Path:
    return tmp_path / "selftest_reference.json"


def _names(report: SelfTestReport) -> list[str]:
    return [c.name for c in report.checks]


# ------------------------------------------------------------------- склад
def test_report_contains_all_four_assertions(reference: Path) -> None:
    report = run_selftest(create_provider(stub=True), reference_path=reference)
    assert _names(report) == ["reference_vector", "unit_norm", "prefix_wiring", "token_budget"]
    assert report.error is None
    assert report.ok
    assert report.dim == 1024
    assert report.duration_ms >= 0.0


def test_report_is_not_a_bare_bool(reference: Path) -> None:
    """Кожна перевірка несе значення, поріг і українське пояснення."""
    report = run_selftest(create_provider(stub=True), reference_path=reference)
    check = next(c for c in report.checks if c.name == "unit_norm")
    assert check.threshold == NORM_TOLERANCE
    assert check.value is not None
    assert "норм" in check.detail.lower()
    assert "Самотест ембедера" in report.format_uk()
    assert json.loads(report.to_json())["model_id"]


# ------------------------------------------- 1. еталон: створення й порівняння
def test_first_run_bootstraps_the_reference(reference: Path) -> None:
    report = run_selftest(create_provider(stub=True), reference_path=reference)
    check = next(c for c in report.checks if c.name == "reference_vector")
    assert check.skipped and check.ok
    assert report.bootstrapped
    assert reference.is_file()


def test_second_run_actually_compares_against_the_reference(reference: Path) -> None:
    run_selftest(create_provider(stub=True), reference_path=reference)
    report = run_selftest(create_provider(stub=True), reference_path=reference)
    check = next(c for c in report.checks if c.name == "reference_vector")
    assert not check.skipped
    assert check.ok and check.value is not None and check.value >= REFERENCE_COS_MIN


def test_drifted_reference_fails_loudly(reference: Path) -> None:
    """Заміна ваг / шаблону / нормалізації мусить валити самотест."""
    run_selftest(create_provider(stub=True), reference_path=reference)
    data = json.loads(reference.read_text(encoding="utf-8"))
    key = next(iter(data))
    rng = np.random.default_rng(7)
    data[key]["query_vector"] = list(rng.normal(size=len(data[key]["query_vector"])))
    reference.write_text(json.dumps(data), encoding="utf-8")

    report = run_selftest(create_provider(stub=True), reference_path=reference)
    assert not report.ok
    failure = next(c for c in report.failures if c.name == "reference_vector")
    assert "еталон" in failure.detail.lower()


def test_reference_of_wrong_dimension_is_a_failure(reference: Path) -> None:
    run_selftest(create_provider(stub=True), reference_path=reference)
    data = json.loads(reference.read_text(encoding="utf-8"))
    key = next(iter(data))
    data[key]["query_vector"] = [0.1, 0.2, 0.3]
    reference.write_text(json.dumps(data), encoding="utf-8")

    report = run_selftest(create_provider(stub=True), reference_path=reference)
    assert not report.ok
    assert "розмірність" in next(c for c in report.failures).detail


def test_bootstrap_can_be_disabled(reference: Path) -> None:
    report = run_selftest(create_provider(stub=True), reference_path=reference, allow_bootstrap=False)
    check = next(c for c in report.checks if c.name == "reference_vector")
    assert check.skipped and check.ok
    assert not reference.exists()


def test_reference_is_keyed_by_model_so_models_coexist(reference: Path) -> None:
    run_selftest(create_provider(stub=True), reference_path=reference)
    run_selftest(create_provider("granite-97m", stub=True), reference_path=reference)
    assert len(json.loads(reference.read_text(encoding="utf-8"))) == 2


# ------------------------------------------------------------- 2. норма
def test_lost_normalisation_is_detected(reference: Path) -> None:
    provider = create_provider(stub=True)

    def unnormalised(texts, *, side):  # noqa: ANN001, ANN202 — локальна диверсія
        return np.full((len(texts), provider.dim), 3.0, dtype=np.float32)

    provider._encode_formatted = unnormalised          # type: ignore[assignment]
    provider._model = __import__("dataclasses").replace(provider.model, normalize=False)
    report = run_selftest(provider, reference_path=reference)
    assert not report.ok
    assert any(c.name == "unit_norm" for c in report.failures)


# ------------------------------------------------------- 3. проводка префікса
def test_prefix_wiring_is_checked_for_models_that_have_a_template(reference: Path) -> None:
    report = run_selftest(create_provider(stub=True), reference_path=reference)
    check = next(c for c in report.checks if c.name == "prefix_wiring")
    assert not check.skipped
    assert check.ok and check.value is not None and check.value > 0


def test_prefix_wiring_is_skipped_when_there_is_no_template(reference: Path) -> None:
    report = run_selftest(create_provider("granite-97m", stub=True), reference_path=reference)
    check = next(c for c in report.checks if c.name == "prefix_wiring")
    assert check.skipped and check.ok


def test_swapped_prefix_wiring_is_detected(reference: Path) -> None:
    """Найтихіша помилка системи: запит форматується як документ."""
    provider = create_provider(stub=True)
    provider.format_text = lambda text, *, side, apply_template=True: text  # type: ignore[assignment]
    report = run_selftest(provider, reference_path=reference)
    assert not report.ok
    failure = next(c for c in report.failures if c.name == "prefix_wiring")
    assert "перевернуто" in failure.detail


# ---------------------------------------------------------- 4. бюджет токенів
def test_token_budget_check_reports_the_ceiling(reference: Path) -> None:
    provider = create_provider(stub=True)
    report = run_selftest(provider, reference_path=reference)
    check = next(c for c in report.checks if c.name == "token_budget")
    assert check.ok
    assert check.value is not None and check.threshold is not None
    assert check.value <= check.threshold


def test_ceiling_above_max_seq_minus_eight_fails(reference: Path) -> None:
    provider = create_provider(stub=True)
    provider._max_input_tokens = provider.model.max_seq   # type: ignore[attr-defined]
    report = run_selftest(provider, reference_path=reference)
    assert not report.ok
    assert "max_seq" in next(c for c in report.failures if c.name == "token_budget").detail


def test_truncation_of_a_hard_cap_chunk_is_visible(reference: Path) -> None:
    """Модель із вікном 512 мусить залишити слід у звіті, а не мовчати."""
    report = run_selftest(create_provider("multilingual-e5-large", stub=True), reference_path=reference)
    check = next(c for c in report.checks if c.name == "token_budget")
    assert check.ok
    assert "Обрізано" in check.detail
    assert report.truncated_inputs >= 1


# ----------------------------------------------------------------- стійкість
def test_broken_provider_yields_a_report_not_a_traceback(reference: Path) -> None:
    provider = create_provider(stub=True)

    def boom(texts, *, side):  # noqa: ANN001, ANN202
        raise RuntimeError("сесія ONNX впала")

    provider._encode_formatted = boom                   # type: ignore[assignment]
    report = run_selftest(provider, reference_path=reference)
    assert not report.ok
    assert report.error is not None and "сесія ONNX впала" in report.error
