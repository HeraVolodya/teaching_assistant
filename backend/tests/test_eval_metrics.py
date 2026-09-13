"""Метрики пошуку — перевірка на прикладах, які можна порахувати руками."""

from __future__ import annotations

import math

import pytest

from app.eval import metrics as M


# ------------------------------------------------------------------ nDCG
def test_ndcg_reproduces_textbook_example():
    """Підручниковий приклад: релевантні {d1, d3}, видано [d1, d2, d3].

    DCG@3  = 1/log2(2) + 0 + 1/log2(4) = 1.0 + 0.5           = 1.5
    IDCG@3 = 1/log2(2) + 1/log2(3)     = 1.0 + 0.63093       = 1.63093
    nDCG@3 = 1.5 / 1.63093                                   = 0.91972
    """
    ranked = ["d1", "d2", "d3"]
    gold = {"d1", "d3"}
    assert M.dcg_at_k(ranked, gold, 3) == pytest.approx(1.5)
    assert M.ndcg_at_k(ranked, gold, 3) == pytest.approx(0.9197207891, abs=1e-9)


def test_ndcg_idcg_uses_min_relevant_and_k():
    """Ідеальний топ-1 при трьох золотих чанках має дати nDCG@1 = 1.0.

    Якби IDCG рахувався по |relevant| = 3, ідеальний ранжувальник отримав би
    0.5 і всі nDCG@1 у звіті були б систематично занижені.
    """
    assert M.ndcg_at_k(["a"], {"a", "b", "c"}, 1) == pytest.approx(1.0)


def test_ndcg_is_one_for_perfect_ranking():
    assert M.ndcg_at_k(["a", "b", "c"], {"a", "b"}, 3) == pytest.approx(1.0)


def test_ndcg_of_single_hit_at_rank_two():
    assert M.ndcg_at_k(["x", "a"], {"a"}, 10) == pytest.approx(1 / math.log2(3))


# ------------------------------------------------------- прості метрики
def test_recall_precision_mrr_hits():
    ranked = ["x", "a", "y", "b"]
    gold = {"a", "b", "c"}
    assert M.recall_at_k(ranked, gold, 2) == pytest.approx(1 / 3)
    assert M.recall_at_k(ranked, gold, 4) == pytest.approx(2 / 3)
    assert M.precision_at_k(ranked, gold, 4) == pytest.approx(0.5)
    assert M.mrr_at_k(ranked, gold, 10) == pytest.approx(0.5)
    assert M.hits_at_k(ranked, gold, 4) == 2
    assert M.hit_rate_at_k(ranked, gold, 1) == 0.0
    assert M.hit_rate_at_k(ranked, gold, 2) == 1.0
    assert M.first_hit_rank(ranked, gold) == 2


def test_precision_divides_by_k_not_by_list_length():
    """Пошук, що повернув один правильний документ, не отримує Precision@10 = 1."""
    assert M.precision_at_k(["a"], {"a"}, 10) == pytest.approx(0.1)


def test_mrr_respects_cutoff():
    assert M.mrr_at_k(["x", "y", "a"], {"a"}, 2) == 0.0
    assert M.mrr_at_k(["x", "y", "a"], {"a"}, 3) == pytest.approx(1 / 3)


def test_duplicate_in_ranking_is_counted_once():
    """Auto-merge і дедуплікація можуть дати той самий uid двічі."""
    assert M.hits_at_k(["a", "a", "b"], {"a", "b"}, 2) == 1
    assert M.recall_at_k(["a", "a", "b"], {"a", "b"}, 3) == pytest.approx(1.0)


def test_average_precision_denominator_is_min_relevant_k():
    assert M.average_precision_at_k(["a", "b"], {"a", "b", "c", "d"}, 2) == pytest.approx(1.0)


def test_metrics_are_nan_without_gold():
    """Питання без золотих чанків не має визначеного Recall — воно про відмову."""
    assert math.isnan(M.recall_at_k(["a"], set(), 5))
    assert math.isnan(M.ndcg_at_k(["a"], set(), 5))
    assert math.isnan(M.mrr_at_k(["a"], set(), 5))
    assert M.first_hit_rank(["a"], set()) is None


# --------------------------------------------------------------- агрегація
def test_evaluate_run_skips_unanswerable_and_counts_missing_as_zero():
    run = {"q1": ["a", "b"], "q2": ["z"]}
    qrels = {"q1": {"a"}, "q2": {"b"}, "q3": set(), "q4": {"c"}}
    result = M.evaluate_run(run, qrels, name="проба", ks=(1, 5))

    assert result.skipped == ("q3",)
    assert result.n_queries == 3           # q1, q2, q4 — q3 не оцінюється
    # q1 влучив, q2 промазав, q4 узагалі відсутній у прогоні → нуль, не пропуск.
    assert result.value("recall@5") == pytest.approx(1 / 3)
    assert set(result.per_query) == {"q1", "q2", "q4"}
    assert result.per_query["q4"]["recall@5"] == 0.0


def test_evaluate_run_latency_and_qps():
    run = {"q1": ["a"], "q2": ["b"]}
    qrels = {"q1": {"a"}, "q2": {"b"}}
    result = M.evaluate_run(
        run, qrels, latencies_ms={"q1": 100.0, "q2": 300.0}, wall_seconds=0.5
    )
    assert result.latency is not None
    assert result.latency.p50_ms == pytest.approx(200.0)
    assert result.latency.qps == pytest.approx(4.0)      # 2 запити / 0.5 с


def test_latency_percentiles_and_serial_qps():
    summary = M.latency_summary([10.0, 20.0, 30.0, 40.0])
    assert summary.p50_ms == pytest.approx(25.0)
    assert summary.max_ms == 40.0
    # Без wall_seconds QPS — послідовна оцінка: 4 запити на 0.1 с сумарно.
    assert summary.qps == pytest.approx(40.0)


def test_latency_summary_of_empty_sample_is_nan_not_crash():
    summary = M.latency_summary([])
    assert summary.count == 0
    assert math.isnan(summary.p95_ms)


# ------------------------------------------------------------------ приріст
def test_relative_gain_reproduces_the_published_plus_14_percent():
    """UNLP 2026: Recall@1 0.6957 → 0.7935, тобто +14.1% відносних.

    Саме цю величину стаття автора зобов'язалась відтворити, тож арифметика
    приросту має бути перевірена окремо від самих метрик.
    """
    base = M.MetricSet("bm25", (1,), {"recall@1": 0.6957}, {}, 0)
    cand = M.MetricSet("rerank", (1,), {"recall@1": 0.7935}, {}, 0)
    assert M.relative_gain(base, cand)["recall@1"] == pytest.approx(14.058, abs=0.01)


def test_relative_gain_survives_zero_baseline():
    base = M.MetricSet("b", (1,), {"recall@1": 0.0}, {}, 0)
    cand = M.MetricSet("c", (1,), {"recall@1": 0.4}, {}, 0)
    assert math.isinf(M.relative_gain(base, cand)["recall@1"])


# ------------------------------------------------------ статистичні тести
def test_paired_t_test_matches_known_critical_value():
    """t = 2.262 при df = 9 — це рівно p ≈ 0.05 у таблиці Стьюдента.

    Неповна бета-функція реалізована власноруч (scipy у постачанні немає), тож
    її треба звірити з відомим числом, а не з собою.
    """
    n = 10
    # t = mean / (sd / sqrt(n)); беремо sd = 1, отже mean = 2.262 / sqrt(10).
    mean = 2.262 / math.sqrt(n)
    candidate = [mean + z for z in _centered_unit_variance(n)]
    baseline = [0.0] * n

    result = M.paired_t_test(candidate, baseline)
    assert result.statistic == pytest.approx(2.262, abs=0.001)
    assert result.p_value == pytest.approx(0.05, abs=0.002)
    assert result.significant(0.06)
    assert not result.significant(0.04)


def _centered_unit_variance(n: int) -> list[float]:
    """n значень із середнім 0 і вибірковим ст. відхиленням 1."""
    raw = [float(i) for i in range(n)]
    mean = sum(raw) / n
    centered = [v - mean for v in raw]
    sd = math.sqrt(sum(v * v for v in centered) / (n - 1))
    return [v / sd for v in centered]


def test_paired_t_test_sees_no_difference():
    values = [0.3, 0.5, 0.7, 0.2]
    result = M.paired_t_test(values, values)
    assert result.p_value == 1.0
    assert not result.significant()


def test_randomization_test_is_deterministic_and_detects_shift():
    better = [0.9] * 20
    worse = [0.4] * 20
    first = M.fisher_randomization_test(better, worse, trials=2000)
    second = M.fisher_randomization_test(better, worse, trials=2000)
    assert first.p_value == second.p_value      # зерно фіксоване — цифра у звіті відтворювана
    assert first.significant()


def test_randomization_test_ignores_pairs_with_nan():
    """Питання без золотих чанків дає nan — пара мусить випасти, а не отруїти тест."""
    result = M.fisher_randomization_test([0.5, math.nan, 0.9], [0.1, 0.2, 0.3], trials=500)
    assert result.n == 2


def test_compare_metric_reports_both_tests():
    base = M.evaluate_run({"q1": ["x"], "q2": ["y"]}, {"q1": {"a"}, "q2": {"b"}}, name="bm25")
    cand = M.evaluate_run({"q1": ["a"], "q2": ["b"]}, {"q1": {"a"}, "q2": {"b"}}, name="hybrid")
    report = M.compare_metric(base, cand, "recall@10", trials=500)
    assert report["baseline_value"] == 0.0
    assert report["candidate_value"] == 1.0
    assert report["t_test"]["n"] == 2
    assert report["randomization"]["n"] == 2


# ------------------------------------------------------------------ експорт
def test_markdown_and_latex_tables():
    rows = [{"конфігурація": "bm25", "Recall@10": 0.5}, {"конфігурація": "гібрид", "Recall@10": 0.75}]
    markdown = M.markdown_table(rows)
    assert markdown.splitlines()[0].startswith("| конфігурація |")
    assert "0.7500" in markdown

    latex = M.latex_table(rows, caption="Таблиця", label="tab:x")
    assert r"\begin{tabular}{lr}" in latex
    assert r"\toprule" in latex and r"\bottomrule" in latex


def test_latex_escapes_underscores():
    latex = M.latex_table([{"metric": "first_hit_rank", "v": 1.0}])
    assert r"first\_hit\_rank" in latex


def test_empty_tables_are_empty_strings():
    assert M.markdown_table([]) == ""
    assert M.latex_table([]) == ""


# -------------------------------------------------------------------- ranx
def test_ranx_is_optional_and_fails_loudly():
    """`ranx` не є обов'язковою; її відсутність має бути гучною, а не тихою."""
    if M.ranx_available():
        scores = M.evaluate_with_ranx({"q": ["a", "b"]}, {"q": {"a"}}, ("recall@2",))
        assert scores["recall@2"] == pytest.approx(1.0)
    else:
        with pytest.raises(RuntimeError, match="ranx"):
            M.evaluate_with_ranx({"q": ["a"]}, {"q": {"a"}})
