"""Прогін конфігурацій пошуку по золотому набору.

Тести працюють на СПРАВЖНЬОМУ конвеєрі пошуку (`HybridRetriever`) над
мініатюрним корпусом із `helpers_retrieval`, у stub-режимі ембедингів і
реранкера — тобто перевіряють машину, яка поїде у звіт, а не мок.
"""

from __future__ import annotations

import pytest

from app.domain import AssistantConfig, ChunkLevel, RetrievalDebug, RetrievedChunk
from app.eval import gold_set as G
from app.eval import metrics as M
from app.eval import runner as R
from app.rerank.reranker import create_reranker
from app.retrieval import fusion
from app.retrieval.hybrid import HybridRetriever
from helpers_retrieval import build_corpus


@pytest.fixture
def corpus(tmp_path):
    return build_corpus(tmp_path)


@pytest.fixture
def retriever(corpus):
    return HybridRetriever(
        corpus.db, provider=corpus.provider, index_dir=corpus.index_dir, reranker=None
    )


def _uid(corpus, title: str, index: int = 0) -> str:
    chunk_id = corpus.leaves(title)[index]
    with corpus.db.connection() as con:
        return con.execute(
            "SELECT chunk_uid FROM chunks WHERE id=?", (chunk_id,)
        ).fetchone()[0]


@pytest.fixture
def questions(corpus):
    return [
        G.GoldQuestion(
            collection_id=corpus.collection_id,
            question="Що таке деривація снаряда?",
            gold_chunk_uids=[_uid(corpus, "Основи балістики", 0)],
            id="q-derivation",
        ),
        G.GoldQuestion(
            collection_id=corpus.collection_id,
            question="Яка максимальна дальність стрільби гаубиці Д-30?",
            gold_chunk_uids=[_uid(corpus, "Методична розробка: гаубиця Д-30", 0)],
            id="q-d30",
        ),
        G.GoldQuestion(
            collection_id=corpus.collection_id,
            question="Як облаштувати польову кухню на 200 осіб?",
            id="q-absent",     # відповіді в корпусі свідомо немає
        ),
    ]


# ------------------------------------------------------- ранжований список
def test_ranked_uids_prefer_reranker_order_and_keep_rrf_for_ties():
    debug = RetrievalDebug(query="q")
    debug.candidates = [
        {"kind": "run", "weights": {}},
        {"kind": "candidate", "chunk_uid": "a", "rrf": 0.9, "rerank": 0.2},
        {"kind": "candidate", "chunk_uid": "b", "rrf": 0.5, "rerank": 0.8},
        {"kind": "candidate", "chunk_uid": "c", "rrf": 0.4, "rerank": 0.8},
        {"kind": "candidate", "rrf": 0.3, "selected": False},   # дубль без uid
    ]
    assert R.ranked_uids_from_debug(debug) == ["b", "c", "a"]


def test_ranked_uids_keep_rrf_order_without_reranker():
    debug = RetrievalDebug(query="q")
    debug.candidates = [
        {"kind": "candidate", "chunk_uid": "a", "rrf": 0.9},
        {"kind": "candidate", "chunk_uid": "b", "rrf": 0.5},
    ]
    assert R.ranked_uids_from_debug(debug) == ["a", "b"]


def test_ranked_uids_respect_depth():
    debug = RetrievalDebug(query="q")
    debug.candidates = [
        {"kind": "candidate", "chunk_uid": f"u{i}", "rrf": 1.0 - i / 100} for i in range(20)
    ]
    assert len(R.ranked_uids_from_debug(debug, depth=5)) == 5


# ---------------------------------------------------- автопідйом ваги sparse
def test_anchor_boost_is_restored_even_after_an_exception():
    original = fusion.ANCHORED_SPARSE_WEIGHT
    with pytest.raises(RuntimeError):
        with R.anchor_boost_disabled():
            assert fusion.ANCHORED_SPARSE_WEIGHT == 0.0
            raise RuntimeError("щось пішло не так")
    assert fusion.ANCHORED_SPARSE_WEIGHT == original


def test_dense_only_arm_really_excludes_sparse_on_an_anchored_query(retriever, corpus):
    """Головна пастка абляції.

    `fusion.resolve_weights` підіймає w_sparse до 1.0 на запиті з позначенням
    («Д-30»), тож `weight_sparse=0.0` у конфігу НЕ дає dense-only плеча —
    саме на артилерійських питаннях, заради яких абляція й робиться.
    """
    query = "Яка дальність стрільби Д-30?"
    config = R._depth_config(
        AssistantConfig(weight_dense=1.0, weight_sparse=0.0, weight_char_ngram=0.0), 20
    )

    _, leaked = retriever.retrieve(query, config, corpus.collection_id)
    sparse_ranks_leaked = [
        row.get("sparse_rank") for row in leaked.candidates if row.get("kind") == "candidate"
    ]
    assert any(rank is not None for rank in sparse_ranks_leaked), (
        "Автопідйом мав повернути sparse-гілку — інакше тест нічого не доводить."
    )

    with R.anchor_boost_disabled():
        _, clean = retriever.retrieve(query, config, corpus.collection_id)
    sparse_ranks_clean = [
        row.get("sparse_rank") for row in clean.candidates if row.get("kind") == "candidate"
    ]
    assert all(rank is None for rank in sparse_ranks_clean)


def test_standard_arms_mark_the_one_sided_arm(corpus):
    arms = {a.name: a for a in R.standard_arms()}
    assert arms["dense"].disable_anchor_boost is True
    assert arms["bm25"].disable_anchor_boost is False   # sparse і так 1.0
    assert arms["hybrid"].disable_anchor_boost is False  # поведінка постачання
    assert arms["hybrid+rerank"].use_reranker is True
    assert arms["bm25"].config.weight_dense == 0.0
    # Глибина кандидатів мусить покривати максимальний k метрик.
    assert arms["hybrid"].config.rerank_top_k >= max(M.DEFAULT_KS)


# ----------------------------------------------------------- прогін плеча
def test_run_arm_finds_the_gold_chunk_and_records_latency(retriever, corpus, questions):
    arm = R.standard_arms()[2]     # hybrid
    result = R.run_arm(retriever, arm, questions, corpus.collection_id)

    assert result.metrics.n_queries == 2               # q-absent пропущено
    assert result.metrics.skipped == ("q-absent",)
    assert result.metrics.value("recall@10") == pytest.approx(1.0)
    assert result.metrics.latency is not None
    assert result.metrics.latency.count == 3           # затримка міряється й для відмов
    assert not result.errors


def test_run_arm_restores_the_injected_reranker(retriever, corpus, questions):
    stub = create_reranker(stub=True)
    retriever.reranker = stub
    R.run_arm(retriever, R.standard_arms()[0], questions, corpus.collection_id)
    assert retriever.reranker is stub


def test_reranker_arm_actually_scores(retriever, corpus, questions):
    arm = R.standard_arms()[3]
    result = R.run_arm(
        retriever, arm, questions, corpus.collection_id, reranker=create_reranker(stub=True)
    )
    assert result.metrics.value("recall@10") == pytest.approx(1.0)
    assert result.arm.use_reranker


def test_run_arm_survives_a_broken_retriever(corpus, questions):
    class _Broken:
        db = corpus.db
        reranker = None

        def retrieve(self, *a, **k):
            raise RuntimeError("індекс пошкоджено")

    result = R.run_arm(_Broken(), R.standard_arms()[0], questions, corpus.collection_id)
    assert len(result.errors) == 3
    assert result.metrics.value("recall@10") == 0.0
    assert all(result.abstained.values())


def test_unanswerable_question_triggers_abstention_on_the_shipping_arm(
    retriever, corpus, questions
):
    result = R.run_arm(
        retriever, R.standard_arms()[3], questions, corpus.collection_id,
        reranker=create_reranker(stub=True),
    )
    assert result.abstained["q-absent"] is True


# --------------------------------------------------------------- auto-merge
def test_final_uids_expand_a_merged_section_back_into_leaves(corpus):
    """Без розгортання кожне спрацювання auto-merge читалося б як промах."""
    from app.db.repositories import ChunkRepo

    parent_id = corpus.parent_ids["Основи балістики"]
    with corpus.db.connection() as con:
        parent = ChunkRepo(con).get(parent_id)
        assert parent is not None and parent.level is ChunkLevel.SECTION
        expanded = R._expand_final_uids(
            con, [RetrievedChunk(chunk=parent, document_title="Основи балістики")]
        )
        leaf_uids = [
            r["chunk_uid"] for r in con.execute(
                "SELECT chunk_uid FROM chunks WHERE parent_id=? AND level='L2' ORDER BY ordinal",
                (parent_id,),
            )
        ]
    assert expanded[0] == parent.chunk_uid
    assert set(leaf_uids) <= set(expanded)


def test_final_uids_do_not_duplicate(corpus):
    from app.db.repositories import ChunkRepo

    with corpus.db.connection() as con:
        leaf = ChunkRepo(con).get(corpus.leaves("Основи балістики")[0])
        assert leaf is not None
        items = [RetrievedChunk(chunk=leaf, document_title="t")] * 2
        expanded = R._expand_final_uids(con, items)
    assert expanded == [leaf.chunk_uid]


# ------------------------------------------------------- сховище прогонів
def test_saving_a_run_persists_questions_because_of_the_foreign_key(
    retriever, corpus, questions
):
    """`eval_results.question_id` має FK на `eval_questions`.

    Прогін по набору, що існує лише в пам'яті, інакше падає на
    `FOREIGN KEY constraint failed` — і саме тому harness зобов'язаний
    зберігати питання разом із результатами.
    """
    result = R.run_arm(retriever, R.standard_arms()[2], questions, corpus.collection_id)
    with corpus.db.transaction() as con:
        run_id = R.EvalRunRepo(con).save(corpus.collection_id, result, questions)

    with corpus.db.connection() as con:
        assert G.EvalQuestionRepo(con).count(corpus.collection_id) == 3
        stored = R.EvalRunRepo(con).results(run_id)
        assert set(stored) == {"q-derivation", "q-d30", "q-absent"}
        assert stored["q-derivation"]["ranked_uids"]
        assert "final_uids" in stored["q-derivation"]["scores"]
        assert stored["q-derivation"]["latency_ms"] >= 0

        meta = R.EvalRunRepo(con).get(run_id)
        assert meta is not None and meta["finished_at"]


def test_metrics_are_recomputable_from_storage(retriever, corpus, questions):
    result = R.run_arm(retriever, R.standard_arms()[2], questions, corpus.collection_id)
    with corpus.db.transaction() as con:
        run_id = R.EvalRunRepo(con).save(corpus.collection_id, result, questions)

    qrels = {q.id: set(q.gold_chunk_uids) for q in questions}
    with corpus.db.connection() as con:
        restored = R.metrics_from_stored(con, run_id, qrels, ks=(1, 3))
    assert restored.value("recall@3") == pytest.approx(result.metrics.value("recall@3"))
    # Метрику, якої в прогоні не рахували, можна дорахувати заднім числом.
    assert "ndcg@3" in restored.values


def test_compare_runs_produces_gain_and_significance(retriever, corpus, questions):
    arms = R.standard_arms()
    results = R.run_suite(
        retriever, [arms[0], arms[3]], questions, corpus.collection_id,
        reranker=create_reranker(stub=True), db=corpus.db,
    )
    qrels = {q.id: set(q.gold_chunk_uids) for q in questions}
    with corpus.db.connection() as con:
        report = R.compare_runs(
            con, results[0].run_id, results[1].run_id, qrels, metric_names=("recall@10",)
        )
    assert report["baseline"]["n_queries"] == 2
    assert "recall@10" in report["relative_gain_pct"]
    test = report["tests"][0]
    assert test["metric"] == "recall@10"
    assert 0.0 <= test["randomization"]["p_value"] <= 1.0


# ------------------------------------------------------------------ сюїта
def test_run_suite_builds_the_article_table(retriever, corpus, questions):
    results = R.run_suite(
        retriever, R.standard_arms(), questions, corpus.collection_id,
        reranker=create_reranker(stub=True),
    )
    assert [r.arm.name for r in results] == ["bm25", "dense", "hybrid", "hybrid+rerank"]

    report = R.suite_report(results)
    assert report["baseline"] == "bm25"
    assert "| конфігурація |" in report["markdown"]
    assert r"\begin{table}" in report["latex"]
    assert set(report["relative_gain_pct"]) == {"dense", "hybrid", "hybrid+rerank"}
    assert all("randomization" in t for t in report["significance"])


def test_suite_rows_carry_distinct_documents(retriever, corpus, questions):
    results = R.run_suite(retriever, R.standard_arms()[2:3], questions, corpus.collection_id)
    row = R.suite_rows(results)[0]
    assert row["конфігурація"] == "hybrid"
    assert "різних документів" in row
    assert "p50, мс" in row


def test_weight_grid_disables_the_boost_everywhere():
    arms = R.weight_grid_arms(sparse=(0.0, 0.6), ngram=(0.4,))
    assert [a.name for a in arms] == ["w_sparse=0/w_ngram=0.4", "w_sparse=0.6/w_ngram=0.4"]
    assert all(a.disable_anchor_boost for a in arms)


def test_final_top_k_ablation_changes_only_that_parameter():
    arms = R.final_top_k_arms(values=(2, 5, 12))
    assert [a.config.final_top_k for a in arms] == [2, 5, 12]
    assert len({a.config.max_per_document for a in arms}) == 1


def test_final_top_k_ablation_is_measurable_on_the_final_list(retriever, corpus, questions):
    """Глибокий ранжований список від `final_top_k` не залежить — саме тому
    цей параметр міряється по фінальному списку й по кількості документів."""
    arms = R.final_top_k_arms(values=(2, 12), use_reranker=False)
    results = R.run_suite(retriever, arms, questions, corpus.collection_id)
    two, twelve = results
    assert two.metrics.value("recall@10") == pytest.approx(twelve.metrics.value("recall@10"))
    assert two.mean_distinct_documents <= twelve.mean_distinct_documents


# ------------------------------------------------------------- вхід контракту
def test_run_retrieval_eval_is_the_contract_entry_point(retriever, corpus, questions):
    result = R.run_retrieval_eval(
        retriever, questions, corpus.collection_id,
        config=AssistantConfig(), reranker=create_reranker(stub=True), db=corpus.db,
    )
    assert isinstance(result.metrics, M.MetricSet)
    assert result.run_id
    assert result.arm.use_reranker is True
    with corpus.db.connection() as con:
        assert R.EvalRunRepo(con).get(result.run_id) is not None


def test_eval_module_does_not_import_torch_or_docling():
    """Контракт, правило 1: API-процес не платить за важкі модулі."""
    import subprocess
    import sys
    from pathlib import Path

    code = (
        "import app.eval, app.eval.runner, app.eval.groundedness, sys;"
        "print('torch' in sys.modules, 'docling' in sys.modules, 'rapidocr' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, check=True,
        cwd=Path(__file__).resolve().parents[1],
    ).stdout
    assert out.strip() == "False False False"
