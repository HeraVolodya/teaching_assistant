"""Harness оцінювання — те, що НДР має опублікувати.

Публічний інтерфейс модуля (див. docs/CONTRACT.md):
    run_retrieval_eval(...) -> ArmResult    (метрики + сирі ранжовані списки)
    плюс скрипти бейк-офів у `backend/scripts/`.

Чотири незалежні частини:
    metrics.py       — Recall@k / Precision@k / MRR@k / nDCG@k / Hits@k,
                       перцентилі затримки, парні тести значущості, LaTeX
    gold_set.py      — золотий набір: імпорт, експорт, валідація,
                       півавтоматична побудова локальною моделлю
    runner.py        — BM25 vs dense vs гібрид vs гібрид+реранкінг,
                       абляції ваг фьюжну і final_top_k, збереження прогонів
    groundedness.py  — суддя заземленості, частка коректних відмов,
                       частка дійсних цитат
    ocr_metrics.py   — CER, WER і українські плутанини (Віха 1)
    table_metrics.py — поклітинний F1 таблиць (Віха 2)

Модуль НЕ імпортує torch, docling чи rapidocr: він має однаково працювати і в
API-процесі, і в CI без жодної завантаженої моделі. Важкі рушії живуть у
скриптах бейк-офу й підключаються опційним імпортом.
"""

from __future__ import annotations

from app.eval.gold_set import (
    EvalQuestionRepo,
    GoldPage,
    GoldQuestion,
    QuestionDraft,
    ValidationReport,
    approve_drafts,
    draft_questions,
    load_csv,
    load_json,
    sample_chunks,
    save_csv,
    save_json,
    validate,
)
from app.eval.groundedness import (
    AbstentionReport,
    CitationValidity,
    Evidence,
    GroundednessReport,
    abstention_stats,
    citation_validity,
    create_judge,
    judge_answer,
    pairwise_compare,
    split_claims,
    summarize_groundedness,
)
from app.eval.metrics import (
    DEFAULT_KS,
    LatencySummary,
    MetricSet,
    StatTest,
    compare_metric,
    evaluate_run,
    fisher_randomization_test,
    latency_summary,
    latex_table,
    markdown_table,
    mrr_at_k,
    ndcg_at_k,
    paired_t_test,
    precision_at_k,
    ranx_available,
    recall_at_k,
    relative_gain,
)
from app.eval.runner import (
    Arm,
    ArmResult,
    EvalRunRepo,
    compare_runs,
    final_top_k_arms,
    metrics_from_stored,
    run_arm,
    run_retrieval_eval,
    run_suite,
    standard_arms,
    suite_report,
    suite_rows,
    weight_grid_arms,
)

__all__ = [
    # метрики
    "DEFAULT_KS", "MetricSet", "LatencySummary", "StatTest",
    "recall_at_k", "precision_at_k", "mrr_at_k", "ndcg_at_k",
    "evaluate_run", "latency_summary", "relative_gain", "compare_metric",
    "paired_t_test", "fisher_randomization_test", "markdown_table", "latex_table",
    "ranx_available",
    # золотий набір
    "GoldQuestion", "GoldPage", "QuestionDraft", "ValidationReport", "EvalQuestionRepo",
    "load_csv", "load_json", "save_csv", "save_json", "validate",
    "sample_chunks", "draft_questions", "approve_drafts",
    # прогони
    "Arm", "ArmResult", "EvalRunRepo", "run_retrieval_eval", "run_arm", "run_suite",
    "standard_arms", "weight_grid_arms", "final_top_k_arms",
    "suite_rows", "suite_report", "compare_runs", "metrics_from_stored",
    # заземленість
    "Evidence", "GroundednessReport", "AbstentionReport", "CitationValidity",
    "create_judge", "judge_answer", "split_claims", "abstention_stats",
    "citation_validity", "pairwise_compare", "summarize_groundedness",
]
