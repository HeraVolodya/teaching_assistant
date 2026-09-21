"""Прогін конфігурацій пошуку по золотому набору й порівняння прогонів.

Це та машина, що виробляє головну таблицю статті:

    BM25-only  vs  dense-only  vs  гібрид  vs  гібрид + реранкінг

плюс дві абляції — ваги фьюжну і `final_top_k` (параметр №1 для налаштування,
план, §7 і «розсуджена суперечність 5»).

ТРИ РІШЕННЯ, ЯКІ ВИЗНАЧАЮТЬ ЧЕСНІСТЬ ЦИФР
-----------------------------------------

1. **Метрики рахуються по ГЛИБОКОМУ ранжованому списку, а не по фінальній
   п'ятірці.** `HybridRetriever.retrieve()` повертає фрагменти вже
   переставленими в порядок читання й U-подібно (`reorder`), тобто в порядку,
   свідомо НЕ ранжованому — подавати його в MRR@10 означало б міряти
   впорядкування контексту замість якості пошуку. Ранжований список береться з
   `RetrievalDebug.candidates`: це ті самі кандидати конвеєра, у порядку
   реранкера (а без нього — RRF).

2. **Auto-merge розгортається назад у листки.** У фінальному списку два сусідні
   листки замінюються L1-батьком, чийого `chunk_uid` у золотому наборі немає
   ніколи (розмічають листки — лише вони індексуються). Без розгортання кожне
   спрацювання auto-merge читалося б як промах, і найкращий режим конвеєра
   виглядав би найгіршим.

3. **«Dense-only» неможливо задати самим лише конфігом.** `fusion.resolve_weights`
   ПІДІЙМАЄ вагу sparse до 1.0, щойно в запиті є лексичний якір (цифра,
   «Д-30», ДСТУ) — і `max(0.0, 1.0)` повертає sparse-гілку в гру саме на тих
   артилерійських питаннях, заради яких абляція робиться. Тому для однобічних
   плечей автопідйом вимикається явно (`anchor_boost_disabled`), а для
   гібридних — лишається, бо це поведінка постачання.
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field, replace
from typing import Any

from app.domain import AssistantConfig, ChunkLevel, RetrievalDebug, new_id, utcnow
from app.eval import metrics as M
from app.eval.gold_set import EvalQuestionRepo, GoldQuestion

__all__ = [
    "EVAL_DEPTH",
    "Arm",
    "ArmResult",
    "EvalRunRepo",
    "anchor_boost_disabled",
    "compare_runs",
    "final_top_k_arms",
    "metrics_from_stored",
    "ranked_uids_from_debug",
    "run_arm",
    "run_retrieval_eval",
    "run_suite",
    "standard_arms",
    "suite_report",
    "suite_rows",
    "weight_grid_arms",
]

# Глибина, до якої зберігається ранжований список. 50 — це `rerank_top_k` за
# замовчуванням: далі конвеєр і сам нічого не бачить, а зберігати більше
# означало б записувати в `eval_results` те, що ніколи не потрапляло в жодну
# метрику (максимальний k у DEFAULT_KS — 10).
EVAL_DEPTH = 50


# --------------------------------------------------------------------- плечі
@dataclass(frozen=True, slots=True)
class Arm:
    """Одна конфігурація пошуку — один стовпчик у таблиці статті."""

    name: str
    config: AssistantConfig
    use_reranker: bool = False
    disable_anchor_boost: bool = False
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return {
            "name": self.name,
            "use_reranker": self.use_reranker,
            "disable_anchor_boost": self.disable_anchor_boost,
            "note": self.note,
            "config": asdict(self.config),
        }


@contextmanager
def anchor_boost_disabled() -> Iterator[None]:
    """Тимчасово прибрати автопідйом ваги sparse на лексичному якорі.

    Патчиться саме КОНСТАНТА `fusion.ANCHORED_SPARSE_WEIGHT`, а не код:
    `resolve_weights` читає її при кожному виклику, тож `max(w_sparse, 0.0)`
    просто повертає вагу з конфігу, а решта логіки лишається недоторканою.
    Потрібно рівно для однобічних плечей абляції (див. шапку модуля, п. 3);
    у постачанні автопідйом обов'язковий і ніколи не вимикається.
    """
    from app.retrieval import fusion

    original = fusion.ANCHORED_SPARSE_WEIGHT
    fusion.ANCHORED_SPARSE_WEIGHT = 0.0
    try:
        yield
    finally:
        fusion.ANCHORED_SPARSE_WEIGHT = original


def _depth_config(base: AssistantConfig, depth: int) -> AssistantConfig:
    """Конфіг із достатньою глибиною кандидатів для метрик @k."""
    return replace(
        base,
        dense_top_k=max(base.dense_top_k, depth),
        sparse_top_k=max(base.sparse_top_k, depth),
        rerank_top_k=max(base.rerank_top_k, depth),
    )


def standard_arms(
    base: AssistantConfig | None = None,
    *,
    depth: int = EVAL_DEPTH,
) -> list[Arm]:
    """Чотири плеча головної таблиці НДР.

    BM25-only — це і є baseline, проти якого стаття автора зобов'язалась
    показати приріст ~15%. Реранкінг стоїть окремим плечем, бо UNLP 2026
    виміряв саме його внесок: Recall@1 0.6957 → 0.7935 (+14.1% відносних).

    УВАГА при читанні `ArmResult.abstained`: на плечах без dense-гілки й без
    реранкера утримання спрацьовує ЗАВЖДИ, і це не дефект. Шкала впевненості
    береться зі скора реранкера, а без нього — із сирого косинуса dense
    (`HybridRetriever._confidence_scores`); у плечі `bm25` немає ні того, ні
    іншого, тож порівнювати нема з чим. Частку відмов міряти треба на плечі
    постачання (`hybrid+rerank`), а не на baseline.
    """
    cfg = _depth_config(base or AssistantConfig(), depth)
    return [
        Arm(
            name="bm25",
            config=replace(cfg, weight_dense=0.0, weight_sparse=1.0, weight_char_ngram=0.0),
            note="Базова лінія: лише FTS5 по лемах (леми 1.0 / форми 0.4 / коди 2.5).",
        ),
        Arm(
            name="dense",
            config=replace(cfg, weight_dense=1.0, weight_sparse=0.0, weight_char_ngram=0.0),
            disable_anchor_boost=True,
            note="Лише вектори. Автопідйом sparse вимкнено — інакше плече не однобічне.",
        ),
        Arm(
            name="hybrid",
            config=replace(cfg, weight_dense=1.0, weight_sparse=0.6, weight_char_ngram=0.4),
            note="Зважений RRF трьох гілок — дефолт постачання, без реранкера.",
        ),
        Arm(
            name="hybrid+rerank",
            config=replace(cfg, weight_dense=1.0, weight_sparse=0.6, weight_char_ngram=0.4),
            use_reranker=True,
            note="Дефолт постачання повністю.",
        ),
    ]


def weight_grid_arms(
    base: AssistantConfig | None = None,
    *,
    sparse: Sequence[float] = (0.0, 0.4, 0.6, 0.8, 1.0),
    ngram: Sequence[float] = (0.0, 0.4),
    depth: int = EVAL_DEPTH,
    use_reranker: bool = False,
) -> list[Arm]:
    """Абляція ваг фьюжну при фіксованій вазі dense = 1.0.

    Саме цю сітку план велить прогнати per-corpus: доказова база чесно
    суперечлива (переможець UNLP не побачив приросту від RRF на перефразованих
    MCQ, команда УКУ виміряла +21% від символьного TF-IDF на українській
    морфології), і вирішувати має вимір на НАШОМУ корпусі, а не цитата.
    """
    cfg = _depth_config(base or AssistantConfig(), depth)
    arms: list[Arm] = []
    for ws in sparse:
        for wn in ngram:
            arms.append(
                Arm(
                    name=f"w_sparse={ws:g}/w_ngram={wn:g}",
                    config=replace(cfg, weight_dense=1.0, weight_sparse=ws, weight_char_ngram=wn),
                    use_reranker=use_reranker,
                    disable_anchor_boost=True,
                    note="Абляція ваг: автопідйом вимкнено, інакше w_sparse нижче 1.0 "
                         "не спостерігається на запитах із позначеннями.",
                )
            )
    return arms


def final_top_k_arms(
    base: AssistantConfig | None = None,
    *,
    values: Sequence[int] = (2, 3, 5, 8, 12),
    depth: int = EVAL_DEPTH,
    use_reranker: bool = True,
) -> list[Arm]:
    """Абляція `final_top_k` — параметр №1 для налаштування (план, §7).

    Конфлікт, який вона розв'язує: UNLP 2026 дав 0.9674 на top-2 проти 0.9346
    на top-10, але їхнє завдання — MCQ по одній сторінці, а наша явна вимога —
    ПОЄДНУВАТИ кілька джерел. Тому дивитись треба не лише на метрику
    ранжування, а й на `distinct_documents` у фінальному списку.
    """
    cfg = _depth_config(base or AssistantConfig(), depth)
    return [
        Arm(
            name=f"final_top_k={k}",
            config=replace(cfg, final_top_k=k),
            use_reranker=use_reranker,
            note="Квота max_per_document і мінімум документів лишаються з базового конфігу.",
        )
        for k in values
    ]


# ------------------------------------------------- витягання ранжованого списку
def ranked_uids_from_debug(debug: RetrievalDebug, *, depth: int = EVAL_DEPTH) -> list[str]:
    """Ранжований список `chunk_uid` із діагностики конвеєра.

    Рядки `candidates` ідуть у порядку RRF і несуть `rerank`, коли реранкер
    працював. Сортування стабільне, тож при рівних скорах реранкера
    зберігається порядок RRF — інакше метрики стрибали б від перестановки
    нічиїх, а не від якості пошуку.
    """
    rows = [
        r for r in debug.candidates
        if r.get("kind") == "candidate" and r.get("chunk_uid")
    ]
    if any(r.get("rerank") is not None for r in rows):
        rows.sort(key=lambda r: -(r.get("rerank") if r.get("rerank") is not None else -math.inf))
    return [str(r["chunk_uid"]) for r in rows][:depth]


def _expand_final_uids(con: sqlite3.Connection, items: Sequence[Any]) -> list[str]:
    """`chunk_uid` фінального списку з розгортанням L1-батьків у листки.

    Після auto-merge у фінальному списку стоїть секція, а золоті мітки завжди
    на листках. Без розгортання найкращий режим конвеєра давав би нульовий
    Recall на фінальному списку — див. шапку модуля, п. 2.
    """
    out: list[str] = []
    for item in items:
        chunk = item.chunk
        if chunk.level is ChunkLevel.LEAF or chunk.id is None:
            out.append(chunk.chunk_uid)
            continue
        rows = con.execute(
            "SELECT chunk_uid FROM chunks WHERE parent_id=? AND level=? ORDER BY ordinal",
            (chunk.id, ChunkLevel.LEAF.value),
        ).fetchall()
        out.append(chunk.chunk_uid)
        out.extend(str(r["chunk_uid"]) for r in rows)
    # Порядок збережено, дублікати прибрано.
    seen: set[str] = set()
    return [u for u in out if not (u in seen or seen.add(u))]


# ---------------------------------------------------------------- прогін плеча
@dataclass(slots=True)
class ArmResult:
    """Усе, що дав один прогін одного плеча."""

    arm: Arm
    metrics: M.MetricSet                    # по глибокому ранжованому списку
    final_metrics: M.MetricSet              # по тому, що реально побачив генератор
    ranked: dict[str, list[str]] = field(default_factory=dict)
    finals: dict[str, list[str]] = field(default_factory=dict)
    latencies_ms: dict[str, float] = field(default_factory=dict)
    abstained: dict[str, bool] = field(default_factory=dict)
    confidence: dict[str, float] = field(default_factory=dict)
    distinct_documents: dict[str, int] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    run_id: str | None = None

    @property
    def mean_distinct_documents(self) -> float:
        values = list(self.distinct_documents.values())
        return sum(values) / len(values) if values else math.nan

    def as_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm.as_dict(),
            "run_id": self.run_id,
            "metrics": self.metrics.as_dict(),
            "final_metrics": self.final_metrics.as_dict(),
            "mean_distinct_documents": self.mean_distinct_documents,
            "abstained": sum(1 for v in self.abstained.values() if v),
            "errors": self.errors,
        }


def run_arm(
    retriever: Any,
    arm: Arm,
    questions: Sequence[GoldQuestion],
    collection_id: str,
    *,
    reranker: Any = None,
    ks: Sequence[int] = M.DEFAULT_KS,
    depth: int = EVAL_DEPTH,
    filters: Any = None,
    language: str = "uk",
) -> ArmResult:
    """Прогнати одне плече по всіх питаннях.

    Реранкер підставляється в ретривер на час прогону і повертається як був:
    `HybridRetriever` приймає його як ін'єктовану залежність, тож «плече без
    реранкера» — це буквально та сама машина з `reranker=None`, а не інший
    код. Виняток на окремому питанні не валить увесь прогін: він записується в
    `errors`, а питання отримує порожній список, тобто чесний нуль.
    """
    ranked: dict[str, list[str]] = {}
    finals: dict[str, list[str]] = {}
    latencies: dict[str, float] = {}
    abstained: dict[str, bool] = {}
    confidence: dict[str, float] = {}
    distinct: dict[str, int] = {}
    errors: dict[str, str] = {}

    previous = retriever.reranker
    retriever.reranker = reranker if arm.use_reranker else None
    boost = anchor_boost_disabled() if arm.disable_anchor_boost else nullcontext()
    started = time.perf_counter()
    try:
        with boost:
            for question in questions:
                t0 = time.perf_counter()
                try:
                    items, debug = retriever.retrieve(
                        question.question,
                        arm.config,
                        collection_id,
                        filters=filters,
                        language=language,
                    )
                except Exception as exc:
                    errors[question.id] = f"{type(exc).__name__}: {exc}"
                    ranked[question.id] = []
                    finals[question.id] = []
                    latencies[question.id] = (time.perf_counter() - t0) * 1000.0
                    abstained[question.id] = True
                    confidence[question.id] = 0.0
                    distinct[question.id] = 0
                    continue
                latencies[question.id] = (time.perf_counter() - t0) * 1000.0
                ranked[question.id] = ranked_uids_from_debug(debug, depth=depth)
                with retriever.db.connection() as con:
                    finals[question.id] = _expand_final_uids(con, items)
                abstained[question.id] = bool(debug.abstained)
                confidence[question.id] = float(debug.abstain_confidence or 0.0)
                distinct[question.id] = int(debug.distinct_documents)
    finally:
        retriever.reranker = previous
    wall = time.perf_counter() - started

    qrels = {q.id: set(q.gold_chunk_uids) for q in questions}
    deep = M.evaluate_run(
        ranked, qrels, name=arm.name, ks=ks, latencies_ms=latencies, wall_seconds=wall,
        extra={"stage": "ranked", "errors": len(errors)},
    )
    final = M.evaluate_run(
        finals, qrels, name=f"{arm.name}/final", ks=ks,
        extra={"stage": "final", "final_top_k": arm.config.final_top_k},
    )
    return ArmResult(
        arm=arm, metrics=deep, final_metrics=final, ranked=ranked, finals=finals,
        latencies_ms=latencies, abstained=abstained, confidence=confidence,
        distinct_documents=distinct, errors=errors,
    )


# ------------------------------------------------------------------- сховище
class EvalRunRepo:
    """Доступ до `eval_runs` і `eval_results`.

    Прогони зберігаються, а не лише друкуються, бо порівнювати доводиться те,
    що виміряли в різні дні різними моделями: міграційний гейт embedding-моделі
    — це буквально «взяти прогін минулого тижня й перевірити нерегресію».
    """

    def __init__(self, con: sqlite3.Connection) -> None:
        self.con = con

    def create(self, collection_id: str, config: Mapping[str, Any]) -> str:
        run_id = new_id()
        self.con.execute(
            "INSERT INTO eval_runs (id,collection_id,config_json) VALUES (?,?,?)",
            (run_id, collection_id, json.dumps(config, ensure_ascii=False, default=str)),
        )
        return run_id

    def add_results(self, run_id: str, result: ArmResult) -> None:
        rows = []
        for qid, uids in result.ranked.items():
            payload = {
                "final_uids": result.finals.get(qid, []),
                "abstained": result.abstained.get(qid, False),
                "confidence": result.confidence.get(qid, 0.0),
                "distinct_documents": result.distinct_documents.get(qid, 0),
                "error": result.errors.get(qid),
            }
            rows.append((
                run_id, qid,
                json.dumps(uids, ensure_ascii=False),
                json.dumps(payload, ensure_ascii=False),
                round(result.latencies_ms.get(qid, 0.0)),
            ))
        self.con.executemany(
            "INSERT INTO eval_results (run_id,question_id,ranked_uids,scores_json,latency_ms)"
            " VALUES (?,?,?,?,?)"
            " ON CONFLICT(run_id,question_id) DO UPDATE SET ranked_uids=excluded.ranked_uids,"
            " scores_json=excluded.scores_json, latency_ms=excluded.latency_ms",
            rows,
        )

    def finish(self, run_id: str, metrics: Mapping[str, Any]) -> None:
        self.con.execute(
            "UPDATE eval_runs SET finished_at=?, metrics_json=? WHERE id=?",
            (utcnow(), json.dumps(metrics, ensure_ascii=False, default=str), run_id),
        )

    def get(self, run_id: str) -> dict[str, Any] | None:
        row = self.con.execute("SELECT * FROM eval_runs WHERE id=?", (run_id,)).fetchone()
        return dict(row) if row else None

    def for_collection(self, collection_id: str) -> list[dict[str, Any]]:
        return [
            dict(r) for r in self.con.execute(
                "SELECT * FROM eval_runs WHERE collection_id=? ORDER BY started_at DESC",
                (collection_id,),
            )
        ]

    def results(self, run_id: str) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for r in self.con.execute(
            "SELECT * FROM eval_results WHERE run_id=? ORDER BY question_id", (run_id,)
        ):
            out[r["question_id"]] = {
                "ranked_uids": json.loads(r["ranked_uids"] or "[]"),
                "scores": json.loads(r["scores_json"] or "{}"),
                "latency_ms": r["latency_ms"],
            }
        return out

    def save(
        self,
        collection_id: str,
        result: ArmResult,
        questions: Sequence[GoldQuestion] = (),
    ) -> str:
        """Створити прогін, записати результати й метрики однією операцією.

        `questions` спочатку записуються в `eval_questions`: `eval_results`
        має на них ЗОВНІШНІЙ КЛЮЧ, тож прогін по набору, який існує лише в
        пам'яті harness'а, інакше падає на `FOREIGN KEY constraint failed`.
        Це не незручність схеми, а її сенс: збережений результат мусить
        лишатися розв'язним — інакше через півроку в таблиці лежатимуть
        цифри, до яких немає питань.
        """
        if questions:
            EvalQuestionRepo(self.con).upsert_many(
                [q for q in questions if q.collection_id == collection_id]
            )
        run_id = self.create(collection_id, result.arm.as_dict())
        self.add_results(run_id, result)
        self.finish(run_id, {
            "metrics": result.metrics.as_dict(),
            "final_metrics": result.final_metrics.as_dict(),
            "mean_distinct_documents": result.mean_distinct_documents,
        })
        result.run_id = run_id
        return run_id


def metrics_from_stored(
    con: sqlite3.Connection,
    run_id: str,
    qrels: Mapping[str, set[str]],
    *,
    name: str | None = None,
    ks: Sequence[int] = M.DEFAULT_KS,
) -> M.MetricSet:
    """Перерахувати метрики збереженого прогону.

    Потрібно рівно тому, що метрики можуть змінитись (нове k, виправлена
    формула), а прогін — ні. Зберігаються сирі ранжовані списки, тому будь-яка
    метрика перераховується заднім числом без повторного пошуку.
    """
    repo = EvalRunRepo(con)
    stored = repo.results(run_id)
    run = {qid: row["ranked_uids"] for qid, row in stored.items()}
    latencies = {qid: float(row["latency_ms"] or 0.0) for qid, row in stored.items()}
    meta = repo.get(run_id) or {}
    config = json.loads(meta.get("config_json") or "{}")
    return M.evaluate_run(
        run, qrels, name=name or str(config.get("name") or run_id), ks=ks,
        latencies_ms=latencies,
    )


def compare_runs(
    con: sqlite3.Connection,
    baseline_run_id: str,
    candidate_run_id: str,
    qrels: Mapping[str, set[str]],
    *,
    metric_names: Sequence[str] = ("recall@10", "mrr@10", "ndcg@10", "recall@1"),
    ks: Sequence[int] = M.DEFAULT_KS,
) -> dict[str, Any]:
    """Порівняти два збережені прогони з парними тестами значущості."""
    base = metrics_from_stored(con, baseline_run_id, qrels, ks=ks)
    cand = metrics_from_stored(con, candidate_run_id, qrels, ks=ks)
    return {
        "baseline": base.as_dict(),
        "candidate": cand.as_dict(),
        "relative_gain_pct": M.relative_gain(base, cand),
        "tests": [M.compare_metric(base, cand, m) for m in metric_names],
    }


def run_retrieval_eval(
    retriever: Any,
    questions: Sequence[GoldQuestion],
    collection_id: str,
    *,
    config: AssistantConfig | None = None,
    reranker: Any = None,
    name: str = "default",
    ks: Sequence[int] = M.DEFAULT_KS,
    depth: int = EVAL_DEPTH,
    db: Any = None,
) -> ArmResult:
    """Оцінити ОДНУ конфігурацію пошуку — вхід модуля за контрактом.

    Це те, що викликає API-ендпоїнт «оцінити асистента» і міграційний гейт
    embedding-моделі. Порівняльна таблиця статті будується `run_suite` по
    `standard_arms()`.
    """
    arm = Arm(
        name=name,
        config=_depth_config(config or AssistantConfig(), depth),
        use_reranker=reranker is not None,
        note="Поточна конфігурація асистента.",
    )
    result = run_arm(
        retriever, arm, questions, collection_id, reranker=reranker, ks=ks, depth=depth
    )
    if db is not None:
        with db.transaction() as con:
            EvalRunRepo(con).save(collection_id, result, questions)
    return result


# --------------------------------------------------------------------- сюїта
def run_suite(
    retriever: Any,
    arms: Sequence[Arm],
    questions: Sequence[GoldQuestion],
    collection_id: str,
    *,
    reranker: Any = None,
    ks: Sequence[int] = M.DEFAULT_KS,
    depth: int = EVAL_DEPTH,
    db: Any = None,
    filters: Any = None,
) -> list[ArmResult]:
    """Прогнати всі плечі. `db` не None → кожне плече зберігається як `eval_run`."""
    results: list[ArmResult] = []
    for arm in arms:
        result = run_arm(
            retriever, arm, questions, collection_id,
            reranker=reranker, ks=ks, depth=depth, filters=filters,
        )
        if db is not None:
            with db.transaction() as con:
                EvalRunRepo(con).save(collection_id, result, questions)
        results.append(result)
    return results


def suite_rows(
    results: Sequence[ArmResult],
    *,
    ks: Sequence[int] = (1, 5, 10),
    include_latency: bool = True,
) -> list[dict[str, Any]]:
    """Рядки порівняльної таблиці: одна конфігурація — один рядок."""
    rows: list[dict[str, Any]] = []
    for r in results:
        row: dict[str, Any] = {"конфігурація": r.arm.name}
        for k in ks:
            row[f"Recall@{k}"] = r.metrics.value(f"recall@{k}")
        row["MRR@10"] = r.metrics.value("mrr@10")
        row["nDCG@10"] = r.metrics.value("ndcg@10")
        row["різних документів"] = r.mean_distinct_documents
        if include_latency and r.metrics.latency is not None:
            row["p50, мс"] = r.metrics.latency.p50_ms
            row["p95, мс"] = r.metrics.latency.p95_ms
        rows.append(row)
    return rows


def suite_report(
    results: Sequence[ArmResult],
    *,
    baseline: str = "bm25",
    metric_names: Sequence[str] = ("recall@1", "recall@10", "mrr@10", "ndcg@10"),
    ks: Sequence[int] = (1, 5, 10),
) -> dict[str, Any]:
    """Повний звіт сюїти: таблиця, приріст над baseline і тести значущості."""
    rows = suite_rows(results, ks=ks)
    by_name = {r.arm.name: r for r in results}
    base = by_name.get(baseline)
    comparisons: list[dict[str, Any]] = []
    gains: dict[str, dict[str, float]] = {}
    if base is not None:
        for r in results:
            if r.arm.name == baseline:
                continue
            gains[r.arm.name] = M.relative_gain(base.metrics, r.metrics)
            comparisons.extend(
                M.compare_metric(base.metrics, r.metrics, m) for m in metric_names
            )
    return {
        "rows": rows,
        "markdown": M.markdown_table(rows),
        "latex": M.latex_table(
            rows,
            caption="Порівняння конфігурацій пошуку на золотому наборі",
            label="tab:retrieval-arms",
        ),
        "baseline": baseline,
        "relative_gain_pct": gains,
        "significance": comparisons,
    }
