"""Метрики пошуку: Recall@k, Precision@k, MRR@k, nDCG@k, Hits@k і затримка.

Це те, що НДР зобов'язалась опублікувати (план, §9 і Віха 13): цифри проти
BM25-baseline, приріст від реранкінгу, і статистична перевірка, що приріст не
є шумом. Тому реалізація тут навмисно самодостатня — на numpy і стандартній
бібліотеці, без жодної зовнішньої залежності.

ЧОМУ НЕ `mteb`
--------------
`mteb` жорстко залежить від `pytrec-eval-terrier`, у якого НЕМАЄ колеса
`win_amd64`: збірка з джерел вимагає MSVC Build Tools на ЦІЛЬОВІЙ машині.
На Mac розробника це непомітно (колесо збереться), на Windows академії — це
провалене встановлення. Ціль постачання — закритий контур і USB, тож будь-яка
залежність, що компілюється при встановленні, дискваліфікована за побудовою.

`ranx` 0.3.21 — навпаки, `py3-none-any` поверх numba, тобто Windows-безпечна.
Вона використовується, ЯКЩО доступна (`ranx_available()`), для перехресної
перевірки й для експорту LaTeX; обов'язковою вона не є ніде, і жоден тест від
неї не залежить.

БІНАРНА РЕЛЕВАНТНІСТЬ І IDCG
----------------------------
Золотий набір розмічає «цей чанк відповідає на це питання» — градацій немає,
тому релевантність бінарна. IDCG рахується по `min(|relevant|, k)`: ідеальний
ранжувальник не може поставити в топ-3 п'ять релевантних чанків, і ділити на
недосяжний ідеал означало б систематично занижувати nDCG@1 і nDCG@3.

ПИТАННЯ БЕЗ ВІДПОВІДІ В КОРПУСІ
-------------------------------
Золотий набір навмисно містить питання, відповіді на яке в матеріалах немає
(план, «Ручна перевірка», п. 4). Для них множина релевантних порожня, Recall
не визначений, і `evaluate_run` їх ПРОПУСКАЄ, повідомляючи про це у
`MetricSet.skipped`. Оцінювати їх треба часткою коректних відмов —
`app/eval/groundedness.py`.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np

__all__ = [
    "DEFAULT_KS",
    "METRIC_NAMES",
    "LatencySummary",
    "MetricSet",
    "StatTest",
    "recall_at_k",
    "precision_at_k",
    "mrr_at_k",
    "ndcg_at_k",
    "dcg_at_k",
    "hits_at_k",
    "hit_rate_at_k",
    "average_precision_at_k",
    "first_hit_rank",
    "query_metrics",
    "evaluate_run",
    "latency_summary",
    "percentiles",
    "relative_gain",
    "paired_t_test",
    "fisher_randomization_test",
    "compare_metric",
    "markdown_table",
    "latex_table",
    "ranx_available",
    "evaluate_with_ranx",
]

# Дефолтні глибини. 1 — бо саме Recall@1 виміряв приріст від реранкінгу в
# UNLP 2026 (0.6957 → 0.7935); 5 — бо стільки фрагментів іде в промпт
# (`AssistantConfig.final_top_k`); 10 — бо це глибина міграційного гейта
# embedding-моделі (реєстр ризиків: нерегресія Recall@10 / MRR / nDCG).
DEFAULT_KS: tuple[int, ...] = (1, 3, 5, 10)

METRIC_NAMES: tuple[str, ...] = (
    "recall", "precision", "mrr", "ndcg", "hit_rate", "hits", "map",
)


def _as_set(relevant: Iterable[str]) -> set[str]:
    return {str(r) for r in relevant}


def _top(ranked: Sequence[str], k: int) -> list[str]:
    """Унікальні ідентифікатори з ПЕРШИХ k позицій.

    Дублікати в ранжованому списку — не теоретична можливість: auto-merge
    підіймає L1-батька, а розгортання його назад у листки може дати той самий
    uid двічі. Дубль СПОЖИВАЄ позицію, а не пропускається: якщо система
    показала в топ-2 один і той самий фрагмент двічі, вона дала одне джерело,
    і метрика має бачити одне. Протилежне правило (брати k унікальних) тихо
    завищувало б Recall саме там, де конвеєр повторюється.
    """
    seen: list[str] = []
    for item in list(ranked)[:k]:
        uid = str(item)
        if uid not in seen:
            seen.append(uid)
    return seen


# ------------------------------------------------------------ базові метрики
def recall_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Частка релевантних, що потрапили в топ-k. `nan`, якщо релевантних немає."""
    gold = _as_set(relevant)
    if not gold:
        return math.nan
    hits = len(gold.intersection(_top(ranked, k)))
    return hits / len(gold)


def precision_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Частка топ-k, що виявилась релевантною. Ділиться на k, а не на довжину
    списку: пошук, що повернув один правильний документ замість десяти, не має
    отримувати Precision@10 = 1.0."""
    gold = _as_set(relevant)
    if k <= 0:
        return math.nan
    if not gold:
        return math.nan
    return len(gold.intersection(_top(ranked, k))) / k


def hits_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> int:
    """КІЛЬКІСТЬ релевантних у топ-k (семантика `ranx.hits`)."""
    return len(_as_set(relevant).intersection(_top(ranked, k)))


def hit_rate_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """1.0, якщо в топ-k є хоч один релевантний (семантика `ranx.hit_rate`).

    Це найближча до життя метрика для нашого продукту: викладачеві потрібен
    хоча б один правильний фрагмент, щоб відповідь була заземленою.
    """
    gold = _as_set(relevant)
    if not gold:
        return math.nan
    return 1.0 if gold.intersection(_top(ranked, k)) else 0.0


def first_hit_rank(ranked: Sequence[str], relevant: Iterable[str]) -> int | None:
    """1-based ранг першого релевантного або None. Основа MRR і діагностики."""
    gold = _as_set(relevant)
    if not gold:
        return None
    # Позиції СИРОГО списку: ранг — це те, скільки фрагментів довелося
    # переглянути, включно з повторами.
    for i, item in enumerate(ranked, start=1):
        if str(item) in gold:
            return i
    return None


def mrr_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Обернений ранг першого влучання, обрізаний глибиною k."""
    gold = _as_set(relevant)
    if not gold:
        return math.nan
    rank = first_hit_rank(ranked, gold)
    if rank is None or rank > k:
        return 0.0
    return 1.0 / rank


def average_precision_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """AP@k — середнє Precision на позиціях влучань. Середнє по запитах = MAP@k.

    Знаменник — `min(|relevant|, k)`, з тієї самої причини, що й в IDCG:
    інакше запит із вісьмома золотими чанками не може отримати AP@5 = 1.0
    навіть за ідеального ранжування.
    """
    gold = _as_set(relevant)
    if not gold:
        return math.nan
    top = _top(ranked, k)
    found = 0
    total = 0.0
    for i, uid in enumerate(top, start=1):
        if uid in gold:
            found += 1
            total += found / i
    denominator = min(len(gold), k)
    return total / denominator if denominator else math.nan


def dcg_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """DCG при бінарних вигодах: Σ 1/log2(i+1) по релевантних позиціях."""
    gold = _as_set(relevant)
    return sum(1.0 / math.log2(i + 1) for i, uid in enumerate(_top(ranked, k), start=1) if uid in gold)


def ndcg_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """nDCG@k з IDCG по `min(|relevant|, k)` — див. шапку модуля."""
    gold = _as_set(relevant)
    if not gold or k <= 0:
        return math.nan
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    if ideal <= 0.0:
        return math.nan
    return dcg_at_k(ranked, gold, k) / ideal


# ------------------------------------------------------------------- агрегація
def query_metrics(
    ranked: Sequence[str],
    relevant: Iterable[str],
    ks: Sequence[int] = DEFAULT_KS,
) -> dict[str, float]:
    """Усі метрики одного запиту. Ключі — `"<метрика>@<k>"`."""
    gold = _as_set(relevant)
    out: dict[str, float] = {}
    for k in ks:
        out[f"recall@{k}"] = recall_at_k(ranked, gold, k)
        out[f"precision@{k}"] = precision_at_k(ranked, gold, k)
        out[f"mrr@{k}"] = mrr_at_k(ranked, gold, k)
        out[f"ndcg@{k}"] = ndcg_at_k(ranked, gold, k)
        out[f"hit_rate@{k}"] = hit_rate_at_k(ranked, gold, k)
        out[f"hits@{k}"] = float(hits_at_k(ranked, gold, k))
        out[f"map@{k}"] = average_precision_at_k(ranked, gold, k)
    rank = first_hit_rank(ranked, gold)
    out["first_hit_rank"] = float(rank) if rank is not None else math.nan
    return out


@dataclass(frozen=True, slots=True)
class LatencySummary:
    """Затримка й пропускна здатність одного прогону."""

    count: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    qps: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "count": self.count, "mean_ms": self.mean_ms, "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms, "p99_ms": self.p99_ms, "min_ms": self.min_ms,
            "max_ms": self.max_ms, "qps": self.qps,
        }


def percentiles(samples: Sequence[float], probs: Sequence[float] = (50, 95, 99)) -> list[float]:
    """Перцентилі з лінійною інтерполяцією (метод numpy за замовчуванням).

    На 50 золотих питаннях p99 — це фактично максимум; інтерполяція чесніша за
    nearest-rank тим, що не вдає, ніби ми виміряли хвіст, якого не бачили.
    """
    if not samples:
        return [math.nan for _ in probs]
    arr = np.asarray(samples, dtype=np.float64)
    return [float(np.percentile(arr, p)) for p in probs]


def latency_summary(samples_ms: Sequence[float], *, wall_seconds: float | None = None) -> LatencySummary:
    """p50/p95/p99 і QPS.

    `wall_seconds` — реальний час прогону: якщо він поданий, QPS рахується по
    ньому (це чесна пропускна здатність, включно з паузами й накладними
    витратами harness'а). Без нього QPS = послідовна оцінка `n / Σ затримок`,
    тобто верхня межа для одного потоку.
    """
    if not samples_ms:
        return LatencySummary(0, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan)
    arr = np.asarray(samples_ms, dtype=np.float64)
    p50, p95, p99 = percentiles(samples_ms)
    total_s = wall_seconds if wall_seconds is not None else float(arr.sum()) / 1000.0
    qps = (len(arr) / total_s) if total_s > 0 else math.nan
    return LatencySummary(
        count=int(arr.size),
        mean_ms=float(arr.mean()),
        p50_ms=p50, p95_ms=p95, p99_ms=p99,
        min_ms=float(arr.min()), max_ms=float(arr.max()),
        qps=qps,
    )


@dataclass(frozen=True, slots=True)
class MetricSet:
    """Результат одного прогону однієї конфігурації по всьому золотому набору."""

    name: str
    ks: tuple[int, ...]
    values: dict[str, float]                      # "recall@5" → середнє
    per_query: dict[str, dict[str, float]]        # qid → {метрика: значення}
    n_queries: int
    skipped: tuple[str, ...] = ()                 # питання без золотих чанків
    latency: LatencySummary | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def value(self, metric: str) -> float:
        return self.values.get(metric, math.nan)

    def column(self, metric: str) -> list[float]:
        """Значення метрики по запитах у СТАЛОМУ порядку — для парних тестів."""
        return [self.per_query[q].get(metric, math.nan) for q in sorted(self.per_query)]

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ks": list(self.ks),
            "values": self.values,
            "n_queries": self.n_queries,
            "skipped": list(self.skipped),
            "latency": self.latency.as_dict() if self.latency else None,
            "extra": self.extra,
        }


def evaluate_run(
    run: Mapping[str, Sequence[str]],
    qrels: Mapping[str, Iterable[str]],
    *,
    name: str = "run",
    ks: Sequence[int] = DEFAULT_KS,
    latencies_ms: Mapping[str, float] | None = None,
    wall_seconds: float | None = None,
    extra: Mapping[str, Any] | None = None,
) -> MetricSet:
    """Порахувати всі метрики по золотому набору.

    `run` — qid → ранжований список `chunk_uid`; `qrels` — qid → золоті uid.
    Питання з порожнім `qrels` не оцінюються (див. шапку) і повертаються у
    `skipped`. Питання, якого немає в `run`, вважається таким, що повернув
    порожній список: мовчазне зникнення запиту з прогону — це найтихіший спосіб
    завищити метрики.
    """
    ks_t = tuple(int(k) for k in ks)
    per_query: dict[str, dict[str, float]] = {}
    skipped: list[str] = []

    for qid, gold in qrels.items():
        gold_set = _as_set(gold)
        if not gold_set:
            skipped.append(qid)
            continue
        per_query[qid] = query_metrics(run.get(qid, ()), gold_set, ks_t)

    values: dict[str, float] = {}
    if per_query:
        keys = sorted({k for row in per_query.values() for k in row})
        for key in keys:
            column = np.asarray([row.get(key, math.nan) for row in per_query.values()], dtype=np.float64)
            with np.errstate(invalid="ignore"):
                mean = float(np.nanmean(column)) if np.any(~np.isnan(column)) else math.nan
            values[key] = mean

    latency = None
    if latencies_ms:
        latency = latency_summary(
            [latencies_ms[q] for q in sorted(latencies_ms)], wall_seconds=wall_seconds
        )

    return MetricSet(
        name=name,
        ks=ks_t,
        values=values,
        per_query=per_query,
        n_queries=len(per_query),
        skipped=tuple(sorted(skipped)),
        latency=latency,
        extra=dict(extra or {}),
    )


def relative_gain(baseline: MetricSet, candidate: MetricSet) -> dict[str, float]:
    """Відносний приріст у ВІДСОТКАХ по кожній метриці.

    Саме ця таблиця відтворює заявлений статтею автора приріст +15% над BM25:
    UNLP 2026 виміряв Recall@1 0.6957 → 0.7935, тобто +14.1% відносних.
    Нуль у знаменнику дає `inf`, а не виняток: baseline, що не знайшов нічого, —
    це теж результат, який має потрапити в таблицю.
    """
    out: dict[str, float] = {}
    for key, base in baseline.values.items():
        new = candidate.values.get(key)
        if new is None or math.isnan(base) or math.isnan(new):
            continue
        if base == 0.0:
            out[key] = math.inf if new > 0 else 0.0
        else:
            out[key] = (new - base) / base * 100.0
    return out


# --------------------------------------------------------- статистичні тести
@dataclass(frozen=True, slots=True)
class StatTest:
    """Результат перевірки «приріст не є шумом»."""

    name: str
    statistic: float
    p_value: float
    mean_difference: float
    n: int
    backend: str = "вбудований"

    def significant(self, alpha: float = 0.05) -> bool:
        return self.p_value < alpha

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "statistic": self.statistic, "p_value": self.p_value,
            "mean_difference": self.mean_difference, "n": self.n, "backend": self.backend,
        }


def _betacf(a: float, b: float, x: float) -> float:
    """Неперервний дріб для неповної бета-функції (метод Лентца)."""
    tiny = 1e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-14:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """Регуляризована неповна бета-функція I_x(a, b).

    Потрібна рівно для одного: p-значення розподілу Стьюдента. Тягти заради
    цього scipy (~90 МБ у офлайн-інсталятор) немає жодних підстав.
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _paired(a: Sequence[float], b: Sequence[float]) -> np.ndarray:
    """Різниці по парах, з викиданням пар, де хоч одне значення `nan`."""
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(
            f"Парний тест вимагає однакової кількості запитів: {x.shape[0]} проти {y.shape[0]}."
        )
    mask = ~(np.isnan(x) | np.isnan(y))
    return x[mask] - y[mask]


def paired_t_test(candidate: Sequence[float], baseline: Sequence[float]) -> StatTest:
    """Двобічний парний t-тест по запитах.

    Парний, а не незалежний: обидві конфігурації бачать ОДИН І ТОЙ САМИЙ набір
    питань, і між запитами дисперсія на порядок більша, ніж між конфігураціями.
    Незалежний тест на тих самих даних просто не побачив би реального приросту.
    """
    diff = _paired(candidate, baseline)
    n = int(diff.size)
    if n < 2:
        return StatTest("paired_t", math.nan, math.nan, float(diff.mean()) if n else math.nan, n)
    mean = float(diff.mean())
    sd = float(diff.std(ddof=1))
    if sd == 0.0:
        # Усі різниці однакові: або приріст детермінований, або його немає.
        return StatTest("paired_t", math.inf if mean else 0.0, 0.0 if mean else 1.0, mean, n)
    t = mean / (sd / math.sqrt(n))
    df = n - 1
    p = _betai(df / 2.0, 0.5, df / (df + t * t))
    return StatTest("paired_t", float(t), float(min(1.0, max(0.0, p))), mean, n)


def fisher_randomization_test(
    candidate: Sequence[float],
    baseline: Sequence[float],
    *,
    trials: int = 10000,
    seed: int = 20260101,
) -> StatTest:
    """Двобічний парний рандомізаційний тест Фішера.

    Це основний тест для IR-метрик, і саме він стоїть у `ranx`: Recall@k і
    nDCG@k обмежені зверху й знизу, розподілені далеко не нормально (маса в
    0 і 1), а t-тест припускає нормальність різниць. Рандомізація нічого не
    припускає — вона просто перебирає, які знаки різниць могли б випасти
    випадково.

    `seed` фіксований: цифра у звіті з НДР має відтворюватись байт-у-байт.
    """
    diff = _paired(candidate, baseline)
    n = int(diff.size)
    if n == 0:
        return StatTest("fisher_randomization", math.nan, math.nan, math.nan, 0)
    observed = float(diff.mean())
    if not np.any(diff):
        return StatTest("fisher_randomization", 0.0, 1.0, 0.0, n)
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(trials, n))
    sampled = (signs * diff).mean(axis=1)
    # +1 у чисельнику й знаменнику: спостережене перестановлення теж є одним із
    # можливих, і без нього p може вийти рівним нулю, чого бути не може.
    p = (int(np.count_nonzero(np.abs(sampled) >= abs(observed))) + 1) / (trials + 1)
    return StatTest("fisher_randomization", observed, float(p), observed, n)


def compare_metric(
    baseline: MetricSet,
    candidate: MetricSet,
    metric: str,
    *,
    trials: int = 10000,
    seed: int = 20260101,
) -> dict[str, Any]:
    """Повний звіт по одній метриці: обидва середні, приріст і два тести."""
    common = sorted(set(baseline.per_query) & set(candidate.per_query))
    base_col = [baseline.per_query[q].get(metric, math.nan) for q in common]
    cand_col = [candidate.per_query[q].get(metric, math.nan) for q in common]
    base_mean = baseline.value(metric)
    cand_mean = candidate.value(metric)
    gain = math.nan
    if not math.isnan(base_mean) and not math.isnan(cand_mean):
        gain = math.inf if base_mean == 0 and cand_mean > 0 else (
            0.0 if base_mean == 0 else (cand_mean - base_mean) / base_mean * 100.0
        )
    return {
        "metric": metric,
        "baseline": baseline.name,
        "candidate": candidate.name,
        "baseline_value": base_mean,
        "candidate_value": cand_mean,
        "relative_gain_pct": gain,
        "n_common_queries": len(common),
        "t_test": paired_t_test(cand_col, base_col).as_dict(),
        "randomization": fisher_randomization_test(
            cand_col, base_col, trials=trials, seed=seed
        ).as_dict(),
    }


# --------------------------------------------------------------- експорт таблиць
def _fmt(value: Any, digits: int = 4) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "—"
        if math.isinf(value):
            return "∞"
        return f"{value:.{digits}f}"
    return str(value)


def markdown_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str] | None = None) -> str:
    """Markdown-таблиця для звіту з НДР. Порожній список — порожній рядок."""
    if not rows:
        return ""
    cols = list(columns) if columns else list(rows[0])
    head = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    body = ["| " + " | ".join(_fmt(r.get(c, "")) for c in cols) + " |" for r in rows]
    return "\n".join([head, sep, *body])


def latex_table(
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str] | None = None,
    *,
    caption: str = "Результати оцінювання пошуку",
    label: str = "tab:retrieval",
) -> str:
    """LaTeX-таблиця (booktabs) для статті.

    `ranx` уміє експортувати LaTeX сама, але робить це у власному форматі й
    лише для своїх об'єктів; тут потрібен той самий вигляд і тоді, коли `ranx`
    не встановлена — а це основний випадок на цільовій машині.
    Підкреслення в назвах метрик екрануються: `recall@5` безпечний, а
    `first_hit_rank` без екранування зламає компіляцію.
    """
    if not rows:
        return ""
    cols = list(columns) if columns else list(rows[0])
    align = "l" + "r" * (len(cols) - 1)

    def esc(value: Any) -> str:
        return str(value).replace("_", r"\_")

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(esc(c) for c in cols) + r" \\",
        r"\midrule",
    ]
    lines += [" & ".join(esc(_fmt(r.get(c, ""))) for c in cols) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ------------------------------------------------------------------------ ranx
@lru_cache(maxsize=1)
def _ranx() -> Any:
    try:
        import ranx  # type: ignore[import-not-found]
    except Exception:
        return None
    return ranx


def ranx_available() -> bool:
    """Чи можна скористатись `ranx` для перехресної перевірки цифр."""
    return _ranx() is not None


def evaluate_with_ranx(
    run: Mapping[str, Sequence[str]],
    qrels: Mapping[str, Iterable[str]],
    metrics: Sequence[str] = ("recall@10", "mrr@10", "ndcg@10"),
) -> dict[str, float]:
    """Ті самі метрики руками `ranx` — перехресна перевірка нашої арифметики.

    Кидає `RuntimeError`, якщо `ranx` не встановлена: це необов'язкова
    залежність (`pip install .[eval]`), і мовчазне повернення порожнього
    словника означало б, що перевірки не було, а звіт про це не сказав.
    """
    ranx = _ranx()
    if ranx is None:
        raise RuntimeError(
            "Пакет `ranx` не встановлено. Він необов'язковий: усі метрики рахує "
            "`app.eval.metrics`. Для перехресної перевірки: pip install 'ranx>=0.3.21'."
        )
    qrels_dict = {q: {uid: 1 for uid in _as_set(gold)} for q, gold in qrels.items() if _as_set(gold)}
    # ranx вимагає СКОРІВ, а не рангів; спадна послідовність відтворює порядок.
    run_dict = {
        q: {uid: float(len(run.get(q, ())) - i) for i, uid in enumerate(run.get(q, ()))}
        for q in qrels_dict
    }
    scores = ranx.evaluate(ranx.Qrels(qrels_dict), ranx.Run(run_dict), list(metrics))
    if isinstance(scores, dict):
        return {k: float(v) for k, v in scores.items()}
    return {metrics[0]: float(scores)}
