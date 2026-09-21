"""Утримання від відповіді: механізм `u_lin` (TMLR 09/2024).

**Чому константний поріг НЕ працює.** Той самий `bge-reranker-v2-m3` дає
релевантній АНГЛІЙСЬКІЙ парі скор 0.9953, а релевантній КИТАЙСЬКІЙ — 0.2093.
Це не різниця в якості, а крос-мовний дрейф самої шкали: голова реранкера
відкалібрована на англомовному тренувальному розподілі. Практичний наслідок
для нас прямий і фатальний: поріг, підібраний на англійських прикладах (або
взятий із README моделі), утримуватиметься від УСЬОГО українського, і асистент
відповідатиме «у матеріалах немає інформації» на питання, відповідь на яке
лежить у першому ж знайденому чанку.

**Механізм.** `u_lin` — гребенева регресія (λ=0.1) на векторі скорів реранкера,
відсортованому ЗА ЗРОСТАННЯМ, що передбачає nDCG@10 конкретного запиту.
Утримуємось, коли `u_lin < τ`. Ключова властивість: модель дивиться не на
абсолютне значення топ-скора, а на ФОРМУ розподілу скорів (є розрив між
першим і рештою? чи всі десять однаково посередні?), а форма переживає
крос-мовний зсув шкали, бо зсув зачіпає всі скори одного запиту однаково.
За вимірами оригінальної роботи це перевершує всі reference-free евристики
(середній nAUC mAP 37.4 проти 28.5 у найкращої), окупається вже на ~38
розмічених запитах і коштує 1.2% часу реранкінгу.

**Калібрування зберігається ПО КОЛЕКЦІЯХ.** Багатоасистентна архітектура дає
це безкоштовно: у кожного асистента свій корпус, своя мовна суміш і свій
розподіл скорів. Одне глобальне калібрування знову змішало б шкали — тобто
відтворило б рівно ту проблему, від якої механізм і рятує.

Поки калібрування немає (менше за `MIN_CALIBRATION_QUERIES` розмічених
запитів) — фолбек на простий поріг з `AssistantConfig.confidence_threshold()`.
Це свідомо консервативно й свідомо тимчасово: фолбек чесно позначається в
`RetrievalDebug` і в звіті, щоб «асистент занадто часто мовчить» діагностувалось
за секунди, а не за тиждень.

Регресія — десяток рядків на numpy. `scikit-learn` заради `Ridge` у закритому
контурі — це +30 МБ колеса, scipy у залежностях і ще один компонент, який
доведеться возити на USB.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np

__all__ = [
    "CALIBRATION_DIR_ENV",
    "MIN_CALIBRATION_QUERIES",
    "MIN_USEFUL_NDCG",
    "N_FEATURES",
    "RIDGE_LAMBDA",
    "AbstentionCalibration",
    "AbstentionDecision",
    "CalibrationSample",
    "CalibrationStore",
    "calibrate_threshold",
    "decide",
    "features_from_scores",
    "fit_ridge",
    "fit_u_lin",
    "ndcg_at_k",
]

log = logging.getLogger(__name__)

CALIBRATION_DIR_ENV = "ASISTENT_CALIBRATION_DIR"

# Довжина ознакового вектора. 10 = nDCG@10: рівно стільки скорів і описують
# ту метрику, яку регресія передбачає.
N_FEATURES = 10
# λ з оригінальної роботи. Не «підібрано» — узято як опубліковане.
RIDGE_LAMBDA = 0.1
# Точка окупності з плану (§7). Нижче за неї регресія перенавчається на шумі
# і поводиться гірше за тупий поріг — тому нижче за неї ми її не вмикаємо.
MIN_CALIBRATION_QUERIES = 38
# Нижче цього nDCG@10 запит вважається непокритим корпусом: фрагменти є, але
# відповіді в них немає.
MIN_USEFUL_NDCG = 0.30

_SAFE_ID = re.compile(r"[^A-Za-z0-9_.-]")


# ------------------------------------------------------------------- ознаки
def features_from_scores(scores: Sequence[float], n_features: int = N_FEATURES) -> np.ndarray:
    """Вектор ознак: `n_features` найкращих скорів, ЗА ЗРОСТАННЯМ.

    Порядок «за зростанням» — не косметика: ознака `x[-1]` завжди означає
    «найкращий кандидат», `x[-2]` — «другий», і ваги регресії лишаються
    інтерпретовними, хоч би скільки кандидатів реально повернув пошук.
    Якщо кандидатів менше за `n_features`, вектор доповнюється СПЕРЕДУ
    найгіршим наявним скором (а не нулем): «кандидатів мало» не означає
    «є кандидати зі скором 0», і нулі зсунули б розподіл ознак.
    """
    arr = np.sort(np.asarray(list(scores), dtype=np.float64))
    if arr.size == 0:
        return np.zeros(n_features, dtype=np.float64)
    if arr.size >= n_features:
        return arr[-n_features:]
    pad = np.full(n_features - arr.size, arr[0], dtype=np.float64)
    return np.concatenate([pad, arr])


def ndcg_at_k(relevances: Sequence[float], k: int = N_FEATURES) -> float:
    """nDCG@k для списку релевантностей У ПОРЯДКУ ВИДАЧІ.

    Живе тут, а не в `app/eval/`, свідомо: калібрування утримання мусить
    рахувати свою мітку тим самим кодом, яким її рахуватиме будь-хто інший,
    і не мати залежності від модуля оцінювання, який вмикається окремо.
    """
    rel = np.asarray(list(relevances), dtype=np.float64)[:k]
    if rel.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, rel.size + 2))
    dcg = float(np.sum((2.0**rel - 1.0) * discounts))
    ideal = np.sort(np.asarray(list(relevances), dtype=np.float64))[::-1][:k]
    idcg = float(np.sum((2.0**ideal - 1.0) / np.log2(np.arange(2, ideal.size + 2))))
    return dcg / idcg if idcg > 0 else 0.0


@dataclass(slots=True)
class CalibrationSample:
    """Один розмічений запит золотого набору."""

    scores: list[float]
    relevances: list[float]

    def target(self, k: int = N_FEATURES) -> float:
        return ndcg_at_k(self.relevances, k)


# ------------------------------------------------------------------ регресія
def fit_ridge(x: np.ndarray, y: np.ndarray, lam: float = RIDGE_LAMBDA) -> tuple[np.ndarray, float]:
    """Гребенева регресія з НЕштрафованим інтерсептом.

    Центрування — не стилістика: без нього λ штрафує і зсув, і при малих
    вибірках модель систематично занижує передбачений nDCG, тобто утримується
    частіше, ніж треба. Розв'язується через `solve`, а не `inv`: обернення
    матриці на майже виродженому `XᵀX` (а він майже вироджений завжди — сусідні
    скори сильно корельовані) втрачає точність без жодної потреби.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.ndim != 2 or x.shape[0] != y.size:
        raise ValueError(f"Невідповідні форми: X={x.shape}, y={y.shape}.")
    x_mean = x.mean(axis=0)
    y_mean = float(y.mean())
    xc = x - x_mean
    a = xc.T @ xc + lam * np.eye(x.shape[1])
    w = np.linalg.solve(a, xc.T @ (y - y_mean))
    return w, y_mean - float(x_mean @ w)


def calibrate_threshold(
    predictions: Sequence[float],
    targets: Sequence[float],
    *,
    min_useful_ndcg: float = MIN_USEFUL_NDCG,
) -> float:
    """Поріг τ, що найкраще розділяє «покриті» й «непокриті» запити.

    Критерій — J Юдена (чутливість + специфічність - 1), а не точність:
    класи тут майже завжди незбалансовані (більшість запитів золотого набору
    покриті), і за точністю виграв би вироджений поріг «ніколи не утримуватись».
    """
    preds = np.asarray(list(predictions), dtype=np.float64)
    labels = np.asarray(list(targets), dtype=np.float64) >= min_useful_ndcg
    if preds.size == 0 or labels.all() or not labels.any():
        return float(min_useful_ndcg)

    candidates = np.unique(np.concatenate([preds, [min_useful_ndcg]]))
    best_tau, best_j = float(min_useful_ndcg), -2.0
    for tau in candidates:
        answered = preds >= tau
        tpr = float(np.mean(answered[labels])) if labels.any() else 0.0
        fpr = float(np.mean(answered[~labels])) if (~labels).any() else 0.0
        j = tpr - fpr
        if j > best_j:
            best_j, best_tau = j, float(tau)
    return best_tau


@dataclass(slots=True)
class AbstentionCalibration:
    """Збережене калібрування `u_lin` для однієї колекції."""

    collection_id: str
    reranker_key: str
    weights: list[float]
    intercept: float
    tau: float
    n_features: int = N_FEATURES
    n_samples: int = 0
    lam: float = RIDGE_LAMBDA
    min_useful_ndcg: float = MIN_USEFUL_NDCG
    train_rmse: float = 0.0
    created_at: str = ""
    version: int = 1

    @property
    def ready(self) -> bool:
        """Чи можна довіряти цій регресії.

        Нижче за точку окупності (~38 запитів) регресія на 10 ознаках
        перенавчається на шумі: вона вивчить конкретні скори конкретних
        запитів, а не форму розподілу. Тоді краще чесний тупий поріг.
        """
        return self.n_samples >= MIN_CALIBRATION_QUERIES and len(self.weights) == self.n_features

    def predict(self, scores: Sequence[float]) -> float:
        """Передбачений nDCG@10 запиту, обрізаний до [0, 1]."""
        x = features_from_scores(scores, self.n_features)
        raw = float(np.dot(np.asarray(self.weights, dtype=np.float64), x)) + self.intercept
        return float(min(1.0, max(0.0, raw)))

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=1)

    @classmethod
    def from_json(cls, raw: str) -> AbstentionCalibration:
        data = json.loads(raw) if raw else {}
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


def fit_u_lin(
    samples: Iterable[CalibrationSample | tuple[Sequence[float], Sequence[float]]],
    *,
    collection_id: str,
    reranker_key: str,
    n_features: int = N_FEATURES,
    lam: float = RIDGE_LAMBDA,
    min_useful_ndcg: float = MIN_USEFUL_NDCG,
) -> AbstentionCalibration:
    """Навчити `u_lin` на золотому наборі колекції.

    Вхід — те, що вже й так є в `eval_questions`: скори реранкера на кандидатах
    запиту й розмічені релевантності. Тобто калібрування не потребує ЖОДНОЇ
    додаткової розмітки понад ту, яку НДР і так робить для звіту про Recall@k.
    """
    prepared = [
        s if isinstance(s, CalibrationSample) else CalibrationSample(list(s[0]), list(s[1]))
        for s in samples
    ]
    if not prepared:
        raise ValueError("Порожній калібрувальний набір: жодного розміченого запиту.")

    x = np.vstack([features_from_scores(s.scores, n_features) for s in prepared])
    y = np.asarray([s.target(n_features) for s in prepared], dtype=np.float64)
    weights, intercept = fit_ridge(x, y, lam)
    predictions = np.clip(x @ weights + intercept, 0.0, 1.0)
    tau = calibrate_threshold(predictions, y, min_useful_ndcg=min_useful_ndcg)
    rmse = float(np.sqrt(np.mean((predictions - y) ** 2)))

    calibration = AbstentionCalibration(
        collection_id=collection_id,
        reranker_key=reranker_key,
        weights=[float(w) for w in weights],
        intercept=float(intercept),
        tau=float(tau),
        n_features=n_features,
        n_samples=len(prepared),
        lam=lam,
        min_useful_ndcg=min_useful_ndcg,
        train_rmse=rmse,
        created_at=datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S"),
    )
    if not calibration.ready:
        log.info(
            "Калібрування утримання для колекції %s навчено на %d запитах — це менше за "
            "точку окупності (%d). До її досягнення діє простий поріг з конфігу асистента.",
            collection_id, len(prepared), MIN_CALIBRATION_QUERIES,
        )
    return calibration


# --------------------------------------------------------------------- рішення
@dataclass(slots=True)
class AbstentionDecision:
    """Рішення «відповідати чи мовчати» разом із поясненням для UI."""

    abstain: bool
    confidence: float
    mode: Literal["u_lin", "threshold", "empty"]
    threshold: float
    supporting: int = 0
    reason: str = ""
    details: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def decide(
    scores: Sequence[float],
    config: object,
    calibration: AbstentionCalibration | None = None,
) -> AbstentionDecision:
    """Утриматись чи відповідати.

    Пороги виконуються В КОДІ, а не в промпті (контракт, правило 8): промптом
    неможливо надійно змусити 12B-модель відмовитись, порогом — можна.

    `config` — це `domain.AssistantConfig`; тип узятий як `object`, щоб модуль
    реранкінгу не залежав від доменного шару за імпортом (контракт: модулі не
    імпортують типи один в одного). Потрібні лише два його методи.
    """
    values = [float(s) for s in scores]
    threshold = float(config.confidence_threshold())        # type: ignore[attr-defined]
    min_supporting = int(config.min_supporting_chunks())    # type: ignore[attr-defined]

    if not values:
        return AbstentionDecision(
            abstain=True,
            confidence=0.0,
            mode="empty",
            threshold=threshold,
            reason="Пошук не повернув жодного фрагмента.",
        )

    if calibration is not None and calibration.ready:
        u_lin = calibration.predict(values)
        abstain = u_lin < calibration.tau
        return AbstentionDecision(
            abstain=abstain,
            confidence=u_lin,
            mode="u_lin",
            threshold=float(calibration.tau),
            supporting=int(sum(1 for s in values if s >= threshold)),
            reason=(
                f"Передбачений nDCG@10 = {u_lin:.3f} нижчий за поріг {calibration.tau:.3f}, "
                "відкалібрований на золотому наборі цієї колекції."
                if abstain
                else f"Передбачений nDCG@10 = {u_lin:.3f} при порозі {calibration.tau:.3f}."
            ),
            details={"top_score": max(values), "n_candidates": float(len(values))},
        )

    supporting = sum(1 for s in values if s >= threshold)
    abstain = supporting < min_supporting
    return AbstentionDecision(
        abstain=abstain,
        confidence=max(values),
        mode="threshold",
        threshold=threshold,
        supporting=supporting,
        reason=(
            f"Калібрування утримання ще немає (треба {MIN_CALIBRATION_QUERIES} розмічених "
            f"запитів). Тимчасовий поріг {threshold:.2f}: опорних фрагментів {supporting}, "
            f"потрібно {min_supporting}."
        ),
        details={"top_score": max(values), "n_candidates": float(len(values))},
    )


# ------------------------------------------------------------------- сховище
class CalibrationStore:
    """Калібрування на диску, по одному JSON на колекцію.

    Свідомо НЕ в SQLite: схема `0001_initial.sql` не має відповідної таблиці,
    а міняти її заборонено контрактом. Файл поруч із даними також простіше
    показати, продіагностувати й перенести між машинами разом зі звітом НДР.
    """

    def __init__(self, directory: Path | str | None = None) -> None:
        if directory is not None:
            self._dir = Path(directory)
        else:
            override = os.environ.get(CALIBRATION_DIR_ENV, "").strip()
            if override:
                self._dir = Path(override).expanduser()
            else:
                from app.config import Paths

                self._dir = Paths.resolve().data_dir / "calibration"

    @property
    def directory(self) -> Path:
        return self._dir

    def path_for(self, collection_id: str) -> Path:
        # Ідентифікатор колекції — UUID-hex, але санітизація тут дешева і
        # закриває шлях `../` назавжди.
        return self._dir / f"{_SAFE_ID.sub('_', collection_id)}.json"

    def load(self, collection_id: str, reranker_key: str | None = None) -> AbstentionCalibration | None:
        path = self.path_for(collection_id)
        if not path.is_file():
            return None
        try:
            calibration = AbstentionCalibration.from_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError) as exc:
            log.warning("Калібрування утримання в %s не читається (%s); діє простий поріг.", path, exc)
            return None
        if reranker_key and calibration.reranker_key != reranker_key:
            # Скори різних реранкерів живуть у різних шкалах — це та сама
            # причина, через яку не працює константний поріг. Калібрування,
            # зняте на іншій моделі, тут гірше за відсутнє.
            log.warning(
                "Калібрування утримання колекції %s знято на іншому реранкері (%s != %s) — "
                "ігнорую. Перекалібруйте на золотому наборі.",
                collection_id, calibration.reranker_key, reranker_key,
            )
            return None
        return calibration

    def save(self, calibration: AbstentionCalibration) -> Path:
        path = self.path_for(calibration.collection_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(calibration.to_json(), encoding="utf-8")
        # Атомарна заміна: недописаний JSON гірший за відсутній — відсутній
        # чесно вмикає фолбек, а обрізаний ловиться лише в рантаймі запиту.
        os.replace(tmp, path)
        return path

    def delete(self, collection_id: str) -> bool:
        path = self.path_for(collection_id)
        if path.is_file():
            path.unlink()
            return True
        return False
