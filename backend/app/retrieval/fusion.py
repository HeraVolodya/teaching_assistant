"""Зважений RRF — злиття трьох гілок пошуку в один ранжований список.

Чому саме RRF, а не зважена сума скорів: три гілки видають скори з абсолютно
різних шкал. Косинус dense-гілки живе в [-1, 1], BM25 — необмежене додатне
число, що росте з довжиною запиту, а символьний TF-IDF — знову [0, 1]. Будь-яка
зважена сума цих чисел мовчки віддає перевагу тій гілці, чия шкала цього дня
більша. RRF працює з РАНГАМИ, тому шкали з рівняння зникають.

Але за це доводиться платити, і плата тут суттєва:

    RRF знищує інформацію про АБСОЛЮТНУ релевантність.

Документ, що стоїть першим у пустому індексі з косинусом 0.11, і документ, що
стоїть першим із косинусом 0.87, отримують РІВНО однаковий внесок 1/(k+1). Саме
абсолютна релевантність потрібна для короткого замикання «немає контексту»
(план, §7: утримання від відповіді), тож сирі топ-скори обох гілок ми НЕСЕМО
ПОРУЧ із RRF-скором у `FusedCandidate`, а не викидаємо.

Автопідйом ваги sparse-гілки: коли в запиті є лексичний якір (цифра,
буквено-цифрове позначення на кшталт «Д-30» чи «2С1», лапки, ДСТУ/ГОСТ),
семантична близькість стає майже марною — «Д-30» і «Д-20» лежать в
ембединговому просторі поруч, а це різні гармати. У такому режимі точний
збіг важить стільки ж, скільки сенс, тому w_sparse підіймається до 1.0.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

__all__ = [
    "Ranked",
    "FusionWeights",
    "FusedCandidate",
    "DEFAULT_RRF_K",
    "ANCHORED_SPARSE_WEIGHT",
    "has_lexical_anchor",
    "resolve_weights",
    "weighted_rrf",
]

# Ранжований список однієї гілки: (chunk_id, СИРИЙ скор гілки), найкращий першим.
Ranked = Sequence[tuple[int, float]]

DEFAULT_RRF_K = 60
# До чого підіймається w_sparse на запиті з лексичним якорем (план, §3).
ANCHORED_SPARSE_WEIGHT = 1.0


# --------------------------------------------------------------- лексичний якір
# Позначення й стандарти, які викладач пише дослівно і які мусять знаходитись
# точним збігом, а не «за сенсом».
_ANCHOR_STANDARD = re.compile(
    r"(?iu)(?:^|[^\w])(ДСТУ|ГОСТ|ТУ|ДБН|ISO|IEC|EN|STANAG|MIL[-\s]?STD|NATO|НАТО)(?:$|[^\w])"
)
_ANCHOR_DIGIT = re.compile(r"\d")
# Токен, у якому є і літера, і цифра: Д-30, 2С1, БМ-21, 9М113, T-72.
_ANCHOR_ALNUM = re.compile(r"(?iu)(?<![\w-])(?=[\w-]*[^\W\d_])(?=[\w-]*\d)[\w-]{2,}(?![\w-])")
# УВАГА: апостроф сюди НЕ входить. Українське слово легітимно містить ' і ’
# (п'ять, об'єкт), тож апостроф у списку лапок зробив би лексичним якорем
# практично кожен український запит і назавжди прибив би dense-гілку.
_ANCHOR_QUOTE = re.compile(r'["«»„“”]')


def has_lexical_anchor(query: str) -> bool:
    """Чи містить запит щось, що мусить знайтися ТОЧНО, а не «за сенсом».

    Чотири незалежні ознаки (план, §3): цифра, буквено-цифрове позначення,
    лапки, назва стандарту.
    """
    if not query:
        return False
    return bool(
        _ANCHOR_DIGIT.search(query)
        or _ANCHOR_ALNUM.search(query)
        or _ANCHOR_QUOTE.search(query)
        or _ANCHOR_STANDARD.search(query)
    )


@dataclass(frozen=True, slots=True)
class FusionWeights:
    """Ваги гілок для одного конкретного запиту."""

    dense: float = 1.0
    sparse: float = 0.6
    ngram: float = 0.4
    k: int = DEFAULT_RRF_K
    anchored: bool = False

    def as_dict(self) -> dict[str, float | bool | int]:
        return {
            "dense": self.dense,
            "sparse": self.sparse,
            "ngram": self.ngram,
            "k": self.k,
            "anchored": self.anchored,
        }


def resolve_weights(
    query: str,
    *,
    weight_dense: float = 1.0,
    weight_sparse: float = 0.6,
    weight_ngram: float = 0.4,
    rrf_k: int = DEFAULT_RRF_K,
) -> FusionWeights:
    """Ваги з конфігу асистента плюс автопідйом sparse на лексичному якорі.

    Значення за замовчуванням узяті з `AssistantConfig` (1.0 / 0.6 / 0.4).
    У §7 плану в схемі конвеєра стоїть 0.7 для sparse — це та сама величина,
    яку harness має підбирати per-corpus; постачений дефолт живе в
    `AssistantConfig`, і саме він тут головний, щоб налаштування асистента
    лишалось ДАНИМИ, а не кодом.
    """
    anchored = has_lexical_anchor(query)
    sparse = max(weight_sparse, ANCHORED_SPARSE_WEIGHT) if anchored else weight_sparse
    return FusionWeights(
        dense=weight_dense, sparse=sparse, ngram=weight_ngram, k=rrf_k, anchored=anchored
    )


@dataclass(slots=True)
class FusedCandidate:
    """Кандидат після злиття.

    Сирі скори гілок несуться поруч навмисно: `rrf_score` придатний лише для
    ВПОРЯДКУВАННЯ, а для рішення «чи взагалі є контекст» потрібна абсолютна
    величина, якої в RRF немає за побудовою.
    """

    chunk_id: int
    rrf_score: float = 0.0
    dense_rank: int | None = None
    sparse_rank: int | None = None
    ngram_rank: int | None = None
    dense_score: float | None = None
    sparse_score: float | None = None
    ngram_score: float | None = None

    @property
    def branches(self) -> int:
        """Скільки гілок знайшли цей чанк. Три з трьох — сильний сигнал."""
        return sum(r is not None for r in (self.dense_rank, self.sparse_rank, self.ngram_rank))

    @property
    def best_rank(self) -> int:
        ranks = [r for r in (self.dense_rank, self.sparse_rank, self.ngram_rank) if r is not None]
        return min(ranks) if ranks else 1 << 30

    def as_dict(self) -> dict[str, float | int | None]:
        return {
            "chunk_id": self.chunk_id,
            "rrf": round(self.rrf_score, 6),
            "dense_rank": self.dense_rank,
            "sparse_rank": self.sparse_rank,
            "ngram_rank": self.ngram_rank,
            "dense": None if self.dense_score is None else round(self.dense_score, 6),
            "sparse": None if self.sparse_score is None else round(self.sparse_score, 6),
            "ngram": None if self.ngram_score is None else round(self.ngram_score, 6),
        }


def weighted_rrf(
    *,
    dense: Ranked = (),
    sparse: Ranked = (),
    ngram: Ranked = (),
    weights: FusionWeights | None = None,
) -> list[FusedCandidate]:
    """Злити три ранжовані списки. Кожен список — найкращий елемент першим.

    Внесок гілки = `w / (k + rank)`, ранг 1-based. `k=60` — стандартне значення
    Cormack et al.; воно навмисно велике, щоб різниця між рангами 1 і 2 не була
    драматичною: на 60 перші місця відрізняються на ~1.6%, і тому одна гілка,
    що впевнено помиляється, не перекриває дві, що обережно праві.
    """
    w = weights or FusionWeights()
    k = max(1, w.k)
    merged: dict[int, FusedCandidate] = {}

    def _absorb(ranked: Ranked, weight: float, field_rank: str, field_score: str) -> None:
        if weight <= 0.0:
            return
        seen: set[int] = set()
        rank = 0
        for chunk_id, raw in ranked:
            cid = int(chunk_id)
            # Одна гілка не має права дати той самий чанк двічі — інакше вона
            # отримує подвійну вагу тихо і безкарно.
            if cid in seen:
                continue
            seen.add(cid)
            rank += 1
            cand = merged.get(cid)
            if cand is None:
                cand = FusedCandidate(chunk_id=cid)
                merged[cid] = cand
            setattr(cand, field_rank, rank)
            setattr(cand, field_score, float(raw))
            cand.rrf_score += weight / (k + rank)

    _absorb(dense, w.dense, "dense_rank", "dense_score")
    _absorb(sparse, w.sparse, "sparse_rank", "sparse_score")
    _absorb(ngram, w.ngram, "ngram_rank", "ngram_score")

    # Детермінований порядок: RRF, далі кількість гілок, що погодились, далі
    # найкращий ранг, далі id. Без останнього тесту не відтворювані.
    return sorted(
        merged.values(),
        key=lambda c: (-c.rrf_score, -c.branches, c.best_rank, c.chunk_id),
    )
