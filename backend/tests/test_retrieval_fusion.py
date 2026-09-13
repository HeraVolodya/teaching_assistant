"""Зважений RRF: арифметика, лексичні якорі, збереження сирих скорів.

Кожне твердження тут закриває конкретну тиху поломку. Найтихіша з них —
втрата сирих скорів гілок: RRF-скор першого місця однаковий і для косинуса
0.87, і для 0.11, тож без сирих скорів коротке замикання «немає контексту»
неможливе за побудовою, і система починає впевнено відповідати на порожньому
контексті.
"""

from __future__ import annotations

import pytest

from app.retrieval.fusion import (
    ANCHORED_SPARSE_WEIGHT,
    DEFAULT_RRF_K,
    FusionWeights,
    has_lexical_anchor,
    resolve_weights,
    weighted_rrf,
)


# ------------------------------------------------------------------ арифметика
def test_rrf_contribution_is_weight_over_k_plus_rank() -> None:
    """Формула має бути рівно `w / (k + rank)`, ранг 1-based."""
    fused = weighted_rrf(
        dense=[(10, 0.9), (11, 0.8)],
        weights=FusionWeights(dense=1.0, sparse=0.0, ngram=0.0, k=60),
    )
    assert fused[0].chunk_id == 10
    assert fused[0].rrf_score == pytest.approx(1.0 / 61)
    assert fused[1].rrf_score == pytest.approx(1.0 / 62)


def test_branch_weights_are_applied() -> None:
    """Гілка з меншою вагою не має перебивати гілку з більшою на тому ж ранзі."""
    fused = weighted_rrf(
        dense=[(1, 0.9)],
        sparse=[(2, 12.0)],
        ngram=[(3, 0.5)],
        weights=FusionWeights(dense=1.0, sparse=0.6, ngram=0.4, k=60),
    )
    assert [c.chunk_id for c in fused] == [1, 2, 3]
    assert fused[0].rrf_score == pytest.approx(1.0 / 61)
    assert fused[1].rrf_score == pytest.approx(0.6 / 61)
    assert fused[2].rrf_score == pytest.approx(0.4 / 61)


def test_agreement_of_branches_beats_a_single_confident_branch() -> None:
    """Два обережні голоси мають важити більше за один упевнений.

    Це і є сенс RRF: `k=60` робить різницю між рангами 1 і 2 мізерною (~1.6%),
    тому гілка, що впевнено помиляється, не перекриває дві, що обережно праві.
    """
    fused = weighted_rrf(
        dense=[(1, 0.99), (2, 0.30)],
        sparse=[(2, 9.0)],
        ngram=[(2, 0.7)],
        weights=FusionWeights(dense=1.0, sparse=0.6, ngram=0.4, k=60),
    )
    assert fused[0].chunk_id == 2
    assert fused[0].branches == 3


def test_raw_scores_survive_fusion() -> None:
    """Сирі скори гілок несуться поруч — на них тримається утримання."""
    fused = weighted_rrf(dense=[(7, 0.87)], sparse=[(7, 11.25)], ngram=[(7, 0.42)])
    cand = fused[0]
    assert cand.dense_score == pytest.approx(0.87)
    assert cand.sparse_score == pytest.approx(11.25)
    assert cand.ngram_score == pytest.approx(0.42)
    assert cand.dense_rank == cand.sparse_rank == cand.ngram_rank == 1


def test_zero_weight_branch_is_ignored_entirely() -> None:
    """Вимкнена гілка не має додавати кандидатів — інакше вага 0 не вимикає її."""
    fused = weighted_rrf(
        dense=[(1, 0.5)], sparse=[(2, 5.0)], weights=FusionWeights(sparse=0.0)
    )
    assert [c.chunk_id for c in fused] == [1]


def test_duplicate_ids_within_one_branch_do_not_double_count() -> None:
    """Гілка, що двічі віддала той самий чанк, не має отримати подвійну вагу."""
    fused = weighted_rrf(dense=[(5, 0.9), (5, 0.8), (6, 0.7)])
    assert [c.chunk_id for c in fused] == [5, 6]
    assert fused[0].rrf_score == pytest.approx(1.0 / 61)
    assert fused[1].rrf_score == pytest.approx(1.0 / 62)


def test_order_is_deterministic_on_ties() -> None:
    """Нічия розв'язується стабільно — інакше тести й діагностика крихкі."""
    first = weighted_rrf(dense=[(9, 0.5)], sparse=[(4, 1.0)], ngram=[(7, 0.1)])
    second = weighted_rrf(ngram=[(7, 0.1)], sparse=[(4, 1.0)], dense=[(9, 0.5)])
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]


def test_empty_input_gives_empty_output() -> None:
    assert weighted_rrf() == []


# ------------------------------------------------------------- лексичний якір
@pytest.mark.parametrize(
    "query",
    [
        "ТТХ Д-30",
        "дальність 15300 метрів",
        "2С1 Гвоздика",
        'що таке "деривація"',
        "оформлення згідно ДСТУ 3008:2015",
        "вимоги ГОСТ",
        "стандарт STANAG 4082",
        "снаряд 9М113",
    ],
)
def test_lexical_anchors_are_detected(query: str) -> None:
    """Позначення, цифри, лапки й назви стандартів мусять знаходитись ТОЧНО.

    «Д-30» і «Д-20» лежать в ембединговому просторі поруч, а це різні гармати:
    саме тут семантична близькість шкодить, і саме тому вага sparse підіймається.
    """
    assert has_lexical_anchor(query) is True


@pytest.mark.parametrize(
    "query",
    [
        "як обчислюється поправка на деривацію",
        "чим гаубиця відрізняється від гармати",
        "п'ять способів визначення кута підвищення",
        "об'єкт спостереження",
        "",
    ],
)
def test_plain_ukrainian_questions_are_not_anchored(query: str) -> None:
    """Апостроф НЕ є якорем.

    Якби апостроф потрапив у список лапок, лексичним якорем став би майже
    кожен український запит («п'ять», «об'єкт»), і dense-гілку було б назавжди
    придушено. Це найдорожча помилка в цьому файлі.
    """
    assert has_lexical_anchor(query) is False


def test_anchor_raises_sparse_weight_to_one() -> None:
    weights = resolve_weights("ТТХ Д-30", weight_sparse=0.6)
    assert weights.anchored is True
    assert weights.sparse == pytest.approx(ANCHORED_SPARSE_WEIGHT)
    assert weights.dense == pytest.approx(1.0)


def test_no_anchor_keeps_configured_sparse_weight() -> None:
    weights = resolve_weights("що таке деривація", weight_sparse=0.6)
    assert weights.anchored is False
    assert weights.sparse == pytest.approx(0.6)
    assert weights.k == DEFAULT_RRF_K


def test_autoraise_never_lowers_a_higher_configured_weight() -> None:
    """Якщо harness уже підняв sparse вище 1.0, автопідйом не має його зрізати."""
    weights = resolve_weights("Д-30", weight_sparse=1.4)
    assert weights.sparse == pytest.approx(1.4)
