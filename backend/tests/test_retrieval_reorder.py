"""Дворівневе впорядкування контексту.

Дві незалежні властивості, і жодну з них не видно на очі у відповіді — вони
виявляються тільки як «модель чомусь погано відповідає»:

  1. Всередині блоку джерела фрагменти йдуть у ПОРЯДКУ ЧИТАННЯ. Реранкер
     віддає їх за спаданням скора, і якщо подати так, мала модель читає
     означення задом наперед: другий уривок посилається на «зазначене вище»,
     якого вище немає.
  2. Між блоками — U-подібне розміщення. «Lost in the middle»: моделі надійно
     дістають з початку й кінця контексту, і на малих моделях провал у
     середині глибший.
"""

from __future__ import annotations

from app.retrieval.reorder import build_blocks, reorder, u_shaped
from tests.helpers_retrieval import make_retrieved


def _chain_pool() -> list:
    """Три сусідні фрагменти однієї секції, подані в порядку спадання скора."""
    return [
        make_retrieved(chunk_id=3, document_id="A", score=0.90, siblings_index=2, text="третій"),
        make_retrieved(chunk_id=1, document_id="A", score=0.85, siblings_index=0, text="перший"),
        make_retrieved(chunk_id=2, document_id="A", score=0.80, siblings_index=1, text="другий"),
    ]


# ------------------------------------------------------- порядок читання
def test_consecutive_siblings_are_restored_to_reading_order() -> None:
    blocks = build_blocks(_chain_pool())
    assert len(blocks) == 1
    assert [r.chunk.siblings_index for r in blocks[0].items] == [0, 1, 2]


def test_chain_grows_both_backwards_and_forwards_from_the_best_fragment() -> None:
    """Побудова ланцюга жадібна й ДВОНАПРАВЛЕНА.

    Насіння тут — середній фрагмент; якби ланцюг ріс лише вперед, перший
    абзац означення відірвався б в окремий блок і поїхав у інший кінець
    контексту.
    """
    items = [
        make_retrieved(chunk_id=2, document_id="A", score=0.95, siblings_index=1),
        make_retrieved(chunk_id=1, document_id="A", score=0.60, siblings_index=0),
        make_retrieved(chunk_id=3, document_id="A", score=0.55, siblings_index=2),
    ]
    blocks = build_blocks(items)
    assert len(blocks) == 1
    assert [r.chunk.id for r in blocks[0].items] == [1, 2, 3]


def test_gap_in_siblings_breaks_the_chain() -> None:
    """Між фрагментами 0 і 5 у підручнику десятки сторінок.

    Склеїти їх в один блок означало б збрехати моделі про суцільність тексту.
    """
    items = [
        make_retrieved(chunk_id=1, document_id="A", score=0.90, siblings_index=0),
        make_retrieved(chunk_id=2, document_id="A", score=0.80, siblings_index=5),
    ]
    assert len(build_blocks(items)) == 2


def test_same_document_different_sections_are_separate_blocks() -> None:
    items = [
        make_retrieved(
            chunk_id=1, document_id="A", score=0.9, header_path="//Розділ 1//", siblings_index=0
        ),
        make_retrieved(
            chunk_id=2, document_id="A", score=0.8, header_path="//Розділ 4//", siblings_index=1
        ),
    ]
    assert len(build_blocks(items)) == 2


def test_different_documents_never_share_a_block() -> None:
    items = [
        make_retrieved(chunk_id=1, document_id="A", score=0.9, siblings_index=0),
        make_retrieved(chunk_id=2, document_id="B", score=0.8, siblings_index=1),
    ]
    blocks = build_blocks(items)
    assert {b.document_id for b in blocks} == {"A", "B"}


def test_block_score_is_the_best_of_its_fragments() -> None:
    blocks = build_blocks(_chain_pool())
    assert blocks[0].score == 0.90
    assert len(blocks[0]) == 3


# --------------------------------------------------------------- U-форма
def test_u_shape_puts_the_best_first_and_the_second_best_last() -> None:
    items = [
        make_retrieved(chunk_id=i, document_id=f"D{i}", score=1.0 - i * 0.1) for i in range(5)
    ]
    blocks = u_shaped(build_blocks(items))
    order = [b.items[0].chunk.id for b in blocks]
    # 0 — найкращий, 1 — другий, 4 — найгірший: слабкі опиняються посередині.
    assert order == [0, 2, 4, 3, 1]


def test_u_shape_is_stable_for_a_single_block() -> None:
    blocks = u_shaped(build_blocks(_chain_pool()))
    assert len(blocks) == 1


# ------------------------------------------------------ повне впорядкування
def test_reorder_keeps_chains_intact_while_reshuffling_blocks() -> None:
    """U-форма переставляє БЛОКИ, а не фрагменти всередині них."""
    items = [
        *_chain_pool(),
        make_retrieved(chunk_id=10, document_id="B", score=0.95, siblings_index=0),
        make_retrieved(chunk_id=20, document_id="C", score=0.70, siblings_index=0),
    ]
    ordered = reorder(items)
    ids = [r.chunk.id for r in ordered]
    # Ланцюг A (0, 1, 2) лишився суцільним і в порядку читання…
    assert ids.index(1) == ids.index(2) - 1 == ids.index(3) - 2
    # …а блоки переставлені U-подібно: B (0.95) → C (0.70) → A (0.90).
    assert ids[0] == 10
    assert ids[1] == 20
    assert ids[-3:] == [1, 2, 3]


def test_ordinals_match_the_final_order_exactly() -> None:
    """`[n]`, яке бачить модель, мусить відповідати позиції в промпті.

    Розбіжність тут означає, що цитата `[3]` вкаже не на той фрагмент —
    тобто відповідь буде виглядати заземленою, але посилання буде хибним.
    Це найгірший клас помилки в усьому продукті.
    """
    ordered = reorder(_chain_pool())
    assert [r.ordinal_in_prompt for r in ordered] == [1, 2, 3]


def test_ordinals_can_be_left_alone() -> None:
    ordered = reorder(_chain_pool(), assign_ordinals=False)
    assert all(r.ordinal_in_prompt is None for r in ordered)


def test_reorder_preserves_every_fragment() -> None:
    items = [
        *_chain_pool(),
        make_retrieved(chunk_id=10, document_id="B", score=0.95, siblings_index=0),
    ]
    assert {r.chunk.id for r in reorder(items)} == {1, 2, 3, 10}


def test_empty_input() -> None:
    assert reorder([]) == []
