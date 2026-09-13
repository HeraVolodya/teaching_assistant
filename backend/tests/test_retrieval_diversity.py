"""Дедуплікація й диверсифікація джерел.

Головне твердження файлу — один документ НЕ МАЄ ПРАВА монополізувати відповідь.
Це не естетика: вимога НДР — «поєднувати декілька джерел», а на FictionalQA
жанрово різноманітні докази дають +17…+47 п.п. проти дублікатів і
перефразувань, причому ефект обернений до розміру моделі (+24.0% на 1B,
+11.2% на 12B при k=5). Ми запускаємо 12B локально, тобто перебуваємо саме в
тій частині кривої, де диверсифікація коштує нуль обчислень і дає найбільше.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.retrieval.diversity import (
    deduplicate,
    distinct_documents,
    diversify,
    hamming64,
    relative_floor_value,
    simhash_similarity,
)
from tests.helpers_retrieval import make_retrieved


# ------------------------------------------------------------------- SimHash
def test_hamming_counts_differing_bits() -> None:
    assert hamming64(0b1011, 0b1001) == 1
    assert hamming64(0, 0) == 0
    assert simhash_similarity(0, 0) == pytest.approx(1.0)


def test_hamming_handles_signed_sqlite_integers() -> None:
    """SQLite INTEGER знакове, тож SimHash повертається з БД від'ємним.

    Наївне порівняння без маски дало б для двох близьких хешів відстань
    десятків бітів і мовчки вимкнуло б дедуплікацію саме на тих чанках, де
    старший біт увімкнено — тобто на половині корпусу.
    """
    a = -1                      # усі 64 біти = 1
    b = (1 << 64) - 2           # ті самі біти, крім наймолодшого
    assert hamming64(a, b) == 1


# ------------------------------------------------------------ дедуплікація
def test_identical_paragraph_from_second_textbook_is_dropped() -> None:
    """Дослівний абзац із конспекта не має з'їдати слот у топ-5."""
    kept, dropped = deduplicate(
        [
            make_retrieved(chunk_id=1, document_id="A", score=0.9, simhash=0xABCDEF),
            make_retrieved(chunk_id=2, document_id="B", score=0.8, simhash=0xABCDEF),
        ]
    )
    assert [r.chunk.id for r in kept] == [1]
    assert [d.chunk_id for d in dropped] == [2]
    assert dropped[0].duplicate_of == 1
    assert dropped[0].hamming == 0


def test_cosine_vetoes_a_simhash_false_positive() -> None:
    """SimHash збігся, а вектори — ні: фрагмент лишається.

    Коротким текстам (підписи до рисунків, заголовки таблиць) SimHash дає
    хибнопозитивні збіги, і саме там втратити фрагмент найдорожче.
    """
    vectors = {
        1: np.array([1.0, 0.0], dtype=np.float32),
        2: np.array([0.0, 1.0], dtype=np.float32),
    }
    kept, dropped = deduplicate(
        [
            make_retrieved(chunk_id=1, document_id="A", score=0.9, simhash=7),
            make_retrieved(chunk_id=2, document_id="B", score=0.8, simhash=7),
        ],
        vectors=vectors,
    )
    assert [r.chunk.id for r in kept] == [1, 2]
    assert dropped == []


def test_cosine_confirms_a_real_duplicate() -> None:
    vectors = {
        1: np.array([1.0, 0.0], dtype=np.float32),
        2: np.array([1.0, 0.0], dtype=np.float32),
    }
    kept, dropped = deduplicate(
        [
            make_retrieved(chunk_id=1, document_id="A", score=0.9, simhash=7),
            make_retrieved(chunk_id=2, document_id="B", score=0.8, simhash=7),
        ],
        vectors=vectors,
    )
    assert [r.chunk.id for r in kept] == [1]
    assert dropped[0].cosine == pytest.approx(1.0)


def test_missing_simhash_never_drops_a_chunk() -> None:
    """Мовчазна втрата фрагмента гірша за показаний дублікат."""
    kept, dropped = deduplicate(
        [
            make_retrieved(chunk_id=1, document_id="A", score=0.9, simhash=None),
            make_retrieved(chunk_id=2, document_id="B", score=0.8, simhash=None),
        ]
    )
    assert len(kept) == 2
    assert dropped == []


def test_distant_simhashes_are_kept() -> None:
    """Різні означення того самого поняття треба ПОКАЗАТИ, а не приховати."""
    kept, _ = deduplicate(
        [
            make_retrieved(chunk_id=1, document_id="A", score=0.9, simhash=0x0F0F0F0F0F0F0F0),
            make_retrieved(chunk_id=2, document_id="B", score=0.8, simhash=0x123456789ABCDE),
        ]
    )
    assert len(kept) == 2


# --------------------------------------------------------- відносний поріг
def test_relative_floor_on_positive_best() -> None:
    assert relative_floor_value(0.9, 0.6) == pytest.approx(0.54)


def test_relative_floor_widens_on_negative_best() -> None:
    """На логітах крос-енкодера найкращий скор буває від'ємним.

    Наївне `best * ratio` дало б поріг ВИЩИЙ за сам best (0.6 × −0.8 = −0.48),
    відкинуло б навіть найкращий фрагмент, і система утримувалась би завжди.
    """
    floor = relative_floor_value(-0.8, 0.6)
    assert floor < -0.8
    assert floor == pytest.approx(-0.8 / 0.6)


# ------------------------------------------------------- диверсифікація
def _monopoly_pool() -> list:
    """Товстий підручник A має 6 кандидатів, B і C — по одному, слабшому."""
    items = [make_retrieved(chunk_id=i, document_id="A", score=0.90 - i * 0.01) for i in range(6)]
    items.append(make_retrieved(chunk_id=100, document_id="B", score=0.80))
    items.append(make_retrieved(chunk_id=200, document_id="C", score=0.78))
    return items


def _per_document(selected: list) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in selected:
        counts[item.chunk.document_id] = counts.get(item.chunk.document_id, 0) + 1
    return counts


def test_quota_is_hard_while_the_pool_can_honour_it() -> None:
    """Три товсті підручники, 5 слотів, квота 2 → рівно 2/2/1, три джерела.

    Це базовий випадок вимоги НДР «поєднувати декілька джерел»: доки в пулі є
    з чого вибирати, жоден документ не бере понад квоту, навіть якщо всі його
    фрагменти сильніші за чужі.
    """
    items = [
        make_retrieved(chunk_id=i, document_id=doc, score=score - i * 0.001)
        for doc, score in (("A", 0.90), ("B", 0.70), ("C", 0.65))
        for i in range(6)
    ]
    selected = diversify(items, final_k=5, max_per_document=2)
    counts = _per_document(selected)
    assert len(selected) == 5
    assert max(counts.values()) <= 2
    assert distinct_documents(selected) == 3


def test_one_document_cannot_monopolise_the_answer() -> None:
    """Товстий підручник із 6 кандидатів проти двох тонких джерел.

    Квоту тут фізично неможливо витримати на всі 5 слотів: B і C дають по
    одному фрагменту, тобто під квотою заповнюються лише 4. П'ятий слот добирає
    прохід 2 — але головна властивість зберігається: три різні документи в
    відповіді, і товстий підручник НЕ забирає більшість слотів.
    """
    selected = diversify(_monopoly_pool(), final_k=5, max_per_document=2)
    counts = _per_document(selected)
    assert len(selected) == 5
    assert distinct_documents(selected) >= 3
    assert counts["A"] <= 3


def test_quota_does_not_starve_a_single_source_answer() -> None:
    """Якщо відповідь чесно лежить в одному підручнику — віддати п'ять із нього.

    Квота, що працює проти якості, гірша за відсутність квоти: питання
    «яка формула поправки на деривацію» цілком може мати відповідь лише в
    одному джерелі.
    """
    items = [make_retrieved(chunk_id=i, document_id="A", score=0.90 - i * 0.01) for i in range(6)]
    selected = diversify(items, final_k=5, max_per_document=2, min_distinct_documents=2)
    assert len(selected) == 5
    assert distinct_documents(selected) == 1


def test_relative_floor_keeps_junk_out_of_the_answer() -> None:
    """Диверсифікація не має права тягнути сміття заради різноманіття."""
    items = [
        make_retrieved(chunk_id=1, document_id="A", score=0.90),
        make_retrieved(chunk_id=2, document_id="A", score=0.85),
        make_retrieved(chunk_id=3, document_id="B", score=0.10),  # нижче 0.6 × 0.90
    ]
    selected = diversify(items, final_k=5, max_per_document=2, relative_score_floor=0.6)
    assert [r.chunk.id for r in selected] == [1, 2]


def test_second_source_is_pulled_in_when_it_clears_the_floor() -> None:
    items = [
        make_retrieved(chunk_id=1, document_id="A", score=0.90),
        make_retrieved(chunk_id=2, document_id="A", score=0.85),
        make_retrieved(chunk_id=3, document_id="A", score=0.80),
        make_retrieved(chunk_id=4, document_id="B", score=0.60),
    ]
    selected = diversify(items, final_k=3, max_per_document=2, min_distinct_documents=2)
    assert distinct_documents(selected) == 2
    assert {r.chunk.id for r in selected} == {1, 2, 4}


def test_result_is_ordered_by_score() -> None:
    selected = diversify(_monopoly_pool(), final_k=5, max_per_document=2)
    scores = [r.score for r in selected]
    assert scores == sorted(scores, reverse=True)


def test_empty_input_and_zero_k() -> None:
    assert diversify([], final_k=5) == []
    assert diversify(_monopoly_pool(), final_k=0) == []
