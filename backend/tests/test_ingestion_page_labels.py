"""Гейти резолвера друкованої мітки сторінки.

Docling не моделює друковану мітку зовсім (`PageItem` — це лише
`{size, image, page_no}`), тому весь цей код — власний. Класичний випадок,
який мусить працювати: римська передмова, далі арабське тіло. Без нього
цитата «с. 147» вказує на 152-й аркуш PDF, і викладач не знаходить фрагмент.
"""

from __future__ import annotations

import pytest

from app.ingestion.page_labels import (
    from_roman,
    labels_from_headers,
    labels_from_ranges,
    resolve_page_labels,
    to_alpha,
    to_roman,
)


# ------------------------------------------------------------------ /PageLabels
def test_roman_preface_then_arabic_body() -> None:
    """Дві гілки дерева /PageLabels: {"S":"r"} і {"S":"D","St":1}."""
    labels = labels_from_ranges([(0, {"S": "r"}), (8, {"S": "D", "St": 1})], page_count=14)
    assert labels[1] == "i"
    assert labels[4] == "iv"
    assert labels[8] == "viii"
    # Тіло починається з арабської «1» на дев'ятому ФІЗИЧНОМУ аркуші.
    assert labels[9] == "1"
    assert labels[14] == "6"


def test_prefix_and_start_offset_are_honoured() -> None:
    """Мітки виду «2-1» у методичках зі складеною нумерацією розділів."""
    labels = labels_from_ranges([(0, {"S": "D", "P": "2-", "St": 1})], page_count=3)
    assert labels == {1: "2-1", 2: "2-2", 3: "2-3"}


def test_upper_roman_and_alpha_styles() -> None:
    labels = labels_from_ranges(
        [(0, {"S": "R"}), (2, {"S": "A"}), (4, {"S": "a", "St": 27})], page_count=6
    )
    assert labels[1] == "I"
    assert labels[2] == "II"
    assert labels[3] == "A"
    assert labels[5] == "aa"


def test_range_without_style_emits_prefix_only() -> None:
    """У специфікації PDF відсутній /S означає «лише префікс»."""
    labels = labels_from_ranges([(0, {"P": "обкладинка"})], page_count=1)
    assert labels[1] == "обкладинка"


@pytest.mark.parametrize(
    ("number", "roman"),
    [(1, "i"), (4, "iv"), (9, "ix"), (14, "xiv"), (40, "xl"), (1990, "mcmxc")],
)
def test_roman_round_trip(number: int, roman: str) -> None:
    assert to_roman(number) == roman
    assert from_roman(roman) == number


def test_from_roman_rejects_ordinary_words() -> None:
    """«мім» і «дім» складаються з римських літер, але числами не є."""
    assert from_roman("гармата") is None
    assert from_roman("") is None


def test_alpha_labels_wrap_past_z() -> None:
    assert to_alpha(1) == "a"
    assert to_alpha(26) == "z"
    assert to_alpha(27) == "aa"


# ------------------------------------------------- регресія по колонтитулах
def test_header_regression_finds_constant_offset() -> None:
    """У книзі мітка й фізичний індекс різняться на СТАЛУ величину."""
    headers = {physical: [f"— {physical + 12} —"] for physical in range(1, 21)}
    result = labels_from_headers(headers, page_count=20)
    assert result is not None
    assert result.source == "header_regression"
    assert result.label_for(1) == "13"
    assert result.label_for(20) == "32"


def test_header_regression_survives_noise_below_twenty_percent() -> None:
    """Один «Розділ 3» у колонтитулі не має перевернути всю нумерацію."""
    headers: dict[int, list[str]] = {p: [str(p + 4)] for p in range(1, 21)}
    headers[7] = ["Розділ 3"]
    headers[13] = ["Розділ 3"]
    result = labels_from_headers(headers, page_count=20)
    assert result is not None
    assert result.label_for(10) == "14"


def test_header_regression_is_rejected_when_pages_disagree() -> None:
    """Згода нижче 80% — не мітки сторінок, а випадкові числа в тексті."""
    headers = {p: [str(p * 3)] for p in range(1, 21)}
    assert labels_from_headers(headers, page_count=20) is None


def test_header_regression_needs_enough_samples() -> None:
    """Три випадкові числа згодні між собою значно частіше, ніж хотілося б."""
    headers = {1: ["5"], 2: ["6"]}
    assert labels_from_headers(headers, page_count=20) is None


# ------------------------------------------------------------------ резолвер
def test_resolver_falls_back_to_physical_index() -> None:
    """Фолбек ніколи не порожній: цитата без номера гірша за приблизний номер."""
    result = resolve_page_labels(None, page_count=3)
    assert result.source == "physical"
    assert [result.label_for(i) for i in (1, 2, 3)] == ["1", "2", "3"]


def test_resolver_prefers_headers_over_physical() -> None:
    headers = {p: [str(p + 100)] for p in range(1, 11)}
    result = resolve_page_labels(None, page_count=10, header_texts=headers)
    assert result.source == "header_regression"
    assert result.label_for(1) == "101"


def test_label_for_unknown_page_never_returns_empty() -> None:
    result = resolve_page_labels(None, page_count=2)
    assert result.label_for(99) == "99"
