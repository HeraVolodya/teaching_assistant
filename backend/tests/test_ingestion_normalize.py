"""Гейти нормалізації українського тексту.

Кожне твердження тут захищає припущення, розбіжність у якому тихо руйнує BM25:
індекс і запит мусять нормалізуватись байт-у-байт однаково.
"""

from __future__ import annotations

import pytest

from app.embeddings import registry
from app.ingestion.normalize_uk import (
    TEXT_PREPROC_VERSION,
    expand_compounds,
    fix_homoglyphs,
    normalize_uk,
    search_terms,
    tokenize,
)


def test_preproc_version_matches_embedding_registry() -> None:
    """Версія нормалізації входить у `embedding_model_key`.

    Якщо ці дві константи розійдуться, зміна правил нормалізації перестане
    інвалідовувати вектори — і корпус тихо стане невідповідним запитам.
    """
    assert TEXT_PREPROC_VERSION == registry.TEXT_PREPROC_VERSION


# ------------------------------------------------------------------ апострофи
@pytest.mark.parametrize(
    "raw",
    ["п’ять", "пʼять", "п`ять", "п´ять", "п′ять", "п'ять"],
)
def test_all_apostrophe_variants_collapse_to_one(raw: str) -> None:
    """Шість написань того самого слова в PDF мають дати один токен."""
    assert normalize_uk(raw) == "п'ять"


def test_apostrophe_stays_inside_the_token() -> None:
    """Апостроф — частина слова, як і в tokenchars таблиці chunk_fts."""
    assert tokenize(normalize_uk("об’єкт")) == ["об'єкт"]


# ------------------------------------------------------------------ гомогліфи
def test_latin_homoglyphs_inside_cyrillic_word_are_fixed() -> None:
    """OCR підставив латинські a, p, o — для людини невидимо, для BM25 смертельно."""
    broken = "гаpматa"  # «гармата» з латинськими p та a
    assert normalize_uk(broken) == "гармата"


def test_english_words_are_never_touched() -> None:
    """У токені немає кирилиці — виправляти нічого."""
    for word in ("computer", "PDF", "NATO", "accuracy"):
        assert normalize_uk(word) == word.casefold()


def test_mixed_script_token_with_unambiguous_latin_is_left_alone() -> None:
    """«PDF-файл»: `d` і `f` не мають кириличних двійників, тож це не слід OCR.

    Без цього правила «p» перетворилось би на «р» і токен став би «рdf-файл» —
    гірше, ніж було.
    """
    assert normalize_uk("PDF-файл") == "pdf-файл"


def test_uppercase_homoglyphs_are_fixed() -> None:
    """Латинські H, K, M, T, B, C у кириличних абревіатурах і словах."""
    assert normalize_uk("HAРОДНА") == "народна"
    assert fix_homoglyphs("BOГOНЬ") == "ВОГОНЬ"


def test_homoglyph_fix_does_not_cross_segment_boundary() -> None:
    """Сегменти всередині складеного токена аналізуються незалежно."""
    assert normalize_uk("IP-адреса") == "ip-адреса"


# ------------------------------------------------------------ тире й переноси
def test_dash_family_is_unified() -> None:
    for dash in "‐‑‒–—―−":
        assert normalize_uk(f"а{dash}б") == "а-б"


def test_soft_hyphen_is_collapsed() -> None:
    assert normalize_uk("гар­мата") == "гармата"


def test_line_break_hyphenation_is_glued() -> None:
    """«гар-\\nмата» — це одне слово, розірване версткою."""
    assert normalize_uk("гар-\nмата") == "гармата"
    assert normalize_uk("гар‐\n   мата") == "гармата"


# --------------------------------------------------------------- складені слова
def test_compound_word_is_emitted_twice() -> None:
    """Запит «технічний» має знаходити «військово-технічний»."""
    assert search_terms("військово-технічний") == [
        "військово-технічний", "військово", "технічний",
    ]


def test_compound_expansion_skips_short_and_numeric_parts() -> None:
    assert expand_compounds("д-30") == ["д-30"]
    assert expand_compounds("по-перше") == ["по-перше", "перше"]


# ------------------------------------------------------------- інваріанти мови
def test_i_and_y_are_not_merged() -> None:
    """и/і фонемно контрастні: «бити» і «біти» — різні слова."""
    assert normalize_uk("бити") != normalize_uk("біти")


def test_ukrainian_specific_letters_survive() -> None:
    assert normalize_uk("Ґанок їжак Єдність") == "ґанок їжак єдність"


def test_normalization_is_idempotent() -> None:
    """Друге застосування нічого не змінює — інакше індекс і запит розійдуться."""
    raw = "Гар-\nмата Д–30 — п’ять об­єктів, HAРОДНА"
    once = normalize_uk(raw)
    assert normalize_uk(once) == once


def test_empty_input_is_safe() -> None:
    assert normalize_uk("") == ""
    assert search_terms("") == []
