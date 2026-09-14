"""Промпт: тир A (правила) і тир B (сендвіч).

Три твердження, які мусять лишатись правдою після будь-якого рефакторингу:
  * правила 1-5 — байт-ідентичний префікс для ВСІХ асистентів;
  * системний промпт до 180 токенів, персона до 250;
  * сендвіч містить питання ДВІЧІ, а блок правил стоїть ПІСЛЯ доказів.
"""

from __future__ import annotations

from app.domain import AssistantConfig
from app.generation.prompt_builder import (
    PERSONA_MAX_TOKENS,
    SYSTEM_MAX_TOKENS,
    SYSTEM_RULES_UK,
    abstain_text,
    build_map_prompt,
    build_prompt,
    build_prompt_bundle,
    build_reduce_prompt,
    build_system_prompt,
    estimate_tokens_uk,
    fit_evidence,
    format_evidence,
    generation_params,
)
from tests.helpers_llm import evidence_two_documents, make_retrieved

QUESTION = "Як враховують поправку на деривацію?"


# ------------------------------------------------------------------- тир A
def test_system_prompt_has_exactly_five_rules() -> None:
    numbered = [ln for ln in SYSTEM_RULES_UK.splitlines() if ln[:2] in
                ("1.", "2.", "3.", "4.", "5.")]
    assert len(numbered) == 5


def test_system_prompt_fits_180_tokens() -> None:
    """Успішність виконання ВСІХ інструкцій падає експоненційно з їх
    кількістю. 2500-токенний промпт NeoLens локальна 12B виконує погано."""
    assert estimate_tokens_uk(SYSTEM_RULES_UK) <= SYSTEM_MAX_TOKENS


def test_rules_prefix_is_byte_identical_across_assistants() -> None:
    """Префіксний KV-кеш перевикористовується лише при БАЙТОВОМУ збігу."""
    a = build_system_prompt("Асистент з балістики. Відповідай стисло.")
    b = build_system_prompt("Асистент з тактики. Наводь приклади.")
    assert a.startswith(SYSTEM_RULES_UK)
    assert b.startswith(SYSTEM_RULES_UK)
    assert a[:len(SYSTEM_RULES_UK)].encode() == b[:len(SYSTEM_RULES_UK)].encode()


def test_persona_is_hard_trimmed_to_250_tokens() -> None:
    persona = "Дуже докладна інструкція асистента. " * 200
    prompt = build_system_prompt(persona)
    persona_part = prompt.split("РОЛЬ АСИСТЕНТА:\n", 1)[1]
    assert estimate_tokens_uk(persona_part) <= PERSONA_MAX_TOKENS + 20
    assert "обрізано" in persona_part


def test_system_prompt_mentions_all_five_themes() -> None:
    lowered = SYSTEM_RULES_UK.lower()
    assert "асистент" in lowered                    # роль
    assert "лише за наданими" in lowered            # заземлення
    assert "[1]" in SYSTEM_RULES_UK                 # формат цитат
    assert "недостатньо" in lowered                 # відмова
    assert "українською" in lowered                 # мова


# ------------------------------------------------------------------- тир B
def test_sandwich_contains_question_twice() -> None:
    """Рівно та розкладка, що виграла UNLP 2026: питання і перед доказами,
    і після них."""
    messages = build_prompt(QUESTION, evidence_two_documents())
    user = messages[1]["content"]
    assert user.count(f"Питання: {QUESTION}") == 2
    assert user.startswith(f"Питання: {QUESTION}")
    assert user.rstrip().endswith("Відповідь:")


def test_rules_block_sits_after_evidence_not_before() -> None:
    """Інструкція перед 2000 токенами доказів для 12B фактично не існує."""
    user = build_prompt(QUESTION, evidence_two_documents())[1]["content"]
    assert user.index("ДЖЕРЕЛА:") < user.index("ПРАВИЛА:")
    assert user.index("ПРАВИЛА:") < user.rindex(f"Питання: {QUESTION}")


def test_tier_b_rules_block_is_40_to_80_tokens() -> None:
    user = build_prompt(QUESTION, evidence_two_documents())[1]["content"]
    rules = user.split("ПРАВИЛА:")[1].split("\n\n")[0]
    assert 30 <= estimate_tokens_uk(rules) <= 80


def test_evidence_uses_ordinals_not_uids() -> None:
    """Модель бачить [n]. 8-hex-схема при 100 тис. чанків має ~69%
    ймовірності колізії, а 32-hex їсть бюджет."""
    evidence = evidence_two_documents()
    block = format_evidence(evidence)
    assert block.startswith("[1] ")
    for rc in evidence:
        assert rc.chunk.chunk_uid not in block
    assert [rc.ordinal_in_prompt for rc in evidence] == [1, 2, 3, 4]


def test_evidence_shows_display_text_not_embed_text() -> None:
    """Генератор бачить санітизоване тіло; контекстний префікс існує для
    пошуку і в промпті лише їв би бюджет."""
    rc = make_retrieved(text="Тіло фрагмента.")
    rc.chunk.context_note = "СЛУЖБОВА КАРТКА ДОКУМЕНТА"
    block = format_evidence([rc])
    assert "Тіло фрагмента." in block
    assert "СЛУЖБОВА КАРТКА" not in block


def test_evidence_header_carries_title_and_pages() -> None:
    block = format_evidence([make_retrieved(page_from=147, page_to=149,
                                            label_from="147", label_to="149")])
    assert "Підручник з балістики" in block
    assert "с. 147–149" in block


def test_citation_map_links_ordinals_to_uids() -> None:
    evidence = evidence_two_documents()
    bundle = build_prompt_bundle(QUESTION, evidence)
    assert set(bundle.citations) == {"1", "2", "3", "4"}
    assert bundle.citations["1"] == evidence[0].chunk.chunk_uid


def test_history_is_included_but_bounded() -> None:
    history = [{"role": "user", "content": "перше питання"},
               {"role": "assistant", "content": "перша відповідь"},
               {"role": "user", "content": "друге питання"},
               {"role": "assistant", "content": "друга відповідь"}]
    user = build_prompt(QUESTION, evidence_two_documents(), history=history)[1]["content"]
    assert "ПОПЕРЕДНІ ХОДИ:" in user
    assert "друге питання" in user
    assert user.index("ПОПЕРЕДНІ ХОДИ:") < user.index("ДЖЕРЕЛА:")


# ------------------------------------------------------------------ бюджет
def test_budget_of_five_chunks_fits_8k_context() -> None:
    """~400 система + ~2150 п'ять чанків + ~150 сендвіч + ~600 відповідь
    ≈ 3300 токенів. Тобто 8k справді достатньо."""
    chunks = [make_retrieved(document_id=f"doc-{i}", ordinal=i, text="слово " * 240)
              for i in range(5)]
    bundle = build_prompt_bundle(QUESTION, chunks)
    assert bundle.estimated_tokens + 600 < 8192


def test_fit_evidence_drops_before_it_truncates() -> None:
    """Порядок поступок: менше діставати → обрізати вікно."""
    chunks = [make_retrieved(document_id=f"doc-{i}", ordinal=i, text="слово " * 900)
              for i in range(8)]
    kept, cap, truncated = fit_evidence(chunks, context_tokens=8192,
                                        max_answer_tokens=600, min_chunks=2)
    assert len(kept) < len(chunks)
    assert kept[0] is chunks[0]          # найкращий за реранкером лишається
    if not truncated:
        assert cap is None


def test_fit_evidence_truncates_when_dropping_is_not_enough() -> None:
    chunks = [make_retrieved(document_id=f"doc-{i}", ordinal=i, text="слово " * 4000)
              for i in range(3)]
    kept, cap, truncated = fit_evidence(chunks, context_tokens=4096,
                                        max_answer_tokens=600, min_chunks=2)
    assert truncated and cap is not None
    assert len(kept) >= 2                # мінімум опорних фрагментів збережено


def test_truncation_is_applied_to_evidence_block() -> None:
    long_chunk = make_retrieved(text="я" * 5000)
    block = format_evidence([long_chunk], char_budget=500)
    assert len(block) < 900
    assert block.endswith("…")


# --------------------------------------------------------------- map-reduce
def test_map_prompt_is_scoped_to_one_document() -> None:
    evidence = evidence_two_documents()[:2]
    user = build_map_prompt(QUESTION, "Балістика та стрільба", evidence)[1]["content"]
    assert "документ «Балістика та стрільба»" in user
    assert "Немає відомостей." in user      # чесний сигнал «тут не покрито»
    assert user.count(f"Питання: {QUESTION}") == 2


def test_reduce_prompt_forbids_inventing_new_markers() -> None:
    user = build_reduce_prompt(QUESTION, [("А", "висновок [1]"), ("Б", "висновок [3]")])[1]["content"]
    assert "[1]" in user and "[3]" in user
    assert "не вигадуй нових номерів" in user
    assert "Збережи всі маркери" in user


# -------------------------------------------------------------- параметри
def test_generation_params_follow_the_plan() -> None:
    p = generation_params(AssistantConfig())
    assert p.temperature == 0.2
    # Бюджет ділиться з ланцюжком міркувань reasoning-моделей, тому він помітно
    # більший за обсяг самої відповіді. Перевіряємо нижню межу, а не точне
    # число: конкретне значення — питання налаштування, а от падіння назад до
    # ~600 знову дало б порожні відповіді на Gemma 4.
    assert p.max_tokens >= 2000
    assert 1.05 <= p.repeat_penalty <= 1.10
    assert p.context_overflow_policy == "stopAtLimit"
    assert "<end_of_turn>" in p.stop
    # «Детальна» відповідь мусить мати БІЛЬШИЙ бюджет за звичайну, а не константу.
    detailed = generation_params(AssistantConfig(), detailed=True).max_tokens
    assert detailed == p.max_tokens * 2


def test_abstain_text_is_honest_and_shows_nearest() -> None:
    text = abstain_text(QUESTION, evidence_two_documents())
    assert "недостатньо інформації" in text
    assert "Балістика та стрільба" in text
