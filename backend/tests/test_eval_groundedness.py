"""Заземленість: суддя, відмови, дійсність цитат, попарне порівняння."""

from __future__ import annotations

import pytest

from app.eval import gold_set as G
from app.eval import groundedness as GR
from helpers_retrieval import build_corpus

EVIDENCE = [
    GR.Evidence(
        ordinal=1,
        title="Методична розробка: гаубиця Д-30",
        page_label="с. 12",
        text=(
            "Гаубиця Д-30 калібру 122 мм має максимальну дальність стрільби "
            "15300 метрів осколково-фугасним снарядом."
        ),
    ),
    GR.Evidence(
        ordinal=2,
        title="Основи балістики",
        page_label="с. 47",
        text="Деривація снаряда — це відхилення снаряда від площини стрільби.",
    ),
]


# ------------------------------------------------------- розбиття на твердження
def test_split_claims_does_not_break_on_ukrainian_abbreviations():
    text = (
        "Дальність становить 15300 м [1]. Див. табл. 3 на с. 12 для поправок. "
        "Деривація — це відхилення снаряда [2]."
    )
    claims = GR.split_claims(text)
    assert len(claims) == 3
    assert "табл. 3" in claims[1] and "с. 12" in claims[1]


def test_split_claims_ignores_fragments_and_empty_text():
    assert GR.split_claims("") == []
    assert GR.split_claims("Так.") == []          # коротше за min_chars
    assert len(GR.split_claims("Це достатньо довге твердження про деривацію.")) == 1


# ------------------------------------------------------------ лексичний суддя
async def test_lexical_judge_supports_a_grounded_claim():
    judge = GR.LexicalJudge()
    verdict = await judge.verdict(
        "Дальність стрільби гаубиці Д-30 становить 15300 метрів [1].", EVIDENCE
    )
    assert verdict.verdict == "ТАК"
    assert verdict.supported
    assert verdict.cited_ordinals == (1,)


async def test_lexical_judge_catches_an_invented_claim():
    """Суддя мусить ловити невідповідність — інакше він не суддя."""
    judge = GR.LexicalJudge()
    verdict = await judge.verdict(
        "Розрахунок гаубиці проходить щорічну атестацію в навчальному центрі.", EVIDENCE
    )
    assert verdict.verdict == "НІ"
    assert not verdict.supported
    assert verdict.score == 0.0


async def test_lexical_judge_catches_a_wrong_number_despite_high_word_overlap():
    """Найдорожча помилка в артилерійському тексті — саме число."""
    judge = GR.LexicalJudge()
    verdict = await judge.verdict(
        "Гаубиця Д-30 калібру 122 мм має максимальну дальність стрільби 18000 метрів.",
        EVIDENCE,
    )
    assert verdict.verdict == "ЧАСТКОВО"
    assert not verdict.supported
    assert "18000" in verdict.raw


# ---------------------------------------------------------------- LLM-суддя
class _ScriptedBackend:
    """Бекенд, що повертає наперед задані відповіді по черзі."""

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []

    async def chat_stream(self, messages, *, params=None, model=None, **_):
        self.prompts.append(messages[-1]["content"])
        reply = self.replies.pop(0) if self.replies else "ТАК"
        for ch in reply:
            yield ch


async def test_llm_judge_parses_the_three_verdicts():
    backend = _ScriptedBackend("ТАК", "НІ", "ЧАСТКОВО")
    judge = GR.LlmJudge(backend)

    yes = await judge.verdict("Твердження перше про дальність.", EVIDENCE)
    no = await judge.verdict("Твердження друге про дальність.", EVIDENCE)
    partly = await judge.verdict("Твердження третє про дальність.", EVIDENCE)

    assert (yes.verdict, yes.supported, yes.score) == ("ТАК", True, 1.0)
    assert (no.verdict, no.supported, no.score) == ("НІ", False, 0.0)
    assert (partly.verdict, partly.supported, partly.score) == ("ЧАСТКОВО", False, 0.5)
    assert all(j.method == "llm" for j in (yes, no, partly))


async def test_llm_judge_prompt_carries_evidence_and_strips_markers():
    backend = _ScriptedBackend("ТАК")
    judge = GR.LlmJudge(backend)
    await judge.verdict("Дальність 15300 метрів [1].", EVIDENCE)

    prompt = backend.prompts[0]
    assert "ФРАГМЕНТИ:" in prompt and "[1] Методична розробка" in prompt
    # Маркер [1] у самому твердженні судді не потрібен — він оцінює зміст.
    assert "Дальність 15300 метрів ." in prompt or "Дальність 15300 метрів." in prompt


async def test_llm_judge_falls_back_when_the_model_writes_an_essay():
    """12B-модель час від часу пише есе замість слова — це не привід зупинятись."""
    backend = _ScriptedBackend("Загалом можна сказати, що питання складне")
    judge = GR.LlmJudge(backend)
    verdict = await judge.verdict("Деривація — це відхилення снаряда.", EVIDENCE)
    assert "не розпарсено" in verdict.method
    assert verdict.verdict in ("ТАК", "ЧАСТКОВО", "НІ")


async def test_llm_judge_falls_back_when_the_backend_dies():
    class _Dead:
        async def chat_stream(self, *a, **k):
            raise RuntimeError("LM Studio не відповідає")
            yield ""  # pragma: no cover

    verdict = await GR.LlmJudge(_Dead()).verdict("Деривація — це відхилення снаряда.", EVIDENCE)
    assert "фолбек" in verdict.method


def test_create_judge_uses_the_lexical_one_in_stub_mode():
    """У stub-режимі заглушка LLM переказує подані фрагменти, тож питати її
    «чи випливає це з фрагментів» означало б завжди отримувати «так»."""
    from app.backends.stub_backend import StubBackend

    assert isinstance(GR.create_judge(StubBackend(), stub=True), GR.LexicalJudge)
    assert isinstance(GR.create_judge(None, stub=False), GR.LexicalJudge)
    assert isinstance(GR.create_judge(StubBackend(), stub=False), GR.LlmJudge)


async def test_judge_answer_aggregates_and_flags_uncited_claims():
    backend = _ScriptedBackend("ТАК", "НІ")
    report = await GR.judge_answer(
        GR.LlmJudge(backend),
        "Яка дальність Д-30?",
        "Дальність становить 15300 метрів [1]. Гармата випускається з 1963 року.",
        EVIDENCE,
    )
    assert report.total == 2
    assert report.supported == 1
    assert report.groundedness == pytest.approx(0.5)
    assert report.uncited_claims == ["Гармата випускається з 1963 року."]
    assert "НЕ ПІДТВЕРДЖЕНО" in report.format_uk()


async def test_summarize_groundedness_over_several_answers():
    judge = GR.LexicalJudge()
    reports = [
        await GR.judge_answer(judge, "q1", "Деривація — це відхилення снаряда [2].", EVIDENCE),
        await GR.judge_answer(judge, "q2", "Гармата бере участь у параді щороку.", EVIDENCE),
    ]
    summary = GR.summarize_groundedness(reports)
    assert summary["answers"] == 2
    assert summary["claims"] == 2
    assert summary["supported"] == 1
    assert summary["methods"] == ["лексичний"]


# ------------------------------------------------------------------- відмови
def test_abstention_stats_counts_both_kinds_of_error():
    questions = [
        G.GoldQuestion(collection_id="c", question="є?", gold_chunk_uids=["u1"], id="a1"),
        G.GoldQuestion(collection_id="c", question="є2?", gold_chunk_uids=["u2"], id="a2"),
        G.GoldQuestion(collection_id="c", question="немає?", id="n1"),
        G.GoldQuestion(collection_id="c", question="немає2?", id="n2"),
    ]
    report = GR.abstention_stats(questions, {"a1": False, "a2": True, "n1": True, "n2": False})

    assert report.correct_refusal_rate == pytest.approx(0.5)
    assert report.false_refusal_rate == pytest.approx(0.5)
    assert report.balanced_accuracy == pytest.approx(0.5)
    assert report.details["a2"] == "хибна відмова"
    assert "Коректних відмов" in report.format_uk()


def test_abstention_stats_punishes_the_always_abstaining_system():
    """Система, що мовчить завжди, має ідеальну частку коректних відмов —
    і саме тому в звіт іде збалансована точність, а не вона."""
    questions = [
        G.GoldQuestion(collection_id="c", question="є?", gold_chunk_uids=["u"], id="a1"),
        G.GoldQuestion(collection_id="c", question="немає?", id="n1"),
    ]
    silent = GR.abstention_stats(questions, {"a1": True, "n1": True})
    assert silent.correct_refusal_rate == 1.0
    assert silent.balanced_accuracy == pytest.approx(0.5)


def test_missing_record_counts_as_answered():
    questions = [G.GoldQuestion(collection_id="c", question="немає?", id="n1")]
    report = GR.abstention_stats(questions, {})
    assert report.unanswerable_refused == 0
    assert report.details["n1"].startswith("відповідь там")


# ------------------------------------------------------------ дійсність цитат
def _seed_chat(corpus, rows):
    """Записати повідомлення асистента й нерозв'язані маркери."""
    from app.db.repositories import ChatRepo, TelemetryRepo

    with corpus.db.transaction() as con:
        chat = ChatRepo(con)
        telemetry = TelemetryRepo(con)
        session = chat.create_session(corpus.assistant_id, "оцінювання")
        for content, mapping, unresolved, abstained in rows:
            message_id = chat.add_message(
                session, "assistant", content, citation_map=mapping, abstained=abstained
            )
            for marker in unresolved:
                telemetry.unresolved_citation(message_id, marker, ["1", "2"])
        return session


def test_citation_validity_counts_hallucinated_markers(tmp_path):
    corpus = build_corpus(tmp_path)
    session = _seed_chat(
        corpus,
        [
            ("Дальність 15300 м [1]. Деривація [2].", {"1": "uidA", "2": "uidB"}, [], False),
            ("Відповідь із вигаданим посиланням [1].", {"1": "uidA"}, ["[42]"], False),
        ],
    )
    with corpus.db.connection() as con:
        validity = GR.citation_validity(con, session_id=session)

    assert validity.messages == 2
    assert validity.emitted == 4          # 3 дійсні + 1 нерозв'язаний
    assert validity.unresolved == 1
    assert validity.valid_ratio == pytest.approx(0.75)
    assert validity.hallucinated_ratio == pytest.approx(0.25)
    assert validity.messages_with_unresolved == 1
    assert "Дійсних цитат" in validity.format_uk()


def test_citation_validity_ignores_abstentions_by_default(tmp_path):
    corpus = build_corpus(tmp_path)
    session = _seed_chat(
        corpus,
        [
            ("Відповідь із цитатою [1].", {"1": "uidA"}, [], False),
            ("У матеріалах немає інформації.", {}, [], True),
        ],
    )
    with corpus.db.connection() as con:
        default = GR.citation_validity(con, session_id=session)
        with_abstained = GR.citation_validity(con, session_id=session, include_abstained=True)

    assert default.messages == 1 and default.uncited_messages == 0
    assert with_abstained.messages == 2 and with_abstained.uncited_messages == 1


def test_citation_validity_flags_markers_missing_from_the_map(tmp_path):
    """Маркер у тексті без запису в мапі — це збій нумерації промпту,
    і він лікується кодом, а не порогом, тому рахується окремо."""
    corpus = build_corpus(tmp_path)
    session = _seed_chat(corpus, [("Двоє джерел [1][3].", {"1": "uidA"}, [], False)])
    with corpus.db.connection() as con:
        validity = GR.citation_validity(con, session_id=session)
    assert validity.unmapped_markers == 1
    assert validity.unresolved == 0


def test_citation_validity_on_empty_history_is_nan_not_crash(tmp_path):
    corpus = build_corpus(tmp_path)
    with corpus.db.connection() as con:
        validity = GR.citation_validity(con)
    assert validity.messages == 0
    assert validity.emitted == 0


# ------------------------------------------------------- попарне порівняння
async def test_pairwise_compare_requires_a_mirrored_verdict():
    """Позиційне упередження судді має читатись як нічия, а не як перемога."""
    consistent = await GR.pairwise_compare(
        _ScriptedBackend("А", "Б"), "питання?", "відповідь А", "відповідь Б"
    )
    assert consistent.winner == "a" and consistent.consistent

    biased = await GR.pairwise_compare(
        _ScriptedBackend("А", "А"), "питання?", "відповідь А", "відповідь Б"
    )
    assert biased.winner == "tie"
    assert not biased.consistent


async def test_pairwise_compare_detects_the_second_candidate():
    verdict = await GR.pairwise_compare(
        _ScriptedBackend("Б", "А"), "питання?", "відповідь А", "відповідь Б"
    )
    assert verdict.winner == "b"


async def test_pairwise_compare_swaps_the_order_in_the_second_call():
    backend = _ScriptedBackend("А", "Б")
    await GR.pairwise_compare(backend, "питання?", "ПЕРША", "ДРУГА")
    first, second = backend.prompts
    assert first.index("ПЕРША") < first.index("ДРУГА")
    assert second.index("ДРУГА") < second.index("ПЕРША")


def test_evidence_from_retrieved_numbers_from_one(tmp_path):
    from app.db.repositories import ChunkRepo
    from app.domain import RetrievedChunk

    corpus = build_corpus(tmp_path)
    with corpus.db.connection() as con:
        chunks = ChunkRepo(con).by_ids(corpus.leaves("Основи балістики")[:2])
    items = [RetrievedChunk(chunk=c, document_title="Основи балістики") for c in chunks]
    evidence = GR.Evidence.from_retrieved(items)

    assert [e.ordinal for e in evidence] == [1, 2]
    assert evidence[0].page_label.startswith("с.")
    assert evidence[0].as_block().startswith("[1] Основи балістики")
