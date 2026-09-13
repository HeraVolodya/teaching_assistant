"""Золотий набір: сховище, імпорт/експорт, валідація, півавтоматична побудова."""

from __future__ import annotations

import json

import pytest

from app.domain import Chunk, ChunkLevel
from app.eval import gold_set as G
from helpers_retrieval import build_corpus


# ------------------------------------------------------------------ значення
def test_gold_page_parses_both_forms():
    assert G.GoldPage.parse(147) == G.GoldPage(page=147)
    assert G.GoldPage.parse("147") == G.GoldPage(page=147)
    parsed = G.GoldPage.parse("abc123def456:147")
    assert parsed.page == 147 and parsed.document_id == "abc123def456"
    assert G.GoldPage.parse({"page": 5, "document_id": "d"}).page == 5


def test_gold_page_rejects_nonsense_with_ukrainian_message():
    with pytest.raises(ValueError, match="золоту сторінку"):
        G.GoldPage.parse("сторінка сто сорок сім")


def test_question_without_gold_chunks_is_marked_unanswerable():
    """Порожній список — це окремий свідомий клас, а не незаповнений рядок."""
    q = G.GoldQuestion(collection_id="c", question="Питання без відповіді?")
    assert not q.answerable
    assert q.id  # ідентифікатор проставляється сам


def test_question_id_is_stable_when_given():
    q = G.GoldQuestion(collection_id="c", question="Так?", id="fixed")
    assert q.id == "fixed"


# --------------------------------------------------------------- імпорт/експорт
def test_csv_round_trip_preserves_everything(tmp_path):
    questions = [
        G.GoldQuestion(
            collection_id="col",
            question="Яка дальність стрільби гаубиці Д-30, м?",
            gold_chunk_uids=["uid1", "uid2"],
            gold_pages=[G.GoldPage(page=147, document_id="doc1")],
            note="перевірено",
        ),
        G.GoldQuestion(collection_id="col", question="Питання без відповіді?"),
    ]
    path = G.save_csv(questions, tmp_path / "gold.csv")
    restored = G.load_csv(path)

    assert [q.question for q in restored] == [q.question for q in questions]
    assert restored[0].gold_chunk_uids == ["uid1", "uid2"]
    assert restored[0].gold_pages[0].page == 147
    assert restored[0].id == questions[0].id
    assert not restored[1].answerable


def test_csv_uses_semicolon_so_commas_in_questions_survive(tmp_path):
    """Кома в українському питанні — норма; роздільник списків має бути іншим."""
    q = G.GoldQuestion(
        collection_id="c",
        question="Що таке деривація, і як її враховують?",
        gold_chunk_uids=["a", "b"],
    )
    path = G.save_csv([q], tmp_path / "g.csv")
    assert ";" in path.read_text(encoding="utf-8-sig")
    assert G.load_csv(path)[0].question == q.question


def test_csv_with_excel_bom_is_read_correctly(tmp_path):
    path = tmp_path / "excel.csv"
    path.write_text(
        "﻿question,gold_chunk_uids,note\nЩо таке деривація?,uid1,\n", encoding="utf-8"
    )
    loaded = G.load_csv(path, collection_id="c")
    assert loaded[0].question == "Що таке деривація?"
    assert loaded[0].gold_chunk_uids == ["uid1"]


def test_json_round_trip(tmp_path):
    q = G.GoldQuestion(collection_id="c", question="Питання?", gold_chunk_uids=["u"])
    path = G.save_json([q], tmp_path / "gold.json")
    assert json.loads(path.read_text(encoding="utf-8"))[0]["question"] == "Питання?"
    assert G.load_json(path)[0].gold_chunk_uids == ["u"]


# ------------------------------------------------------------------ сховище
def test_repo_upsert_and_qrels(tmp_path):
    corpus = build_corpus(tmp_path)
    with corpus.db.transaction() as con:
        repo = G.EvalQuestionRepo(con)
        q = G.GoldQuestion(
            collection_id=corpus.collection_id, question="Що таке деривація?",
            gold_chunk_uids=["uid-a"], gold_pages=[G.GoldPage(3, "doc")],
        )
        repo.upsert_many([q])
        q.question = "Що таке деривація снаряда?"
        repo.upsert_many([q])          # той самий id — оновлення, не дубль

    with corpus.db.connection() as con:
        repo = G.EvalQuestionRepo(con)
        assert repo.count(corpus.collection_id) == 1
        stored = repo.get(q.id)
        assert stored is not None
        assert stored.question.endswith("снаряда?")
        assert stored.gold_pages[0].page == 3
        assert repo.qrels(corpus.collection_id) == {q.id: {"uid-a"}}


def test_repo_clear_only_touches_its_collection(tmp_path):
    corpus = build_corpus(tmp_path)
    other = "інша-колекція"
    with corpus.db.transaction() as con:
        # Колекції з таким id немає, тому пишемо лише в реальну.
        G.EvalQuestionRepo(con).upsert_many([
            G.GoldQuestion(collection_id=corpus.collection_id, question="А?"),
            G.GoldQuestion(collection_id=corpus.collection_id, question="Б?"),
        ])
    with corpus.db.transaction() as con:
        G.EvalQuestionRepo(con).clear(other)
    with corpus.db.connection() as con:
        assert G.EvalQuestionRepo(con).count(corpus.collection_id) == 2


# ----------------------------------------------------------------- валідація
def test_validate_accepts_real_uids(tmp_path):
    corpus = build_corpus(tmp_path)
    with corpus.db.connection() as con:
        chunk_id = corpus.leaves("Основи балістики")[0]
        uid = con.execute("SELECT chunk_uid FROM chunks WHERE id=?", (chunk_id,)).fetchone()[0]
        questions = [
            G.GoldQuestion(collection_id=corpus.collection_id, question="Що таке деривація?",
                           gold_chunk_uids=[uid])
        ]
        report = G.validate(con, questions, require_minimum=False)
    assert report.ok, report.format_uk()


def test_validate_reports_stale_uid_after_reindex(tmp_path):
    """Найважливіша перевірка: мітка, яку більше не вдається розв'язати.

    `chunk_uid` — це blake2b від тексту, тож зміна тексту при переіндексації
    робить стару мітку недійсною. Тихо зарахувати це як промах означало б
    написати у звіт неправду.
    """
    corpus = build_corpus(tmp_path)
    with corpus.db.connection() as con:
        report = G.validate(
            con,
            [G.GoldQuestion(collection_id=corpus.collection_id, question="Що?",
                            gold_chunk_uids=["0" * 32])],
            require_minimum=False,
        )
    kinds = {i.kind for i in report.issues}
    assert "stale_uid" in kinds
    assert not report.ok
    assert "blake2b" in report.format_uk()


def test_validate_rejects_uid_from_another_collection(tmp_path):
    from helpers_retrieval import CORPUS, add_collection

    corpus = build_corpus(tmp_path)
    other_id = add_collection(corpus, name="Інженерна", specs=CORPUS[:1])
    with corpus.db.connection() as con:
        foreign_uid = con.execute(
            "SELECT chunk_uid FROM chunks WHERE collection_id=? AND level='L2' LIMIT 1",
            (other_id,),
        ).fetchone()[0]
        report = G.validate(
            con,
            [G.GoldQuestion(collection_id=corpus.collection_id, question="Що?",
                            gold_chunk_uids=[foreign_uid])],
            require_minimum=False,
        )
    assert "foreign_uid" in {i.kind for i in report.issues}


def test_validate_rejects_gold_on_a_section_parent(tmp_path):
    """Розмічати треба листок: індексується лише L2."""
    corpus = build_corpus(tmp_path)
    with corpus.db.connection() as con:
        parent_uid = con.execute(
            "SELECT chunk_uid FROM chunks WHERE id=?",
            (corpus.parent_ids["Основи балістики"],),
        ).fetchone()[0]
        report = G.validate(
            con,
            [G.GoldQuestion(collection_id=corpus.collection_id, question="Що?",
                            gold_chunk_uids=[parent_uid])],
            require_minimum=False,
        )
    assert "not_leaf" in {i.kind for i in report.issues}


def test_validate_flags_duplicates_and_too_few_questions(tmp_path):
    corpus = build_corpus(tmp_path)
    with corpus.db.connection() as con:
        report = G.validate(
            con,
            [
                G.GoldQuestion(collection_id=corpus.collection_id, question="Що таке деривація?"),
                G.GoldQuestion(collection_id=corpus.collection_id, question="що  таке   ДЕРИВАЦІЯ?"),
                G.GoldQuestion(collection_id=corpus.collection_id, question=""),
            ],
            require_minimum=True,
        )
    kinds = {i.kind for i in report.issues}
    assert {"duplicate", "empty_question", "no_gold", "too_few"} <= kinds
    assert report.unanswerable == 3
    assert str(G.MIN_CALIBRATION_QUESTIONS) in report.format_uk()


# ------------------------------------------- півавтоматична побудова набору
def test_sample_chunks_is_deterministic_and_covers_every_document(tmp_path):
    corpus = build_corpus(tmp_path)
    with corpus.db.connection() as con:
        first = G.sample_chunks(con, corpus.collection_id, count=4, min_chars=40)
        second = G.sample_chunks(con, corpus.collection_id, count=4, min_chars=40)

    assert [c.chunk_uid for c in first] == [c.chunk_uid for c in second]
    # Кругова видача: перші 4 фрагменти мають бути з 4 різних документів.
    assert len({c.document_id for c in first}) == 4


def test_sample_chunks_skips_short_and_non_leaf(tmp_path):
    corpus = build_corpus(tmp_path)
    with corpus.db.connection() as con:
        picked = G.sample_chunks(con, corpus.collection_id, count=50, min_chars=40)
    assert picked
    assert all(c.level is ChunkLevel.LEAF for c in picked)
    assert all(len(c.display_text) >= 40 for c in picked)


class _QuestionBackend:
    """Фейковий бекенд, що поводиться як модель, яка вміє ставити питання."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompts: list[str] = []

    async def chat_stream(self, messages, *, params=None, model=None, **_):
        self.prompts.append(messages[-1]["content"])
        for token in self.reply.split(" "):
            yield token + " "


async def test_draft_questions_uses_model_output_when_it_is_a_question(tmp_path):
    corpus = build_corpus(tmp_path)
    with corpus.db.connection() as con:
        chunks = G.sample_chunks(con, corpus.collection_id, count=2, min_chars=40)

    backend = _QuestionBackend("Яка максимальна дальність стрільби гаубиці Д-30?")
    drafts = await G.draft_questions(backend, chunks)

    assert len(drafts) == 2
    assert all(d.source == "llm" for d in drafts)
    assert all(d.question.endswith("?") for d in drafts)
    assert all(d.needs_review for d in drafts)
    assert "Матеріал:" in backend.prompts[0]


async def test_draft_questions_falls_back_to_template_on_refusal(tmp_path):
    """Заглушка LLM відповідає відмовою — набір усе одно мусить будуватись."""
    from app.backends.stub_backend import StubBackend

    corpus = build_corpus(tmp_path)
    with corpus.db.connection() as con:
        chunks = G.sample_chunks(con, corpus.collection_id, count=2, min_chars=40)

    drafts = await G.draft_questions(StubBackend(), chunks)
    assert all(d.source == "шаблон" for d in drafts)
    assert all(d.question.endswith("?") for d in drafts)
    assert all("недостатньо інформації" not in d.question for d in drafts)


async def test_draft_questions_survives_dead_backend():
    class _Dead:
        async def chat_stream(self, *a, **k):
            raise RuntimeError("LM Studio не відповідає")
            yield ""  # pragma: no cover

    chunk = Chunk(
        document_id="d", collection_id="c", ordinal=0, level=ChunkLevel.LEAF,
        display_text="Деривація снаряда — це відхилення від площини стрільби.",
        header_path="//Розділ 1//Балістика//",
    )
    drafts = await G.draft_questions(_Dead(), [chunk])
    assert drafts[0].source == "шаблон"
    assert "Балістика" in drafts[0].question


def test_approve_drafts_keeps_only_selected_and_carries_the_gold_label():
    drafts = [
        G.QuestionDraft(chunk_uid="u1", question="Питання один?", document_id="d1",
                        document_title="Підручник", page_label="с. 147", excerpt="…"),
        G.QuestionDraft(chunk_uid="u2", question="Питання два?", document_id="d1",
                        document_title="Підручник", page_label="с. 148", excerpt="…"),
    ]
    approved = G.approve_drafts(drafts, "col", approved_uids=["u2"])
    assert len(approved) == 1
    assert approved[0].gold_chunk_uids == ["u2"]
    assert approved[0].gold_pages[0].page == 148
    assert approved[0].collection_id == "col"

    assert len(G.approve_drafts(drafts, "col")) == 2      # None → затверджено все
