"""Гібридний пошук: екранування FTS5, метаданні фільтри, повний конвеєр.

Усе проганяється у stub-режимі (`create_provider(stub=True)` + ехо-реранкер),
без жодної завантаженої моделі. Заглушка ембедингів дає косинус, приблизно
рівний лексичному перекриттю, тому тести перевіряють справжній конвеєр, а не
власні моки.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from app.domain import AssistantConfig, ChunkLevel
from app.rerank.reranker import create_reranker
from app.retrieval.hybrid import (
    EXACT_SELECTIVITY_THRESHOLD,
    HybridRetriever,
    MetadataFilter,
    analyze_query,
    build_match_expression,
    char_ngram_scores,
    fts_term,
)
from app.retrieval.vector_index import build_collection_index
from tests.helpers_retrieval import Corpus, DocSpec, add_collection, build_corpus


@pytest.fixture()
def corpus(tmp_path: Path) -> Corpus:
    c = build_corpus(tmp_path)
    build_collection_index(c.db, c.collection_id, index_dir=c.index_dir)
    return c


@pytest.fixture()
def retriever(corpus: Corpus):
    r = HybridRetriever(
        corpus.db,
        provider=corpus.provider,
        index_dir=corpus.index_dir,
        reranker=create_reranker(stub=True),
    )
    yield r
    r.close()


def _cfg(**kwargs) -> AssistantConfig:
    return AssistantConfig(**kwargs)


# ============================================================ екранування FTS5
def test_fts_term_wraps_in_double_quotes() -> None:
    assert fts_term("Д-30") == '"Д-30"'


def test_fts_term_doubles_embedded_quotes() -> None:
    """Лапки всередині терміна мусять подвоюватись, інакше вираз обривається."""
    assert fts_term('він сказав "стій"') == '"він сказав ""стій"""'


def test_unescaped_designation_is_parsed_as_not(tmp_path: Path) -> None:
    """Доказ, заради якого існує `fts_term`.

    Голий `Д-30` FTS5 читає як `Д NOT 30` і кидає `no such column: 30`.
    Запит викладача «ТТХ Д-30» без екранування — це 500-та помилка в чаті.
    """
    migration = Path(__file__).resolve().parents[1] / "app" / "db" / "migrations" / "0001_initial.sql"
    con = sqlite3.connect(tmp_path / "fts.db")
    con.executescript(migration.read_text(encoding="utf-8"))
    con.execute("INSERT INTO chunk_fts(rowid, lemmas, forms, codes) VALUES (1,'гаубиця','','Д-30')")

    with pytest.raises(sqlite3.OperationalError):
        con.execute("SELECT rowid FROM chunk_fts WHERE chunk_fts MATCH 'Д-30'").fetchall()

    rows = con.execute(
        "SELECT rowid FROM chunk_fts WHERE chunk_fts MATCH ?", (fts_term("Д-30"),)
    ).fetchall()
    assert [r[0] for r in rows] == [1]


def test_match_expression_is_or_of_escaped_terms() -> None:
    assert build_match_expression(["Д-30", "гаубиця"]) == '"Д-30" OR "гаубиця"'


def test_match_expression_drops_duplicates_and_stubs() -> None:
    """Односимвольні токени лише роздувають вираз, нічого не додаючи."""
    assert build_match_expression(["гармата", "гармата", "я", " "]) == '"гармата"'


def test_empty_match_expression() -> None:
    assert build_match_expression([]) == ""


# ============================================================== аналіз запиту
def test_query_analysis_extracts_designations_verbatim() -> None:
    """Позначення беруться з СИРОГО запиту.

    Casefold і нормалізація тире змінили б «Д-30», а воно мусить лишитись
    побайтово тим, що написав викладач: у `codes` індексуються саме такі рядки.
    """
    terms = analyze_query("ТТХ гаубиці Д-30")
    assert "Д-30" in terms.codes
    assert terms.anchored is True
    assert terms.normalized == terms.normalized.casefold()


def test_query_analysis_keeps_apostrophe_words_whole() -> None:
    terms = analyze_query("п'ять об'єктів спостереження")
    assert "п'ять" in terms.forms


def test_match_terms_merge_lemmas_forms_and_codes_without_repeats() -> None:
    terms = analyze_query("дальність стрільби Д-30")
    assert len(terms.match_terms) == len(set(terms.match_terms))


# ================================================= символьна n-грамна гілка
def test_char_ngrams_beat_word_matching_on_ukrainian_morphology() -> None:
    """«гаубиці» і «гаубиця» — різні словесні токени й майже ті самі 4-грами.

    Саме на цьому команда УКУ виміряла +21% відносних для символьного TF-IDF
    проти словесного; гілка існує рівно заради цієї властивості.
    """
    ranked = char_ngram_scores(
        "гаубиці",
        {1: "гаубиця д-30 калібру 122 мм", 2: "автомобіль вантажний бортовий"},
        top_k=2,
    )
    assert ranked[0][0] == 1


def test_char_ngrams_survive_a_typo() -> None:
    ranked = char_ngram_scores(
        "деривацвя снаряда",   # друкарська помилка
        {1: "деривація снаряда це відхилення", 2: "кут підвищення гармати"},
        top_k=2,
    )
    assert ranked[0][0] == 1


def test_char_ngrams_on_empty_pool() -> None:
    assert char_ngram_scores("запит", {}, top_k=5) == []


# ========================================================= метаданий фільтр
def test_empty_filter_is_empty() -> None:
    assert MetadataFilter().is_empty() is True
    assert MetadataFilter(languages=("uk",)).is_empty() is False


def test_filter_sql_qualifies_columns_with_the_alias() -> None:
    where, params = MetadataFilter(doc_types=("textbook",), year_from=2018).sql()
    assert "f.doc_type IN (?)" in where
    assert "f.year >= ?" in where
    assert params == ["textbook", 2018]


def test_language_filter_returns_only_that_language(retriever, corpus: Corpus) -> None:
    final, debug = retriever.retrieve(
        "projectile drift", _cfg(), corpus.collection_id,
        filters=MetadataFilter(languages=("en",)),
    )
    assert final
    assert {r.chunk.language for r in final} == {"en"}
    assert debug.candidates[0]["selectivity"] == pytest.approx(2 / corpus.leaf_count())


def test_document_filter_restricts_to_one_textbook(retriever, corpus: Corpus) -> None:
    """«Шукай лише в цій методичці» — базовий сценарій викладача."""
    wanted = corpus.doc("Методична розробка: гаубиця Д-30")
    final, _ = retriever.retrieve(
        "поправка на деривацію", _cfg(), corpus.collection_id,
        filters=MetadataFilter(document_ids=(wanted,)),
    )
    assert final
    assert {r.chunk.document_id for r in final} == {wanted}


def test_doc_type_filter(retriever, corpus: Corpus) -> None:
    final, _ = retriever.retrieve(
        "таблиці стрільби", _cfg(), corpus.collection_id,
        filters=MetadataFilter(doc_types=("methodical",)),
    )
    assert final
    assert {r.chunk.document_id for r in final} == {
        corpus.doc("Методична розробка: гаубиця Д-30")
    }


def test_year_range_filter(retriever, corpus: Corpus) -> None:
    final, _ = retriever.retrieve(
        "снаряд", _cfg(), corpus.collection_id,
        filters=MetadataFilter(year_from=2021, year_to=2021),
    )
    assert final
    assert {r.chunk.document_id for r in final} == {
        corpus.doc("Методична розробка: гаубиця Д-30")
    }


def test_filters_combine_conjunctively(retriever, corpus: Corpus) -> None:
    """Українська методичка 2015 року не існує → порожньо, а не «щось схоже»."""
    final, debug = retriever.retrieve(
        "снаряд", _cfg(), corpus.collection_id,
        filters=MetadataFilter(languages=("uk",), years=(2015,)),
    )
    assert final == []
    assert debug.abstained is True
    assert debug.abstain_confidence == 0.0


def test_narrow_filter_switches_to_exact_scan(retriever, corpus: Corpus) -> None:
    """Селективність < 0.15 → рівень L2 стратегії: точний скан, не ANN.

    На вузькому фільтрі обхід HNSW вироджується в блукання серед відкинутих
    сусідів: overfetch росте швидше за економію від графа.
    """
    lecture = corpus.doc("Конспект лекцій з балістики")
    final, debug = retriever.retrieve(
        "деривація", _cfg(), corpus.collection_id,
        filters=MetadataFilter(document_ids=(lecture,)),
    )
    run = debug.candidates[0]
    assert run["selectivity"] < EXACT_SELECTIVITY_THRESHOLD
    assert run["filter_strategy"] == "L2"
    assert [r.chunk.document_id for r in final] == [lecture]


def test_broad_filter_uses_ann_with_overfetch(retriever, corpus: Corpus) -> None:
    _, debug = retriever.retrieve(
        "деривація", _cfg(), corpus.collection_id,
        filters=MetadataFilter(languages=("uk",)),
    )
    assert debug.candidates[0]["filter_strategy"] == "L1"


def test_no_filter_uses_the_physical_partition(retriever, corpus: Corpus) -> None:
    """Одна колекція = один файл індексу: фільтр «свій/чужий» коштує нуль."""
    _, debug = retriever.retrieve("деривація", _cfg(), corpus.collection_id)
    run = debug.candidates[0]
    assert run["filter_strategy"] == "L0"
    assert run["selectivity"] == pytest.approx(1.0)


# ================================================================ конвеєр
def test_end_to_end_finds_the_right_fragment(retriever, corpus: Corpus) -> None:
    final, debug = retriever.retrieve(
        "Яка поправка на деривацію для Д-30?", _cfg(), corpus.collection_id
    )
    assert final
    assert debug.abstained is False
    assert "Д-30" in final[0].chunk.display_text
    assert final[0].chunk.document_id == corpus.doc("Методична розробка: гаубиця Д-30")


def test_all_three_branches_contribute(retriever, corpus: Corpus) -> None:
    _, debug = retriever.retrieve("дальність стрільби гаубиці", _cfg(), corpus.collection_id)
    assert debug.dense_count > 0
    assert debug.sparse_count > 0
    assert debug.ngram_count > 0
    assert debug.fused_count > 0


def test_designation_query_survives_fts_escaping_end_to_end(retriever, corpus: Corpus) -> None:
    """Той самий «ТТХ Д-30», але через увесь конвеєр, а не через `fts_term`."""
    final, debug = retriever.retrieve("ТТХ Д-30", _cfg(), corpus.collection_id)
    assert debug.sparse_count > 0
    assert final
    assert debug.candidates[0]["weights"]["anchored"] is True
    assert debug.candidates[0]["weights"]["sparse"] == pytest.approx(1.0)


def test_citations_carry_page_labels(retriever, corpus: Corpus) -> None:
    """Без номера сторінки цитата не виконує вимогу НДР."""
    final, _ = retriever.retrieve("деривація снаряда", _cfg(), corpus.collection_id)
    assert all(r.chunk.citation_label().startswith("с. ") for r in final)


def test_ordinals_are_assigned_for_the_prompt(retriever, corpus: Corpus) -> None:
    """Модель бачить `[n]`, а не UUID (контракт, правило 7)."""
    final, _ = retriever.retrieve("деривація снаряда", _cfg(), corpus.collection_id)
    assert [r.ordinal_in_prompt for r in final] == list(range(1, len(final) + 1))


def test_verbatim_duplicate_from_the_lecture_notes_is_dropped(
    retriever, corpus: Corpus
) -> None:
    """Конспект несе дослівний абзац підручника — він не має з'їдати слот."""
    _, debug = retriever.retrieve(
        "Деривація снаряда — це відхилення снаряда від площини стрільби",
        _cfg(final_top_k=5),
        corpus.collection_id,
    )
    dropped = debug.candidates[0]["duplicates_dropped"]
    assert dropped, "дослівний дублікат мав бути виявлений"
    assert dropped[0]["hamming"] == 0


def test_no_document_takes_more_than_its_quota(retriever, corpus: Corpus) -> None:
    """Наскрізна перевірка вимоги «поєднувати декілька джерел»."""
    final, debug = retriever.retrieve(
        "таблиці стрільби гармати снаряд дальність деривація",
        _cfg(final_top_k=5, max_per_document=2, relative_score_floor=0.0),
        corpus.collection_id,
    )
    counts: dict[str, int] = {}
    for item in final:
        counts[item.chunk.document_id] = counts.get(item.chunk.document_id, 0) + 1
    assert debug.distinct_documents >= 2
    assert max(counts.values()) <= 3   # квота 2 + добір проходу 2


def test_off_topic_question_abstains(retriever, corpus: Corpus) -> None:
    """Поріг виконується В КОДІ, а не проханням у промпті (контракт, правило 8).

    Промптом неможливо надійно змусити 12B-модель відмовитись; порогом — можна.
    """
    final, debug = retriever.retrieve(
        "рецепт борщу з пампушками", _cfg(confidence_required="high"), corpus.collection_id
    )
    assert debug.abstained is True
    assert debug.abstain_confidence is not None
    assert debug.abstain_confidence < 0.55
    # Фрагменти все одно повертаються — екран «Чому ця відповідь» має показати,
    # що саме система знайшла і чому визнала це недостатнім.
    assert isinstance(final, list)


def test_confidence_level_moves_the_abstention_threshold(retriever, corpus: Corpus) -> None:
    query = "поправка на деривацію"
    _, low = retriever.retrieve(query, _cfg(confidence_required="low"), corpus.collection_id)
    _, high = retriever.retrieve(query, _cfg(confidence_required="high"), corpus.collection_id)
    assert low.abstain_confidence == pytest.approx(high.abstain_confidence)
    assert low.abstained is False


def test_pipeline_works_without_a_reranker(corpus: Corpus) -> None:
    """CI і stub-режим мусять працювати без ваг реранкера взагалі.

    Тут перевіряється саме те, на чому конвеєр колись мовчки утримувався
    завжди: впевненість і лічильник опорних фрагментів мусять жити на ОДНІЙ
    шкалі. RRF-скор першого місця — це ~1/61 ≈ 0.016, тобто нижче будь-якого
    з порогів 0.15 / 0.35 / 0.55.
    """
    plain = HybridRetriever(corpus.db, provider=corpus.provider, index_dir=corpus.index_dir)
    try:
        final, debug = plain.retrieve(
            "Деривація снаряда — це відхилення снаряда від площини стрільби",
            _cfg(), corpus.collection_id,
        )
        assert final
        assert debug.abstained is False
        assert debug.abstain_confidence > 0.35
    finally:
        plain.close()


def test_automerge_lifts_two_neighbouring_leaves_to_their_parent(
    retriever, corpus: Corpus
) -> None:
    """«Retrieve small, show medium»: два сусідні листки → одна секція L1.

    Показувати моделі два шматки з діркою посередині гірше, ніж показати
    секцію цілком; один фрагмент так НЕ розширюється — це роздуло б контекст
    удвічі без приросту.
    """
    final, debug = retriever.retrieve(
        "деривація снаряда таблиці стрільби гармати заряду",
        _cfg(final_top_k=5, relative_score_floor=0.0),
        corpus.collection_id,
    )
    levels = {r.chunk.level for r in final}
    assert ChunkLevel.SECTION in levels
    parents = {r.chunk.id for r in final if r.chunk.level is ChunkLevel.SECTION}
    assert parents <= set(corpus.parent_ids.values())
    # Листки, злиті в батька, лишаються поясненими на екрані «Чому ця відповідь».
    merged = [c for c in debug.candidates if c.get("merged_into") in parents]
    assert len(merged) >= 2
    assert all(c["selected"] for c in merged)


def test_context_budget_is_respected(retriever, corpus: Corpus) -> None:
    """Бюджет у СИМВОЛАХ: токен-номінований бюджет змінюється вдвічі при зміні
    генератора (українська fertility 2.16–3.90 ток/слово)."""
    final, _ = retriever.retrieve(
        "деривація снаряда", _cfg(final_top_k=5, relative_score_floor=0.0),
        corpus.collection_id, char_budget=200,
    )
    assert len(final) == 1   # перший фрагмент проходить завжди, решта — ні


def test_debug_reports_every_stage(retriever, corpus: Corpus) -> None:
    """`RetrievalDebug` живить екран «Чому ця відповідь» — він не декоративний."""
    final, debug = retriever.retrieve("поправка на деривацію", _cfg(), corpus.collection_id)
    assert debug.query == "поправка на деривацію"
    assert debug.reranked_count > 0
    assert debug.final_count == len(final)
    assert debug.distinct_documents == len({r.chunk.document_id for r in final})
    assert set(debug.latency_ms) >= {
        "analyze", "filter", "dense", "sparse", "ngram", "fuse", "dedup", "rerank", "select",
    }
    # Кожен кандидат, що дійшов до відповіді, позначений як обраний — інакше
    # екран «Чому ця відповідь» показував би фрагменти без зв'язку з цитатами.
    selected = [c for c in debug.candidates if c.get("selected")]
    assert selected
    assert {c["chunk_uid"] for c in selected} <= {r.chunk.chunk_uid for r in final}
    assert debug.to_json()


def test_unknown_collection_is_a_loud_error(retriever) -> None:
    with pytest.raises(KeyError, match="не знайдено"):
        retriever.retrieve("запит", _cfg(), "немає-такої")


def test_assistants_are_isolated_physically(corpus: Corpus) -> None:
    """Фрагмент чужого асистента НЕ МАЄ ПРАВА протекти у відповідь.

    Ізоляція тут фізична — один файл індексу на колекцію (рівень L0), — а не
    метаданий фільтр. Тому вона коштує нуль і не втрачає recall: у другій
    колекції лежить дослівно та сама тема, і все одно жоден її фрагмент не
    з'являється у відповіді першої.
    """
    secret = DocSpec(
        title="Таємна методичка іншої кафедри",
        doc_type="methodical",
        language="uk",
        year=2024,
        bodies=(
            "Поправка на деривацію для гаубиці Д-30 у таємних таблицях стрільби "
            "визначається за окремою методикою.",
        ),
    )
    other_id = add_collection(corpus, name="Інша кафедра", specs=(secret,))
    build_collection_index(corpus.db, corpus.collection_id, index_dir=corpus.index_dir)
    build_collection_index(corpus.db, other_id, index_dir=corpus.index_dir)

    forbidden = corpus.doc("Таємна методичка іншої кафедри")
    r = HybridRetriever(
        corpus.db, provider=corpus.provider, index_dir=corpus.index_dir,
        reranker=create_reranker(stub=True),
    )
    try:
        final, debug = r.retrieve(
            "поправка на деривацію для Д-30", _cfg(relative_score_floor=0.0),
            corpus.collection_id,
        )
        assert final
        assert forbidden not in {item.chunk.document_id for item in final}
        assert all(c.get("chunk_uid") is None or forbidden not in str(c) for c in debug.candidates)

        # …і симетрично: друга колекція бачить лише своє.
        other_final, _ = r.retrieve(
            "поправка на деривацію для Д-30", _cfg(), other_id
        )
        assert {item.chunk.document_id for item in other_final} == {forbidden}
    finally:
        r.close()


def test_retriever_works_before_the_index_is_built(tmp_path: Path) -> None:
    """Документ щойно завантажили, воркер ще будує граф — чат мусить відповідати."""
    corpus = build_corpus(tmp_path)   # без build_collection_index
    r = HybridRetriever(
        corpus.db, provider=corpus.provider, index_dir=corpus.index_dir,
        reranker=create_reranker(stub=True),
    )
    try:
        final, debug = r.retrieve("гаубиця Д-30 дальність", _cfg(), corpus.collection_id)
        assert final
        assert debug.dense_count > 0
    finally:
        r.close()


def test_retrieval_never_pulls_torch_into_the_api_process() -> None:
    """Пошук живе в API-процесі, тож його імпорти — це холодний старт.

    `import torch` коштує 1.5–4 с на Windows (контракт, правило 1). Лематизація
    підтягується лениво саме тому: модуль приймання важчий, а перший запит
    може почекати 100 мс, старт застосунку — ні.
    """
    import subprocess
    import sys as _sys

    code = (
        "import app.retrieval.hybrid, app.retrieval.vector_index, sys;"
        "print(any(m in sys.modules for m in ('torch', 'docling', 'rapidocr')))"
    )
    out = subprocess.run(
        [_sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "False"
