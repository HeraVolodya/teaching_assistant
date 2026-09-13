"""Механічна перевірка цитат — єдиний доступний детектор галюцинацій.

`logprobs` у LM Studio не реалізовано, тому перплексійної оцінки немає.
Лишається саме це: кожен [n] мусить існувати в множині доказів.
"""

from __future__ import annotations

from app.domain import Citation
from app.generation.citations import (
    extract_ordinals,
    merge_page_ranges,
    parse_citations,
    quote_for,
    record_unresolved,
)
from tests.helpers_llm import evidence_two_documents, make_retrieved


class FakeTelemetry:
    """Мінімальний двійник TelemetryRepo — модуль цитат не має знати про БД."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, str, list[str]]] = []

    def unresolved_citation(self, message_id: str, emitted: str, available) -> None:
        self.rows.append((message_id, emitted, list(available)))


def evidence():
    ev = evidence_two_documents()
    for i, rc in enumerate(ev, start=1):
        rc.ordinal_in_prompt = i
    return ev


# ------------------------------------------------------------------ розбір
def test_resolves_markers_and_keeps_order_of_appearance() -> None:
    text = "Деривація зростає з дальністю [3]. Поправку беруть із таблиць [1]."
    cleaned, citations, unresolved = parse_citations(text, evidence())
    assert not unresolved
    assert [c.ordinal for c in citations] == [3, 1]
    assert citations[0].document_title == "Правила стрільби та управління вогнем"
    assert cleaned == text


def test_marker_outside_evidence_is_removed_and_reported() -> None:
    """Головна перевірка модуля: [7] не існує → його немає у відповіді, і
    факт записано в телеметрію разом зі списком ДОСТУПНИХ номерів."""
    text = "Перше твердження [1]. Друге твердження [7]. Третє [2]."
    cleaned, citations, unresolved = parse_citations(text, evidence())

    assert unresolved == ["[7]"]
    assert "[7]" not in cleaned
    assert "[1]" in cleaned and "[2]" in cleaned
    assert [c.ordinal for c in citations] == [1, 2]
    # Після видалення маркера не лишається «Друге твердження ." »
    assert "твердження ." not in cleaned


def test_unresolved_is_logged_with_available_set() -> None:
    telemetry = FakeTelemetry()
    _, _, unresolved = parse_citations("Твердження [9] і [1].", evidence())
    written = record_unresolved(telemetry, "msg-1", unresolved, ["1", "2", "3", "4"])
    assert written == 1
    assert telemetry.rows == [("msg-1", "[9]", ["1", "2", "3", "4"])]


def test_comma_form_is_accepted() -> None:
    """Мала модель пише [1, 2] — карати її за кому дорожче, ніж розібрати."""
    cleaned, citations, unresolved = parse_citations("Твердження [1, 2].", evidence())
    assert not unresolved
    assert [c.ordinal for c in citations] == [1, 2]
    assert "[1][2]" in cleaned


def test_partially_invalid_group_keeps_valid_part() -> None:
    cleaned, citations, unresolved = parse_citations("Твердження [2,99].", evidence())
    assert unresolved == ["[99]"]
    assert [c.ordinal for c in citations] == [2]
    assert "[2]" in cleaned and "99" not in cleaned


def test_answer_without_markers_yields_no_citations() -> None:
    cleaned, citations, unresolved = parse_citations("Відповідь без посилань.", evidence())
    assert citations == [] and unresolved == []
    assert cleaned == "Відповідь без посилань."


def test_citation_carries_pages_bboxes_and_quote() -> None:
    _, citations, _ = parse_citations("Твердження [1].", evidence())
    c = citations[0]
    assert c.page_from == 147 and c.page_to == 148
    assert c.page_label == "с. 147–148"
    assert c.bboxes and c.bboxes[0]["page"] == 147
    assert c.quote
    assert len(c.chunk_uid) == 32          # стабільний контентний ідентифікатор


def test_extract_ordinals_dedups_in_order() -> None:
    assert extract_ordinals("[2] текст [1] ще [2] і [3, 1]") == [2, 1, 3]


def test_quote_is_cut_on_sentence_boundary() -> None:
    text = "Перше речення. " + "Довге друге речення. " * 40
    q = quote_for(text, limit=60)
    assert len(q) <= 61
    assert q.endswith((".", "…"))


# ---------------------------------------------------- злиття діапазонів
def test_adjacent_page_ranges_of_same_file_merge() -> None:
    """с. 10-12 + с. 13-15 → с. 10-15. Викладач бачить одне посилання."""
    a = make_retrieved(document_id="d", ordinal=1, page_from=10, page_to=12)
    b = make_retrieved(document_id="d", ordinal=2, page_from=13, page_to=15)
    a.ordinal_in_prompt, b.ordinal_in_prompt = 1, 2
    _, citations, _ = parse_citations("Твердження [1] і [2].", [a, b])

    merged = merge_page_ranges(citations)
    assert len(merged) == 1
    assert (merged[0].page_from, merged[0].page_to) == (10, 15)
    assert merged[0].page_label == "с. 10–15"
    assert merged[0].ordinal == 1


def test_non_adjacent_ranges_stay_separate() -> None:
    a = make_retrieved(document_id="d", ordinal=1, page_from=10, page_to=12)
    b = make_retrieved(document_id="d", ordinal=2, page_from=40, page_to=41)
    a.ordinal_in_prompt, b.ordinal_in_prompt = 1, 2
    _, citations, _ = parse_citations("[1] [2]", [a, b])
    assert len(merge_page_ranges(citations)) == 2


def test_different_documents_never_merge() -> None:
    a = make_retrieved(document_id="d1", title="Книга А", page_from=10, page_to=12)
    b = make_retrieved(document_id="d2", title="Книга Б", page_from=13, page_to=15)
    a.ordinal_in_prompt, b.ordinal_in_prompt = 1, 2
    _, citations, _ = parse_citations("[1] [2]", [a, b])
    merged = merge_page_ranges(citations)
    assert len(merged) == 2
    assert {c.document_title for c in merged} == {"Книга А", "Книга Б"}


def test_merge_prefers_printed_labels_over_physical_indices() -> None:
    """Викладач шукає «с. 147», надруковану внизу сторінки, а не 152-гу
    сторінку файлу."""
    a = make_retrieved(document_id="d", page_from=152, page_to=153,
                       label_from="147", label_to="148")
    b = make_retrieved(document_id="d", page_from=154, page_to=155,
                       label_from="149", label_to="150")
    a.ordinal_in_prompt, b.ordinal_in_prompt = 1, 2
    _, citations, _ = parse_citations("[1] [2]", [a, b])
    assert merge_page_ranges(citations)[0].page_label == "с. 147–150"


def test_merge_is_stable_for_empty_input() -> None:
    assert merge_page_ranges([]) == []


def test_merge_tolerates_missing_pages() -> None:
    c = Citation(ordinal=1, chunk_uid="u", document_id="d", document_title="Т",
                 page_from=None, page_to=None, page_label="", quote="")
    assert merge_page_ranges([c]) == [c]
