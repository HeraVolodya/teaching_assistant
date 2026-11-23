"""Правила моделі під кожну форму файлу — і заглушка, що їх розуміє.

Заглушка (`ASISTENT_STUB=1`) — несуча конструкція: CI й машина без моделі
проходять увесь шлях «прохання → файл». Для цього вона мусить упізнати форму
з тих самих маркерів, що стоять у правилах, — розбіжність рядків тихо
перетворила б презентацію в заглушці на суцільний абзац.
"""

from __future__ import annotations

from app.backends import stub_backend
from app.backends.stub_backend import compose_stub_answer
from app.export.intent import DocumentRequest
from app.export.outline import parse_markdown
from app.export.tasks import SHAPE_MARKERS, document_task, shape_of
from app.generation.prompt_builder import build_prompt_bundle
from tests.helpers_llm import evidence_two_documents


def test_stub_knows_every_shape_marker() -> None:
    assert {marker: shape for shape, marker in SHAPE_MARKERS.items()} == stub_backend._DOCUMENT_SHAPES


def test_rules_carry_the_marker_and_start_like_tier_b() -> None:
    """«ПРАВИЛА:» на початку — заглушка ріже по ньому хвіст промпту від тіла
    останнього джерела."""
    for request in (DocumentRequest("pptx", "т"), DocumentRequest("docx", "т"),
                    DocumentRequest("xlsx", "т", table=True), DocumentRequest("docx", "т", table=True)):
        task = document_task(request)
        assert task.rules.startswith("ПРАВИЛА:")
        assert SHAPE_MARKERS[shape_of(request)] in task.rules
        assert "[n]" in task.rules
        # Модель має знати, що файл збирає застосунок: інакше вона відповідала
        # «я не можу створити файл PPTX, бо генерую лише текст».
        assert "збере застосунок" in task.rules


def test_slide_count_and_budget() -> None:
    five = document_task(DocumentRequest("pptx", "т", slides=5))
    assert "рівно 5 слайдів" in five.rules
    assert "5–7 слайдів" in document_task(DocumentRequest("pptx", "т")).rules
    twelve = document_task(DocumentRequest("pptx", "т", slides=12))
    assert five.max_tokens < twelve.max_tokens <= 2900
    # Презентація просить таблицю (де доречно) і питання для самоперевірки.
    assert "таблицею" in five.rules and "Питання для самоперевірки" in five.rules
    assert "таблицею" in document_task(DocumentRequest("docx", "т")).rules


def test_table_in_word_is_asked_as_a_table() -> None:
    assert shape_of(DocumentRequest("docx", "т", table=True)) == "table"
    assert shape_of(DocumentRequest("docx", "т")) == "text"


def _stub_outline(request: DocumentRequest):
    bundle = build_prompt_bundle("Зроби файл про деривацію", evidence_two_documents(),
                                 task=document_task(request))
    return parse_markdown(compose_stub_answer(bundle.messages))


def test_stub_returns_slides_for_a_presentation() -> None:
    outline = _stub_outline(DocumentRequest("pptx", "деривація"))
    assert outline.title.startswith("Матеріали за запитом")
    assert [b.kind for b in outline.blocks][:2] == ["heading", "bullets"]
    assert "[1]" in outline.blocks[1].items[0]
    # Тіло останнього джерела не захопило правил і повтору питання.
    assert all("ПРАВИЛА" not in item for b in outline.blocks for item in b.items)


def test_stub_returns_a_table_for_a_table() -> None:
    outline = _stub_outline(DocumentRequest("xlsx", "деривація", table=True))
    table = outline.blocks[0]
    assert table.kind == "table" and table.header == ["Положення", "Джерело"]
    assert [row[1] for row in table.rows] == ["[1]", "[2]", "[3]", "[4]"]


def test_stub_returns_sections_for_a_document() -> None:
    outline = _stub_outline(DocumentRequest("docx", "деривація"))
    kinds = [b.kind for b in outline.blocks]
    assert kinds[:2] == ["heading", "paragraph"] and outline.has_content
