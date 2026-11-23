"""Збирання .pptx / .docx / .xlsx і перевідкриття тими самими бібліотеками.

Перевідкриття — не формальність: файл, який python-pptx записав, а PowerPoint
не відкриє, виглядає в тестах як успіх. Тому тут читається те, що побачить
викладач: назви слайдів, стилі абзаців, комірки, список джерел.
"""

from __future__ import annotations

import io
from datetime import date

import pytest
from docx import Document
from openpyxl import load_workbook
from pptx import Presentation
from pptx.util import Inches

from app.export.outline import parse_markdown
from app.export.render import MEDIA_TYPES, SourceRef, document_filename, render_document

SOURCES = {
    1: SourceRef(1, "boyova_Trofymenko", "с. 93"),
    2: SourceRef(2, "Бойовий статут артилерії", "с. 15"),
    3: SourceRef(3, "Ніде не цитований", "с. 1"),
}
DAY = date(2026, 11, 23)

SLIDES = """# Обов'язки старшого офіцера батареї

## Слайд 1. Прийом команд [1]
- Приймає та записує всі команди [1].
- Передає команди без змін [1].

## Топогеодезична підготовка [2]
- Визначає координати вогневої позиції [2].
"""

DOCUMENT = """# Вогнева позиція батареї

## Визначення
**Відкрита вогнева позиція:** матеріальна частина не укрита [1].

**Порядок дій:**
- Оцінює район [2].
1. Підготовка до ведення вогню [1].

| Елемент | Відстань | Джерело |
| --- | --- | --- |
| Між гарматами | 100–200 м | [2] |
"""

TABLE = """# Види вогню

| Вид вогню | Визначення | Дальність, км | Джерело |
| --- | --- | --- | --- |
| Зосереджений | По одній цілі [1] | 15,3 | [1] |
| Загороджувальний | Суцільна завіса | 122 мм | [1], [2] |
"""


def render(text: str, fmt: str):
    return render_document(parse_markdown(text, fallback_title="запасна"), fmt, SOURCES, today=DAY)


# ------------------------------------------------------------------ pptx
def test_presentation_has_title_content_and_sources_slides() -> None:
    rendered = render(SLIDES, "pptx")
    assert rendered.media_type == MEDIA_TYPES["pptx"]
    assert rendered.stats == {"slides": 2}
    prs = Presentation(io.BytesIO(rendered.data))
    assert prs.slide_width == Inches(13.333)          # 16:9
    titles = [s.shapes.title.text_frame.text for s in prs.slides]
    # «Слайд 1.» і маркери з заголовка прибрано, назва документа — на титулі.
    assert titles == ["Обов'язки старшого офіцера батареї", "Прийом команд",
                      "Топогеодезична підготовка", "Джерела"]
    texts = ["\n".join(sh.text_frame.text for sh in s.shapes if sh.has_text_frame)
             for s in prs.slides]
    assert "Приймає та записує всі команди [1]." in texts[1]
    assert "Джерела: [1] boyova Trofymenko, с. 93" in texts[1]
    # Лише процитовані джерела — «Ніде не цитований» у списку не з'являється.
    assert "Бойовий статут артилерії, с. 15" in texts[3]
    assert "Ніде не цитований" not in texts[3]


def test_overfull_slide_is_continued_not_shrunk() -> None:
    items = "\n".join(f"- Теза номер {i} з поясненням [1]." for i in range(1, 15))
    prs = Presentation(io.BytesIO(render(f"# Д\n\n## Розділ\n{items}", "pptx").data))
    titles = [s.shapes.title.text_frame.text for s in prs.slides]
    assert titles[1:4] == ["Розділ", "Розділ (продовження)", "Розділ (продовження)"]


def test_table_becomes_a_table_slide() -> None:
    prs = Presentation(io.BytesIO(render(TABLE, "pptx").data))
    tables = [sh.table for s in prs.slides for sh in s.shapes if sh.has_table]
    assert len(tables) == 1
    assert tables[0].cell(0, 0).text == "Вид вогню"
    assert tables[0].cell(1, 0).text == "Зосереджений"


def test_new_slides_added_by_the_teacher_are_wide_too() -> None:
    """Макети шаблону 4:3 розтягнуто: доданий вручну слайд не має заголовка
    в лівих трьох чвертях."""
    prs = Presentation(io.BytesIO(render(SLIDES, "pptx").data))
    layout_title = next(ph for ph in prs.slide_layouts[1].placeholders if ph.placeholder_format.idx == 0)
    assert layout_title.left + layout_title.width > Inches(12)


# ------------------------------------------------------------------ docx
def test_word_document_structure() -> None:
    rendered = render(DOCUMENT, "docx")
    assert rendered.stats == {"sections": 1, "tables": 1, "rows": 1}
    doc = Document(io.BytesIO(rendered.data))
    styled = [(p.style.name, p.text) for p in doc.paragraphs if p.text]
    assert styled[0] == ("Title", "Вогнева позиція батареї")
    assert ("Heading 1", "Визначення") in styled
    assert ("Heading 2", "Порядок дій") in styled
    assert ("List Bullet", "Оцінює район [2].") in styled
    assert ("Normal", "1. Підготовка до ведення вогню [1].") in styled
    bold = [r.text for p in doc.paragraphs for r in p.runs if r.bold]
    assert "Відкрита вогнева позиція:" in bold
    assert [c.text for c in doc.tables[0].rows[0].cells] == ["Елемент", "Відстань", "Джерело"]
    # Список джерел — лише процитовані, у порядку номерів; плюс примітка.
    assert [t for _, t in styled if t.startswith("[")] == [
        "[1] boyova Trofymenko, с. 93", "[2] Бойовий статут артилерії, с. 15"]
    assert "звірте положення з джерелами" in styled[-1][1]
    assert doc.core_properties.title == "Вогнева позиція батареї"


def test_word_fonts_are_not_left_to_the_theme() -> None:
    """Тематичний шрифт (asciiTheme) Word ставить вище за явну назву."""
    doc = Document(io.BytesIO(render(DOCUMENT, "docx").data))
    heading = doc.styles["Heading 1"].element.rPr.rFonts
    assert not any("Theme" in key or "theme" in key for key in heading.attrib)
    assert doc.styles["Normal"].font.name == "Times New Roman"


# ------------------------------------------------------------------ xlsx
def test_spreadsheet_cells_sources_and_numbers() -> None:
    rendered = render(TABLE, "xlsx")
    assert rendered.stats == {"rows": 2, "tables": 1}
    book = load_workbook(io.BytesIO(rendered.data))
    assert book.sheetnames == ["Таблиця", "Джерела"]
    sheet = book["Таблиця"]
    assert sheet["A1"].value == "Види вогню"
    assert [c.value for c in sheet[3]] == ["Вид вогню", "Визначення", "Дальність, км", "Джерело"]
    row = [c.value for c in sheet[4]]
    assert row[1] == "По одній цілі [1]"           # маркер у тексті лишається
    assert row[2] == pytest.approx(15.3)            # «15,3» — число, не рядок
    assert row[3] == "[1] boyova Trofymenko, с. 93"  # голий маркер розгорнуто
    second = [c.value for c in sheet[5]]
    assert second[2] == "122 мм"                    # з одиницями — текст
    assert second[3].startswith("[1] boyova") and "[2] Бойовий статут" in second[3]
    assert sheet.freeze_panes == "A4"
    # Справжня «Таблиця Excel»: стиль зі смугами, фільтр і сортування в шапці.
    table = sheet.tables["Table1"]
    assert table.ref == "A3:D5"
    assert table.tableStyleInfo.name == "TableStyleMedium4" and table.tableStyleInfo.showRowStripes
    assert sheet.print_title_rows == "$3:$3"           # шапка на кожному друкованому аркуші
    assert sheet.page_setup.orientation == "landscape"
    assert book["Джерела"].tables["Sources"].ref == "A1:C3"
    sources = list(book["Джерела"].iter_rows(values_only=True))
    assert sources[1] == (1, "boyova Trofymenko", "с. 93")
    assert all("Ніде не цитований" not in str(r) for r in sources)


def test_answer_without_tables_becomes_a_contents_sheet() -> None:
    book = load_workbook(io.BytesIO(render(SLIDES, "xlsx").data))
    sheet = book["Зміст"]
    values = [r[0] for r in sheet.iter_rows(values_only=True) if r[0]]
    assert values[0] == "Обов'язки старшого офіцера батареї"
    assert "• Приймає та записує всі команди [1]." in values


# --------------------------------------------------------------- назви
@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("Що таке деривація? [1]", "Що таке деривація.docx"),
        ('Види "вогню": таблиця/огляд', "Види вогню таблиця огляд.docx"),
        ("", "Документ.docx"),
        ("А" * 120, "А" * 80 + ".docx"),
    ],
)
def test_filename_is_safe_for_windows(title: str, expected: str) -> None:
    assert document_filename(title, "docx") == expected


def test_nested_items_are_indented_in_word_and_powerpoint() -> None:
    text = "# Д\n\n## Розділ\n- Ознайомити з:\n - напрямком стрільби [1]\n- Доповісти [2]"
    doc = Document(io.BytesIO(render(text, "docx").data))
    styles = {p.text: p.style.name for p in doc.paragraphs}
    assert styles["напрямком стрільби [1]"] == "List Bullet 2"
    assert styles["Доповісти [2]"] == "List Bullet"
    prs = Presentation(io.BytesIO(render(text, "pptx").data))
    body = prs.slides[1].placeholders[1].text_frame.paragraphs
    assert [(p.text, p.level) for p in body] == [
        ("Ознайомити з:", 0), ("напрямком стрільби [1]", 1), ("Доповісти [2]", 0)]


# ------------------------------------------------------- інтерактивність pptx
INTERACTIVE = """# Основи бойової роботи

## Основні поняття [1]
- Бойова робота — дії на вогневій позиції [1]
- Вогнева позиція — ділянка місцевості [2]
- Найменший приціл — перелітання гребеня [1]

## Обов'язки СОБ
- Приймає команди [1]
- Визначає координати [2]

## Порівняння позицій
| Позиція | Перевага |
| --- | --- |
| Закрита | Укрита [2] |

## Питання для самоперевірки
- Що таке вогнева позиція? — Ділянка місцевості для ведення вогню [2]
- Хто визначає координати? — Старший офіцер батареї [2]
"""

_P = "{http://schemas.openxmlformats.org/presentationml/2006/main}"


def _clicks(slide) -> int:
    return len([n for n in slide._element.iter(f"{_P}cTn") if n.get("nodeType") == "clickEffect"])


def test_presentation_is_interactive() -> None:
    """Перевірено й у справжньому PowerPoint (COM): ефекти на місці, файл без
    «відновлення»; тут — те саме на рівні розмітки, щоб CI це стеріг."""
    rendered = render(INTERACTIVE, "pptx")
    assert rendered.stats == {"slides": 4, "questions": 2}
    prs = Presentation(io.BytesIO(rendered.data))
    slides = list(prs.slides)
    titles = [s.shapes.title.text_frame.text for s in slides]
    assert titles[:2] == ["Основи бойової роботи", "Зміст"]
    contents = slides[1]

    # Зміст: кнопка на кожен розділ, клік веде на перший слайд розділу.
    targets = [sh.click_action.target_slide for sh in contents.shapes
               if sh.click_action.target_slide is not None]
    assert [t.shapes.title.text_frame.text for t in targets] == [
        "Основні поняття", "Обов'язки СОБ", "Порівняння позицій", "Питання для самоперевірки"]

    # Кнопка «☰ Зміст» на кожному слайді після змісту веде назад.
    for slide in slides[2:]:
        back = [sh for sh in slide.shapes if sh.click_action.target_slide is not None]
        assert back and back[0].click_action.target_slide.slide_id == contents.slide_id
        assert back[0].text_frame.text == "☰ Зміст"

    # Тези — по одній на клік; таблиця — статична; відповіді — на клік.
    by_title = dict(zip(titles, slides, strict=True))
    assert _clicks(by_title["Основні поняття"]) == 3
    assert _clicks(by_title["Обов'язки СОБ"]) == 2
    assert _clicks(by_title["Порівняння позицій"]) == 0
    quiz = by_title["Питання для самоперевірки"]
    assert _clicks(quiz) == 2
    quiz_text = " ".join(sh.text_frame.text for sh in quiz.shapes if sh.has_text_frame)
    assert "1. Що таке вогнева позиція?" in quiz_text and "Відповідь: Старший офіцер батареї" in quiz_text

    # Перехід «вицвітання» й номер «n / усього» — на кожному слайді.
    for number, slide in enumerate(slides, start=1):
        assert slide._element.find(f"{_P}transition") is not None
        texts = [sh.text_frame.text for sh in slide.shapes if sh.has_text_frame]
        assert f"{number} / {len(slides)}" in texts


def test_few_sections_get_no_contents_slide() -> None:
    prs = Presentation(io.BytesIO(render(SLIDES, "pptx").data))
    assert "Зміст" not in [s.shapes.title.text_frame.text for s in prs.slides]


# ------------------------------------------------------- оформлення docx
def test_word_has_clickable_contents_captions_and_running_header() -> None:
    text = (
        "# Вогнева позиція\n\n## Визначення\nТекст [1].\n\n## Види\n"
        "| Позиція | Перевага |\n| --- | --- |\n| Закрита | Укрита [2] |\n\n## Безпека\nТекст [2]."
    )
    doc = Document(io.BytesIO(render(text, "docx").data))
    body = doc.element.body
    w = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    anchors = [h.get(f"{w}anchor") for h in body.iter(f"{w}hyperlink")]
    bookmarks = [b.get(f"{w}name") for b in body.iter(f"{w}bookmarkStart")]
    assert anchors == ["_section1", "_section2", "_section3"]
    assert bookmarks == anchors                           # кожне посилання має ціль
    texts = [p.text for p in doc.paragraphs]
    assert "Зміст" in texts
    assert "Таблиця 1 – Види" in texts                    # підпис над таблицею
    assert doc.sections[0].header.paragraphs[0].text == "Вогнева позиція"
