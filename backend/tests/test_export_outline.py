"""Розбір Markdown моделі в структуру документа.

Мала модель пише Markdown «приблизно»: загортає в ```, ставить «**Розділ**»
замість заголовка, дописує власний список джерел. Файл без цієї поблажливості
вийшов би з зірочками й двома списками джерел.
"""

from __future__ import annotations

from app.export.outline import (
    markers,
    parse_markdown,
    plain,
    plain_math,
    runs,
    strip_markers,
    strip_preamble,
)

# Скорочений справжній вивід Gemma 4 E4B на «зроби презентацію на 5 слайдів».
SLIDES = """# Обов'язки старшого офіцера батареї

## Слайд 1. Прийом та передача команд [1]
- Приймає та записує всі команди, що подаються на вогневу позицію [1].
- Команди щодо підготовки вогню гармати передає без змін [1].

## Топогеодезична підготовка [2], [3]
- Визначає координати вогневої позиції [2].
"""


def test_title_comes_from_the_first_h1() -> None:
    outline = parse_markdown(SLIDES, fallback_title="запасна")
    assert outline.title == "Обов'язки старшого офіцера батареї"
    assert [b.kind for b in outline.blocks] == ["heading", "bullets", "heading", "bullets"]
    assert outline.blocks[1].items[1].endswith("[1].")
    assert outline.has_content


def test_fallback_title_without_h1() -> None:
    outline = parse_markdown("Просто абзац [1].", fallback_title="Що таке деривація? [2]")
    assert outline.title == "Що таке деривація?"
    assert outline.blocks[0].kind == "paragraph"


def test_code_fence_wrapper_is_ignored() -> None:
    outline = parse_markdown("```markdown\n# Назва\n- пункт [1]\n```")
    assert outline.title == "Назва"
    assert outline.blocks[0].items == ["пункт [1]"]


def test_bold_line_becomes_a_subheading() -> None:
    outline = parse_markdown("# Д\n\n**Загальні положення:**\n- пункт [1]")
    assert outline.blocks[0].kind == "heading" and outline.blocks[0].level == 3
    assert outline.blocks[0].text == "Загальні положення"


def test_table_with_separator_and_ragged_rows() -> None:
    text = (
        "# Види вогню\n\n"
        "| Вид | Визначення | Джерело |\n"
        "| :--- | :--- | :--- |\n"
        "| Зосереджений | По одній цілі [1] | [1] |\n"
        "| Вогневий вал | Без джерела |\n"
        "| Мішок | Три сторони | [2] | зайве |\n"
    )
    table = parse_markdown(text).blocks[0]
    assert table.kind == "table"
    assert table.header == ["Вид", "Визначення", "Джерело"]
    assert table.rows[1] == ["Вогневий вал", "Без джерела", ""]
    assert table.rows[2] == ["Мішок", "Три сторони", "[2] зайве"]


def test_models_own_source_list_is_dropped() -> None:
    """Справжній список джерел додає збирач — із назвами й сторінками з бази."""
    text = "# Д\n\nТекст [1].\n\n## Джерела\n- [1] Підручник\n\n## Висновок\nВсе [1]."
    outline = parse_markdown(text)
    headings = [b.text for b in outline.blocks if b.kind == "heading"]
    assert headings == ["Висновок"]
    trailing = parse_markdown("# Д\n\nТекст [1].\n\nДжерела: [1], [2]")
    assert [b.text for b in trailing.blocks] == ["Текст [1]."]


def test_only_headings_is_no_content() -> None:
    assert not parse_markdown("# Назва\n\n## Розділ").has_content


def test_list_item_continuation_line() -> None:
    outline = parse_markdown("- перший рядок\n  продовження [1]\n- другий")
    assert outline.blocks[0].items == ["перший рядок продовження [1]", "другий"]


def test_marker_helpers() -> None:
    assert markers("а [2] б [1] в [2]") == [2, 1]
    # Група з розділювачами зникає цілком — без «висячої» коми в заголовку.
    assert strip_markers("Топогеодезична підготовка [2], [3]") == "Топогеодезична підготовка"
    assert strip_markers("Текст [1].") == "Текст."
    assert plain("**Жирний** і *курсив* та `код`") == "Жирний і курсив та код"
    assert runs("**Ж:** текст") == [("Ж:", True, False), (" текст", False, False)]
    # Підкреслення в назвах документів — не курсив.
    assert plain("boyova_Trofymenko") == "boyova_Trofymenko"


def test_nested_items_keep_their_level() -> None:
    """Підпункти моделі («    *   …») після `parse_citations` мають один пробіл
    відступу — і цього досить, щоб не сплющити їх до верхнього рівня."""
    outline = parse_markdown(
        "- Ознайомити з:\n - напрямком стрільби [1]\n - точками наводки [1]\n- Доповісти"
    )
    block = outline.blocks[0]
    assert block.items == ["Ознайомити з:", "напрямком стрільби [1]", "точками наводки [1]", "Доповісти"]
    assert [block.depth(i) for i in range(4)] == [0, 1, 1, 0]


def test_preamble_before_the_title_is_cut() -> None:
    """Живий випадок: модель почала з «Я не можу створити файл…», а далі
    написала саме те, що треба. У файл і в історію вступ не йде."""
    text = (
        "Я не можу створити файл формату PPTX, оскільки я генерую лише текст. Однак, я "
        "підготую для Вас структуру презентації у форматі Markdown.\n\n"
        "```markdown\n# Основи бойової роботи\n\n## Терміни\n- Вогнева позиція [1]\n```"
    )
    assert strip_preamble(text).startswith("# Основи бойової роботи")
    assert "```" not in strip_preamble(text)
    outline = parse_markdown(text)
    assert outline.title == "Основи бойової роботи"
    assert [b.kind for b in outline.blocks] == ["heading", "bullets"]


def test_content_before_the_title_is_not_a_preamble() -> None:
    text = "- пункт [1]\n\n# Назва\n\nабзац [1]"
    assert strip_preamble(text) == text


def test_formulas_become_readable_symbols() -> None:
    """Живий слайд мав «$\\alpha$ осі» — так Docling витягує формули підручників.
    У чаті це малює KaTeX, а у файлі мусять бути звичайні символи."""
    assert plain_math(r"Для визначення $\alpha$ осі") == "Для визначення α осі"
    assert plain_math(r"азимут ($\text{Аосі}$)") == "азимут (Аосі)"
    assert plain_math(r"$V_0$ і $\Delta Д$") == "V₀ і ΔД"
    assert plain_math(r"$x^2 + y^{2}$") == "x² + y²"
    assert plain_math("ціна 5$ і 10$ разом") == "ціна 5$ і 10$ разом"   # гроші — не формула
    assert runs(r"**Кут** $\alpha$") == [("Кут", True, False), (" α", False, False)]
    assert plain(r"$\alpha$ осі") == "α осі"
