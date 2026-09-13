"""Гейти двоступеневого чанкування.

Найдорожчі твердження тут:
  * стек заголовків штовхається за MARKDOWN-РІВНЕМ — стрибок H1→H3 має
    зберігати H1 у шляху; при поп-за-глибиною він би зник;
  * атомарний блок фізично не можна розрізати;
  * перекриття 300 символів існує ЛИШЕ на рекурсивному фолбеку, а на справжній
    межі його немає — інакше індекс роздувається на ~20% майже-дублікатами.
"""

from __future__ import annotations

from app.domain import ChunkLevel
from app.ingestion.chunker import (
    ChunkConfig,
    _protect_atoms,
    _recursive_split,
    _restore_atoms,
    build_chapters,
    calibrate_ch_per_tok,
    chunk_markdown,
    hamming64,
    header_path_of,
    simhash64,
)
from app.ingestion.docling_pipeline import ParseOptions, _parse_text_source

CFG = ChunkConfig()
PROSE = "Траєкторія снаряда залежить від кута підвищення та початкової швидкості. "


def build(md: str, **kwargs) -> object:
    return chunk_markdown(md, title="Підручник", document_id="d1", collection_id="c1", **kwargs)


# ------------------------------------------------------------- шлях заголовків
def test_header_path_keeps_h1_when_document_jumps_to_h3() -> None:
    """H1 → H3 без проміжного H2. Стек штовхається за рівнем, не за глибиною."""
    tree = build("# Розділ 2\n\n### 2.3. Балістика\n\n" + PROSE * 4)
    leaf = tree.leaves()[-1]
    assert leaf.header_path == "//Розділ 2//2.3. Балістика//"


def test_sibling_heading_pops_only_its_own_level() -> None:
    md = (
        "# Розділ 2\n\n### 2.3. Балістика\n\n" + PROSE * 3
        + "\n\n## 2.4. Стрільба\n\n" + PROSE * 3
    )
    paths = {c.header_path for c in build(md).leaves()}
    assert "//Розділ 2//2.3. Балістика//" in paths
    assert "//Розділ 2//2.4. Стрільба//" in paths


def test_header_path_feeds_embed_and_rerank_text() -> None:
    """`header_path` — це +23.8% MRR@5 (план, §5). Він мусить доїхати в обидва тексти."""
    leaf = build("# Розділ 2\n\n### 2.3. Балістика\n\n" + PROSE * 4).leaves()[-1]
    assert "Розділ 2 › 2.3. Балістика" in leaf.embed_text
    assert leaf.rerank_text.startswith("2.3. Балістика")
    # Крос-енкодери чутливі до шуму в префіксі — картки документа в них немає.
    assert "Підручник" not in leaf.rerank_text
    assert "Підручник" in leaf.embed_text


def test_header_path_of_is_stable_and_escaped() -> None:
    assert header_path_of([]) == "//"
    assert header_path_of(["Розділ 2", "2.3"]) == "//Розділ 2//2.3//"
    assert "/" not in header_path_of(["А/Б"]).strip("/").replace("//", "")


# ------------------------------------------------------------------- бюджети
def test_leaves_respect_the_soft_maximum() -> None:
    tree = build("# Тема\n\n" + PROSE * 200)
    for leaf in tree.leaves():
        assert len(leaf.display_text) <= CFG.leaf_hard_max
    assert len(tree.leaves()) > 1


def test_long_paragraph_is_split_rather_than_kept_whole() -> None:
    """Абзац на 3000 символів — не атомарна одиниця, жорсткий максимум не для нього."""
    tree = build("# Тема\n\n" + PROSE * 45)
    assert len(tree.leaves()) >= 2


def test_section_level_never_enters_the_index() -> None:
    tree = build("# Тема\n\n" + PROSE * 120)
    for chunk in tree.chunks:
        assert chunk.is_indexable == (chunk.level is ChunkLevel.LEAF)


def test_section_ceiling_splits_huge_sections_into_several_parents() -> None:
    tree = build("# Тема\n\n" + "\n\n".join(PROSE * 6 for _ in range(40)))
    sections = [c for c in tree.chunks if c.level is ChunkLevel.SECTION]
    assert len(sections) >= 2
    for section in sections:
        assert len(section.display_text) <= CFG.section_max + CFG.leaf_soft_max


# ------------------------------------------------------------- атомарні блоки
def test_markdown_table_becomes_one_atomic_self_describing_unit() -> None:
    md = (
        "# Таблиці стрільби\n\n"
        "| Дальність | Приціл | Поправка |\n| --- | --- | --- |\n"
        "| 1000 | 12 | 0,5 |\n| 2000 | 24 | 1,2 |\n"
    )
    leaf = build(md).leaves()[0]
    # Кожен рядок несе власні заголовки колонок — тому переживає будь-який розріз.
    assert "1000, Приціл = 12" in leaf.display_text
    assert "2000, Поправка = 1,2" in leaf.display_text


def test_atomic_block_is_never_cut_by_recursive_split() -> None:
    """Плейсхолдер робить розріз усередині блоку фізично неможливим."""
    block = "```mermaid\n" + "\n".join(f"A{i} --> B{i}" for i in range(60)) + "\n```"
    text = PROSE * 30 + "\n\n" + block + "\n\n" + PROSE * 30
    parts = _recursive_split(text, CFG)
    assert len(parts) > 1
    holders = [p for p in parts if "```mermaid" in p]
    assert len(holders) == 1
    assert holders[0].count("```") == 2, "блок Mermaid розрізано навпіл"


def test_protect_and_restore_round_trip() -> None:
    text = "перед\n\n```py\nкод\n```\n\nпісля"
    protected, blocks = _protect_atoms(text)
    assert "```" not in protected
    assert len(blocks) == 1
    assert _restore_atoms(protected, blocks) == text


def test_huge_table_is_split_only_between_rows() -> None:
    """Триплетний рядок самоописний, тому розріз МІЖ рядками нічого не втрачає."""
    rows = "\n".join(f"{i}, Приціл = {i * 2}. {i}, Поправка = 0,{i}." for i in range(400))
    md = "# Таблиця\n\n" + "\n".join(
        ["| Дальність | Приціл | Поправка |", "| --- | --- | --- |"]
        + [f"| {i} | {i * 2} | 0,{i} |" for i in range(400)]
    )
    leaves = build(md).leaves()
    assert len(leaves) > 1
    assert rows  # табличка справді велика
    for leaf in leaves:
        for line in leaf.display_text.splitlines():
            assert line.strip() == "" or "=" in line or line.startswith("["), line


# ------------------------------------------------------------- злиття малих
def test_small_trailing_part_is_merged_into_its_neighbour() -> None:
    md = "# Тема\n\n" + PROSE * 25 + "\n\nКоротка кінцівка."
    leaves = build(md).leaves()
    assert all(
        len(leaf.display_text) >= CFG.leaf_min or len(leaves) == 1 for leaf in leaves
    )


def test_heading_only_fragment_does_not_become_its_own_chunk() -> None:
    md = "# Тема\n\n" + "\n\n".join(["Підзаголовок без крапки", PROSE * 20])
    for leaf in build(md).leaves():
        assert leaf.display_text.strip() != "Підзаголовок без крапки"


# --------------------------------------------------------------- перекриття
def test_no_overlap_on_a_real_element_boundary() -> None:
    """На межі елемента перекриття — чиста втрата: роздуває індекс і плодить дублікати."""
    first, second = PROSE * 18, "Друга частина тексту. " * 60
    tree = build(f"# Тема\n\n{first}\n\n{second}")
    leaves = tree.leaves()
    assert len(leaves) >= 2
    assert not leaves[1].display_text.startswith(leaves[0].display_text[-100:])


def test_overlap_exists_only_on_the_recursive_fallback() -> None:
    parts = _recursive_split(PROSE * 120, CFG)
    assert len(parts) > 1
    tail = parts[0][-CFG.fallback_overlap:]
    assert any(word and word in parts[1][:CFG.fallback_overlap + 40]
               for word in tail.split(". ")[-1:])


# ------------------------------------------------------- метадані й порядок
def test_leaf_ordinals_are_contiguous_for_auto_merge() -> None:
    """`ChunkRepo.neighbours` шукає сусідів за ordinal BETWEEN — розриви його ламають."""
    tree = build("# А\n\n" + PROSE * 60 + "\n\n# Б\n\n" + PROSE * 60)
    ordinals = [c.ordinal for c in tree.leaves()]
    assert ordinals == list(range(ordinals[0], ordinals[0] + len(ordinals)))


def test_parents_are_emitted_before_children() -> None:
    tree = build("# А\n\n" + PROSE * 60)
    levels = [c.level for c in tree.chunks]
    assert levels.index(ChunkLevel.SECTION) < levels.index(ChunkLevel.LEAF)
    assert levels[0] is ChunkLevel.DOCUMENT_CARD


def test_parent_ids_are_applied_after_insertion() -> None:
    tree = build("# А\n\n" + PROSE * 60)
    for index, chunk in enumerate(tree.chunks, start=1):
        chunk.id = index
    assert tree.apply_parent_ids() == len(tree.parent_uid)
    card = tree.chunks[0]
    for leaf in tree.leaves():
        assert leaf.parent_id is not None
        assert leaf.parent_id != card.id


def test_positional_metadata_is_populated() -> None:
    leaf = build("# А\n\n" + PROSE * 60).leaves()[0]
    assert leaf.page_from is not None and leaf.page_to is not None
    assert leaf.page_label_from and leaf.page_label_to
    assert leaf.char_to > leaf.char_from
    assert leaf.siblings_count >= 1
    assert leaf.chunk_uid and len(leaf.chunk_uid) == 32


def test_document_card_is_raw_structure_not_a_summary() -> None:
    """Абляція UNLP 2026: LLM-конспект замість сирого префікса ПОГІРШУЄ (0.9346→0.9177)."""
    tree = build("# Балістика\n\n## Розділ 1\n\n" + PROSE * 5)
    card = tree.chunks[0]
    assert card.level is ChunkLevel.DOCUMENT_CARD
    assert "Документ: Підручник" in card.display_text
    assert "Зміст:" in card.display_text
    assert "Балістика" in card.display_text


def test_context_note_does_not_duplicate_the_header_path() -> None:
    leaf = build("# Розділ\n\n" + PROSE * 5).leaves()[0]
    assert leaf.context_note == "Підручник"
    assert leaf.embed_text.count("Розділ") == 1


# ----------------------------------------------------------------- simhash
def test_simhash_is_stable_and_signed_for_sqlite() -> None:
    value = simhash64(PROSE * 3)
    assert value == simhash64(PROSE * 3)
    assert -(2 ** 63) <= value < 2 ** 63


def test_near_duplicates_are_close_in_hamming_distance() -> None:
    a = simhash64(PROSE * 10)
    b = simhash64(PROSE * 10 + "Додано одне речення.")
    c = simhash64("Зовсім інший текст про організацію зв'язку у підрозділі. " * 10)
    assert hamming64(a, b) < hamming64(a, c)


# ------------------------------------------------------------ ch_per_tok
def test_ch_per_tok_lands_near_the_planned_value_for_ukrainian() -> None:
    """План: 2200 символів ÷ 1.93 ch/tok ≈ 1140 токенів Qwen."""
    value = calibrate_ch_per_tok(PROSE * 50)
    assert 1.6 <= value <= 2.4


def test_ch_per_tok_uses_a_real_tokenizer_when_given() -> None:
    value = calibrate_ch_per_tok("абвгд " * 100, tokenizer=lambda s: s.split())
    assert abs(value - 6.0) < 0.1


# ------------------------------------------------------------------ розділи
def test_chapter_tree_is_a_valid_nested_set() -> None:
    parsed = _parse_text_source(
        "# Розділ 1\n\nтекст\n\n## 1.1\n\nтекст\n\n### 1.1.1\n\nтекст\n\n# Розділ 2\n\nтекст\n",
        ParseOptions(), source_path="t.md",
    )
    chapters = build_chapters(parsed, "d1")
    assert [c.title for c in chapters] == ["Розділ 1", "1.1", "1.1.1", "Розділ 2"]
    for chapter in chapters:
        assert chapter.lft < chapter.rgt
    root = chapters[0]
    child = chapters[1]
    assert root.lft < child.lft and child.rgt < root.rgt
    assert chapters[3].lft > root.rgt


def test_empty_document_produces_no_leaves() -> None:
    tree = chunk_markdown("", title="Порожній", document_id="d", collection_id="c")
    assert tree.leaves() == []
