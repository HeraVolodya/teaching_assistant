"""Фабрики доказів для тестів LLM і генерації.

Не `conftest.py` навмисно: сусідні модулі пишуть свої тести паралельно, і
спільний conftest — це файл, який двоє редагують одночасно.
"""

from __future__ import annotations

from app.domain import BBox, Chunk, ChunkLevel, RetrievedChunk


def make_chunk(
    *,
    document_id: str = "doc-1",
    ordinal: int = 0,
    text: str = "Текст фрагмента.",
    header_path: str = "//Розділ 1//1.1. Тема//",
    page_from: int = 10,
    page_to: int = 12,
    label_from: str | None = None,
    label_to: str | None = None,
) -> Chunk:
    return Chunk(
        document_id=document_id,
        collection_id="col-1",
        ordinal=ordinal,
        level=ChunkLevel.LEAF,
        display_text=text,
        header_path=header_path,
        page_from=page_from,
        page_to=page_to,
        page_label_from=label_from,
        page_label_to=label_to,
        bboxes=[BBox(page=page_from, left=10.0, top=20.0, right=300.0, bottom=90.0)],
    )


def make_retrieved(
    *,
    document_id: str = "doc-1",
    title: str = "Підручник з балістики",
    ordinal: int = 0,
    text: str = "Текст фрагмента.",
    page_from: int = 10,
    page_to: int = 12,
    label_from: str | None = None,
    label_to: str | None = None,
    rerank_score: float = 0.8,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=make_chunk(document_id=document_id, ordinal=ordinal, text=text,
                         page_from=page_from, page_to=page_to,
                         label_from=label_from, label_to=label_to),
        document_title=title,
        rerank_score=rerank_score,
        fused_score=rerank_score,
    )


def evidence_two_documents() -> list[RetrievedChunk]:
    """Дві книги по два фрагменти — мінімальний набір для перевірки вимоги
    «поєднувати декілька джерел одночасно»."""
    return [
        make_retrieved(document_id="doc-1", title="Балістика та стрільба", ordinal=1,
                       text="Деривація — це відхилення снаряда вбік від площини стрільби. "
                            "Її величина зростає з дальністю.",
                       page_from=147, page_to=148, rerank_score=0.91),
        make_retrieved(document_id="doc-1", title="Балістика та стрільба", ordinal=2,
                       text="Поправка на деривацію вводиться в установки кутоміра. "
                            "Таблиці стрільби подають її в тисячних.",
                       page_from=149, page_to=150, rerank_score=0.84),
        make_retrieved(document_id="doc-2", title="Правила стрільби та управління вогнем",
                       ordinal=1,
                       text="Поправку на деривацію обчислюють за таблицями стрільби "
                            "для кожного заряду окремо.",
                       page_from=52, page_to=53, rerank_score=0.77),
        make_retrieved(document_id="doc-3", title="Методичні рекомендації з підготовки",
                       ordinal=1,
                       text="Під час підготовки установок деривацію враховують разом "
                            "із поправкою на вітер.",
                       page_from=8, page_to=9, rerank_score=0.71),
    ]
