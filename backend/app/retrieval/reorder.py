"""Дворівневе впорядкування контексту перед подачею в модель.

Порт `Reorderer` з NeoLens
(`assistant-api/src/ai/assistants/helpers/context_postprocessing/reorderers.py`),
плюс U-подібне розміщення між блоками.

РІВЕНЬ 1 — ПОРЯДОК ЧИТАННЯ ВСЕРЕДИНІ БЛОКУ ДЖЕРЕЛА
--------------------------------------------------
Реранкер видає фрагменти у порядку спадання скора. Якщо подати їх моделі саме
так, то два сусідні абзаци одного означення можуть прийти в порядку 2, 1 —
і мала локальна модель читає означення задом наперед. Вона не «розуміє», що
це один текст; вона бачить два уривки, другий з яких посилається на «зазначене
вище», якого вище немає.

Тому фрагменти з однаковими `(document_id, header_path)` і послідовними
`siblings_index` збираються в ЛАНЦЮГ і подаються в порядку читання документа.
Побудова ланцюга жадібна й двонаправлена: від найкращого за скором фрагмента
йдемо вперед (`+1`) і назад (`−1`), доки в наборі є сусіди.

РІВЕНЬ 2 — U-ПОДІБНЕ РОЗМІЩЕННЯ МІЖ БЛОКАМИ
-------------------------------------------
«Lost in the middle»: моделі надійно дістають інформацію з ПОЧАТКУ і КІНЦЯ
контексту, а середина провалюється, і на малих моделях провал глибший.
Тому найкращий блок ставимо першим, другий за якістю — ОСТАННІМ, третій —
другим, четвертий — передостаннім. Найслабші блоки опиняються в середині,
де їх усе одно прочитають найгірше — і це саме те, чого ми хочемо.

Важливо, що переставляються БЛОКИ, а не окремі фрагменти: розірвати ланцюг
означень заради U-форми — це зруйнувати рівень 1 заради рівня 2.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from app.domain import RetrievedChunk

__all__ = ["SourceBlock", "build_blocks", "u_shaped", "reorder"]


@dataclass(slots=True)
class SourceBlock:
    """Суцільний ланцюг фрагментів однієї секції одного документа."""

    document_id: str
    header_path: str
    items: list[RetrievedChunk] = field(default_factory=list)

    @property
    def score(self) -> float:
        """Скор блоку — найкращий скор його фрагментів."""
        return max((r.score for r in self.items), default=float("-inf"))

    @property
    def document_title(self) -> str:
        return self.items[0].document_title if self.items else ""

    def __len__(self) -> int:
        return len(self.items)


def build_blocks(items: Sequence[RetrievedChunk]) -> list[SourceBlock]:
    """Зібрати фрагменти в блоки за (document_id, header_path) і сусідством.

    Один документ може дати КІЛЬКА блоків, якщо знайдені фрагменти лежать у
    різних секціях або не є сусідами — і це правильно: між ними в підручнику
    десятки сторінок, тож склеювати їх в один блок означало б збрехати моделі
    про суцільність тексту.
    """
    remaining = sorted(items, key=lambda r: -r.score)
    used: set[int] = set()
    blocks: list[SourceBlock] = []

    # Індекс сусідства: (документ, шлях заголовків) -> siblings_index -> фрагмент
    by_section: dict[tuple[str, str], dict[int, RetrievedChunk]] = {}
    for r in remaining:
        key = (r.chunk.document_id, r.chunk.header_path)
        by_section.setdefault(key, {})[r.chunk.siblings_index] = r

    for seed in remaining:
        if id(seed) in used:
            continue
        key = (seed.chunk.document_id, seed.chunk.header_path)
        siblings = by_section[key]
        chain: list[RetrievedChunk] = [seed]
        used.add(id(seed))

        # назад
        idx = seed.chunk.siblings_index - 1
        while idx in siblings and id(siblings[idx]) not in used:
            node = siblings[idx]
            chain.insert(0, node)
            used.add(id(node))
            idx -= 1
        # вперед
        idx = seed.chunk.siblings_index + 1
        while idx in siblings and id(siblings[idx]) not in used:
            node = siblings[idx]
            chain.append(node)
            used.add(id(node))
            idx += 1

        blocks.append(SourceBlock(document_id=key[0], header_path=key[1], items=chain))

    return blocks


def u_shaped(blocks: Sequence[SourceBlock]) -> list[SourceBlock]:
    """Найкращий — на початок, другий за якістю — в кінець, і так по черзі."""
    ordered = sorted(blocks, key=lambda b: -b.score)
    head: list[SourceBlock] = []
    tail: list[SourceBlock] = []
    for i, block in enumerate(ordered):
        (head if i % 2 == 0 else tail).append(block)
    tail.reverse()
    return head + tail


def reorder(
    items: Sequence[RetrievedChunk], *, assign_ordinals: bool = True
) -> list[RetrievedChunk]:
    """Повний дворівневий порядок і, за потреби, нумерація `[n]` для промпту.

    `ordinal_in_prompt` призначається ТУТ і більше ніде: модель бачить `[n]`,
    а не UUID (контракт, правило 7), і номер мусить відповідати фактичному
    порядку блоків у промпті, інакше цитата `[3]` вкаже не на той фрагмент.
    """
    blocks = u_shaped(build_blocks(items))
    ordered: list[RetrievedChunk] = [r for block in blocks for r in block.items]
    if assign_ordinals:
        for n, chunk in enumerate(ordered, start=1):
            chunk.ordinal_in_prompt = n
    return ordered
