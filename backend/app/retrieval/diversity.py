"""Дедуплікація майже-дублікатів і диверсифікація джерел.

ЧОМУ ЦЕ НАЙДЕШЕВШИЙ ВАЖІЛЬ ВИСОКОЇ ВІДДАЧІ САМЕ ДЛЯ НАС
------------------------------------------------------
На FictionalQA (корпус без параметричного витоку, тобто модель не могла знати
відповідь наперед; генератори 1B–12B) виміряно: дублікати й перефразування ТОГО
САМОГО факту не дають генератору жодного приросту, а жанрово різноманітні
докази дають +17…+47 п.п. При k=5 приріст становить +24.0% на моделі 1B,
+17.3% на 3B, +15.2% на 8B і +11.2% на 12B.

Ефект ОБЕРНЕНИЙ до розміру моделі. Ми запускаємо MamayLM-Gemma-3-12B на
8–12 ГБ VRAM, тобто перебуваємо рівно в тій частині кривої, де диверсифікація
дає найбільше. При цьому вона коштує нуль додаткових обчислень — це порядок
уже наявного списку.

Другий, незалежний аргумент: вимога НДР — «поєднувати декілька джерел».
Без квоти на документ один товстий підручник монополізує всі 5 слотів просто
тому, що він товстіший, і система формально виконує пошук, але фактично
перестає бути тим, що замовили.

ПАСТКА, ЯКУ ЗАКРИВАЄ ВІДНОСНИЙ ПОРІГ
------------------------------------
Диверсифікація без порога — це вказівка «принеси щось із іншого документа
будь-якою ціною». Якщо в інших документах немає нічого релевантного, вона
притягне сміття і витіснить ним хороший фрагмент. Тому кандидат нижче
`relative_score_floor × найкращий` не бере участі в диверсифікації взагалі.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.domain import RetrievedChunk

__all__ = [
    "DEFAULT_MAX_HAMMING",
    "DEFAULT_MIN_COSINE",
    "DroppedDuplicate",
    "hamming64",
    "simhash_similarity",
    "relative_floor_value",
    "deduplicate",
    "diversify",
    "distinct_documents",
]

# SimHash-64: 3 біти з 64 — це ~95% збіг зважених ознак. Ширше вікно починає
# зливати різні означення того самого поняття з різних підручників, а це саме
# те, що ми хочемо ПОКАЗАТИ викладачеві, а не приховати.
DEFAULT_MAX_HAMMING = 3
# Косинус — друга, незалежна перевірка. SimHash дає хибнопозитивні збіги на
# коротких текстах (підписи до рисунків, заголовки таблиць), і саме там
# втратити фрагмент найдорожче.
DEFAULT_MIN_COSINE = 0.93


def hamming64(a: int, b: int) -> int:
    """Відстань Хеммінга між двома 64-бітними SimHash."""
    return int((int(a) ^ int(b)) & 0xFFFFFFFFFFFFFFFF).bit_count()


def simhash_similarity(a: int, b: int) -> float:
    """Частка спільних бітів у [0, 1] — зручно для діагностики."""
    return 1.0 - hamming64(a, b) / 64.0


@dataclass(slots=True)
class DroppedDuplicate:
    """Що і чому викинуто — це живить екран «Чому ця відповідь»."""

    chunk_id: int
    duplicate_of: int
    hamming: int | None = None
    cosine: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "duplicate_of": self.duplicate_of,
            "hamming": self.hamming,
            "cosine": None if self.cosine is None else round(self.cosine, 4),
        }


def deduplicate(
    items: Sequence[RetrievedChunk],
    *,
    vectors: Mapping[int, np.ndarray] | None = None,
    max_hamming: int = DEFAULT_MAX_HAMMING,
    min_cosine: float = DEFAULT_MIN_COSINE,
) -> tuple[list[RetrievedChunk], list[DroppedDuplicate]]:
    """Прибрати майже-дублікати МІЖ підручниками, зберігаючи кращий за скором.

    Два незалежні сигнали, і потрібні обидва:
      * SimHash-64 (`chunks.simhash`, порахований на етапі чанкування) —
        дешевий, але грубий;
      * косинус канонічних векторів — точний, але вимагає векторів у пам'яті.

    Якщо вектора немає (наприклад, чанк ще не переіндексований новою моделлю),
    рішення приймається лише за SimHash. Якщо немає й SimHash — чанк НЕ
    відкидається: мовчазна втрата фрагмента гірша за показаний дублікат.

    Порядок вхідного списку вважається впорядкуванням за якістю: перший
    представник групи виживає.
    """
    kept: list[RetrievedChunk] = []
    dropped: list[DroppedDuplicate] = []

    for item in items:
        chunk_id = item.chunk.id
        simhash = item.chunk.simhash
        duplicate: DroppedDuplicate | None = None

        for survivor in kept:
            s_hash = survivor.chunk.simhash
            if simhash is None or s_hash is None:
                continue
            bits = hamming64(simhash, s_hash)
            if bits > max_hamming:
                continue
            cos: float | None = None
            if vectors is not None and chunk_id in vectors and survivor.chunk.id in vectors:
                cos = float(np.dot(vectors[chunk_id], vectors[survivor.chunk.id]))
                if cos < min_cosine:
                    # SimHash збігся, а вектори — ні: це хибнопозитив, лишаємо.
                    continue
            duplicate = DroppedDuplicate(
                chunk_id=int(chunk_id or 0),
                duplicate_of=int(survivor.chunk.id or 0),
                hamming=bits,
                cosine=cos,
            )
            break

        if duplicate is None:
            kept.append(item)
        else:
            dropped.append(duplicate)

    return kept, dropped


def relative_floor_value(best: float, ratio: float) -> float:
    """Абсолютний поріг із відносного, стійкий до знака скора.

    Наївне `best * ratio` ламається на від'ємних скорах, а вони тут реальні:
    крос-енкодер може повертати логіти, і тоді найкращий кандидат має скор
    −0.8, а `0.6 × (−0.8) = −0.48` виявляється ВИЩИМ за нього — поріг викидає
    навіть найкращий фрагмент і система утримується завжди. Тому для
    від'ємного best поріг розширюється (ділення), а не звужується.
    """
    if ratio <= 0.0:
        return float("-inf")
    if best > 0.0:
        return best * ratio
    if best < 0.0:
        return best / ratio
    return 0.0


@dataclass(slots=True)
class _DocBucket:
    document_id: str
    taken: int = 0
    items: list[RetrievedChunk] = field(default_factory=list)


def diversify(
    items: Sequence[RetrievedChunk],
    *,
    final_k: int = 5,
    max_per_document: int = 2,
    min_distinct_documents: int = 2,
    relative_score_floor: float = 0.6,
) -> list[RetrievedChunk]:
    """Відібрати фінальні `final_k` так, щоб жоден документ не монополізував їх.

    Алгоритм у три проходи, і кожен існує через конкретну відмову:

      1. Жадібний прохід із квотою `max_per_document` — основний механізм.
         При 5 слотах і квоті 2 це гарантує щонайменше 3 різні документи,
         якщо вони взагалі є в пулі.
      2. Добір понад квоту, якщо після проходу 1 слоти лишились порожніми.
         Без нього питання, відповідь на яке чесно лежить в одному підручнику,
         поверталося б із двома фрагментами замість п'яти — тобто квота
         працювала б проти якості.
      3. Примусовий обмін заради `min_distinct_documents`, якщо пул дозволяє.
         Потрібен, коли квоту довелося послабити на проході 2.

    Кандидати нижче відносного порога не беруть участі в жодному проході.
    """
    if not items:
        return []
    if final_k <= 0:
        return []

    ordered = sorted(items, key=lambda r: -r.score)
    floor = relative_floor_value(ordered[0].score, relative_score_floor)
    eligible = [r for r in ordered if r.score >= floor]
    if not eligible:  # захист від NaN у скорах
        eligible = ordered[:final_k]

    quota = max(1, max_per_document)
    per_doc: dict[str, int] = {}
    selected: list[RetrievedChunk] = []
    leftovers: list[RetrievedChunk] = []

    # --- прохід 1: квота на документ ---
    for item in eligible:
        doc = item.chunk.document_id
        if len(selected) >= final_k:
            leftovers.append(item)
            continue
        if per_doc.get(doc, 0) >= quota:
            leftovers.append(item)
            continue
        selected.append(item)
        per_doc[doc] = per_doc.get(doc, 0) + 1

    # --- прохід 2: добір понад квоту, якщо лишились порожні слоти ---
    for item in leftovers:
        if len(selected) >= final_k:
            break
        selected.append(item)
        per_doc[item.chunk.document_id] = per_doc.get(item.chunk.document_id, 0) + 1

    # --- прохід 3: дотягнути кількість різних документів ---
    want_docs = min(min_distinct_documents, len({r.chunk.document_id for r in eligible}))
    if len({r.chunk.document_id for r in selected}) < want_docs:
        chosen_ids = {id(r) for r in selected}
        for candidate in eligible:
            if len({r.chunk.document_id for r in selected}) >= want_docs:
                break
            if id(candidate) in chosen_ids:
                continue
            docs_now = {r.chunk.document_id for r in selected}
            if candidate.chunk.document_id in docs_now:
                continue
            # Жертвуємо найгіршим фрагментом із найбільш представленого
            # документа, а не просто останнім: інакше можна викинути єдиного
            # представника другого джерела й нічого не виграти.
            counts: dict[str, int] = {}
            for r in selected:
                counts[r.chunk.document_id] = counts.get(r.chunk.document_id, 0) + 1
            victim = min(
                (r for r in selected if counts[r.chunk.document_id] > 1),
                key=lambda r: r.score,
                default=None,
            )
            if victim is None:
                break
            selected.remove(victim)
            chosen_ids.discard(id(victim))
            selected.append(candidate)
            chosen_ids.add(id(candidate))

    selected.sort(key=lambda r: -r.score)
    return selected[:final_k]


def distinct_documents(items: Iterable[RetrievedChunk]) -> int:
    """Скільки різних документів представлено. Це метрика НДР, не дрібниця."""
    return len({r.chunk.document_id for r in items})
