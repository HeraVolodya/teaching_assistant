"""Розбір маркерів [n] і МЕХАНІЧНА перевірка цитат.

Це головний — і, через обмеження LM Studio, єдиний — детектор галюцинацій у
системі. `logprobs` у LM Studio не реалізовано (параметри приймаються,
`choices[0].logprobs` завжди NULL), тому перплексійна оцінка впевненості
НЕДОСТУПНА. Замінників рівно два, і обидва механічні:
    1) скор реранкера як міра релевантності доказів;
    2) перевірка, що КОЖЕН маркер [n] існує в множині поданих доказів.

Маркер, якого в множині немає, — це не «дрібна неточність форматування», а
пряме свідчення, що модель вигадала джерело. Такий маркер ВИДАЛЯЄТЬСЯ з
тексту, а сам факт логується в `unresolved_citations` разом зі списком тих
номерів, які БУЛИ доступні. Це безкоштовна телеметрія галюцинацій: у звіті
з НДР із неї будується графік «частка вигаданих посилань на 100 відповідей»,
і жодного окремого експерименту для цього ставити не треба.

Друга задача файлу — злиття суміжних діапазонів сторінок ОДНОГО файлу:
с. 10-12 + с. 13-15 → с. 10-15. Правило NeoLens: зливаємо, якщо
`range_to + 1 >= other.range_from`. Викладач бачить одне посилання замість
двох, а клік по ньому відкриває той самий PDF на тій самій сторінці.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from app.domain import Citation, RetrievedChunk

__all__ = [
    "CITATION_RE", "parse_citations", "extract_ordinals", "merge_page_ranges",
    "citations_from_chunks", "quote_for", "record_unresolved",
]

# Приймаємо і [1], і [1,2], і [1, 2] — малі моделі пишуть по-різному, і
# карати їх за кому дорожче, ніж її розібрати.
CITATION_RE = re.compile(r"\[(\d{1,3}(?:\s*,\s*\d{1,3})*)\]")

# Порожній «слід» після видалення маркера: пробіл перед крапкою тощо.
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?…»])")
_MULTI_SPACE = re.compile(r"[ \t]{2,}")
_SPACE_BEFORE_NEWLINE = re.compile(r"[ \t]+\n")

QUOTE_MAX_CHARS = 320


def extract_ordinals(text: str) -> list[int]:
    """Усі номери в порядку першої появи, з розкриттям форми [1, 2]."""
    seen: list[int] = []
    for group in CITATION_RE.findall(text):
        for raw in group.split(","):
            n = int(raw.strip())
            if n not in seen:
                seen.append(n)
    return seen


def quote_for(chunk_text: str, limit: int = QUOTE_MAX_CHARS) -> str:
    """Коротка цитата для картки джерела — по межі речення, якщо вдається."""
    text = " ".join(chunk_text.split())
    if len(text) <= limit:
        return text
    cut = text[:limit]
    for sep in (". ", "! ", "? ", "; "):
        idx = cut.rfind(sep)
        if idx > limit // 2:
            return cut[: idx + 1]
    space = cut.rfind(" ")
    return (cut[:space] if space > limit // 2 else cut).rstrip() + "…"


def _citation_from(rc: RetrievedChunk, ordinal: int) -> Citation:
    ch = rc.chunk
    return Citation(
        ordinal=ordinal,
        chunk_uid=ch.chunk_uid,
        document_id=ch.document_id,
        document_title=rc.document_title or "Без назви",
        page_from=ch.page_from,
        page_to=ch.page_to,
        page_label=ch.citation_label(),
        quote=quote_for(ch.display_text),
        bboxes=[b.as_dict() for b in ch.bboxes],
        language=ch.language,
    )


def citations_from_chunks(chunks: Sequence[RetrievedChunk]) -> list[Citation]:
    """Усі докази як цитати — для екрана «Чому ця відповідь»."""
    return [_citation_from(rc, rc.ordinal_in_prompt or i)
            for i, rc in enumerate(chunks, start=1)]


def _normalise_mapping(
    mapping: Mapping[int | str, RetrievedChunk] | Sequence[RetrievedChunk],
) -> dict[int, RetrievedChunk]:
    if isinstance(mapping, Mapping):
        return {int(k): v for k, v in mapping.items()}
    return {rc.ordinal_in_prompt or i: rc for i, rc in enumerate(mapping, start=1)}


def _cleanup(text: str) -> str:
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = _MULTI_SPACE.sub(" ", text)
    text = _SPACE_BEFORE_NEWLINE.sub("\n", text)
    return text.strip()


def parse_citations(
    text: str,
    mapping: Mapping[int | str, RetrievedChunk] | Sequence[RetrievedChunk],
) -> tuple[str, list[Citation], list[str]]:
    """Публічний контракт модуля.

    Повертає (очищений текст, цитати в порядку появи, нерозв'язані маркери).

    Нерозв'язані маркери ВИДАЛЯЮТЬСЯ з тексту: лишити «[7]», за яким нічого
    немає, означає показати викладачеві посилання, що нікуди не веде, — а
    довіра до системи тримається саме на тому, що кожне посилання клікабельне
    й веде на конкретну сторінку конкретного файлу.
    """
    available = _normalise_mapping(mapping)
    used: dict[int, Citation] = {}
    unresolved: list[str] = []

    def replace(match: re.Match[str]) -> str:
        kept: list[int] = []
        for raw in match.group(1).split(","):
            n = int(raw.strip())
            rc = available.get(n)
            if rc is None:
                marker = f"[{n}]"
                if marker not in unresolved:
                    unresolved.append(marker)
                continue
            if n not in used:
                used[n] = _citation_from(rc, n)
            kept.append(n)
        return "".join(f"[{n}]" for n in kept)

    cleaned = _cleanup(CITATION_RE.sub(replace, text))
    ordered = [used[n] for n in extract_ordinals(cleaned) if n in used]
    return cleaned, ordered, unresolved


def merge_page_ranges(citations: Sequence[Citation]) -> list[Citation]:
    """Злити суміжні діапазони сторінок одного документа.

    Правило NeoLens: зливаємо, якщо `page_to + 1 >= other.page_from`.
    Порядковий номер зливка — найменший із злитих, щоб маркери [n] у тексті
    лишались чинними.
    """
    by_doc: dict[str, list[Citation]] = {}
    order: list[str] = []
    for c in citations:
        if c.document_id not in by_doc:
            by_doc[c.document_id] = []
            order.append(c.document_id)
        by_doc[c.document_id].append(c)

    out: list[Citation] = []
    for doc in order:
        group = sorted(by_doc[doc], key=lambda c: (c.page_from is None, c.page_from or 0))
        current: Citation | None = None
        for c in group:
            if current is None:
                current = c
                continue
            if (current.page_to is not None and c.page_from is not None
                    and current.page_to + 1 >= c.page_from):
                merged_to = max(current.page_to, c.page_to or c.page_from)
                current = Citation(
                    ordinal=min(current.ordinal, c.ordinal),
                    chunk_uid=current.chunk_uid,
                    document_id=current.document_id,
                    document_title=current.document_title,
                    page_from=current.page_from,
                    page_to=merged_to,
                    page_label=_merged_label(current, c, merged_to),
                    quote=current.quote,
                    bboxes=[*current.bboxes, *c.bboxes],
                    language=current.language,
                )
                continue
            out.append(current)
            current = c
        if current is not None:
            out.append(current)
    return sorted(out, key=lambda c: c.ordinal)


def _merged_label(a: Citation, b: Citation, merged_to: int) -> str:
    """Мітка зливка. Друковані мітки важливіші за фізичні індекси: викладач
    шукає «с. 147», надруковану внизу сторінки, а не 152-гу сторінку файлу."""
    start = a.page_label.replace("с. ", "").split("–")[0].strip() if a.page_label else ""
    end = b.page_label.replace("с. ", "").split("–")[-1].strip() if b.page_label else ""
    if not start:
        start = str(a.page_from or "")
    if not end:
        end = str(merged_to)
    if not start:
        return ""
    return f"с. {start}" if start == end else f"с. {start}–{end}"


def record_unresolved(
    telemetry: object,
    message_id: str,
    unresolved: Sequence[str],
    available: Sequence[str] | Mapping[int | str, object],
) -> int:
    """Записати нерозв'язані маркери разом зі списком ДОСТУПНИХ.

    `telemetry` типізовано як `object` навмисно: цей модуль не імпортує
    `TelemetryRepo`, щоб генерація лишалась тестованою без БД. Потрібен лише
    метод `unresolved_citation(message_id, emitted, available)`.
    """
    if not unresolved:
        return 0
    recorder = getattr(telemetry, "unresolved_citation", None)
    if recorder is None:
        return 0
    avail = [str(k) for k in (available.keys() if isinstance(available, Mapping) else available)]
    for marker in unresolved:
        recorder(message_id, marker, avail)
    return len(unresolved)
