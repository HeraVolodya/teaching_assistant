"""Структура документа → байти .pptx / .docx / .xlsx.

Бібліотеки (python-pptx, python-docx, openpyxl — усі MIT) імпортуються ЛИШЕ
всередині функцій збирання: API-процес стартує за ~0,6 с і не мусить платити
за lxml і Pillow, доки викладач не попросив файл (контракт, правило 1).

ДЖЕРЕЛА В КОЖНОМУ ФАЙЛІ. Маркери [n] лишаються в тексті, як у чаті, а наприкінці
йде список «[n] документ, сторінка» — лише тих джерел, на які текст справді
посилається. Файл, винесений із застосунку, мусить і далі показувати, звідки
кожне твердження: саме на цьому тримається довіра до відповіді.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date

from app.export.intent import DocumentFormat
from app.export.outline import Outline, markers, plain, strip_markers

__all__ = [
    "EXTENSIONS",
    "MEDIA_TYPES",
    "RenderedFile",
    "SourceRef",
    "cited_sources",
    "document_filename",
    "render_document",
]

MEDIA_TYPES: dict[DocumentFormat, str] = {
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}
EXTENSIONS: dict[DocumentFormat, str] = {"pptx": ".pptx", "docx": ".docx", "xlsx": ".xlsx"}

PROVENANCE = (
    "Сформовано асистентом викладача {day} за матеріалами бази знань. "
    "Перед використанням звірте положення з джерелами."
)


@dataclass(frozen=True, slots=True)
class SourceRef:
    """Одне джерело зі списку: номер [n], документ, сторінки."""

    ordinal: int
    title: str
    label: str = ""

    @property
    def nice_title(self) -> str:
        # Назви з бази — часто імена файлів: «Бойовий_статут_артилерії_СВ».
        return re.sub(r"\s+", " ", self.title.replace("_", " ")).strip() or "Без назви"

    def full(self) -> str:
        return f"{self.nice_title}, {self.label}" if self.label else self.nice_title

    def short(self, limit: int = 34) -> str:
        title = self.nice_title
        if len(title) > limit:
            title = title[: limit - 1].rstrip() + "…"
        return f"{title}, {self.label}" if self.label else title


@dataclass(frozen=True, slots=True)
class RenderedFile:
    data: bytes
    filename: str
    media_type: str
    # Що вийшло: {"slides": 6} / {"sections": 4} / {"rows": 10}. Для підпису
    # на картці файлу в чаті.
    stats: dict[str, int] = field(default_factory=dict)


def cited_sources(outline: Outline, sources: Mapping[int, SourceRef]) -> list[SourceRef]:
    """Джерела, на які текст справді посилається, — у порядку номерів."""
    used = sorted(n for n in markers(outline.text()) if n in sources)
    return [sources[n] for n in used]


_UNSAFE_FILENAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')


def document_filename(title: str, fmt: DocumentFormat) -> str:
    """Назва файлу з назви документа: без символів, яких не терпить Windows."""
    name = _UNSAFE_FILENAME.sub(" ", plain(strip_markers(title)))
    name = re.sub(r"\s+", " ", name).strip(" .")
    if len(name) > 80:
        name = name[:80].rsplit(" ", 1)[0].rstrip(" .,;—–-") or name[:80]
    return f"{name or 'Документ'}{EXTENSIONS[fmt]}"


def render_document(
    outline: Outline,
    fmt: DocumentFormat,
    sources: Mapping[int, SourceRef],
    *,
    today: date | None = None,
) -> RenderedFile:
    """Зібрати файл. Форма вмісту (розділи, слайди, таблиця) — уже в самому
    тексті: модель отримала інструкцію саме під цей формат (`tasks.py`)."""
    day = (today or date.today()).strftime("%d.%m.%Y")
    cited = cited_sources(outline, sources)
    note = PROVENANCE.format(day=day)
    if fmt == "pptx":
        from app.export.render_pptx import render_pptx

        data, stats = render_pptx(outline, sources, cited, day=day)
    elif fmt == "docx":
        from app.export.render_docx import render_docx

        data, stats = render_docx(outline, cited, note=note)
    elif fmt == "xlsx":
        from app.export.render_xlsx import render_xlsx

        data, stats = render_xlsx(outline, sources, cited, note=note)
    else:  # pragma: no cover — Literal не пропустить, але API приймає рядок
        raise ValueError(f"Невідомий формат файлу: {fmt}")
    return RenderedFile(
        data=data,
        filename=document_filename(outline.title, fmt),
        media_type=MEDIA_TYPES[fmt],
        stats=stats,
    )
