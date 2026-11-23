"""Файли на прохання викладача: презентація (.pptx), документ (.docx), таблиця (.xlsx).

Публічний інтерфейс модуля (див. docs/CONTRACT.md):
    detect_document_request(text) -> DocumentRequest | None
    document_task(request) -> AnswerTask
    parse_markdown(text, fallback_title=...) -> Outline
    render_document(outline, fmt, sources) -> RenderedFile
    render_message(con, message_id, fmt=None) -> RenderedFile

Як це працює. Чат розпізнає прохання (`intent.py`), шукає в матеріалах ТЕМУ без
командних слів, і та сама генерація пише Markdown за правилами формату
(`tasks.py`) — він стрімиться в чат із посиланнями [n], як звичайна відповідь.
Файл збирається з цього тексту (`outline.py` → `render_*.py`) і щоразу заново
на завантаженні (`store.py`), тож байтів на диску немає.

Модуль НЕ імпортує python-pptx / python-docx / openpyxl на рівні модуля:
лише всередині збирання, щоб старт API лишався швидким (контракт, правило 1).
"""

from __future__ import annotations

from app.export.intent import FORMATS, DocumentFormat, DocumentRequest, detect_document_request
from app.export.outline import Outline, parse_markdown
from app.export.render import MEDIA_TYPES, RenderedFile, SourceRef, document_filename, render_document
from app.export.tasks import document_task

__all__ = [
    "FORMATS",
    "MEDIA_TYPES",
    "DocumentFormat",
    "DocumentRequest",
    "Outline",
    "RenderedFile",
    "SourceRef",
    "detect_document_request",
    "document_filename",
    "document_task",
    "parse_markdown",
    "render_document",
]
