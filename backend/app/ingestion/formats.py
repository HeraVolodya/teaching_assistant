"""Які файли приймає база знань і яким шляхом кожен із них читається.

Модуль навмисно ЛЕГКИЙ: його імпортує API-процес (перевірка розширення при
завантаженні), а правило 1 контракту забороняє тягнути туди хоч щось важке.

ТРИ ШЛЯХИ ЧИТАННЯ — і вибір між ними визначає, на що зможе послатися цитата.

  * «paged» — PDF напряму, а Word / PowerPoint / DjVu через PDF-копію
    (`convert.py`). Тут є справжні сторінки й рамки абзаців, тож цитата
    відкриває документ на потрібній сторінці з підсвіткою. DOCX свідомо не
    читається «як є»: `msword_backend` Docling ніколи не створює
    `ProvenanceItem`, і номер сторінки для нього неможливий у принципі. PDF,
    який зберіг сам Word, має ті самі сторінки, що викладач бачить при друці.
  * «text» — txt/md напряму, а таблиці Excel/CSV і XML через Markdown-копію.
    Для таблиці «сторінка» — це аркуш: цитата веде на «арк. Норми», а не на
    вигадану друковану сторінку, якої в Excel немає.
  * копія не потрібна — PDF і текст читаються з оригіналу.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

__all__ = [
    "ACCEPTED_SUFFIXES",
    "DJVU_SUFFIXES",
    "PDF_SUFFIXES",
    "SHEET_SUFFIXES",
    "SLIDE_SUFFIXES",
    "TEXT_SUFFIXES",
    "WORD_SUFFIXES",
    "XML_SUFFIXES",
    "SourceKind",
    "accepted_label",
    "sniff_xml",
    "source_kind",
]

SourceKind = Literal["pdf", "text", "word", "slides", "djvu", "sheet", "xml"]

PDF_SUFFIXES = frozenset({".pdf"})
TEXT_SUFFIXES = frozenset({".txt", ".md", ".markdown", ".text"})
WORD_SUFFIXES = frozenset({".doc", ".docx", ".docm", ".dot", ".dotx", ".rtf", ".odt"})
SLIDE_SUFFIXES = frozenset({".ppt", ".pptx", ".pptm", ".pps", ".ppsx", ".odp"})
DJVU_SUFFIXES = frozenset({".djvu", ".djv"})
SHEET_SUFFIXES = frozenset({".xlsx", ".xlsm", ".xls", ".csv", ".ods"})
XML_SUFFIXES = frozenset({".xml"})

ACCEPTED_SUFFIXES = frozenset(
    PDF_SUFFIXES | TEXT_SUFFIXES | WORD_SUFFIXES | SLIDE_SUFFIXES
    | DJVU_SUFFIXES | SHEET_SUFFIXES | XML_SUFFIXES
)

# Підписи XML-файлів Office 2003. Word і Excel уміли зберігати документ у
# «XML-форматі», і на кафедрах такі файли досі трапляються з розширенням .xml.
_WORDML_MARKS = (b"schemas.microsoft.com/office/word/2003/wordml", b'progid="Word.Document"')
_SPREADSHEETML_MARKS = (b"urn:schemas-microsoft-com:office:spreadsheet", b'progid="Excel.Sheet"')


def sniff_xml(head: bytes) -> SourceKind:
    """XML Office 2003 → «word»/«sheet»; будь-який інший XML → «xml»."""
    if any(mark in head for mark in _SPREADSHEETML_MARKS):
        return "sheet"
    if any(mark in head for mark in _WORDML_MARKS):
        return "word"
    return "xml"


def source_kind(path: str | Path) -> SourceKind | None:
    """Шлях читання файлу за розширенням (для .xml — ще й за вмістом).

    None — формат не приймається взагалі.
    """
    suffix = Path(path).suffix.lower()
    if suffix in PDF_SUFFIXES:
        return "pdf"
    if suffix in TEXT_SUFFIXES:
        return "text"
    if suffix in WORD_SUFFIXES:
        return "word"
    if suffix in SLIDE_SUFFIXES:
        return "slides"
    if suffix in DJVU_SUFFIXES:
        return "djvu"
    if suffix in SHEET_SUFFIXES:
        return "sheet"
    if suffix in XML_SUFFIXES:
        try:
            with Path(path).open("rb") as fh:
                return sniff_xml(fh.read(4096))
        except OSError:
            return "xml"
    return None


def accepted_label() -> str:
    """Перелік для повідомлення про помилку — групами, а не 25 розширеннями."""
    return (
        "PDF, Word (DOC, DOCX, RTF, ODT), PowerPoint (PPT, PPTX), DjVu, "
        "Excel (XLS, XLSX, CSV, ODS), XML, TXT і Markdown"
    )
