"""Довільний XML → Markdown-копія для текстового конвеєра.

XML буває двох родів, і читати їх треба по-різному:

  * ДАНІ — `<гармата><назва>Д-30</назва><калібр>122</калібр></гармата>`.
    Значення без імені тега безглузде («122» — чого?), тож короткі листки
    пишуться як «калібр: 122», а кожен запис — окремим абзацом;
  * ТЕКСТ — абзаци книги чи настанови (`<p>…</p>`, змішаний вміст із
    `<b>`, `<i>`). Тут ім'я тега — шум, і елемент іде суцільним текстом.

XML-файли Office 2003 (Word / Excel) сюди не потрапляють: їх розпізнає
`formats.sniff_xml` і відправляє тим самим шляхом, що й DOC/XLS.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

__all__ = ["xml_to_markdown"]

_WS_RE = re.compile(r"\s+")
# Листок довший за це або з такою кількістю слів — уже абзац тексту, а не поле.
PROSE_CHARS = 120
PROSE_WORDS = 12


def _local(tag: object) -> str:
    name = str(tag)
    return name.rsplit("}", 1)[-1] if "}" in name else name


def _clean(text: str | None) -> str:
    return _WS_RE.sub(" ", text or "").strip()


def xml_to_markdown(path: str | Path) -> str:
    """Розібрати XML і повернути текст, придатний для `_parse_text_source`."""
    root = ET.parse(path).getroot()
    lines: list[str] = []

    def walk(element: ET.Element) -> None:
        children = [c for c in element if isinstance(c.tag, str)]
        own = _clean(element.text)
        if children and (own or any(_clean(c.tail) for c in children)):
            # Змішаний вміст: абзац із вкрапленнями розмітки.
            text = _clean("".join(element.itertext()))
            if text:
                lines.extend((text, ""))
            return
        if not children:
            if own:
                prose = len(own) > PROSE_CHARS or own.count(" ") >= PROSE_WORDS
                lines.append(own if prose else f"{_local(element.tag)}: {own}")
            return
        before = len(lines)
        for child in children:
            walk(child)
        if len(lines) > before and lines[-1] != "":
            lines.append("")

    walk(root)
    text = "\n".join(lines).strip()
    return f"{text}\n" if text else ""
