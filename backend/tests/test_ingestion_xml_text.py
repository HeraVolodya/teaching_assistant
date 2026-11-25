"""Довільний XML → текст: дані з іменами полів, проза — суцільно."""

from __future__ import annotations

from app.ingestion.formats import sniff_xml, source_kind
from app.ingestion.xml_text import xml_to_markdown

DATA = """<?xml version="1.0" encoding="utf-8"?>
<каталог xmlns="urn:test">
  <гармата><назва>Д-30</назва><калібр>122 мм</калібр></гармата>
  <гармата><назва>2С1</назва><калібр>122 мм</калібр></гармата>
  <примітка>Дальність наведено для осколково-фугасного снаряда за нормальних
    умов, без урахування <b>поправок</b> на метеоумови.</примітка>
</каталог>
"""


def test_data_fields_keep_their_names_and_records_are_paragraphs(tmp_path) -> None:
    path = tmp_path / "каталог.xml"
    path.write_text(DATA, encoding="utf-8")
    text = xml_to_markdown(path)
    assert "назва: Д-30\nкалібр: 122 мм\n\nназва: 2С1" in text
    # Змішаний вміст — одним реченням, без імен тегів і без розриву на <b>.
    assert "без урахування поправок на метеоумови." in text
    assert "примітка:" not in text and "{urn:test}" not in text


def test_windows_1251_declaration_is_honoured(tmp_path) -> None:
    path = tmp_path / "старий.xml"
    path.write_bytes('<?xml version="1.0" encoding="windows-1251"?><а><б>Гаубиця</б></а>'
                     .encode("cp1251"))
    assert "б: Гаубиця" in xml_to_markdown(path)


def test_office_2003_xml_goes_the_office_way(tmp_path) -> None:
    assert sniff_xml(b'<?mso-application progid="Excel.Sheet"?><Workbook '
                     b'xmlns="urn:schemas-microsoft-com:office:spreadsheet">') == "sheet"
    assert sniff_xml(b'<?mso-application progid="Word.Document"?><w:wordDocument>') == "word"
    assert sniff_xml(b"<catalog/>") == "xml"
    path = tmp_path / "таблиця.xml"
    path.write_bytes(b'<?xml version="1.0"?><?mso-application progid="Excel.Sheet"?><Workbook/>')
    assert source_kind(path) == "sheet"
