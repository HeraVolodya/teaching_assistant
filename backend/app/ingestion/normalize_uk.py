"""Нормалізація українського тексту — ОДНА функція для індексації і для запиту.

Найтихіша поломка в усій системі живе саме тут. Якщо індексація нормалізує
текст бодай трохи інакше, ніж запит, BM25 просто перестає знаходити: без
винятку, без падіння тесту, без сліду в логах. Тому нормалізація — одна
функція, а `TEXT_PREPROC_VERSION` входить у ключ версіювання ембедингів
(див. `app/embeddings/registry.py`), щоб зміна правил робила старі вектори
несумісними ГОЛОСНО.

Порядок кроків (він не довільний):
  1. Unicode NFC.
  2. Уніфікація апострофів і тире, видалення м'яких переносів.
  3. Склеювання слів, розірваних переносом на кінці рядка. Робиться ПІСЛЯ
     уніфікації тире (щоб ловити й U+2010 на кінці рядка) і ПЕРЕД виправленням
     гомогліфів — бо гомогліфний фільтр аналізує токен цілком, а розірваний
     токен дає хибний вердикт.
  4. Виправлення гомогліфів.
  5. casefold().

Що свідомо НЕ робиться: и/і не зливаються (фонемно контрастні), ґ/і/ї/є
зберігаються (у FTS5 за це відповідає `remove_diacritics 2`, що зачіпає лише
латиницю).
"""

from __future__ import annotations

import re
import unicodedata

__all__ = [
    "TEXT_PREPROC_VERSION",
    "APOSTROPHE",
    "normalize_uk",
    "tokenize",
    "search_terms",
    "expand_compounds",
    "fix_homoglyphs",
    "cyrillic_ratio",
    "has_unambiguous_cyrillic",
    "is_cyrillic",
]

# Версія конвеєра нормалізації. МАЄ збігатися з
# app.embeddings.registry.TEXT_PREPROC_VERSION — це перевіряє тест
# test_ingestion_normalize.py::test_preproc_version_matches_embedding_registry.
# Піднімати щоразу, коли змінюються правила нижче: інакше старі вектори і
# старий FTS-індекс тихо стають несумісними з новими запитами.
TEXT_PREPROC_VERSION = 1

APOSTROPHE = "'"  # U+0027 — канонічний апостроф після нормалізації

# --------------------------------------------------------------- символьні мапи
# П'ять форм апострофа, які реально трапляються в українських PDF: типографська
# лапка з Word, «правильний» ʼ зі шрифтів UA, гравіс і акут із друкарських
# машинок і прайм із математичних шрифтів.
_APOSTROPHES: dict[int, str] = {
    0x2019: APOSTROPHE,  # ’ right single quotation mark
    0x02BC: APOSTROPHE,  # ʼ modifier letter apostrophe
    0x0060: APOSTROPHE,  # ` grave accent
    0x00B4: APOSTROPHE,  # ´ acute accent
    0x2032: APOSTROPHE,  # ′ prime
}
# U+2010..U+2015 — уся родина дефісів і тире, U+2212 — математичний мінус.
_DASHES: dict[int, str] = {cp: "-" for cp in range(0x2010, 0x2016)}
_DASHES[0x2212] = "-"

_TRANSLATION: dict[int, str | None] = {
    **_APOSTROPHES,
    **_DASHES,
    0x00AD: None,  # м'який перенос згортається (видаляється) повністю
    0x200B: None,  # zero width space — типовий сміттєвий символ конвертерів
    0x00A0: " ",   # нерозривний пробіл: у токенізації поводиться як пробіл
}

# Латинські гомогліфи → кириличні. OCR українських підручників плутає їх
# постійно; для людини різниця невидима, для BM25 — смертельна.
# Y додано понад мінімальний перелік: це той самий клас помилки, а правило
# «немає однозначно латинської літери» не дає йому зіпсувати англійські слова.
_HOMOGLYPHS: dict[str, str] = {
    "i": "і", "I": "І",
    "e": "е", "E": "Е",
    "o": "о", "O": "О",
    "a": "а", "A": "А",
    "c": "с", "C": "С",
    "p": "р", "P": "Р",
    "x": "х", "X": "Х",
    "y": "у", "Y": "У",
    "B": "В", "H": "Н", "K": "К", "M": "М", "T": "Т",
}
# Кириличні літери, які САМІ є гомогліфами: їх наявність у токені нічого не
# доводить, тому вони не рахуються за доказ «це кирилиця».
_AMBIGUOUS_CYRILLIC = frozenset(_HOMOGLYPHS.values())

_CYRILLIC_RANGES = ((0x0400, 0x04FF), (0x0500, 0x052F), (0x2DE0, 0x2DFF), (0xA640, 0xA69F))

# Сегмент слова: суцільний пробіг літер/цифр. Дефіс і апостроф сюди НЕ входять,
# тому «PDF-файл» аналізується як два незалежні сегменти, і латинська частина
# лишається недоторканою.
_SEGMENT_RE = re.compile(r"[^\W_]+")

# Токен: сегменти, зчеплені апострофом або дефісом. «п'ять», «Д-30»,
# «військово-технічний» лишаються цілими — рівно як у tokenchars FTS5.
_TOKEN_RE = re.compile(r"[^\W_]+(?:['\-][^\W_]+)*")

# Перенос на кінці рядка: «гар-\nмата» → «гармата». Виконується після
# уніфікації тире. Плата за це — справжній складений термін, розірваний на межі
# рядка («військово-\nтехнічний»), склеюється в одне слово; у підручниках
# переносів на порядок більше, ніж таких випадків, тож обмін вигідний.
_HYPHEN_BREAK_RE = re.compile(r"(?<=[^\W\d_])-[ \t]*\r?\n[ \t]*(?=[^\W\d_])")


def is_cyrillic(ch: str) -> bool:
    """Чи належить символ кириличним блокам Unicode."""
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _CYRILLIC_RANGES)


_is_cyrillic = is_cyrillic  # внутрішній псевдонім, лишений для стислості нижче


def has_unambiguous_cyrillic(text: str) -> bool:
    """Чи є в тексті хоч одна кирилична літера, що НЕ є гомогліфом латиниці."""
    return any(_is_cyrillic(ch) and ch not in _AMBIGUOUS_CYRILLIC for ch in text)


def _fix_segment(segment: str) -> str:
    """Виправити гомогліфи в одному сегменті слова.

    Три умови, і всі три обов'язкові:
      * у сегменті вже є однозначно кирилична літера — інакше це англійське
        слово, і чіпати його не можна;
      * у сегменті є латинська літера — інакше нічого виправляти;
      * УСІ латинські літери сегмента мають кириличний двійник. Якщо є хоч одна
        літера без двійника (d, f, g, b, l, ...), це справді змішаний токен на
        кшталт «PDFфайл» або коду деталі, а не слід OCR: гомогліфна підміна за
        визначенням підставляє тільки схожі літери.
    """
    if not has_unambiguous_cyrillic(segment):
        return segment
    latin = [ch for ch in segment if "A" <= ch <= "Z" or "a" <= ch <= "z"]
    if not latin or any(ch not in _HOMOGLYPHS for ch in latin):
        return segment
    return "".join(_HOMOGLYPHS.get(ch, ch) for ch in segment)


def fix_homoglyphs(text: str) -> str:
    """Замінити латинські гомогліфи на кириличні всередині кириличних слів."""
    return _SEGMENT_RE.sub(lambda m: _fix_segment(m.group(0)), text)


def normalize_uk(text: str) -> str:
    """Канонічна форма українського тексту. Ідемпотентна.

    Застосовується ІДЕНТИЧНО до тіла чанка при індексації і до запиту при
    пошуку. Будь-яка розбіжність тихо руйнує BM25.
    """
    if not text:
        return ""
    s = unicodedata.normalize("NFC", text)
    s = s.translate(_TRANSLATION)
    s = _HYPHEN_BREAK_RE.sub("", s)
    s = fix_homoglyphs(s)
    return s.casefold()


def tokenize(text: str) -> list[str]:
    """Розбити НОРМАЛІЗОВАНИЙ текст на токени.

    Апостроф і дефіс — частина токена, як і в `tokenchars` таблиці chunk_fts:
    інакше «п'ять» перетворюється на «п» + «ять», а «Д-30» на «д» + «30».
    """
    return _TOKEN_RE.findall(text)


def expand_compounds(token: str) -> list[str]:
    """Подвійна емісія складених слів.

    «військово-технічний» → ["військово-технічний", "військово", "технічний"].
    Запит «технічний» має знаходити чанк, де слово трапилось лише у складі
    терміна; без цього ціла родина українських складених прикметників
    невидима для BM25.
    """
    if "-" not in token:
        return [token]
    out = [token]
    for part in token.split("-"):
        if len(part) >= 3 and not part.isdigit() and part not in out:
            out.append(part)
    return out


def search_terms(text: str) -> list[str]:
    """Повний конвеєр: нормалізація → токенізація → подвійна емісія складених.

    Це те, що подається у FTS5 (колонка `forms`) і те, з чого будується запит.
    Позначення (Д-30, ДСТУ 3008:2015) сюди теж потрапляють як звичайні токени —
    їх окремо перехоплює `app.ingestion.lemmatize`.
    """
    terms: list[str] = []
    for token in tokenize(normalize_uk(text)):
        terms.extend(expand_compounds(token))
    return terms


def cyrillic_ratio(text: str) -> float:
    """Частка кириличних літер серед усіх літер. 0.0, якщо літер немає."""
    letters = 0
    cyr = 0
    for ch in text:
        if ch.isalpha():
            letters += 1
            if _is_cyrillic(ch):
                cyr += 1
    return cyr / letters if letters else 0.0
