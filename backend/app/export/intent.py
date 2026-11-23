"""Розпізнати прохання «зроби презентацію / документ / таблицю».

ЧОМУ ПРАВИЛАМИ, А НЕ ВИКЛИКОМ ІНСТРУМЕНТІВ МОДЕЛІ
Перевірено 23.11.2026 на живій Gemma 4 E4B: інструменти (tool calling) вона
викликає правильно. Але маршрутизація моделлю коштує ~500 токенів схеми в
КОЖНОМУ запиті, недоступна заглушці й CI без моделі, а її помилку не видно:
відповідь просто приходить текстом. Прохання про файл має сталу форму —
дієслово-наказ і назва формату («склади таблицю», «зроби презентацію»,
«збережи у Word»), тож правило миттєве, детерміноване й перевіряється тестом.

Ціни помилок різні, і правила під них підібрані. Зайве спрацювання коштує
мало: викладач однаково бачить зміст у чаті, просто з файлом поруч. Тому
досить наказу з назвою формату. А от питання «Як скласти таблицю поправок?»
файлом ставати не мусить — інфінітив у питальному реченні не є проханням.

ЩО ПОВЕРТАЄТЬСЯ
Формат і ТЕМА — те, що шукати в матеріалах. Командні слова з теми прибрано:
«зроби презентацію на 5 слайдів» у пошуковому запиті лише шуміло б у BM25.
Порожня тема («зроби з цього презентацію») означає «тема попереднього
питання» — її підставляє чат, бо лише він знає історію.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

__all__ = [
    "FORMATS",
    "MAX_SLIDES",
    "MIN_SLIDES",
    "DocumentFormat",
    "DocumentRequest",
    "detect_document_request",
]

DocumentFormat = Literal["pptx", "docx", "xlsx"]
FORMATS: tuple[DocumentFormat, ...] = ("pptx", "docx", "xlsx")

# Межі кількості слайдів. Верхня — від вікна моделі: 8K контексту мінус
# джерела лишають ~2,5 тис. токенів на вивід, тобто ~12 слайдів по 4 тези.
MIN_SLIDES = 2
MAX_SLIDES = 12


@dataclass(frozen=True, slots=True)
class DocumentRequest:
    format: DocumentFormat
    # Що шукати в матеріалах. "" — тема не названа («зроби з цього таблицю»).
    topic: str
    # Скільки слайдів назвав викладач; None — на розсуд моделі (5–7).
    slides: int | None = None
    # Просили саме ТАБЛИЦЮ. Для xlsx це завжди так; для docx — «таблицю у
    # Word»: документ, головне в якому таблиця, а не розділи з абзацами.
    table: bool = False


# ------------------------------------------------------------------ словник
_W = r"[\w'-]"   # символ слова разом з апострофом і дефісом («п'ять», «план-конспект»)

# Накази (і майбутній час «зробиш?» як ввічливе прохання).
_IMPERATIVES = (
    "зроби", "зробіть", "зробиш", "створи", "створіть", "створиш",
    "склади", "складіть", "складеш", "сформуй", "сформуйте", "сформуєш",
    "підготуй", "підготуйте", "підготуєш", "згенеруй", "згенеруйте",
    "оформи", "оформіть", "оформиш", "експортуй", "експортуйте",
    "збережи", "збережіть", "перетвори", "перетворіть", "переведи", "переведіть",
    "конвертуй", "конвертуйте", "напиши", "напишіть", "розроби", "розробіть",
    "виведи", "виведіть", "запиши", "запишіть", "упорядкуй", "систематизуй",
)
_INFINITIVES = (
    "зробити", "створити", "скласти", "сформувати", "підготувати", "згенерувати",
    "оформити", "експортувати", "зберегти", "перетворити", "перевести",
    "конвертувати", "написати", "розробити", "вивести", "записати",
)
# Слова, після яких інфінітив стає проханням: «можеш скласти таблицю?».
_MODALS = (
    "можеш", "можете", "зможеш", "зможете", "прошу", "просимо", "треба", "потрібно",
    "необхідно", "хочу", "хочемо", "хотів би", "хотіла б", "допоможи", "допоможіть",
    "можна", "варто", "слід",
)
# «Потрібна презентація про …» — прохання без дієслова.
_NEEDS = r"потрібн[аеоийі]|потрібен|необхідн[аеоийі]"

# Питальне речення — не прохання, навіть з інфінітивом після «можна».
_QUESTION_START = re.compile(
    r"^\s*(?:як|яким\s+чином|що|чому|навіщо|коли|де|куди|звідки|хто|який|яка|яке|які|"
    r"якого|якої|яких|скільки|чим|чи|в\s+чому|у\s+чому|для\s+чого)(?![\w'-])",
    re.IGNORECASE,
)
# …крім звертання до самого асистента: «Чи можеш скласти таблицю?» — прохання,
# а «Чи можна скласти таблицю без …?» — питання про можливість.
_ASKS_ASSISTANT = re.compile(r"^\s*чи\s+(?:можеш|можете|зможеш|зможете)(?![\w'-])", re.IGNORECASE)

# Назви форматів-ПРЕДМЕТІВ (те, що просять створити).
# Правопис — поблажливий: «ПРИЗЕНТАЦІЮ», «презинтація», «презентацию» — живі
# написання викладачів (перше — з першого ж прохання в застосунку, 23.11.2026,
# і тоді воно мовчки стало звичайною відповіддю без файлу).
_NOUNS: tuple[tuple[DocumentFormat, str], ...] = (
    ("pptx", r"пр[еиі]з[еиі]нтац[іи](?:я|ю|ї|єю|и)|слайд(?:и|ів|ами)?"),
    ("xlsx", r"(?:електронн\w*\s+)?таблиц(?:я|ю|і|ею|у)"),
    ("docx", r"документ(?:а|і|ом)?|конспект(?:а|у|ом)?(?:\s+(?:заняття|уроку|лекції|семінару))?"
             r"|план(?:-конспект)?\s+(?:заняття|уроку|лекції|семінару)|реферат(?:у)?"
             r"|доповід(?:ь|і)|довідк(?:а|у|и)|пам'ятк(?:а|у|и)|методичк(?:а|у|и)"
             r"|методичн\w*\s+(?:розробк|рекомендаці)\w*"),
)
# Назви ПРОГРАМ мають пріоритет: «зроби таблицю у Word» — це документ Word.
_APPS: tuple[tuple[DocumentFormat, str], ...] = (
    ("pptx", r"power\s*point|павер\s*поінт\w*|пауер\s*поінт\w*|pptx"),
    ("xlsx", r"excel|ексел\w*|ексель|xlsx"),
    ("docx", r"word|ворд\w*|docx"),
)
# Прикметники перед назвою формату («коротку презентацію») — частина команди,
# а не теми.
_ADJECTIVES = (
    r"(?:(?:коротк|стисл|детальн|докладн|невелик|прост|повн|загальн|навчальн|окрем|"
    r"гарн|зручн|підсумков|порівняльн|зведен|нов|електронн)\w*\s+){0,2}"
)
# Прийменник перед назвою формату робить її ДЖЕРЕЛОМ чи темою, а не
# предметом: «зроби висновок з таблиці», «конспект про документ».
_NOT_OBJECT = {"з", "із", "зі", "від", "до", "про", "по", "щодо", "для", "без", "після"}

# Скільки слів після наказу шукаємо назву формату. Вісім покривають
# «Зроби, будь ласка, для курсантів другого курсу коротку презентацію».
_WINDOW_WORDS = 8

_NUMBER_WORDS = {
    "два": 2, "дві": 2, "три": 3, "чотири": 4, "п'ять": 5, "шість": 6, "сім": 7,
    "вісім": 8, "дев'ять": 9, "десять": 10, "одинадцять": 11, "дванадцять": 12,
}
_NUMBER = r"(?:\d{1,2}(?:\s*[-–—]\s*\d{1,2})?|" + "|".join(_NUMBER_WORDS) + r")"
_SLIDES_COUNT = re.compile(
    rf"(?:(?:на|з|із|зі|у|в)\s+)?({_NUMBER})\s+слайд\w*", re.IGNORECASE,
)

# Слова, що відсилають до попереднього ходу, а не називають тему.
_DEICTIC = {
    "з", "із", "зі", "у", "в", "на", "це", "цього", "цієї", "цю", "ця", "цей", "цим", "цими",
    "оце", "оцього", "того", "те", "та", "ту", "усе", "все", "всього", "усього", "її", "його",
    "їх", "неї", "нього", "них", "відповідь", "відповіді", "відповіддю", "попередню",
    "попередньої", "останню", "останньої", "вище", "вищенаведене", "вищенаведеного",
    "наведене", "наведеного", "мені", "нам", "для", "мене", "нас", "будь", "ласка",
    "тему", "теми", "свою", "свого", "мою", "моєї", "текст", "тексту", "матеріал",
    "матеріалу", "відповідно", "також", "ще", "раз", "один", "формат", "форматі",
    "вигляді", "файл", "файлі", "файлом", "файлу", "як", "і", "й", "а", "але",
    "тепер", "зараз", "знову", "заново", "теж", "так", "само",
}

_TOPIC_MARKERS = re.compile(
    r"(?:на\s+тему|за\s+темою|по\s+темі|з\s+теми|тема)\s*[:—–-]?\s*", re.IGNORECASE,
)
_LEADING_CONNECTORS = re.compile(
    r"^(?:\s|[,.:;—–\-«»\"“”„']|про(?![\w'-])|щодо(?![\w'-])|стосовно(?![\w'-])|"
    r"відносно(?![\w'-])|по(?![\w'-])|з\s+питання(?![\w'-])|будь\s+ласка|будь-ласка|"
    # Адресат — не тема: «конспект для курсантів про …» шукає «…».
    r"для\s+(?:курсант|студент|слухач|учн|викладач|офіцер|сержант|солдат|військовослужбовц)\w*"
    r"(?:\s+\d+(?:-?го)?\s+курсу)?(?![\w'-]))+",
    re.IGNORECASE,
)
_POLITE = re.compile(r"будь[\s-]+ласка|(?<![\w'-])(?:мені|нам)(?![\w'-])", re.IGNORECASE)
_QUOTES = str.maketrans({c: " " for c in "«»\"“”„"})
_APOSTROPHES = str.maketrans({"’": "'", "ʼ": "'", "`": "'", "‘": "'"})
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n+")


def _alternation(words: tuple[str, ...]) -> str:
    # Довші першими: «хотів би» не має програвати «хочу».
    return "|".join(
        re.escape(w).replace("\\ ", " ").replace(" ", r"\s+")
        for w in sorted(words, key=len, reverse=True)
    )


_IMPERATIVE_RE = re.compile(
    rf"(?<![\w'-])(?<!не\s)({_alternation(_IMPERATIVES)})(?![\w'-])", re.IGNORECASE,
)
_MODAL_RE = re.compile(
    rf"(?<![\w'-])(?:{_alternation(_MODALS)})\s+(?:{_W}+\s+){{0,2}}?"
    rf"({_alternation(_INFINITIVES)})(?![\w'-])",
    re.IGNORECASE,
)
_NEEDS_RE = re.compile(rf"(?<![\w'-])(?:{_NEEDS})(?![\w'-])", re.IGNORECASE)


def _bounded(pattern: str) -> str:
    return rf"(?<![\w'-])(?:{pattern})(?![\w'-])"


_NOUN_RES = [(fmt, re.compile(_bounded(p), re.IGNORECASE)) for fmt, p in _NOUNS]
_APP_RES = [(fmt, re.compile(_bounded(p), re.IGNORECASE)) for fmt, p in _APPS]
_TABLE_NOUN_RE = re.compile(dict(_NOUNS)["xlsx"], re.IGNORECASE)
# Те, що стоїть ПЕРЕД назвою формату й належить команді, а не темі:
# «у вигляді таблиці», «у форматі Word», «коротку презентацію», «10 слайдів».
_FORMAT_PREFIX_RE = re.compile(
    r"(?:(?<![\w'-])(?:у|в|як)\s+(?:(?:форматі|вигляді|файл\w*)\s+)?)?"
    # «файл презентації» — «файл» теж частина команди.
    + r"(?:(?<![\w'-])файл\w*\s+)?"
    + rf"(?:{_NUMBER}\s+)?" + _ADJECTIVES + rf"(?:{_NUMBER}\s+)?$",
    re.IGNORECASE,
)
# Назви програм у темі не бувають ніколи — їх вирізаємо всюди.
_APP_PHRASE_RE = re.compile(
    r"(?<![\w'-])(?:(?:у|в|як)\s+(?:(?:форматі|вигляді|файл\w*)\s+)?)?(?:"
    + "|".join(p for _, p in _APPS) + r")(?![\w'-])",
    re.IGNORECASE,
)


# ------------------------------------------------------------------ розбір
def detect_document_request(message: str) -> DocumentRequest | None:
    """Прохання про файл або None, якщо це звичайне питання."""
    text = (message or "").translate(_APOSTROPHES)
    sentences = [s.strip() for s in _SENTENCE_END.split(text) if s and s.strip()]
    for index, sentence in enumerate(sentences):
        hit = _match(sentence)
        if hit is None:
            continue
        fmt, trigger_end, span = hit
        topic = _topic(sentence, trigger_end, span)
        if not topic:
            # «Що таке СОБ? Зроби про це презентацію.» — тема в сусідньому реченні.
            others = [s for j, s in enumerate(sentences) if j != index and _match(s) is None]
            topic = _clean_topic(" ".join(others))
        slides = _slides(text) if fmt == "pptx" else None
        named = sentence[trigger_end:][span[0]:span[1]]
        table = fmt == "xlsx" or (fmt == "docx" and bool(_TABLE_NOUN_RE.fullmatch(named)))
        return DocumentRequest(format=fmt, topic=topic, slides=slides, table=table)
    return None


_Span = tuple[int, int]


def _match(sentence: str) -> tuple[DocumentFormat, int, _Span] | None:
    """(формат, кінець тригера, межі назви формату в хвості) — якщо речення
    просить файл."""
    triggers: list[re.Match[str]] = list(_IMPERATIVE_RE.finditer(sentence))
    if _ASKS_ASSISTANT.search(sentence) or not _QUESTION_START.search(sentence):
        triggers += list(_MODAL_RE.finditer(sentence))
        triggers += list(_NEEDS_RE.finditer(sentence))
    for trigger in sorted(triggers, key=lambda m: m.start()):
        tail = sentence[trigger.end():]
        noun = _object_format(tail)
        app = _app_format(tail)
        if app is not None:
            # Програма визначає формат, але вирізати з теми треба й предмет:
            # «склади таблицю у Word» — тема без «таблицю».
            return app[0], trigger.end(), noun[1] if noun is not None else app[1]
        if noun is not None:
            return noun[0], trigger.end(), noun[1]
    return None


def _app_format(tail: str) -> tuple[DocumentFormat, _Span] | None:
    """Назва програми будь-де після наказу: «склади список … в Excel»."""
    found = [(m.start(), m.end(), fmt) for fmt, rx in _APP_RES for m in rx.finditer(tail)]
    if not found:
        return None
    start, end, fmt = min(found)
    return fmt, (start, end)


def _object_format(tail: str) -> tuple[DocumentFormat, _Span] | None:
    """Перша назва формату-предмета у вікні з кількох слів після наказу."""
    window = _first_words(tail, _WINDOW_WORDS)
    found: list[tuple[int, int, DocumentFormat]] = []
    for fmt, rx in _NOUN_RES:
        for m in rx.finditer(window):
            before = window[: m.start()].split()
            if before and before[-1].strip(",.:;—–-«»\"").lower() in _NOT_OBJECT:
                continue
            found.append((m.start(), m.end(), fmt))
    if not found:
        return None
    start, end, fmt = min(found)
    return fmt, (start, end)


def _first_words(text: str, count: int) -> str:
    matches = list(re.finditer(rf"{_W}+", text))
    if len(matches) <= count:
        return text
    return text[: matches[count - 1].end()]


def _topic(sentence: str, trigger_end: int, span: _Span) -> str:
    # Усе, що до тригера, — звертання й увічливість («Будь ласка, зроби …»),
    # а не тема: тема йде ПІСЛЯ наказу. Назву формату вирізаємо лише ту, що
    # визначила формат: «конспект про таблиці стрільби» мусить шукати таблиці.
    rest = sentence[trigger_end:]
    head = _FORMAT_PREFIX_RE.sub("", rest[: span[0]])
    rest = f"{head} {rest[span[1]:]}"
    rest = _APP_PHRASE_RE.sub(" ", rest)
    rest = _SLIDES_COUNT.sub(" ", rest)
    return _clean_topic(rest)


def _clean_topic(raw: str) -> str:
    text = raw.translate(_APOSTROPHES)
    found = list(_TOPIC_MARKERS.finditer(text))
    if found:
        # «конспект для курсантів на тему «Вогнева позиція»» — тема після маркера.
        text = text[found[-1].end():]
    text = _POLITE.sub(" ", text)
    text = text.translate(_QUOTES)
    text = _LEADING_CONNECTORS.sub("", text)
    text = re.sub(r"\s+", " ", text).strip(" ,.;:!?—–-")
    meaningful = [
        w for w in re.findall(rf"{_W}+", text)
        # Абревіатура з двох літер — теж тема: «ВП», «БК».
        if (w.lower() not in _DEICTIC and len(w.strip("'-")) >= 3) or (len(w) >= 2 and w.isupper())
    ]
    return text if meaningful else ""


def _slides(text: str) -> int | None:
    m = _SLIDES_COUNT.search(text)
    if m is None:
        return None
    raw = m.group(1).lower()
    # «8–10 слайдів» — беремо верхню межу: просили «не менше вісьмох».
    value = _NUMBER_WORDS[raw] if raw in _NUMBER_WORDS else max(int(n) for n in re.findall(r"\d+", raw))
    return max(MIN_SLIDES, min(MAX_SLIDES, value))
