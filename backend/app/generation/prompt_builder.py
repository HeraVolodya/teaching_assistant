"""Складання промпту: дворівнева стратегія.

ЧОМУ ТАК КОРОТКО. Успішність виконання ВСІХ інструкцій падає приблизно
експоненційно з їх кількістю. Системний промпт NeoLens — ~2500 токенів; на
локальній моделі 8-14B він виконується погано і непередбачувано. Промпт
системи-переможця UNLP 2026 — близько 80 слів. Тому:

  ТИР A — статичний системний промпт до 180 токенів, РІВНО П'ЯТЬ правил:
      1) роль, 2) заземлення, 3) формат цитат, 4) відмова, 5) мова.
    Правила 1-5 — БАЙТ-ІДЕНТИЧНИЙ префікс для ВСІХ асистентів. Це не
    естетика: LM Studio перевикористовує префіксний KV-кеш, і при перемиканні
    асистента незмінний префікс економить повний prefill. Персона асистента
    йде ПІСЛЯ правил і має жорсткий ліміт 250 токенів.

  ТИР B — «сендвіч»: питання йде І ПЕРЕД джерелами, І ПІСЛЯ них, а блок
    правил на 40-60 токенів стоїть ПІСЛЯ доказів і безпосередньо перед
    генерацією. Це рівно та розкладка, що виграла UNLP 2026, і рівно та
    позиція, яку малі моделі реально виконують: інструкція перед 2000
    токенами доказів для 12B-моделі фактично не існує.

БЮДЖЕТ (Gemma, українська ≈ 2.35 токена на слово):
    ~400 система + ~2150 (5 чанків) + ~150 сендвіч + ~600 відповідь ≈ 3300.
Тобто 8k контексту справді достатньо для одноразової заземленої відповіді,
а 16k купує близько трьох ходів історії.

ПОРОГИ ТУТ НЕ ЖИВУТЬ. Промптом неможливо надійно змусити 12B-модель
відмовитись — відмову виконує код (`AssistantConfig.confidence_threshold`,
утримання в ретривері). Правило 4 лише дає моделі дозвіл сказати «не знаю»,
коли код уже пропустив запит далі.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.backends.base import DEFAULT_STOP, ChatParams, Messages
from app.domain import AssistantConfig, RetrievedChunk

__all__ = [
    "PERSONA_MAX_TOKENS",
    "SYSTEM_MAX_TOKENS",
    "SYSTEM_RULES_UK",
    "UK_CHARS_PER_TOKEN",
    "UK_TOKENS_PER_WORD",
    "PromptBundle",
    "abstain_text",
    "assign_ordinals",
    "build_map_prompt",
    "build_prompt",
    "build_prompt_bundle",
    "build_reduce_prompt",
    "build_system_prompt",
    "citation_map",
    "estimate_tokens_uk",
    "fit_evidence",
    "format_evidence",
    "generation_params",
]

# Токенізаційний податок української: Gemma 2.35 токена на слово.
# (Qwen3 — 3.62-3.90, тобто на ~60% більше на той самий текст. Саме тому
# генератор — родина Gemma, а Qwen лишається для ембедингів і реранкінгу.)
UK_TOKENS_PER_WORD = 2.35
UK_CHARS_PER_TOKEN = 3.2

SYSTEM_MAX_TOKENS = 180
PERSONA_MAX_TOKENS = 250

# ---------------------------------------------------------------------------
# БАЙТ-ІДЕНТИЧНИЙ ПРЕФІКС. Не редагувати заради «краще звучить»: будь-яка
# зміна одного байта інвалідує префіксний KV-кеш для ВСІХ асистентів одразу.
# ---------------------------------------------------------------------------
SYSTEM_RULES_UK = (
    "1. Ти — навчальний асистент викладача Національної академії сухопутних військ.\n"
    "2. Відповідай лише за наданими нижче уривками з навчальних матеріалів; "
    "власних знань не додавай.\n"
    "3. Після кожного твердження став номер джерела у квадратних дужках: [1], [2]. "
    "Інших номерів не вигадуй.\n"
    "4. Якщо уривків недостатньо, прямо напиши про це — це правильна відповідь, а не невдача.\n"
    "5. Відповідай українською, навіть якщо джерела англійською або російською."
)

# Блок правил ТИРУ B: після доказів, перед генерацією. 40-60 токенів.
_TIER_B_RULES_COMPACT = (
    "ПРАВИЛА: спирайся лише на ДЖЕРЕЛА вище; після кожного твердження став [n]; "
    "якщо джерел недостатньо — скажи прямо; українською, стисло, один раз."
)
_TIER_B_RULES_FULL = (
    "ПРАВИЛА: спирайся лише на ДЖЕРЕЛА вище; після кожного твердження став [n]; "
    "поєднай відомості з різних джерел, якщо вони доповнюють одна одну; "
    "не переказуй питання; якщо джерел недостатньо — скажи прямо; "
    "українською, стисло, один раз. Після відповіді зупинись."
)

_WORD = re.compile(r"\S+")


def estimate_tokens_uk(text: str) -> int:
    """Оцінка довжини в токенах БЕЗ токенайзера.

    Точний токенайзер у API-процесі був би зайвою залежністю (а Gemma-шний і
    зовсім недоступний офлайн без ваг). Беремо максимум із двох незалежних
    оцінок — за словами і за символами — бо занижена оцінка бюджету коштує
    дорожче за завищену: вона дає `contextOverflowPolicy=stopAtLimit` посеред
    відповіді.
    """
    if not text:
        return 0
    words = len(_WORD.findall(text))
    return max(int(words * UK_TOKENS_PER_WORD), int(len(text) / UK_CHARS_PER_TOKEN)) + 1


def _trim_to_tokens(text: str, limit: int) -> tuple[str, bool]:
    """Обрізати по межі слова до бюджету токенів. True → обрізали."""
    if estimate_tokens_uk(text) <= limit:
        return text, False
    max_chars = int(limit * UK_CHARS_PER_TOKEN)
    cut = text[:max_chars]
    space = cut.rfind(" ")
    if space > max_chars // 2:
        cut = cut[:space]
    return cut.rstrip() + "…", True


def build_system_prompt(persona: str = "") -> str:
    """Тир A. Правила 1-5 незмінні; персона — після них, з жорстким лімітом."""
    if not persona or not persona.strip():
        return SYSTEM_RULES_UK
    trimmed, was_cut = _trim_to_tokens(persona.strip(), PERSONA_MAX_TOKENS)
    suffix = ("\n(Опис ролі обрізано до ліміту 250 токенів.)" if was_cut else "")
    return f"{SYSTEM_RULES_UK}\n\nРОЛЬ АСИСТЕНТА:\n{trimmed}{suffix}"


# ------------------------------------------------------------------ докази
def assign_ordinals(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Присвоїти [1]…[n] у порядку подання.

    Модель бачить ПОРЯДКОВІ НОМЕРИ, а не chunk_uid: 8-hex-схема NeoLens при
    100 тис. чанків має ~69% ймовірності колізії (це лише 32 біти), а
    32-hex-ідентифікатор з'їдає бюджет і провокує помилки переписування.
    """
    for i, rc in enumerate(chunks, start=1):
        rc.ordinal_in_prompt = i
    return chunks


def citation_map(chunks: list[RetrievedChunk]) -> dict[str, str]:
    """Мапа «[n] → chunk_uid», яка зберігається НА КОЖНЕ повідомлення асистента
    і накопичується між ходами: уточнювальне питання може цитувати чанк,
    знайдений два ходи тому."""
    return {str(rc.ordinal_in_prompt): rc.chunk.chunk_uid
            for rc in chunks if rc.ordinal_in_prompt}


def _source_header(rc: RetrievedChunk) -> str:
    label = rc.chunk.citation_label()
    path = rc.chunk.header_path.strip("/").replace("//", " › ").strip()
    bits = [rc.document_title or "Без назви"]
    if path:
        bits.append(path)
    if label:
        bits.append(label)
    return f"[{rc.ordinal_in_prompt}] " + " — ".join(bits)


def format_evidence(chunks: list[RetrievedChunk], *, char_budget: int | None = None) -> str:
    """Блок ДЖЕРЕЛА.

    Генератор бачить `display_text` — санітизоване тіло. Він НЕ бачить
    `embed_text`: контекстний префікс і картка документа існують для пошуку,
    а в промпті вони лише їли б бюджет і провокували переказ службової
    інформації замість відповіді. Санітизація контексту перед подачею в
    модель була найвищою за важелем інтервенцією в on-device POC.
    """
    # Захист від виклику в обхід build_prompt — але ЛИШЕ якщо номерів немає
    # взагалі. Перенумерувати вже призначені означало б зламати map-reduce,
    # де кожен документ обробляється окремо, а номери мусять лишатись
    # глобальними на весь запит.
    if all(rc.ordinal_in_prompt is None for rc in chunks):
        assign_ordinals(chunks)
    parts: list[str] = []
    for rc in chunks:
        body = rc.chunk.display_text.strip()
        if char_budget is not None and len(body) > char_budget:
            body = body[:char_budget].rstrip() + "…"
        parts.append(f"{_source_header(rc)}\n{body}")
    return "\n\n".join(parts)


@dataclass(slots=True)
class PromptBundle:
    messages: Messages
    evidence: list[RetrievedChunk]
    citations: dict[str, str] = field(default_factory=dict)
    estimated_tokens: int = 0
    dropped: int = 0
    truncated: bool = False
    strategy: str = "direct"        # direct | trimmed | truncated | map-reduce


def _user_message_tier_b(
    question: str,
    evidence_block: str,
    *,
    tier: str = "compact",
    history_block: str = "",
) -> str:
    """Сендвіч: питання — джерела — правила — питання."""
    rules = _TIER_B_RULES_FULL if tier == "full" else _TIER_B_RULES_COMPACT
    head = f"Питання: {question.strip()}"
    blocks = [head]
    if history_block:
        blocks.append(f"ПОПЕРЕДНІ ХОДИ:\n{history_block}")
    blocks.append(f"ДЖЕРЕЛА:\n{evidence_block}")
    blocks.append(rules)
    blocks.append(f"{head}\nВідповідь:")
    return "\n\n".join(blocks)


def _history_block(history: list[dict[str, str]] | None, *, max_turns: int = 2) -> str:
    if not history:
        return ""
    tail = history[-max_turns * 2:]
    lines = []
    for m in tail:
        role = "Викладач" if m.get("role") == "user" else "Асистент"
        text, _ = _trim_to_tokens((m.get("content") or "").strip(), 120)
        if text:
            lines.append(f"{role}: {text}")
    return "\n".join(lines)


def build_prompt(
    question: str,
    chunks: list[RetrievedChunk],
    config: AssistantConfig | None = None,
    *,
    persona: str = "",
    history: list[dict[str, str]] | None = None,
) -> Messages:
    """Публічний контракт модуля: питання + докази → повідомлення для LLM."""
    return build_prompt_bundle(
        question, chunks, config, persona=persona, history=history
    ).messages


def build_prompt_bundle(
    question: str,
    chunks: list[RetrievedChunk],
    config: AssistantConfig | None = None,
    *,
    persona: str = "",
    history: list[dict[str, str]] | None = None,
    char_budget: int | None = None,
) -> PromptBundle:
    """Те саме, але з мапою цитат і оцінкою бюджету — це потрібно генератору."""
    cfg = config or AssistantConfig()
    evidence = assign_ordinals(list(chunks))
    system = build_system_prompt(persona)
    user = _user_message_tier_b(
        question,
        format_evidence(evidence, char_budget=char_budget),
        tier=cfg.prompt_tier,
        history_block=_history_block(history),
    )
    messages: Messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return PromptBundle(
        messages=messages,
        evidence=evidence,
        citations=citation_map(evidence),
        estimated_tokens=estimate_tokens_uk(system) + estimate_tokens_uk(user),
        truncated=char_budget is not None,
        strategy="truncated" if char_budget is not None else "direct",
    )


# ------------------------------------------------------------- map-reduce
def build_map_prompt(
    question: str,
    document_title: str,
    chunks: list[RetrievedChunk],
    config: AssistantConfig | None = None,
    *,
    char_budget: int | None = None,
) -> Messages:
    """Крок MAP: часткова відповідь за ОДНИМ документом.

    Кожен виклик лишається малим, а «який документ що сказав» стає
    структурно явним — і тому цитованим. Це і є причина брати map-reduce
    замість refine (див. `generator.py`).
    """
    cfg = config or AssistantConfig()
    evidence_block = format_evidence(chunks, char_budget=char_budget)
    length = "трьома-чотирма" if cfg.prompt_tier == "full" else "двома-трьома"
    user = (
        f"Питання: {question.strip()}\n\n"
        f"ДЖЕРЕЛА (документ «{document_title}»):\n{evidence_block}\n\n"
        "ПРАВИЛА: випиши лише те, що цей документ каже по суті питання; "
        "збережи маркери [n]; якщо документ не відповідає — напиши рівно "
        f"«Немає відомостей.»; {length} реченнями.\n\n"
        f"Питання: {question.strip()}\nЧастковий висновок:"
    )
    return [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": user},
    ]


def build_reduce_prompt(
    question: str,
    partials: list[tuple[str, str]],
    config: AssistantConfig | None = None,
    *,
    persona: str = "",
) -> Messages:
    """Крок REDUCE: звести часткові висновки в одну відповідь.

    Маркери [n] у часткових висновках уже вказують на ті самі докази, тому
    reduce не має права їх переписувати — лише зберігати. Саме тут refine
    вироджувався б у повторення відповіді №1 із тихим ігноруванням джерел
    2..n, що прямо руйнує вимогу «поєднувати кілька джерел».
    """
    cfg = config or AssistantConfig()
    blocks = []
    for i, (title, text) in enumerate(partials, start=1):
        blocks.append(f"ЧАСТКОВИЙ ВИСНОВОК {i} (документ «{title}»):\n{text.strip()}")
    body = "\n\n".join(blocks)
    rules = _TIER_B_RULES_FULL if cfg.prompt_tier == "full" else _TIER_B_RULES_COMPACT
    user = (
        f"Питання: {question.strip()}\n\n"
        f"{body}\n\n"
        f"{rules} Збережи всі маркери [n] з часткових висновків; "
        "не вигадуй нових номерів.\n\n"
        f"Питання: {question.strip()}\nЗведена відповідь:"
    )
    return [
        {"role": "system", "content": build_system_prompt(persona)},
        {"role": "user", "content": user},
    ]


# ----------------------------------------------------------------- відмова
def abstain_text(question: str, near: list[RetrievedChunk] | None = None) -> str:
    """Чесна відмова. Формується КОДОМ, а не моделлю.

    Причина проста: якщо ретривер уже вирішив утриматись, звертатись по цю
    відповідь до LLM означає дати їй шанс вигадати. Дешевше і чесніше —
    сказати прямо й показати, що знайшлося найближче.
    """
    lines = [
        "У наданих матеріалах недостатньо інформації, щоб відповісти на це запитання."
    ]
    if near:
        lines.append("Найближче, що знайшлося в матеріалах:")
        for rc in near[:3]:
            label = rc.chunk.citation_label()
            title = rc.document_title or "Без назви"
            lines.append(f"— {title}{(', ' + label) if label else ''}")
        lines.append(
            "Якщо потрібної теми немає в колекції, додайте відповідний документ."
        )
    return "\n".join(lines)


# --------------------------------------------------------------- параметри
def generation_params(config: AssistantConfig | None = None, *,
                      detailed: bool = False) -> ChatParams:
    """Параметри проти розбігання (план, §10)."""
    cfg = config or AssistantConfig()
    return ChatParams(
        temperature=cfg.temperature,
        # Подвоєння, а НЕ константа. Раніше тут стояло 1200 — рівно вдвічі від
        # тодішнього дефолту 600. Щойно дефолт піднявся (див. AssistantConfig.
        # max_tokens), константа стала МЕНШОЮ за звичайний режим: «детальна
        # відповідь» отримувала тісніший бюджет, ніж коротка, і в reasoning-
        # моделей обривалася першою. Співвідношення тримаємо, число — ні.
        max_tokens=cfg.max_tokens * 2 if detailed else cfg.max_tokens,
        repeat_penalty=cfg.repeat_penalty,
        stop=DEFAULT_STOP,
    )


# ----------------------------------------------------------------- бюджет
def fit_evidence(
    chunks: list[RetrievedChunk],
    *,
    context_tokens: int,
    max_answer_tokens: int,
    persona_tokens: int = 0,
    min_chunks: int = 2,
    reserve_tokens: int = 250,
) -> tuple[list[RetrievedChunk], int | None, bool]:
    """Стратегія переповнення, у порядку: менше діставати → обрізати вікно.

    Третій крок (map-reduce) робить уже генератор, коли навіть обрізане вікно
    не влазить. Порядок не довільний: викидання найгіршого за реранкером
    кандидата коштує найменше, обрізання тіла — більше (можна відрізати саме
    те речення), а map-reduce коштує ще одного проходу моделі на документ.

    Повертає (докази, ліміт символів на чанк або None, чи довелось обрізати).
    """
    # Накладні: системний промпт + персона + сендвіч (питання двічі, правила).
    overhead = SYSTEM_MAX_TOKENS + persona_tokens + reserve_tokens
    budget = context_tokens - max_answer_tokens - overhead
    if budget <= 0:
        return chunks[:min_chunks], 600, True

    kept = list(chunks)

    def total(items: list[RetrievedChunk], cap: int | None = None) -> int:
        return sum(estimate_tokens_uk(
            rc.chunk.display_text[:cap] if cap else rc.chunk.display_text
        ) + 24 for rc in items)

    # 1) менше діставати
    while len(kept) > min_chunks and total(kept) > budget:
        kept.pop()
    if total(kept) <= budget:
        return kept, None, False

    # 2) обрізати вікно кожного чанка
    for cap in (2000, 1400, 1000, 700, 500):
        if total(kept, cap) <= budget:
            return kept, cap, True
    return kept[:min_chunks], 500, True
