"""Заземленість відповіді: суддя на ЛОКАЛЬНІЙ моделі, відмови, дійсність цитат.

Три незалежні сигнали, і вони навмисно різної ціни:

  1. **Дійсність цитат** — БЕЗ судді взагалі. Рахується з таблиці
     `unresolved_citations`, яку генератор наповнює безкоштовно на кожній
     відповіді. Це найдешевша й найнадійніша цифра всього оцінювання:
     маркер, за яким немає фрагмента, — це доведена галюцинація, а не думка
     ще однієї моделі.
  2. **Частка коректних відмов** — теж без судді. Питання, відповіді на яке в
     корпусі свідомо немає (`GoldQuestion` з порожнім `gold_chunk_uids`),
     мусить дати утримання. Порогом У КОДІ, а не проханням у промпті
     (контракт, правило 8), тож це вимірювана властивість, а не сподівання.
  3. **Заземленість тверджень** — суддя на локальній моделі через LM Studio.
     Найдорожча й найменш надійна з трьох, тому вона стоїть ТРЕТЬОЮ, а не
     першою: `logprobs` у LM Studio не реалізовано (план, §10), тож
     перплексійна оцінка недоступна, і все, що можна отримати від судді, —
     текстовий вердикт ТАК / НІ / ЧАСТКОВО.

МЕТОДОЛОГІЧНА ПАСТКА, ЯКУ ТУТ ОБХОДЯТЬ (план, §9)
-------------------------------------------------
В абляції «префікс шляху заголовків увімкнено / вимкнено» на ОДНОМУ пулі
згода двох анотаторів падає з Cohen's κ 0.45 до 0.04. Причина не в анотаторах:
зняття ланцюга заголовків прибирає сам сигнал, за яким можна розрізнити
варіанти, тож обидва анотатори починають вгадувати. Наслідок для коду:

    АБСОЛЮТНОЮ LLM-суддею на обрізаному тексті префікс НЕ ОЦІНЮВАТИ.

Для порівняння варіантів чанкування й префіксів існує `pairwise_compare()` —
попарне порівняння двох кандидатів на одному питанні, з обов'язковою
перестановкою позицій (position-swap): малі моделі мають сильне позиційне
упередження, і без перестановки вимірюється воно, а не якість.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

__all__ = [
    "Evidence",
    "ClaimVerdict",
    "GroundednessReport",
    "AbstentionReport",
    "CitationValidity",
    "PairwiseVerdict",
    "Judge",
    "LexicalJudge",
    "LlmJudge",
    "JUDGE_SYSTEM_UK",
    "PAIRWISE_SYSTEM_UK",
    "split_claims",
    "create_judge",
    "judge_answer",
    "abstention_stats",
    "citation_validity",
    "pairwise_compare",
    "summarize_groundedness",
]


# --------------------------------------------------------------------- дані
@dataclass(frozen=True, slots=True)
class Evidence:
    """Фрагмент, поданий моделі. `ordinal` — це те саме [n], що бачила модель."""

    ordinal: int
    text: str
    title: str = ""
    page_label: str = ""

    def as_block(self) -> str:
        head = f"[{self.ordinal}]"
        if self.title:
            head += f" {self.title}"
        if self.page_label:
            head += f" — {self.page_label}"
        return f"{head}\n{self.text}"

    @classmethod
    def from_retrieved(cls, items: Sequence[Any]) -> list["Evidence"]:
        """Побудувати докази з `list[RetrievedChunk]`.

        `ordinal_in_prompt` виставляє генератор; коли його ще немає (виклик до
        складання промпту), нумеруємо з одиниці в порядку списку — саме так це
        зробить `prompt_builder`.
        """
        out: list[Evidence] = []
        for i, item in enumerate(items, start=1):
            chunk = item.chunk
            out.append(
                cls(
                    ordinal=int(item.ordinal_in_prompt or i),
                    text=chunk.display_text,
                    title=getattr(item, "document_title", "") or "",
                    page_label=chunk.citation_label(),
                )
            )
        return out


@dataclass(frozen=True, slots=True)
class ClaimVerdict:
    """Вердикт про одне твердження відповіді."""

    claim: str
    verdict: str                 # ТАК | ЧАСТКОВО | НІ | НЕВІДОМО
    supported: bool
    score: float                 # 1.0 / 0.5 / 0.0 — вага у частці заземленості
    method: str                  # llm | лексичний
    raw: str = ""
    cited_ordinals: tuple[int, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "claim": self.claim, "verdict": self.verdict, "supported": self.supported,
            "score": self.score, "method": self.method, "cited": list(self.cited_ordinals),
        }


@dataclass(slots=True)
class GroundednessReport:
    """Підсумок по одній відповіді."""

    question: str
    verdicts: list[ClaimVerdict] = field(default_factory=list)
    evidence_count: int = 0
    method: str = "llm"

    @property
    def total(self) -> int:
        return len(self.verdicts)

    @property
    def supported(self) -> int:
        return sum(1 for v in self.verdicts if v.supported)

    @property
    def groundedness(self) -> float:
        """Зважена частка заземлених тверджень (ЧАСТКОВО важить 0.5)."""
        if not self.verdicts:
            return math.nan
        return sum(v.score for v in self.verdicts) / len(self.verdicts)

    @property
    def uncited_claims(self) -> list[str]:
        """Твердження без жодного маркера [n] — кандидати на невидиму вигадку."""
        return [v.claim for v in self.verdicts if not v.cited_ordinals]

    def format_uk(self) -> str:
        lines = [
            f"Заземленість: {self.groundedness:.2f} "
            f"({self.supported} із {self.total} тверджень, суддя: {self.method}).",
        ]
        for v in self.verdicts:
            if not v.supported:
                lines.append(f"  НЕ ПІДТВЕРДЖЕНО [{v.verdict}]: {v.claim[:120]}")
        if self.uncited_claims:
            lines.append(f"  Тверджень без цитат: {len(self.uncited_claims)}.")
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "groundedness": self.groundedness,
            "supported": self.supported,
            "total": self.total,
            "evidence_count": self.evidence_count,
            "method": self.method,
            "uncited_claims": len(self.uncited_claims),
            "verdicts": [v.as_dict() for v in self.verdicts],
        }


# ------------------------------------------------------- розбиття на твердження
# Скорочення, після яких крапка НЕ закінчує речення. Без цього «див. рис. 3.4»
# перетворюється на три «твердження», і заземленість рахується по сміттю.
_ABBREVIATIONS = {
    "с", "стор", "рис", "табл", "мал", "арт", "див", "напр", "тис", "млн", "млрд",
    "гр", "п", "пп", "ст", "розд", "им", "т", "тобто", "зокрема", "мм", "см", "км",
    "кг", "хв", "год", "обр", "зр",
}
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?…])\s+(?=[«\"(\[]?[А-ЯЁЇІЄҐA-Z0-9])")
_MARKER_RE = re.compile(r"\[(\d{1,3})\]")


def split_claims(text: str, *, min_chars: int = 12) -> list[str]:
    """Розбити відповідь на твердження — по одному реченню на твердження.

    Це свідомо ГРУБИЙ поділ. Точніший (виділення атомарних фактів окремим
    викликом LLM) подвоює кількість викликів моделі й додає власний шар
    галюцинацій між відповіддю і суддею; на 12B-моделі це коштує більше, ніж
    дає. Речення — одиниця, яку модель і так породжує цілком.
    """
    cleaned = " ".join(text.split())
    if not cleaned:
        return []
    parts = _SENTENCE_BOUNDARY.split(cleaned)

    merged: list[str] = []
    for part in parts:
        if merged:
            tail = merged[-1].rstrip(".").rsplit(" ", 1)[-1].casefold().strip("«»\"()")
            # Крапка після скорочення або після одинокої цифри («розділ 3.»)
            # межею речення не є.
            if tail in _ABBREVIATIONS or tail.isdigit():
                merged[-1] = f"{merged[-1]} {part}"
                continue
        merged.append(part)

    return [p.strip() for p in merged if len(p.strip()) >= min_chars]


# ------------------------------------------------------------------- судді
class Judge(Protocol):
    """Контракт судді. Асинхронний, бо LM Studio асинхронний."""

    name: str

    async def verdict(self, claim: str, evidence: Sequence[Evidence]) -> ClaimVerdict: ...


_WORD_RE = re.compile(r"(?u)[^\W\d_]{3,}")
# Слова, спільні для будь-яких двох українських речень: у перекритті вони дають
# хибне «підтверджено» рівно тоді, коли твердження вигадане.
_STOPISH = {
    "який", "яка", "яке", "які", "цей", "цього", "цієї", "той", "тому", "тобто",
    "його", "їхній", "може", "мають", "має", "бути", "було", "буде", "також",
    "щодо", "після", "перед", "через", "разом", "тільки", "лише", "дуже",
    "згідно", "відповідно", "наведені", "матеріали", "матеріалів",
}


def _content_words(text: str) -> set[str]:
    return {w.casefold() for w in _WORD_RE.findall(text)} - _STOPISH


class LexicalJudge:
    """Детермінований суддя без моделі: перекриття змістовних слів і чисел.

    Це не заміна LLM-судді, а ПІДЛОГА. Він потрібен у трьох випадках, і в усіх
    трьох альтернатива — не мати вимірювання взагалі:
      * `ASISTENT_STUB=1` (CI без моделей) — увесь застосунок мусить працювати
        наскрізь без жодного завантаженого гігабайта;
      * LM Studio недоступний на машині, де запускають harness;
      * вердикт моделі не розпарсився (12B-моделі час від часу пишуть есе
        замість «ТАК»).

    Числа перевіряються окремо й жорстко: «дальність 15300 м» проти
    «дальність 15800 м» дає високе словесне перекриття й нульове числове —
    а в артилерійському підручнику помилка саме в числі найдорожча.
    """

    name = "лексичний"

    def __init__(self, *, threshold: float = 0.5) -> None:
        self.threshold = threshold

    async def verdict(self, claim: str, evidence: Sequence[Evidence]) -> ClaimVerdict:
        cited = tuple(int(m) for m in _MARKER_RE.findall(claim))
        text = _MARKER_RE.sub(" ", claim)
        claim_words = _content_words(text)
        pool = " ".join(e.text for e in evidence)
        evidence_words = _content_words(pool)

        overlap = (
            len(claim_words & evidence_words) / len(claim_words) if claim_words else 0.0
        )
        claim_numbers = set(re.findall(r"\d+(?:[.,]\d+)?", text))
        evidence_numbers = set(re.findall(r"\d+(?:[.,]\d+)?", pool))
        numbers_ok = claim_numbers.issubset(evidence_numbers)

        if not claim_words and not claim_numbers:
            return ClaimVerdict(claim, "НЕВІДОМО", False, 0.0, self.name, cited_ordinals=cited)
        if overlap >= self.threshold and numbers_ok:
            return ClaimVerdict(claim, "ТАК", True, 1.0, self.name, cited_ordinals=cited)
        if overlap >= self.threshold and not numbers_ok:
            missing = sorted(claim_numbers - evidence_numbers)
            return ClaimVerdict(
                claim, "ЧАСТКОВО", False, 0.5, self.name,
                raw=f"числа поза джерелами: {', '.join(missing)}", cited_ordinals=cited,
            )
        return ClaimVerdict(
            claim, "НІ", False, 0.0, self.name,
            raw=f"перекриття {overlap:.2f} < {self.threshold:.2f}", cited_ordinals=cited,
        )


# Промпт судді: одна задача, три дозволені відповіді, жодних пояснень.
# Довший промпт на 12B-моделі не покращує вердикт, зате збільшує шанс есе
# замість слова — а есе доводиться відкидати у фолбек.
JUDGE_SYSTEM_UK = (
    "Ти — суворий перевіряльник фактів. Тобі дають ФРАГМЕНТИ навчальних "
    "матеріалів і ОДНЕ твердження.\n"
    "Визнач, чи випливає твердження САМЕ З ЦИХ фрагментів. Власні знання не "
    "використовуй.\n"
    "Відповідай рівно одним словом:\n"
    "ТАК — твердження повністю підтверджується фрагментами;\n"
    "ЧАСТКОВО — підтверджується лише частина твердження;\n"
    "НІ — не підтверджується або суперечить фрагментам."
)

_VERDICT_RE = re.compile(r"\b(ТАК|ЧАСТКОВО|НІ)\b", re.IGNORECASE)


class LlmJudge:
    """Суддя на локальній моделі через LM Studio.

    Бекенд ІН'ЄКТУЄТЬСЯ, а не створюється тут: вибір «LM Studio чи заглушка»
    належить викликачеві (контракт модуля оцінювання), і harness має право
    судити тією самою моделлю, що відповідала, або навмисно іншою.
    """

    name = "llm"

    def __init__(
        self,
        backend: Any,
        *,
        model: str | None = None,
        fallback: Judge | None = None,
        max_evidence_chars: int = 6000,
    ) -> None:
        self.backend = backend
        self.model = model
        self.fallback = fallback or LexicalJudge()
        self.max_evidence_chars = max_evidence_chars

    def build_messages(self, claim: str, evidence: Sequence[Evidence]) -> list[dict[str, str]]:
        blocks: list[str] = []
        used = 0
        for item in evidence:
            block = item.as_block()
            if used + len(block) > self.max_evidence_chars:
                break
            blocks.append(block)
            used += len(block)
        body = (
            "ФРАГМЕНТИ:\n" + "\n\n".join(blocks)
            + "\n\nТВЕРДЖЕННЯ:\n" + _MARKER_RE.sub("", claim).strip()
            + "\n\nВердикт (ТАК / ЧАСТКОВО / НІ):"
        )
        return [
            {"role": "system", "content": JUDGE_SYSTEM_UK},
            {"role": "user", "content": body},
        ]

    async def verdict(self, claim: str, evidence: Sequence[Evidence]) -> ClaimVerdict:
        from app.backends.base import ChatParams

        cited = tuple(int(m) for m in _MARKER_RE.findall(claim))
        # temperature 0.0: суддя не має бути креативним; 8 токенів — це «ЧАСТКОВО»
        # з запасом, і водночас жорсткий бар'єр проти есе.
        params = ChatParams(temperature=0.0, max_tokens=8)
        text = ""
        try:
            async for token in self.backend.chat_stream(
                self.build_messages(claim, evidence), params=params, model=self.model
            ):
                text += token
        except Exception as exc:  # noqa: BLE001 — недоступність судді не має валити прогін
            fallback = await self.fallback.verdict(claim, evidence)
            return ClaimVerdict(
                fallback.claim, fallback.verdict, fallback.supported, fallback.score,
                f"{self.fallback.name} (фолбек: {type(exc).__name__})",
                raw=fallback.raw, cited_ordinals=cited,
            )

        match = _VERDICT_RE.search(text)
        if match is None:
            fallback = await self.fallback.verdict(claim, evidence)
            return ClaimVerdict(
                fallback.claim, fallback.verdict, fallback.supported, fallback.score,
                f"{self.fallback.name} (вердикт не розпарсено)",
                raw=text.strip()[:200], cited_ordinals=cited,
            )
        word = match.group(1).upper()
        score = {"ТАК": 1.0, "ЧАСТКОВО": 0.5, "НІ": 0.0}[word]
        return ClaimVerdict(
            claim, word, word == "ТАК", score, self.name,
            raw=text.strip()[:200], cited_ordinals=cited,
        )


def create_judge(
    backend: Any = None,
    *,
    stub: bool | None = None,
    model: str | None = None,
) -> Judge:
    """Створити суддю. `stub=None` означає «дивись на ASISTENT_STUB».

    У stub-режимі суддею стає лексичний: заглушка LLM за побудовою переказує
    подані фрагменти, тож питати її «чи випливає це з фрагментів» — це міряти
    власну заглушку й отримувати завжди «так».
    """
    from app.backends.stub_backend import is_stub_enabled

    use_stub = is_stub_enabled() if stub is None else stub
    if use_stub or backend is None:
        return LexicalJudge()
    return LlmJudge(backend, model=model)


async def judge_answer(
    judge: Judge,
    question: str,
    answer: str,
    evidence: Sequence[Evidence],
) -> GroundednessReport:
    """Перевірити кожне твердження відповіді проти поданих фрагментів."""
    claims = split_claims(answer)
    report = GroundednessReport(
        question=question, evidence_count=len(evidence), method=getattr(judge, "name", "?")
    )
    for claim in claims:
        report.verdicts.append(await judge.verdict(claim, evidence))
    return report


# ------------------------------------------------------------------- відмови
@dataclass(slots=True)
class AbstentionReport:
    """Частка коректних відмов і — головне — частка ХИБНИХ.

    Обидві потрібні разом. Система, що утримується завжди, має ідеальну частку
    коректних відмов і нульову користь; саме тому `false_refusal_rate`
    (утримання на питанні, відповідь на яке в матеріалах Є) стоїть поруч і
    вважається помилкою тієї ж ваги.
    """

    unanswerable_total: int = 0
    unanswerable_refused: int = 0
    answerable_total: int = 0
    answerable_refused: int = 0
    details: dict[str, str] = field(default_factory=dict)

    @property
    def correct_refusal_rate(self) -> float:
        if not self.unanswerable_total:
            return math.nan
        return self.unanswerable_refused / self.unanswerable_total

    @property
    def false_refusal_rate(self) -> float:
        if not self.answerable_total:
            return math.nan
        return self.answerable_refused / self.answerable_total

    @property
    def balanced_accuracy(self) -> float:
        """Середнє двох часток — єдина цифра, яку не можна накрутити утриманням."""
        correct = self.correct_refusal_rate
        answered = 1.0 - self.false_refusal_rate if self.answerable_total else math.nan
        if math.isnan(correct) or math.isnan(answered):
            return math.nan
        return (correct + answered) / 2.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "unanswerable_total": self.unanswerable_total,
            "unanswerable_refused": self.unanswerable_refused,
            "answerable_total": self.answerable_total,
            "answerable_refused": self.answerable_refused,
            "correct_refusal_rate": self.correct_refusal_rate,
            "false_refusal_rate": self.false_refusal_rate,
            "balanced_accuracy": self.balanced_accuracy,
        }

    def format_uk(self) -> str:
        return (
            f"Коректних відмов: {self.unanswerable_refused}/{self.unanswerable_total} "
            f"({self.correct_refusal_rate:.2f}); хибних відмов: "
            f"{self.answerable_refused}/{self.answerable_total} "
            f"({self.false_refusal_rate:.2f}); збалансована точність: "
            f"{self.balanced_accuracy:.2f}."
        )


def abstention_stats(
    questions: Sequence[Any],
    abstained: Mapping[str, bool],
) -> AbstentionReport:
    """Звести утримання по золотому набору.

    `questions` — `list[GoldQuestion]`; питання без золотих фрагментів вважається
    таким, на яке відповіді в корпусі немає (див. `gold_set.GoldQuestion`).
    Питання, якого немає в `abstained`, вважається таким, що НЕ утрималось:
    відсутній запис — це прогін, який щось повернув і не позначив утримання.
    """
    report = AbstentionReport()
    for q in questions:
        refused = bool(abstained.get(q.id, False))
        if q.answerable:
            report.answerable_total += 1
            if refused:
                report.answerable_refused += 1
                report.details[q.id] = "хибна відмова"
        else:
            report.unanswerable_total += 1
            if refused:
                report.unanswerable_refused += 1
            else:
                report.details[q.id] = "відповідь там, де її бути не могло"
    return report


# ------------------------------------------------------------ дійсність цитат
@dataclass(frozen=True, slots=True)
class CitationValidity:
    """Частка дійсних цитат — рахується без судді, з телеметрії генератора."""

    messages: int
    emitted: int
    unresolved: int
    messages_with_unresolved: int
    uncited_messages: int
    unmapped_markers: int = 0

    @property
    def valid_ratio(self) -> float:
        return (self.emitted - self.unresolved) / self.emitted if self.emitted else math.nan

    @property
    def hallucinated_ratio(self) -> float:
        return self.unresolved / self.emitted if self.emitted else math.nan

    def as_dict(self) -> dict[str, Any]:
        return {
            "messages": self.messages, "emitted": self.emitted,
            "unresolved": self.unresolved,
            "messages_with_unresolved": self.messages_with_unresolved,
            "uncited_messages": self.uncited_messages,
            "unmapped_markers": self.unmapped_markers,
            "valid_ratio": self.valid_ratio,
            "hallucinated_ratio": self.hallucinated_ratio,
        }

    def format_uk(self) -> str:
        return (
            f"Дійсних цитат: {self.emitted - self.unresolved} із {self.emitted} "
            f"({self.valid_ratio:.3f}); повідомлень із нерозв'язаними маркерами: "
            f"{self.messages_with_unresolved} із {self.messages}; "
            f"відповідей без жодної цитати: {self.uncited_messages}."
        )


def citation_validity(
    con: sqlite3.Connection,
    *,
    session_id: str | None = None,
    include_abstained: bool = False,
) -> CitationValidity:
    """Порахувати дійсність цитат із `chat_messages` і `unresolved_citations`.

    Арифметика тримається на тому, ЯК працює генератор: `parse_citations`
    ВИДАЛЯЄ нерозв'язаний маркер із тексту (посилання, що нікуди не веде,
    показувати не можна) і логує його один раз у `unresolved_citations`.
    Отже:

        видані маркери = унікальні [n], що лишились у тексті + рядки таблиці
        дійсні         = ті, що лишились у тексті

    Утримання за замовчуванням не рахується: відповідь-відмова цитат не має за
    визначенням, і її нуль у знаменнику завищив би «частку відповідей без
    цитат» рівно на кількість коректних відмов.
    """
    where = ["role='assistant'"]
    params: list[Any] = []
    if session_id is not None:
        where.append("session_id=?")
        params.append(session_id)
    if not include_abstained:
        where.append("abstained=0")
    rows = con.execute(
        f"SELECT id, content, citation_map_json FROM chat_messages WHERE {' AND '.join(where)}",
        tuple(params),
    ).fetchall()

    messages = len(rows)
    emitted = 0
    uncited = 0
    unmapped = 0
    for row in rows:
        markers = {int(m) for m in _MARKER_RE.findall(row["content"] or "")}
        emitted += len(markers)
        if not markers:
            uncited += 1
        # Мапа [n] → chunk_uid зберігається на кожне повідомлення. Маркер, який
        # у тексті лишився, а в мапі його немає, — це збій нумерації промпту, а
        # не галюцинація моделі; він мусить бути видимим окремо, бо лікується
        # інакше (виправленням коду, а не порогом).
        mapping = json.loads(row["citation_map_json"] or "{}")
        if mapping:
            unmapped += sum(1 for m in markers if str(m) not in mapping)

    ids = [r["id"] for r in rows]
    unresolved = 0
    with_unresolved = 0
    if ids:
        marks = ",".join("?" * len(ids))
        unresolved = int(
            con.execute(
                f"SELECT count(*) FROM unresolved_citations WHERE message_id IN ({marks})",
                tuple(ids),
            ).fetchone()[0]
        )
        with_unresolved = int(
            con.execute(
                "SELECT count(DISTINCT message_id) FROM unresolved_citations"
                f" WHERE message_id IN ({marks})",
                tuple(ids),
            ).fetchone()[0]
        )
    return CitationValidity(
        messages=messages,
        emitted=emitted + unresolved,
        unresolved=unresolved,
        messages_with_unresolved=with_unresolved,
        uncited_messages=uncited,
        unmapped_markers=unmapped,
    )


# ------------------------------------------------------- попарне порівняння
PAIRWISE_SYSTEM_UK = (
    "Ти — методист, що порівнює дві відповіді на одне запитання за навчальними "
    "матеріалами.\n"
    "Оцінюй лише те, наскільки відповідь спирається на матеріали, точна й повна.\n"
    "Довжина й стиль значення не мають.\n"
    "Відповідай рівно одним словом: А, Б або НІЧИЯ."
)

_PAIRWISE_RE = re.compile(r"\b(А|Б|НІЧИЯ)\b")


@dataclass(frozen=True, slots=True)
class PairwiseVerdict:
    """Результат попарного порівняння з перестановкою позицій."""

    winner: str            # a | b | tie
    direct: str            # вердикт при порядку (A=a, B=b)
    swapped: str           # вердикт при порядку (A=b, B=a)
    consistent: bool       # чи збігаються вердикти після перестановки

    def as_dict(self) -> dict[str, Any]:
        return {
            "winner": self.winner, "direct": self.direct,
            "swapped": self.swapped, "consistent": self.consistent,
        }


async def _ask_pairwise(backend: Any, question: str, first: str, second: str, model: str | None) -> str:
    from app.backends.base import ChatParams

    body = (
        f"Запитання: {question}\n\nВІДПОВІДЬ А:\n{first}\n\nВІДПОВІДЬ Б:\n{second}\n\n"
        "Яка відповідь краще спирається на навчальні матеріали? (А / Б / НІЧИЯ):"
    )
    text = ""
    async for token in backend.chat_stream(
        [
            {"role": "system", "content": PAIRWISE_SYSTEM_UK},
            {"role": "user", "content": body},
        ],
        params=ChatParams(temperature=0.0, max_tokens=6),
        model=model,
    ):
        text += token
    match = _PAIRWISE_RE.search(text.upper())
    return match.group(1) if match else "НІЧИЯ"


async def pairwise_compare(
    backend: Any,
    question: str,
    answer_a: str,
    answer_b: str,
    *,
    model: str | None = None,
) -> PairwiseVerdict:
    """Попарне порівняння двох кандидатів — ЄДИНИЙ дозволений спосіб оцінювати
    абляцію префікса й стратегії чанкування (див. шапку модуля).

    Питається двічі, з перестановкою: якщо вердикт після перестановки не
    змінився на дзеркальний, це позиційне упередження судді, і результат
    чесніше зарахувати як нічию, ніж як перемогу.
    """
    direct = await _ask_pairwise(backend, question, answer_a, answer_b, model)
    swapped = await _ask_pairwise(backend, question, answer_b, answer_a, model)

    if direct == "А" and swapped == "Б":
        winner = "a"
    elif direct == "Б" and swapped == "А":
        winner = "b"
    else:
        winner = "tie"
    consistent = winner != "tie" or (direct == "НІЧИЯ" and swapped == "НІЧИЯ")
    return PairwiseVerdict(winner=winner, direct=direct, swapped=swapped, consistent=consistent)


def summarize_groundedness(reports: Iterable[GroundednessReport]) -> dict[str, Any]:
    """Середні по набору відповідей — рядок таблиці у звіті з НДР."""
    items = list(reports)
    if not items:
        return {"answers": 0}
    values = [r.groundedness for r in items if not math.isnan(r.groundedness)]
    return {
        "answers": len(items),
        "mean_groundedness": sum(values) / len(values) if values else math.nan,
        "claims": sum(r.total for r in items),
        "supported": sum(r.supported for r in items),
        "uncited_claims": sum(len(r.uncited_claims) for r in items),
        "methods": sorted({r.method for r in items}),
    }
