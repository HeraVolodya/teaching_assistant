"""Золотий набір: побудова, зберігання, валідація, півавтоматичне наповнення.

Золотий набір — це не «тестові дані», а окремий артефакт НДР. Він виконує
три роботи одразу:
  1. дає цифри Recall@k / MRR / nDCG проти BM25-baseline (план, §9);
  2. слугує калібрувальним набором для утримання `u_lin` — окупність настає
     приблизно на 38 розмічених запитах (план, §7);
  3. є міграційним гейтом embedding-моделі: без нерегресії Recall@10 / MRR /
     nDCG на ньому вказівник на нову модель не перемикається (реєстр ризиків).

ПАСТКА, ЗАРАДИ ЯКОЇ `chunk_uid` КОНТЕНТНИЙ
------------------------------------------
Золоті мітки прив'язані до фрагментів. Якби ідентифікатор фрагмента був
випадковим UUID (як `documents.id`), то БУДЬ-ЯКА переіндексація — нова версія
Docling, інший OCR-рушій, змінена стратегія чанкування — видала б нові
ідентифікатори, і ввесь золотий набір, зібраний викладачем за тижні, мовчки
перетворився б на нулі: Recall = 0 при ідеальному пошуку.

Саме тому `domain.chunk_uid_for(document_id, ordinal, text)` — це blake2b від
ВМІСТУ. Доки текст фрагмента не змінився, мітка лишається дійсною навіть після
повної перебудови бази. Зворотний бік цієї медалі: якщо текст ЗМІНИВСЯ (кращий
OCR виправив «і» на «i»), мітка стає недійсною — і це чесно, бо це вже інший
фрагмент. `validate()` знаходить такі мітки й називає їх `stale_uid`, а не
мовчки рахує їх як промах.

ПІВАВТОМАТИЧНА ПОБУДОВА
-----------------------
Розмітити 50 питань руками — це день роботи викладача. Тому: узяти N
випадкових фрагментів (стратифіковано за документами, з фіксованим зерном),
згенерувати до кожного питання ЛОКАЛЬНОЮ моделлю, показати викладачеві —
і зарахувати лише те, що він затвердив. Модель тут пише ЧЕРНЕТКУ питання, а
не золоту мітку: мітка відома за побудовою, бо ми знаємо, з якого фрагмента
питання зроблено.
"""

from __future__ import annotations

import csv
import json
import random
import re
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.domain import Chunk, ChunkLevel, new_id

__all__ = [
    "DRAFT_SYSTEM_UK",
    "MIN_CALIBRATION_QUESTIONS",
    "RECOMMENDED_QUESTIONS",
    "EvalQuestionRepo",
    "GoldPage",
    "GoldQuestion",
    "QuestionDraft",
    "ValidationIssue",
    "ValidationReport",
    "approve_drafts",
    "draft_questions",
    "load_csv",
    "load_json",
    "sample_chunks",
    "save_csv",
    "save_json",
    "validate",
]

# Скільки питань потрібно на колекцію. 38 — точка окупності u_lin (план, §7),
# 50 — рекомендація плану для міграційного гейта й для стабільних середніх.
MIN_CALIBRATION_QUESTIONS = 38
RECOMMENDED_QUESTIONS = 50

_LIST_SEPARATOR = ";"
# Ідентифікатор документа не звужується до 32 hex навмисно: у CSV, який руками
# заповнює викладач, там цілком може стояти скорочення на кшталт `балістика`.
_PAGE_RE = re.compile(r"^\s*(?:(?P<doc>[^\s:]+)\s*:\s*)?(?P<page>\d{1,6})\s*$")


# ------------------------------------------------------------------- значення
@dataclass(frozen=True, slots=True)
class GoldPage:
    """Золота сторінка. `document_id` не обов'язковий, але без нього номер
    сторінки в багатодокументній колекції неоднозначний, тож імпорт із CSV
    приймає обидві форми: `147` і `<document_id>:147`."""

    page: int
    document_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"page": self.page, "document_id": self.document_id}

    def as_text(self) -> str:
        return f"{self.document_id}:{self.page}" if self.document_id else str(self.page)

    @classmethod
    def parse(cls, raw: Any) -> GoldPage:
        if isinstance(raw, GoldPage):
            return raw
        if isinstance(raw, int):
            return cls(page=int(raw))
        if isinstance(raw, dict):
            return cls(page=int(raw["page"]), document_id=raw.get("document_id"))
        m = _PAGE_RE.match(str(raw))
        if not m:
            raise ValueError(
                f"Не вдалося розібрати золоту сторінку {raw!r}. "
                "Очікується «147» або «<document_id>:147»."
            )
        return cls(page=int(m.group("page")), document_id=m.group("doc"))


@dataclass(slots=True)
class GoldQuestion:
    """Одне питання золотого набору.

    Порожній `gold_chunk_uids` — це НЕ незаповнений рядок, а окремий свідомий
    клас: питання, відповіді на яке в корпусі немає. Такі питання не входять у
    Recall/MRR/nDCG (вони там не визначені) і оцінюються часткою коректних
    відмов — див. `app/eval/groundedness.py`.
    """

    collection_id: str
    question: str
    gold_chunk_uids: list[str] = field(default_factory=list)
    gold_pages: list[GoldPage] = field(default_factory=list)
    note: str = ""
    id: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            self.id = new_id()
        self.gold_chunk_uids = [str(u).strip() for u in self.gold_chunk_uids if str(u).strip()]
        self.gold_pages = [GoldPage.parse(p) for p in self.gold_pages]

    @property
    def answerable(self) -> bool:
        """Чи очікуємо ми взагалі відповідь на це питання."""
        return bool(self.gold_chunk_uids)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "collection_id": self.collection_id,
            "question": self.question,
            "gold_chunk_uids": list(self.gold_chunk_uids),
            "gold_pages": [p.as_dict() for p in self.gold_pages],
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, collection_id: str | None = None) -> GoldQuestion:
        uids = data.get("gold_chunk_uids") or data.get("chunk_uids") or []
        if isinstance(uids, str):
            uids = [u for u in uids.split(_LIST_SEPARATOR) if u.strip()]
        pages = data.get("gold_pages") or data.get("pages") or []
        if isinstance(pages, str):
            pages = [p for p in pages.split(_LIST_SEPARATOR) if p.strip()]
        return cls(
            id=str(data.get("id") or ""),
            collection_id=str(collection_id or data.get("collection_id") or ""),
            question=str(data.get("question", "")).strip(),
            gold_chunk_uids=list(uids),
            gold_pages=[GoldPage.parse(p) for p in pages],
            note=str(data.get("note", "")),
        )


# --------------------------------------------------------------- сховище
class EvalQuestionRepo:
    """Доступ до `eval_questions`.

    SQL живе тут, а не в `app/db/repositories.py`, з двох причин: репозиторії —
    спільний фундамент, який модулі не редагують (контракт), і таблиць
    оцінювання там немає взагалі. Отже, це «єдине місце, що знає SQL» для
    свого модуля, за тим самим принципом.
    """

    def __init__(self, con: sqlite3.Connection) -> None:
        self.con = con

    def upsert_many(self, questions: Sequence[GoldQuestion]) -> list[str]:
        rows = [
            (
                q.id, q.collection_id, q.question,
                json.dumps(q.gold_chunk_uids, ensure_ascii=False),
                json.dumps([p.as_dict() for p in q.gold_pages], ensure_ascii=False),
                q.note,
            )
            for q in questions
        ]
        self.con.executemany(
            "INSERT INTO eval_questions (id,collection_id,question,gold_chunk_uids,gold_pages,note)"
            " VALUES (?,?,?,?,?,?)"
            " ON CONFLICT(id) DO UPDATE SET question=excluded.question,"
            " gold_chunk_uids=excluded.gold_chunk_uids, gold_pages=excluded.gold_pages,"
            " note=excluded.note",
            rows,
        )
        return [q.id for q in questions]

    def get(self, question_id: str) -> GoldQuestion | None:
        row = self.con.execute(
            "SELECT * FROM eval_questions WHERE id=?", (question_id,)
        ).fetchone()
        return self._row(row) if row else None

    def for_collection(self, collection_id: str) -> list[GoldQuestion]:
        rows = self.con.execute(
            "SELECT * FROM eval_questions WHERE collection_id=? ORDER BY created_at, id",
            (collection_id,),
        ).fetchall()
        return [self._row(r) for r in rows]

    def qrels(self, collection_id: str) -> dict[str, set[str]]:
        """`question_id → золоті chunk_uid` — прямий вхід `metrics.evaluate_run`."""
        return {q.id: set(q.gold_chunk_uids) for q in self.for_collection(collection_id)}

    def count(self, collection_id: str) -> int:
        return int(
            self.con.execute(
                "SELECT count(*) FROM eval_questions WHERE collection_id=?", (collection_id,)
            ).fetchone()[0]
        )

    def delete(self, question_id: str) -> None:
        self.con.execute("DELETE FROM eval_questions WHERE id=?", (question_id,))

    def clear(self, collection_id: str) -> None:
        self.con.execute("DELETE FROM eval_questions WHERE collection_id=?", (collection_id,))

    @staticmethod
    def _row(r: sqlite3.Row) -> GoldQuestion:
        return GoldQuestion(
            id=r["id"],
            collection_id=r["collection_id"],
            question=r["question"],
            gold_chunk_uids=json.loads(r["gold_chunk_uids"] or "[]"),
            gold_pages=[GoldPage.parse(p) for p in json.loads(r["gold_pages"] or "[]")],
            note=r["note"] or "",
        )


# ------------------------------------------------------------ імпорт / експорт
def load_json(path: Path | str, *, collection_id: str | None = None) -> list[GoldQuestion]:
    """Прочитати золотий набір із JSON (список об'єктів або {"questions": [...]})."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    items = data.get("questions", []) if isinstance(data, dict) else data
    return [GoldQuestion.from_dict(item, collection_id=collection_id) for item in items]


def load_csv(path: Path | str, *, collection_id: str | None = None) -> list[GoldQuestion]:
    """Прочитати золотий набір із CSV.

    Колонки: `question`, `gold_chunk_uids`, `gold_pages`, `note`, необов'язково
    `id` і `collection_id`. Списки розділені `;` — не комою: кома всередині
    українського питання зустрічається постійно, і CSV з комою-роздільником
    усередині поля переживає лише той, хто ніколи не відкривав його в Excel.

    `utf-8-sig` навмисно: Excel на Windows зберігає CSV із BOM, і без цього
    перша колонка називалася б `\\ufeffquestion`, а всі питання були б порожні.
    """
    with Path(path).open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        return [
            GoldQuestion.from_dict({k: (v or "") for k, v in row.items()}, collection_id=collection_id)
            for row in reader
            if any((v or "").strip() for v in row.values())
        ]


def save_json(questions: Sequence[GoldQuestion], path: Path | str) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps([q.as_dict() for q in questions], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out


def save_csv(questions: Sequence[GoldQuestion], path: Path | str) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", "collection_id", "question", "gold_chunk_uids", "gold_pages", "note"])
        for q in questions:
            writer.writerow([
                q.id, q.collection_id, q.question,
                _LIST_SEPARATOR.join(q.gold_chunk_uids),
                _LIST_SEPARATOR.join(p.as_text() for p in q.gold_pages),
                q.note,
            ])
    return out


# --------------------------------------------------------------- валідація
@dataclass(frozen=True, slots=True)
class ValidationIssue:
    question_id: str
    kind: str        # stale_uid | foreign_uid | not_leaf | empty_question | duplicate | no_gold | bad_page
    detail: str
    severity: str = "error"   # error | warning

    def as_dict(self) -> dict[str, str]:
        return {
            "question_id": self.question_id, "kind": self.kind,
            "detail": self.detail, "severity": self.severity,
        }


@dataclass(slots=True)
class ValidationReport:
    checked: int
    issues: list[ValidationIssue] = field(default_factory=list)
    unanswerable: int = 0

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def format_uk(self) -> str:
        lines = [
            f"Перевірено питань: {self.checked} "
            f"(з них без відповіді в корпусі: {self.unanswerable})."
        ]
        if not self.issues:
            lines.append("Проблем не знайдено.")
        for issue in self.issues:
            mark = "ПОМИЛКА" if issue.severity == "error" else "УВАГА"
            lines.append(f"  [{mark}] {issue.kind}: {issue.detail}")
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "checked": self.checked,
            "unanswerable": self.unanswerable,
            "ok": self.ok,
            "issues": [i.as_dict() for i in self.issues],
        }


def validate(
    con: sqlite3.Connection,
    questions: Sequence[GoldQuestion],
    *,
    collection_id: str | None = None,
    require_minimum: bool = True,
) -> ValidationReport:
    """Перевірити золотий набір проти реальної бази.

    Найважливіша перевірка — `stale_uid`: мітка, яку не вдається розв'язати.
    Вона майже завжди означає не помилку викладача, а переіндексацію зі зміною
    тексту фрагмента, і мовчазне зарахування її як промаху зробило б наступний
    звіт із НДР неправдивим.
    """
    report = ValidationReport(checked=len(questions))
    seen_questions: dict[str, str] = {}

    for q in questions:
        text = q.question.strip()
        if not text:
            report.issues.append(
                ValidationIssue(q.id, "empty_question", "Порожній текст питання.")
            )
        else:
            key = " ".join(text.casefold().split())
            if key in seen_questions:
                report.issues.append(
                    ValidationIssue(
                        q.id, "duplicate",
                        f"Питання дослівно повторює {seen_questions[key]}: {text[:60]!r}.",
                        severity="warning",
                    )
                )
            else:
                seen_questions[key] = q.id

        if not q.gold_chunk_uids:
            report.unanswerable += 1
            report.issues.append(
                ValidationIssue(
                    q.id, "no_gold",
                    f"Питання без золотих фрагментів (перевірка утримання): {text[:60]!r}.",
                    severity="warning",
                )
            )

        for uid in q.gold_chunk_uids:
            row = con.execute(
                "SELECT collection_id, level, document_id FROM chunks WHERE chunk_uid=?", (uid,)
            ).fetchone()
            if row is None:
                report.issues.append(
                    ValidationIssue(
                        q.id, "stale_uid",
                        f"Фрагмент {uid} не знайдено. `chunk_uid` — це blake2b від "
                        "тексту, тож або документ видалено, або текст фрагмента "
                        "змінився при переіндексації (інший OCR чи інше чанкування).",
                    )
                )
                continue
            target = collection_id or q.collection_id
            if target and row["collection_id"] != target:
                report.issues.append(
                    ValidationIssue(
                        q.id, "foreign_uid",
                        f"Фрагмент {uid} належить іншій колекції ({row['collection_id']}). "
                        "Колекції ізольовані фізично, тож пошук його не поверне ніколи.",
                    )
                )
            if row["level"] != ChunkLevel.LEAF.value:
                report.issues.append(
                    ValidationIssue(
                        q.id, "not_leaf",
                        f"Фрагмент {uid} має рівень {row['level']}, а індексуються лише L2. "
                        "Розмічати треба листок; L1-батько потрапить у відповідь сам через auto-merge.",
                    )
                )

        for page in q.gold_pages:
            if page.page < 1:
                report.issues.append(
                    ValidationIssue(q.id, "bad_page", f"Недійсний номер сторінки: {page.page}.")
                )

    answerable = report.checked - report.unanswerable
    if require_minimum and answerable < MIN_CALIBRATION_QUESTIONS:
        report.issues.append(
            ValidationIssue(
                "", "too_few",
                f"Питань із золотими фрагментами: {answerable}. Утримання `u_lin` окупається "
                f"від {MIN_CALIBRATION_QUESTIONS}, для стабільних середніх у звіті потрібно "
                f"близько {RECOMMENDED_QUESTIONS}.",
                severity="warning",
            )
        )
    return report


# ------------------------------------------- півавтоматична побудова набору
def sample_chunks(
    con: sqlite3.Connection,
    collection_id: str,
    *,
    count: int = RECOMMENDED_QUESTIONS,
    seed: int = 20260101,
    min_chars: int = 250,
    max_per_document: int = 8,
) -> list[Chunk]:
    """Вибрати фрагменти, з яких варто робити питання.

    Три правила, кожне з ціною помилки:
      * лише L2 — інші рівні не індексуються, тож золота мітка на них
        недосяжна за побудовою;
      * `min_chars` — фрагмент коротший за 250 символів здебільшого є
        заголовком або підписом, і питання з нього виходить беззмістовне;
      * стратифікація за документами — без неї випадкова вибірка з корпусу, де
        один підручник на 900 сторінок і три методички по 30, дасть золотий
        набір ПРО ОДИН ПІДРУЧНИК, і диверсифікація джерел лишиться невиміряною.

    Зерно фіксоване: вибірка має відтворюватись, інакше два прогони harness'а
    міряють різні речі.
    """
    rows = con.execute(
        "SELECT id, document_id FROM chunks"
        " WHERE collection_id=? AND level=? AND length(display_text) >= ?"
        " ORDER BY document_id, ordinal",
        (collection_id, ChunkLevel.LEAF.value, min_chars),
    ).fetchall()
    if not rows:
        return []

    by_document: dict[str, list[int]] = {}
    for row in rows:
        by_document.setdefault(str(row["document_id"]), []).append(int(row["id"]))

    rng = random.Random(seed)
    pools: dict[str, list[int]] = {}
    for doc_id, ids in sorted(by_document.items()):
        shuffled = list(ids)
        rng.shuffle(shuffled)
        pools[doc_id] = shuffled[:max_per_document]

    # Кругова видача по документах: перші N питань покривають максимум різних
    # документів, тож обрізаний набір лишається збалансованим.
    picked: list[int] = []
    order = sorted(pools)
    depth = 0
    while len(picked) < count:
        added = False
        for doc_id in order:
            pool = pools[doc_id]
            if depth < len(pool):
                picked.append(pool[depth])
                added = True
                if len(picked) >= count:
                    break
        if not added:
            break
        depth += 1

    from app.db.repositories import ChunkRepo

    return ChunkRepo(con).by_ids(picked)


@dataclass(slots=True)
class QuestionDraft:
    """Чернетка питання. Золота мітка вже відома — вона за побудовою."""

    chunk_uid: str
    question: str
    document_id: str
    document_title: str
    page_label: str
    excerpt: str
    source: str = "llm"          # llm | шаблон
    needs_review: bool = True
    model_id: str = ""

    def to_question(self, collection_id: str, *, note: str = "") -> GoldQuestion:
        pages = []
        digits = re.findall(r"\d+", self.page_label)
        if digits:
            pages = [GoldPage(page=int(digits[0]), document_id=self.document_id)]
        return GoldQuestion(
            collection_id=collection_id,
            question=self.question.strip(),
            gold_chunk_uids=[self.chunk_uid],
            gold_pages=pages,
            note=note or f"чернетка: {self.source}; джерело: {self.document_title}",
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "chunk_uid": self.chunk_uid, "question": self.question,
            "document_id": self.document_id, "document_title": self.document_title,
            "page_label": self.page_label, "excerpt": self.excerpt,
            "source": self.source, "needs_review": self.needs_review, "model_id": self.model_id,
        }


# Промпт навмисно короткий: успішність виконання інструкцій малою моделлю падає
# приблизно експоненційно з їх кількістю (план, §10). П'ять рядків, одна задача.
DRAFT_SYSTEM_UK = (
    "Ти — методист військового навчального закладу. Отримавши фрагмент "
    "навчального матеріалу, сформулюй РІВНО ОДНЕ конкретне запитання "
    "українською мовою, відповідь на яке міститься саме в цьому фрагменті.\n"
    "Не згадуй слова «фрагмент», «текст» чи «уривок».\n"
    "Не додавай відповіді, пояснень і нумерації.\n"
    "Виведи лише саме запитання одним рядком, що закінчується знаком питання."
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+")


def _template_question(chunk: Chunk) -> str:
    """Детермінований запасний варіант, коли модель не дала питання.

    Потрібен не лише для `ASISTENT_STUB=1`: 12B-модель час від часу видає
    відповідь замість питання, і без фолбека півавтоматична побудова просто
    зупинялась би посеред 50 фрагментів.
    """
    tail = chunk.header_path.strip("/").split("//")[-1].strip()
    first = _SENTENCE_SPLIT.split(" ".join(chunk.display_text.split()))[0]
    words = first.split()
    topic = " ".join(words[:9]).rstrip(",.;:")
    if tail:
        return f"Що навчальні матеріали зазначають про «{topic}» (розділ «{tail}»)?"
    return f"Що навчальні матеріали зазначають про «{topic}»?"


def _looks_like_question(text: str) -> bool:
    """Одне речення, що закінчується знаком питання, і не є відмовою заглушки."""
    cleaned = " ".join(text.split())
    if not cleaned or len(cleaned) > 400:
        return False
    if "недостатньо інформації" in cleaned.casefold():
        return False
    return cleaned.endswith("?")


async def _collect(stream: Any) -> str:
    parts: list[str] = []
    async for token in stream:
        parts.append(token)
    return "".join(parts)


async def draft_questions(
    backend: Any,
    chunks: Sequence[Chunk],
    *,
    titles: dict[str, str] | None = None,
    model: str | None = None,
    max_excerpt_chars: int = 1200,
) -> list[QuestionDraft]:
    """Згенерувати чернетки питань локальною моделлю через LM Studio.

    `backend` — будь-що, що задовольняє протокол `app.backends.base.LlmBackend`
    (у CI і в stub-режимі це `StubBackend`). Модуль оцінювання не імпортує
    фабрику бекенда навмисно: вибір «LM Studio чи заглушка» належить
    викликачеві, а не harness'у.

    Кожна чернетка позначена `needs_review=True`. Це не формальність:
    затвердження викладачем — єдине, що відрізняє золотий набір від
    самозбувного пророцтва, у якому модель питає рівно те, що вміє знайти.
    """
    from app.backends.base import ChatParams

    out: list[QuestionDraft] = []
    names = titles or {}
    # Низька температура: питання має бути точним, а не оригінальним.
    params = ChatParams(temperature=0.1, max_tokens=120)

    for chunk in chunks:
        excerpt = chunk.display_text[:max_excerpt_chars]
        messages = [
            {"role": "system", "content": DRAFT_SYSTEM_UK},
            {"role": "user", "content": f"Матеріал:\n{excerpt}\n\nЗапитання:"},
        ]
        text = ""
        try:
            text = await _collect(backend.chat_stream(messages, params=params, model=model))
        except Exception:
            # Модель недоступна — це не привід зупиняти побудову набору:
            # шаблонні чернетки викладач однаково переписує.
            text = ""
        first_line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
        if _looks_like_question(first_line):
            question, source = first_line, "llm"
        else:
            question, source = _template_question(chunk), "шаблон"
        out.append(
            QuestionDraft(
                chunk_uid=chunk.chunk_uid,
                question=question,
                document_id=chunk.document_id,
                document_title=names.get(chunk.document_id, ""),
                page_label=chunk.citation_label(),
                excerpt=excerpt[:300],
                source=source,
                model_id=str(model or ""),
            )
        )
    return out


def approve_drafts(
    drafts: Iterable[QuestionDraft],
    collection_id: str,
    *,
    approved_uids: Iterable[str] | None = None,
) -> list[GoldQuestion]:
    """Перетворити затверджені чернетки на золоті питання.

    `approved_uids=None` означає «затверджено все» — так поводиться скрипт, коли
    викладач уже відредагував файл чернеток руками. Явний список потрібен UI,
    де затвердження поштучне.
    """
    allowed = None if approved_uids is None else {str(u) for u in approved_uids}
    out: list[GoldQuestion] = []
    for draft in drafts:
        if allowed is not None and draft.chunk_uid not in allowed:
            continue
        if not draft.question.strip():
            continue
        out.append(draft.to_question(collection_id))
    return out
