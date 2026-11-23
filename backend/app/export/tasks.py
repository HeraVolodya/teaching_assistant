"""Інструкції моделі під кожну форму файлу.

Формулювання перевірено на живій Gemma 4 E4B (23.11.2026, див. `outline.py`).
Три місця, без яких результат помітно гірший:
  * «Перший рядок — # Назва …» — інакше назву документа нема звідки взяти;
  * для таблиці — «переглянь усі джерела по черзі й нічого не пропускай»: без
    цього мала модель бере рядки з першого джерела й зупиняється (4 з 10);
  * «не нумеруй слайди словом «Слайд»» — модель охоче пише «## Слайд 1. …».

Правила ПОЧИНАЮТЬСЯ з «ПРАВИЛА:» навмисно: заглушка (`stub_backend.py`)
відрізає по цьому слову хвіст промпту від тіла останнього джерела, а маркери
`SHAPE_MARKERS` підказують їй, яку форму віддати. Тест звіряє обидва боки.
"""

from __future__ import annotations

from typing import Literal

from app.export.intent import DocumentRequest
from app.generation.prompt_builder import AnswerTask

__all__ = ["SHAPE_MARKERS", "Shape", "document_task", "shape_of"]

Shape = Literal["slides", "text", "table"]

SHAPE_MARKERS: dict[Shape, str] = {
    "slides": "ФОРМАТ — ПРЕЗЕНТАЦІЯ",
    "text": "ФОРМАТ — ДОКУМЕНТ",
    "table": "ФОРМАТ — ТАБЛИЦЯ",
}

_COMMON = "ПРАВИЛА: спирайся лише на ДЖЕРЕЛА вище; після кожного твердження став [n]; українською."
# Без цього модель, побачивши «файл PPTX» у проханні, відповідала «Я не можу
# створити файл формату PPTX, бо генерую лише текст…» (живий випадок
# 23.11.2026) — хоча файл збирає застосунок, а від неї потрібен лише зміст.
_APP_BUILDS = (
    "Файл ({kind}) збере застосунок із твого Markdown — одразу почни з «# …», "
    "без вступу й без слів про файли чи формати."
)
_KINDS = {"pptx": "PowerPoint", "docx": "Word", "xlsx": "Excel"}

# Бюджет виводу. Замір: 5 слайдів по 3 тези ≈ 800 токенів, таблиця на 10 рядків
# ≈ 900. Запас потрібен, бо обрізаний посередині файл гірший за повільніший.
_TOKENS_PER_SLIDE = 260
_QUIZ_TOKENS = 300          # три питання з короткими відповідями
_TEXT_TOKENS = 2400


def shape_of(request: DocumentRequest) -> Shape:
    if request.format == "pptx":
        return "slides"
    return "table" if request.table else "text"


def document_task(request: DocumentRequest) -> AnswerTask:
    task = _shape_task(request)
    rules = f"{task.rules} {_APP_BUILDS.format(kind=_KINDS[request.format])}"
    return AnswerTask(rules=rules, max_tokens=task.max_tokens)


def _shape_task(request: DocumentRequest) -> AnswerTask:
    shape = shape_of(request)
    marker = SHAPE_MARKERS[shape]
    if shape == "slides":
        count = f"рівно {request.slides}" if request.slides else "5–7"
        # Таблиця й питання — те, що робить презентацію робочою на занятті:
        # порівняння читається таблицею, а питання з відповіддю «на клік»
        # (render_pptx) — це готове усне опитування.
        rules = (
            f"{_COMMON}\n{marker} у Markdown. Перший рядок — «# Назва презентації». "
            f"Далі {count} слайдів: кожен — рядок «## Заголовок слайда» і під ним 3–5 коротких "
            "тез «- …» з [n] після кожної. Якщо в джерелах є порівняння чи характеристики — "
            "один зі слайдів оформи таблицею «| … | … |». Наприкінці — «## Питання для "
            "самоперевірки»: 3 питання, кожне рядком «- Питання? — Коротка відповідь [n]». "
            "Не нумеруй слайди словом «Слайд»."
        )
        budget = _QUIZ_TOKENS + 300 + _TOKENS_PER_SLIDE * (request.slides or 7)
        return AnswerTask(rules=rules, max_tokens=min(2900, budget))
    if shape == "table":
        rules = (
            f"{_COMMON}\n{marker} у Markdown. Перший рядок — «# Назва таблиці». Далі одна "
            "таблиця «| … | … | Джерело |»: рядок на КОЖЕН елемент, описаний у БУДЬ-ЯКОМУ "
            "з джерел, — переглянь усі джерела по черзі й нічого не пропускай; у стовпці "
            "«Джерело» — [n]. Лише таблиця, без пояснень."
        )
        return AnswerTask(rules=rules, max_tokens=_TEXT_TOKENS)
    rules = (
        f"{_COMMON}\n{marker} у Markdown. Перший рядок — «# Назва документа». Далі 3–6 "
        "розділів «## Назва розділу» зі зв'язним текстом абзацами; переліки — списком «- …»; "
        "порівняння, характеристики й переліки з властивостями — таблицею «| … | … |»; "
        "поєднуй відомості з усіх доречних джерел. Без звертань і пояснень про себе."
    )
    return AnswerTask(rules=rules, max_tokens=_TEXT_TOKENS)
