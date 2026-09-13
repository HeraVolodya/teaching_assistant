"""Детермінована заглушка LLM для ASISTENT_STUB=1.

Це не «мок для одного тесту», а несуча конструкція. Патерн LLAMA_CPP_STUB із
on-device POC NeoLens: увесь застосунок — UI, SSE, цитати, телеметрія —
має працювати наскрізь без жодної завантаженої моделі. Саме це дозволяє:
  * ганяти CI без GPU і без 1.5 ГБ ваг (єдиний компонент, що потребує GPU, —
    генерація, і вона живе ПОЗА процесом, тож межа тривіально мокається);
  * показувати інтерфейс на машині, де LM Studio ще не встановлено;
  * писати тести генерації, які перевіряють НАШ код, а не примхи семплера.

Заглушка НЕ вигадує: вона будує осмислену українську відповідь із поданих
фрагментів і ставить маркери [n] рівно тих джерел, які реально бачила.
Тому тест «маркер поза множиною доказів видаляється» перевіряє наш
пост-процесор, а не випадковість.
"""

from __future__ import annotations

import asyncio
import os
import re
from collections.abc import AsyncIterator

from app.backends.base import (
    BackendHealth,
    ChatParams,
    LoadConfig,
    Messages,
    ModelInfo,
    ModelNotLoaded,
)

__all__ = ["StubBackend", "STUB_MODEL_KEY", "is_stub_enabled"]

STUB_MODEL_KEY = "asistent-stub-uk"

# Заголовок блоку джерела, який складає app/generation/prompt_builder.py:
#   [3] Балістика та стрільба — с. 147–149
_SOURCE_HEADER = re.compile(r"^\[(\d{1,3})\]\s*(.+?)\s*$", re.MULTILINE)
_MARKER = re.compile(r"\[(\d{1,3})\]")
_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")


def is_stub_enabled() -> bool:
    """ASISTENT_STUB=1 → детерміновані заглушки в усьому застосунку."""
    return os.environ.get("ASISTENT_STUB", "").strip().lower() in ("1", "true", "yes", "on")


def _split_source_blocks(text: str) -> list[tuple[int, str, str]]:
    """Витягти (порядковий номер, заголовок, тіло) з блоку ДЖЕРЕЛА."""
    blocks: list[tuple[int, str, str]] = []
    matches = list(_SOURCE_HEADER.finditer(text))
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        # Хвіст промпту після джерел (правила, повторене питання) тілом не є.
        body = body.split("\nПРАВИЛА")[0].split("\nПитання:")[0].strip()
        blocks.append((int(m.group(1)), m.group(2).strip(), body))
    return blocks


def _first_sentences(text: str, limit: int = 2) -> str:
    """Перші речення тіла — рівно те, що зробила б модель у стислій відповіді."""
    cleaned = " ".join(text.split())
    if not cleaned:
        return ""
    parts = [p for p in _SENTENCE_END.split(cleaned) if p]
    out = " ".join(parts[:limit]).strip()
    return out[:400].rstrip()


def compose_stub_answer(messages: Messages) -> str:
    """Побудувати відповідь із того, що реально є в промпті.

    Три режими, у порядку перевірки:
      1. є блоки ДЖЕРЕЛА → зведена відповідь із маркерами [n];
      2. немає блоків, але є маркери [n] (це крок REDUCE у map-reduce, де
         «джерелами» слугують часткові відповіді) → зберегти ВСІ маркери,
         бо саме їх втрата і є та регресія, яку тест мусить ловити;
      3. нічого → чесне «недостатньо інформації».
    """
    user_parts = [m.get("content", "") for m in messages if m.get("role") == "user"]
    prompt = "\n\n".join(user_parts)

    question = ""
    qm = re.search(r"^Питання:\s*(.+)$", prompt, re.MULTILINE)
    if qm:
        question = qm.group(1).strip()

    blocks = _split_source_blocks(prompt)
    if blocks:
        lead = f"Щодо запиту «{question}» надані матеріали дають таке." if question \
            else "Надані матеріали дають таке."
        sentences: list[str] = []
        for ordinal, title, body in blocks:
            gist = _first_sentences(body) or f"фрагмент із джерела «{title}»"
            sentences.append(f"{gist} [{ordinal}]")
        body_text = " ".join(sentences)
        tail = ("Наведені положення узгоджуються між собою і разом відповідають на "
                "поставлене запитання.") if len(blocks) > 1 else \
               "Це єдиний фрагмент, що стосується запитання."
        return f"{lead} {body_text} {tail}"

    ordinals = list(dict.fromkeys(int(m) for m in _MARKER.findall(prompt)))
    if ordinals:
        marks = " ".join(f"[{o}]" for o in ordinals)
        return (
            "Зведена відповідь за частковими висновками з усіх опрацьованих документів. "
            f"Кожен із документів підтверджує окремий бік запитання: {marks}. "
            "Розбіжностей між джерелами не виявлено."
        )

    return ("У наданих матеріалах недостатньо інформації, щоб відповісти на це "
            "запитання. Уточніть формулювання або додайте відповідний документ "
            "до колекції асистента.")


def tokenize_for_stream(text: str) -> list[str]:
    """Порізати відповідь на «токени» так, як це робив би llama.cpp.

    Пробіл лишається на початку токена — саме так поводяться SentencePiece-
    токенайзери, і саме така склейка перевіряє, що ми ніде не робимо
    `" ".join(deltas)` замість `"".join(deltas)`.
    """
    return re.findall(r"\s*\S+", text)


class StubBackend:
    """Реалізує протокол `LlmBackend` без жодної моделі."""

    name = "stub"

    def __init__(self, *, delay_s: float = 0.0, fail_first: int = 0) -> None:
        self.delay_s = delay_s
        # Скільки перших викликів має впасти — для тестів відновлення.
        self.fail_first = fail_first
        self.calls: list[Messages] = []
        self.loaded: LoadConfig | None = None

    async def health(self) -> BackendHealth:
        return BackendHealth(
            ok=True, base_url=None, api_version="stub",
            models=(self._model_info(),),
            detail="Режим заглушки: моделі не завантажуються, відповіді детерміновані.",
        )

    async def list_models(self) -> list[ModelInfo]:
        return [self._model_info()]

    async def load_model(self, cfg: LoadConfig) -> ModelInfo:
        self.loaded = cfg
        return self._model_info(loaded_context=cfg.context_length)

    def _model_info(self, *, loaded_context: int | None = None) -> ModelInfo:
        return ModelInfo(
            key=STUB_MODEL_KEY, display_name="Заглушка (українська)", kind="llm",
            engine="stub", loaded=True, max_context=32768,
            loaded_context=loaded_context or 8192, quantization="none",
        )

    async def chat_stream(
        self,
        messages: Messages,
        *,
        params: ChatParams | None = None,
        model: str | None = None,
        cancel: asyncio.Event | None = None,
        warm: bool = True,
    ) -> AsyncIterator[str]:
        self.calls.append(list(messages))
        if self.fail_first > 0:
            self.fail_first -= 1
            raise ModelNotLoaded(model or STUB_MODEL_KEY)

        p = params or ChatParams()
        text = compose_stub_answer(messages)
        for emitted, token in enumerate(tokenize_for_stream(text)):
            if cancel is not None and cancel.is_set():
                return
            # Ліміт токенів дотримується, як у справжньої моделі: інакше тест
            # «max_tokens поважається» проходив би хибно.
            if emitted >= p.max_tokens:
                return
            if self.delay_s:
                await asyncio.sleep(self.delay_s)
            yield token
