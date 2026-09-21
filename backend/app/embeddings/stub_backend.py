"""Детермінований бекенд-заглушка ембедингів (`ASISTENT_STUB=1`).

Це патерн `LLAMA_CPP_STUB` з on-device POC NeoLens: увесь застосунок — UI,
індексація, пошук, оцінювання, CI — має працювати наскрізь БЕЗ жодної
завантаженої моделі. 1.5 ГБ ваг не місце в тестовому прогоні.

Три властивості, без яких заглушка була б безглуздою:

1. **Стабільність між запусками і між процесами.** Жодного `random`, жодного
   `hash()` (PYTHONHASHSEED рандомізує його), жодного `id()`. Лише blake2b від
   байтів тексту. Індекс, побудований учора, лишається чинним сьогодні.

2. **Схожі тексти → схожі вектори.** Псевдовипадковий вектор із seed = hash(text)
   дає ортогональні вектори для будь-яких двох різних рядків, тобто пошук
   перетворюється на лотерею і тести пошуку нічого не перевіряють. Тому основа —
   *мішок хешованих символьних 3-грам і словоформ* (hashing trick зі знаком):
   косинус між векторами приблизно дорівнює лексичному перекриттю. Це заодно
   реалістично для української, де морфологія робить точний збіг слів рідкісним,
   а збіг 3-грам — частим.

3. **Заглушка мусить ловити ті самі помилки, що й справжня модель.** Найтихіша
   з них — перевернута або загублена проводка префікса (`selftest`, §4.3). Тому
   заглушка ЗНАЄ шаблон із реєстру: якщо текст прийшов на бік, який ВИМАГАЄ
   префікса, але префікса в ньому немає, вектор навмисно деградується
   детермінованим шумом — рівно так, як справжня модель віддає «правдоподібний»
   вектор поза своїм розподілом. Без цієї деталі самотест у stub-режимі
   проходив би вхолосту й перевіряв би нуль.
"""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata

import numpy as np

from app.embeddings.provider import (
    DEFAULT_MAX_INPUT_TOKENS,
    BaseEmbeddingProvider,
    Side,
    l2_normalize,
)
from app.embeddings.registry import UK_RETRIEVAL_TASK, EmbeddingModel

__all__ = ["STUB_CHARS_PER_TOKEN", "StubEmbeddingProvider"]

# Українська fertility під токенайзером Qwen — ~1.93 символа на токен (план, §5).
# Заглушка не має токенайзера, тож рахує токени за цим коефіцієнтом. Число
# свідомо збігається з тим, на якому побудований бюджет чанка в символах.
STUB_CHARS_PER_TOKEN = 1.93

# Скільки ваги віддано «відбитку всього тексту» відносно мішка n-грам.
# Потрібен, щоб (а) порожній рядок мав ненульовий вектор, (б) два тексти з
# однаковим мішком 3-грам не збігалися байт-у-байт. Мале число: більше — і
# схожість між спорідненими текстами почала б тонути в шумі.
_IDENTITY_WEIGHT = 0.12

# Наскільки псується вектор, коли обов'язковий префікс не застосовано.
# Не 1.0 і не 0.0: справжня модель без префікса віддає вектор, який ще корелює
# зі змістом, але помітно гірше — саме це й треба відтворити.
_MISSING_PREFIX_NOISE = 0.45

_WORD_RE = re.compile(r"\w+", re.UNICODE)
_APOSTROPHES = str.maketrans({"’": "'", "ʼ": "'", "`": "'", "´": "'", "′": "'"})


def _normalize(text: str) -> str:
    """Мінімальна нормалізація, навмисно НЕ залежна від `app.ingestion`.

    Заглушка не має права тягнути модуль приймання: він пишеться паралельно,
    важчий і має власну версію нормалізації, яка входить у `model_key`.
    """
    text = unicodedata.normalize("NFC", text).translate(_APOSTROPHES)
    return " ".join(text.casefold().split())


def _grams(text: str) -> list[str]:
    """Символьні 3-грами + цілі словоформи.

    Символьні n-грами дають морфологічну стійкість («гаубиці» ~ «гаубиця»),
    цілі словоформи — точність на позначеннях типу `д-30`.
    """
    norm = _normalize(text)
    out: list[str] = [f"w:{w}" for w in _WORD_RE.findall(norm)]
    padded = f"  {norm} "
    out.extend(f"c:{padded[i:i + 3]}" for i in range(len(padded) - 2))
    return out


def _digest(payload: str) -> bytes:
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=16).digest()


def _hashed_bag(grams: list[str], dim: int) -> np.ndarray:
    """Hashing trick зі знаком: n-грама → (індекс, ±1), вага = 1 + ln(tf)."""
    vec = np.zeros(dim, dtype=np.float64)
    if not grams:
        return vec
    counts: dict[str, int] = {}
    for g in grams:
        counts[g] = counts.get(g, 0) + 1
    for gram, tf in counts.items():
        d = _digest(gram)
        idx = int.from_bytes(d[:4], "little") % dim
        sign = 1.0 if d[4] & 1 else -1.0
        vec[idx] += sign * (1.0 + math.log(tf))
    return vec


def _pseudo_vector(payload: str, dim: int) -> np.ndarray:
    """Стабільний псевдовипадковий одиничний вектор із хешу рядка.

    Використовується як «відбиток тексту» і як шум деградації. Розгортається
    з blake2b у режимі XOF, тому не залежить ні від версії numpy, ні від
    платформи, ні від PYTHONHASHSEED.
    """
    need = dim * 4
    raw = hashlib.blake2b(payload.encode("utf-8"), digest_size=64).digest()
    stream = bytearray()
    counter = 0
    while len(stream) < need:
        stream.extend(hashlib.blake2b(raw + counter.to_bytes(4, "little"), digest_size=64).digest())
        counter += 1
    ints = np.frombuffer(bytes(stream[:need]), dtype="<u4").astype(np.float64)
    # рівномірне [-1, 1) — достатньо, бо далі йде L2-нормалізація
    return (ints / 2147483648.0) - 1.0


class StubEmbeddingProvider(BaseEmbeddingProvider):
    """Ембедер без моделі. Той самий контракт форматування, що й у ONNX."""

    def __init__(
        self,
        model: EmbeddingModel,
        *,
        task: str = UK_RETRIEVAL_TASK,
        max_input_tokens: int | None = None,
    ) -> None:
        super().__init__(model, task=task)
        # Стеля -8 токенів — це вимога самотесту §4.4, а не косметика:
        # `tokenizer(chunk).length <= max_seq - 8`.
        ceiling = max(8, model.max_seq - 8)
        self._max_input_tokens = min(max_input_tokens or DEFAULT_MAX_INPUT_TOKENS, ceiling)

    # ------------------------------------------------------------ властивості
    @property
    def max_input_tokens(self) -> int:
        return self._max_input_tokens

    def count_tokens(self, text: str) -> int:
        """Оцінка довжини в токенах за українською fertility."""
        if not text:
            return 1  # навіть порожній вхід дає щонайменше EOS
        return max(1, math.ceil(len(text) / STUB_CHARS_PER_TOKEN))

    # ---------------------------------------------------------------- префікс
    def template_affixes(self, side: Side) -> tuple[str, str]:
        """Константні частини шаблону реєстру для цього боку.

        Виводяться з самого шаблону через маркер-заповнювач, тому лишаються
        побайтово вірними навіть якщо шаблон у реєстрі зміниться. Жодного
        дублювання рядка `"Query:"` у коді — саме на такому дублюванні й
        народжуються тихі розбіжності.
        """
        marker = "\x00"
        if side == "query":
            rendered = self._model.format_query(marker, self._task)
        else:
            rendered = self._model.format_document(marker)
        if marker not in rendered:      # шаблону немає — текст іде як є
            return "", ""
        head, _, tail = rendered.partition(marker)
        return head, tail

    def _strip_template(self, text: str, side: Side) -> tuple[str, bool]:
        """Відрізати службову обгортку. Повертає (зміст, чи був префікс).

        Інструкція — це метадані задачі, а не зміст документа; справжня модель
        її «розуміє», а не вбудовує лексично. Заглушка тому просто прибирає її
        з мішка n-грам — інакше константні 40 символів англійської інструкції
        домінували б над коротким українським запитом.
        """
        head, tail = self.template_affixes(side)
        if not head and not tail:
            return text, True           # вимоги немає → вважаємо виконаною
        core = text
        matched = True
        if head:
            if core.startswith(head):
                core = core[len(head):]
            else:
                matched = False
        if tail:
            if core.endswith(tail):
                core = core[: len(core) - len(tail)]
            else:
                matched = False
        return (core, True) if matched else (text, False)

    # ------------------------------------------------------------- кодування
    def _encode_formatted(self, texts: list[str], *, side: Side) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float64)
        for row, text in enumerate(texts):
            core, has_prefix = self._strip_template(text, side)

            tokens = self.count_tokens(core)
            truncated = tokens > self._max_input_tokens
            if truncated:
                core = core[: int(self._max_input_tokens * STUB_CHARS_PER_TOKEN)]
                tokens = self._max_input_tokens
            self.stats.note(tokens=tokens, truncated=truncated)

            bag = l2_normalize(_hashed_bag(_grams(core), self.dim))[0].astype(np.float64)
            ident = l2_normalize(_pseudo_vector(f"id::{core}", self.dim))[0].astype(np.float64)
            vec = bag + _IDENTITY_WEIGHT * ident

            if not has_prefix:
                # Імітація «вектор поза розподілом моделі»: зміст ще видно,
                # але косинус із релевантним документом просідає. Саме це й
                # ловить перевірка проводки префікса в selftest.
                noise = l2_normalize(_pseudo_vector(f"noprefix::{core}", self.dim))[0]
                vec = l2_normalize(vec)[0] + _MISSING_PREFIX_NOISE * noise.astype(np.float64)

            out[row] = vec
        self.stats.batches += 1
        return out.astype(np.float32)
