"""Реранкінг мовною моделлю LM Studio (listwise) — на GPU замість CPU-крос-енкодера.

ЧОМУ. Виміряно на живих даних (підручник «Бойова робота», 232 с., 329 фрагментів):
  * Qwen3-Reranker-0.6B (int8 ONNX) на CPU оцінює 50 кандидатів ~80 с — це 95 %
    часу всієї відповіді на просте питання;
  * і робить це погано саме на питаннях «що таке X»: визначенню «Гарматна обслуга —
    найменший вогневий підрозділ артилерії» він дав 0.15, а загальним абзацам вступу,
    які лише ЗГАДУЮТЬ артилерійські підрозділи, — 0.86; сторінці про орієнтири — 0.77.
    До того ж його int8-скори шумні: математично еквівалентне обчислення (інший склад
    батча) міняє 2–3 з топ-5.
  * Gemma 4 E4B, що вже піднята в LM Studio для відповіді, оцінює ті самі 50 кандидатів
    одним запитом за ~2.5 с на GPU (prefill ~2000 ток/с) і знаходить те визначення.

ЯК. Один запит на всіх кандидатів: питання + стислі уривки (речення з термінами питання,
визначення — першими), модель повертає до `max_listed` номерів з оцінкою 1–3. Решта
кандидатів отримує 0. Оцінки переводяться в [0, 1] (3 → 0.9, 2 → 0.6, 1 → 0.3) із малим
кроком за позицією, щоб зберегти порядок моделі. Пороги утримання `AssistantConfig`
(0.15 / 0.35 / 0.55) на цій шкалі означають: low — достатньо дотичного уривка, medium —
потрібен суттєвий, high — два суттєві (`min_supporting_chunks`).

ЗАПАСНИЙ ШЛЯХ. LM Studio не відповідає, модель не піднята, відповідь не розібрати —
кандидати оцінює CPU-крос-енкодер (`fallback`, створюється ліниво при першій потребі:
без нього процес не тримає зайвих 1.2 ГБ ваг і ~19 ГБ піку на батчі).

Синхронний HTTP свідомо: пошук виконується в окремому потоці, і реранкер мусить
лишатися звичайним `score(query, docs) -> list[float]` за контрактом модуля.
"""

from __future__ import annotations

import logging
import re
import threading
from collections.abc import Callable, Sequence
from typing import Any

import httpx

__all__ = ["GRADE_SCORES", "LlmListwiseReranker", "build_snippet", "parse_grades"]

log = logging.getLogger(__name__)

# Оцінка моделі → скор на шкалі порогів утримання.
GRADE_SCORES: dict[int, float] = {3: 0.9, 2: 0.6, 1: 0.3}
# Крок за позицією всередині однієї оцінки: зберігає порядок моделі, не виходячи
# за межі смуги оцінки (8 позицій × 0.005 = 0.04 < 0.3).
POSITION_STEP = 0.005

# Скільки символів уривка на кандидата і скільки токенів лишити на інструкцію й вихід.
DEFAULT_SNIPPET_CHARS = 420
MIN_SNIPPET_CHARS = 160
PROMPT_RESERVE_TOKENS = 900
# Консервативна оцінка для української в токенайзері Gemma (символів на токен):
# занижена оцінка бюджету дала б `stopAtLimit` посеред переліку.
CHARS_PER_TOKEN = 2.6

_SENTENCE_RE = re.compile(r"(?<=[.!?;])\s+")
_WORD_RE = re.compile(r"[^\W\d_][^\W_'’ʼ-]*", re.UNICODE)
_DEFINITION_RE = re.compile(r"^[^.?!]{2,80}?\s[-—–]\s")
_GRADE_RE = re.compile(r"(\d{1,3})\s*[:=]\s*([0-3])")
_NUMBER_RE = re.compile(r"\d{1,3}")

_INSTRUCTION = (
    "Завдання: вибери до {k} уривків, які найкраще допомагають ВІДПОВІСТИ на питання, "
    "найкращий першим. Формат відповіді — лише «номер:оцінка» через кому, наприклад "
    "«12:3, 4:2». Оцінка 3 — уривок прямо відповідає (визначення, точні дані, порядок дій); "
    "2 — суттєво допомагає; 1 — лише дотичний. Уривки, що тільки згадують тему, не "
    "відповідаючи на питання, не включай. Якщо жоден уривок не допомагає — напиши 0."
)


def _stems(text: str) -> set[str]:
    """Грубі основи слів запиту: перші 5 літер слів від 4 літер.

    Українська словозміна ламає точний збіг («підрозділ» / «підрозділу» /
    «підрозділи»), а повна лематизація тут зайва: основа потрібна лише, щоб
    вибрати з довгого фрагмента речення, варті показу моделі.
    """
    return {w[:5].lower() for w in _WORD_RE.findall(text) if len(w) >= 4}


def build_snippet(text: str, query: str, limit: int = DEFAULT_SNIPPET_CHARS) -> str:
    """Стислий уривок фрагмента для listwise-оцінки.

    Перший рядок `rerank_text` — назва розділу (`Chunk.build_rerank_text`); вона
    лишається префіксом, бо саме з неї видно «Вступ», «Висновки», словник термінів.
    Із тіла беруться речення з термінами питання, визначення («X — …») першими;
    у підсумку — в порядку тексту. Без жодного збігу — початок фрагмента.
    """
    head, _, body = text.partition("\n")
    head = " ".join(head.split())
    if not body or len(head) > 60:
        # Немає назви розділу (порожній шлях заголовків): перший рядок — це вже
        # текст фрагмента, і губити його не можна.
        head, body = "", text
    title = f"(розділ: {head}) " if head else ""
    sentences = [s for s in _SENTENCE_RE.split(" ".join(body.split())) if s]
    stems = _stems(query)
    budget = max(1, limit - len(title))

    scored: list[tuple[int, int, int]] = []
    for index, sentence in enumerate(sentences):
        hits = len(stems & _stems(sentence))
        if hits:
            bonus = 1 if _DEFINITION_RE.match(sentence) else 0
            scored.append((hits + bonus, -len(sentence), index))
    picked: list[int] = []
    used = 0
    for _, neg_len, index in sorted(scored, reverse=True):
        if used + (-neg_len) > budget and picked:
            continue
        picked.append(index)
        used += -neg_len + 3
        if used >= budget:
            break
    body_text = " … ".join(sentences[i] for i in sorted(picked)) if picked else " ".join(sentences)
    return (title + body_text)[:limit]


def parse_grades(text: str, count: int, *, limit: int = 8) -> list[tuple[int, int]] | None:
    """Розібрати «номер:оцінка, …» → [(індекс з 0, оцінка), …] у порядку моделі.

    Порожній список — модель свідомо сказала «жоден не допомагає» (`0`).
    None — відповідь не розібрати (тоді спрацьовує запасний реранкер).
    Номери без оцінки (модель забула формат) — оцінка 2: модель їх таки вибрала.
    """
    pairs = [(int(n), int(g)) for n, g in _GRADE_RE.findall(text)]
    if not pairs:
        numbers = [int(n) for n in _NUMBER_RE.findall(text)]
        if not numbers:
            return None
        if all(n == 0 for n in numbers):
            return []
        pairs = [(n, 2) for n in numbers]
    out: list[tuple[int, int]] = []
    seen: set[int] = set()
    for number, grade in pairs:
        if not 1 <= number <= count or number in seen or grade == 0:
            continue
        seen.add(number)
        out.append((number - 1, grade))
        if len(out) >= limit:
            break
    return out


class LlmListwiseReranker:
    """Реалізує протокол `Reranker` через LM Studio (OpenAI-сумісна поверхня)."""

    name = "llm"

    def __init__(
        self,
        *,
        base_url: str | None = None,
        model: str | None = None,
        fallback: Callable[[], Any] | None = None,
        timeout_s: float = 90.0,
        max_listed: int = 8,
        snippet_chars: int = DEFAULT_SNIPPET_CHARS,
        context_tokens: int = 8192,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/") if base_url else None
        self._model = model
        self._fallback_factory = fallback
        self._fallback: Any = None
        self._timeout_s = timeout_s
        self._max_listed = max_listed
        self._snippet_chars = snippet_chars
        self._context_tokens = context_tokens
        self._transport = transport
        self._lock = threading.Lock()
        # Модель, що відмовилась від `reasoning_effort`: більше не шлемо.
        self._no_reasoning_cfg: set[str] = set()
        self.last_mode: str = "llm"   # "llm" | "fallback" — для діагностики

    # ------------------------------------------------------------ контракт
    @property
    def model_key(self) -> str:
        return f"llm:{self._model or 'auto'}"

    def score(
        self,
        query: str,
        docs: list[str],
        *,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> list[float]:
        if not docs:
            return []
        try:
            scores = self._llm_scores(query, docs)
            self.last_mode = "llm"
        except Exception as exc:
            log.warning("LLM-реранкінг не вдався (%s) — оцінюю запасним крос-енкодером.", exc)
            scores = self._fallback_scores(query, docs, on_progress)
            self.last_mode = "fallback"
        if on_progress is not None:
            on_progress(len(docs), len(docs))
        return scores

    def close(self) -> None:
        closer = getattr(self._fallback, "close", None)
        if callable(closer):
            closer()

    # ------------------------------------------------------------ LM Studio
    def _client(self) -> httpx.Client:
        return httpx.Client(transport=self._transport, timeout=httpx.Timeout(
            connect=2.0, read=self._timeout_s, write=30.0, pool=2.0))

    def _base_urls(self) -> list[str]:
        if self._base_url:
            return [self._base_url]
        from app.backends.lmstudio_client import candidate_base_urls

        return candidate_base_urls()

    def _resolve(self, client: httpx.Client) -> tuple[str, str, int, bool]:
        """(base_url, модель, контекст, чи можна вимкнути міркування) — зі стану сервера.

        Модель та сама, що відповідає (налаштована, інакше вже піднята, інакше
        єдина наявна): вона вже в пам'яті, а друга модель витіснила б першу.
        """
        last_error: Exception | None = None
        for base in self._base_urls():
            try:
                resp = client.get(f"{base}/api/v1/models", timeout=5.0)
                resp.raise_for_status()
                payload = resp.json()
            except (httpx.HTTPError, ValueError) as exc:
                last_error = exc
                continue
            items = payload.get("models") or payload.get("data") or []
            chat = [m for m in items if isinstance(m, dict)
                    and str(m.get("type", "llm")).lower() in ("llm", "vlm")]
            chosen = None
            if self._model:
                chosen = next((m for m in chat if str(m.get("key") or m.get("id")) == self._model), None)
            if chosen is None:
                loaded = [m for m in chat if m.get("loaded_instances")]
                chosen = loaded[0] if loaded else (chat[0] if len(chat) == 1 else None)
            if chosen is None:
                raise RuntimeError("у LM Studio немає однозначної мовної моделі для реранкінгу")
            key = str(chosen.get("key") or chosen.get("id"))
            instances = chosen.get("loaded_instances") or []
            context = int((instances[0] or {}).get("context_length") or 0) if instances else 0
            caps = chosen.get("capabilities")
            reasoning = caps.get("reasoning") if isinstance(caps, dict) else None
            options = (reasoning or {}).get("allowed_options") or [] if isinstance(reasoning, dict) else []
            return base, key, context or self._context_tokens, "off" in options
        raise RuntimeError(f"LM Studio не відповідає: {last_error}")

    def _llm_scores(self, query: str, docs: Sequence[str]) -> list[float]:
        with self._lock, self._client() as client:
            base, model, context, can_disable_reasoning = self._resolve(client)

            available = max(0, context - PROMPT_RESERVE_TOKENS) * CHARS_PER_TOKEN
            per_doc = min(self._snippet_chars, int(available / max(1, len(docs))))
            count = len(docs)
            if per_doc < MIN_SNIPPET_CHARS:
                # Вікно моделі замале для всіх: оцінюємо стільки перших (за RRF),
                # скільки влазить; решта лишається з нулем.
                count = max(1, int(available / MIN_SNIPPET_CHARS))
                per_doc = MIN_SNIPPET_CHARS
            snippets = [build_snippet(d, query, per_doc) for d in docs[:count]]
            listing = "\n".join(f"[{i}] {s}" for i, s in enumerate(snippets, start=1))
            prompt = (f"Питання: {query.strip()}\n\nУРИВКИ:\n{listing}\n\n"
                      f"{_INSTRUCTION.format(k=self._max_listed)}\n\nПитання: {query.strip()}\nВідповідь:")
            body: dict[str, Any] = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 12 * self._max_listed + 16,
                "stream": False,
            }
            if can_disable_reasoning and model not in self._no_reasoning_cfg:
                body["reasoning_effort"] = "none"
            resp = client.post(f"{base}/v1/chat/completions", json=body)
            if resp.status_code == 400 and "reasoning" in resp.text.lower() and "reasoning_effort" in body:
                self._no_reasoning_cfg.add(model)
                body.pop("reasoning_effort")
                resp = client.post(f"{base}/v1/chat/completions", json=body)
            resp.raise_for_status()
            choices = resp.json().get("choices") or []
            text = str(((choices[0] or {}).get("message") or {}).get("content") or "") if choices else ""

        grades = parse_grades(text, count, limit=self._max_listed)
        if grades is None:
            raise RuntimeError(f"відповідь моделі не розібрати: {text[:120]!r}")
        scores = [0.0] * len(docs)
        for position, (index, grade) in enumerate(grades):
            scores[index] = max(0.0, GRADE_SCORES[grade] - POSITION_STEP * position)
        return scores

    # ------------------------------------------------------------ запасний шлях
    def _fallback_scores(
        self, query: str, docs: list[str], on_progress: Callable[[int, int], None] | None
    ) -> list[float]:
        if self._fallback is None and self._fallback_factory is not None:
            self._fallback = self._fallback_factory()
        if self._fallback is None:
            raise RuntimeError(
                "Реранкінг недоступний: LM Studio не відповідає, а запасного реранкера немає."
            )
        return list(self._fallback.score(query, docs))
