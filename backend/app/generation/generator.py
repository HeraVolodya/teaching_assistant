"""Генерація відповіді: стрімінг, утримання, MAP-REDUCE.

ЧОМУ MAP-REDUCE, А НЕ REFINE. У NeoLens стояв `CompactAndRefine`, і при вікні
GPT-4.1 у 1 млн токенів гілка refine НІКОЛИ не виконувалась — тому й не
шкодила. При 8-32k вона виконуватиметься на більшості багатоджерельних
питань, і там вона хибна з трьох причин:
  * строго послідовна → затримка лінійна за кількістю джерел;
  * на малих локальних моделях запобіжник «якщо контекст не допомагає,
    просто повтори наявну відповідь» вироджується в БУКВАЛЬНЕ повторення
    відповіді №1 із тихим ігноруванням джерел 2..n — а це ПРЯМО руйнує
    вимогу «поєднувати декілька джерел одночасно», заради якої проєкт існує;
  * on-device POC уже задокументував ці патології саме в цьому режимі.

Map-reduce тримає кожен виклик малим, робить «який документ що сказав»
структурно явним (і тому цитованим) і дає чесний сигнал «тут не покрито» на
КОЖЕН документ, який живить телеметрію утримання.

СТРАТЕГІЯ ПЕРЕПОВНЕННЯ, у порядку зростання ціни:
    менше діставати → обрізати вікно → map-reduce.

УТРИМАННЯ. Якщо ретривер повернув `abstained`, LLM не викликається взагалі:
кликати модель по відповідь, коли докази вже визнано непридатними, — це
дати їй шанс вигадати. Замість цього — чесне «У наданих матеріалах
недостатньо інформації» плюс те, що знайшлося найближче. Поріг виконується
В КОДІ (`AssistantConfig.confidence_threshold`), а не в промпті: промптом
неможливо надійно змусити 12B-модель відмовитись.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Sequence
from typing import Any

from app.backends.base import (
    ChatParams,
    LlmBackend,
    LlmCancelled,
    LlmError,
    Messages,
)
from app.domain import (
    AnswerChunk,
    AssistantConfig,
    Citation,
    GenerationResult,
    RetrievalDebug,
    RetrievedChunk,
)
from app.generation.citations import (
    merge_page_ranges,
    parse_citations,
    record_unresolved,
)
from app.generation.prompt_builder import (
    abstain_text,
    assign_ordinals,
    build_map_prompt,
    build_prompt_bundle,
    build_reduce_prompt,
    estimate_tokens_uk,
    fit_evidence,
    generation_params,
)

__all__ = ["CancelToken", "Generator"]


class CancelToken:
    """Кооперативне скасування без залежності від asyncio.Event.

    Свідомо мінімальний тип: скасування має бути перевіряним і з синхронного
    коду API-шару, і з тесту, і без запущеного event loop.
    """

    __slots__ = ("_set",)

    def __init__(self) -> None:
        self._set = False

    def cancel(self) -> None:
        self._set = True

    def is_set(self) -> bool:
        return self._set


def _group_by_document(
    chunks: Sequence[RetrievedChunk],
) -> list[tuple[str, str, list[RetrievedChunk]]]:
    """(document_id, назва, чанки) у порядку першої появи.

    Порядок появи = порядок реранкера, тож найрелевантніший документ іде
    першим і в map-кроках, і у зведенні.
    """
    order: list[str] = []
    buckets: dict[str, list[RetrievedChunk]] = {}
    titles: dict[str, str] = {}
    for rc in chunks:
        doc = rc.chunk.document_id
        if doc not in buckets:
            buckets[doc] = []
            order.append(doc)
            titles[doc] = rc.document_title or "Без назви"
        buckets[doc].append(rc)
    return [(doc, titles[doc], buckets[doc]) for doc in order]


class Generator:
    """`Generator.answer(...) -> AsyncIterator[AnswerChunk]`.

    Стрімить `AnswerChunk` чотирьох видів:
        token     — чергова дельта тексту (склеювати БЕЗ роздільника);
        debug     — етап map-reduce, для індикатора в UI;
        citations — payload {"citations": [...], "merged": [...], "unresolved": [...]};
        done      — payload = GenerationResult;
        error     — payload = код помилки, text = повідомлення українською.
    """

    # Нижче цього ліміту символів на чанк обрізання вже коштує сенсу —
    # дешевше піти в map-reduce, ніж подавати моделі уривки уривків.
    MIN_USEFUL_CHUNK_CHARS = 700
    MAP_MAX_TOKENS = 220
    NO_INFO_MARKER = "немає відомостей"

    def __init__(
        self,
        backend: LlmBackend,
        *,
        config: AssistantConfig | None = None,
        model_id: str | None = None,
        persona: str = "",
        context_tokens: int = 8192,
        telemetry: object | None = None,
    ) -> None:
        self.backend = backend
        self.config = config or AssistantConfig()
        self.model_id = model_id
        self.persona = persona
        self.context_tokens = context_tokens
        self.telemetry = telemetry

    # ------------------------------------------------------------- публічне
    async def answer(
        self,
        question: str,
        chunks: Sequence[RetrievedChunk],
        *,
        debug: RetrievalDebug | None = None,
        history: list[dict[str, str]] | None = None,
        cancel: CancelToken | None = None,
        message_id: str | None = None,
        detailed: bool = False,
        force_map_reduce: bool = False,
    ) -> AsyncIterator[AnswerChunk]:
        started = time.perf_counter()
        evidence = list(chunks)
        # Номери [n] призначаються ОДИН раз на весь запит. У map-reduce кожен
        # документ обробляється окремо, але маркери мусять лишатись
        # глобальними — інакше [1] у зведенні означатиме різні чанки.
        assign_ordinals(evidence)

        # --- утримання: жодного виклику моделі ---------------------------
        if not evidence or (debug is not None and debug.abstained):
            async for ac in self._abstain(question, evidence, debug, started):
                yield ac
            return

        params = generation_params(self.config, detailed=detailed)
        kept, cap, truncated = fit_evidence(
            evidence,
            context_tokens=self.context_tokens,
            max_answer_tokens=params.max_tokens,
            persona_tokens=estimate_tokens_uk(self.persona),
            # Мінімум 2: один фрагмент — це вже не «поєднання кількох джерел»,
            # а саме воно є явною вимогою до системи.
            min_chunks=max(2, self.config.min_supporting_chunks()),
        )
        docs_before = len({rc.chunk.document_id for rc in evidence})
        docs_after = len({rc.chunk.document_id for rc in kept})
        # Головний тригер map-reduce — не «замало місця», а ВТРАТА ДОКУМЕНТА:
        # якщо підгонка під контекст викинула ціле джерело, прямий прохід уже
        # порушив вимогу «поєднувати декілька джерел одночасно».
        needs_map = force_map_reduce or docs_after < docs_before or (
            truncated and cap is not None and cap < self.MIN_USEFUL_CHUNK_CHARS
            and docs_after > 1
        )

        try:
            if needs_map:
                async for ac in self._answer_map_reduce(
                    question, evidence, params=params, cancel=cancel,
                    debug=debug, message_id=message_id, started=started,
                ):
                    yield ac
            else:
                async for ac in self._answer_direct(
                    question, kept, params=params, cap=cap, history=history,
                    cancel=cancel, debug=debug, message_id=message_id, started=started,
                    dropped=len(evidence) - len(kept),
                ):
                    yield ac
        except LlmCancelled:
            yield AnswerChunk(kind="done", payload=GenerationResult(
                text="", abstained=False, debug=debug, model_id=self.model_id))
        except LlmError as exc:
            self._event("generation_failed", started, ok=False, error_code=type(exc).__name__)
            yield AnswerChunk(kind="error", text=str(exc), payload=type(exc).__name__)
            yield AnswerChunk(kind="done", payload=GenerationResult(
                text="", abstained=False, debug=debug, model_id=self.model_id))

    # ------------------------------------------------------------ утримання
    async def _abstain(
        self,
        question: str,
        evidence: list[RetrievedChunk],
        debug: RetrievalDebug | None,
        started: float,
    ) -> AsyncIterator[AnswerChunk]:
        text = abstain_text(question, evidence)
        for token in _stream_text(text):
            yield AnswerChunk(kind="token", text=token)
        # Показуємо те, що знайшлося найближче: викладач має бачити, ЧОМУ
        # система вирішила, що відповіді немає, а не голе «не знаю».
        near = [
            Citation(
                ordinal=i, chunk_uid=rc.chunk.chunk_uid, document_id=rc.chunk.document_id,
                document_title=rc.document_title or "Без назви",
                page_from=rc.chunk.page_from, page_to=rc.chunk.page_to,
                page_label=rc.chunk.citation_label(), quote="",
                bboxes=[b.as_dict() for b in rc.chunk.bboxes], language=rc.chunk.language,
            )
            for i, rc in enumerate(evidence[:3], start=1)
        ]
        yield AnswerChunk(kind="citations",
                          payload={"citations": [], "merged": near, "unresolved": []})
        self._event("generation_abstained", started)
        yield AnswerChunk(kind="done", payload=GenerationResult(
            text=text, citations=[], unresolved=[], abstained=True, debug=debug,
            model_id=self.model_id, ttft_ms=0, tokens_out=0,
        ))

    # -------------------------------------------------------- прямий прохід
    async def _answer_direct(
        self,
        question: str,
        evidence: list[RetrievedChunk],
        *,
        params: ChatParams,
        cap: int | None,
        history: list[dict[str, str]] | None,
        cancel: CancelToken | None,
        debug: RetrievalDebug | None,
        message_id: str | None,
        started: float,
        dropped: int,
    ) -> AsyncIterator[AnswerChunk]:
        bundle = build_prompt_bundle(
            question, evidence, self.config, persona=self.persona,
            history=history, char_budget=cap,
        )
        if dropped or cap:
            yield AnswerChunk(kind="debug", payload={
                "stage": "fit", "dropped": dropped, "char_cap": cap,
                "prompt_tokens": bundle.estimated_tokens,
            })
        text, ttft_ms, count = "", None, 0
        async for delta, first_ms in self._stream(bundle.messages, params, cancel):
            if ttft_ms is None:
                ttft_ms = first_ms
            count += 1
            text += delta
            yield AnswerChunk(kind="token", text=delta)

        for ac in self._finalize(text, bundle.evidence, debug=debug,
                                 message_id=message_id, ttft_ms=ttft_ms,
                                 tokens_out=count, started=started, strategy="direct"):
            yield ac

    # ----------------------------------------------------------- map-reduce
    async def _answer_map_reduce(
        self,
        question: str,
        evidence: list[RetrievedChunk],
        *,
        params: ChatParams,
        cancel: CancelToken | None,
        debug: RetrievalDebug | None,
        message_id: str | None,
        started: float,
    ) -> AsyncIterator[AnswerChunk]:
        groups = _group_by_document(evidence)
        map_params = ChatParams(
            temperature=params.temperature, max_tokens=self.MAP_MAX_TOKENS,
            repeat_penalty=params.repeat_penalty, stop=params.stop,
        )
        partials: list[tuple[str, str]] = []
        covered: list[RetrievedChunk] = []

        for _doc_id, title, items in groups:
            yield AnswerChunk(kind="debug", payload={
                "stage": "map", "document": title, "chunks": len(items)})
            # Кожен MAP-виклик має власний бюджет: сенс map-reduce саме в
            # тому, що жоден окремий виклик не переповнює вікно.
            fitted, cap, _ = fit_evidence(
                items, context_tokens=self.context_tokens,
                max_answer_tokens=self.MAP_MAX_TOKENS,
                persona_tokens=0, min_chunks=1,
            )
            piece = ""
            async for delta, _ in self._stream(
                build_map_prompt(question, title, fitted, self.config, char_budget=cap),
                map_params, cancel,
            ):
                piece += delta
            piece = piece.strip()
            # «Немає відомостей.» — це корисний сигнал, а не сміття: він
            # живить телеметрію покриття по документах.
            if not piece or self.NO_INFO_MARKER in piece.lower():
                self._event("map_no_information", started, meta={"document": title})
                continue
            partials.append((title, piece))
            covered.extend(items)

        if not partials:
            async for ac in self._abstain(question, evidence, debug, started):
                yield ac
            return

        yield AnswerChunk(kind="debug", payload={"stage": "reduce", "parts": len(partials)})
        messages = build_reduce_prompt(question, partials, self.config, persona=self.persona)
        text, ttft_ms, count = "", None, 0
        async for delta, first_ms in self._stream(messages, params, cancel):
            if ttft_ms is None:
                ttft_ms = first_ms
            count += 1
            text += delta
            yield AnswerChunk(kind="token", text=delta)

        # Розв'язуємо цитати проти ПОВНОЇ множини доказів, а не лише проти
        # тих документів, що дали частковий висновок: маркер [n] із
        # відкинутого документа має бути видалений як нерозв'язаний, а не
        # тихо зіставлений з чужим чанком.
        for ac in self._finalize(text, evidence, debug=debug, message_id=message_id,
                                 ttft_ms=ttft_ms, tokens_out=count, started=started,
                                 strategy="map-reduce"):
            yield ac

    # ------------------------------------------------------------ службове
    async def _stream(
        self,
        messages: Messages,
        params: ChatParams,
        cancel: CancelToken | None,
    ) -> AsyncIterator[tuple[str, int]]:
        """Обгортка над бекендом: віддає (дельта, мс до першого токена)."""
        t0 = time.perf_counter()
        first_ms: int | None = None
        async for delta in self.backend.chat_stream(
            messages, params=params, model=self.model_id
        ):
            if cancel is not None and cancel.is_set():
                raise LlmCancelled
            if first_ms is None:
                first_ms = int((time.perf_counter() - t0) * 1000)
            yield delta, first_ms

    def _finalize(
        self,
        text: str,
        evidence: Sequence[RetrievedChunk],
        *,
        debug: RetrievalDebug | None,
        message_id: str | None,
        ttft_ms: int | None,
        tokens_out: int,
        started: float,
        strategy: str,
    ) -> list[AnswerChunk]:
        cleaned, citations, unresolved = parse_citations(text, evidence)
        merged = merge_page_ranges(citations)

        # ПОРОЖНЯ ВІДПОВІДЬ — ЦЕ ЗБІЙ, А НЕ РЕЗУЛЬТАТ.
        # Утримання йде через `_abstain` і сюди не потрапляє, тож нуль видимих
        # токенів тут означає, що модель нічого не віддала. Найчастіша причина —
        # reasoning-модель (Gemma 4) вичерпала `max_tokens` на ланцюжок міркувань:
        # `content` не почався, стрім завершився штатно, і без цієї гілки викладач
        # бачив порожню бульбашку без жодного пояснення. Саме так цей дефект
        # і ховався: у логах чисто, HTTP 200, `abstained: false`.
        if not cleaned.strip():
            self._event("generation_empty", started, ok=False, error_code="EMPTY_GENERATION",
                        meta={"strategy": strategy, "tokens_out": tokens_out})
            return [
                AnswerChunk(
                    kind="error",
                    text=(
                        "Модель не повернула тексту відповіді. Найімовірніше, увесь бюджет "
                        f"max_tokens={self.config.max_tokens} пішов на внутрішні міркування "
                        "моделі. Збільште max_tokens у налаштуваннях асистента або візьміть "
                        "модель без ланцюжка міркувань."
                    ),
                    payload="EMPTY_GENERATION",
                ),
                AnswerChunk(kind="done", payload=GenerationResult(
                    text="", citations=[], unresolved=[], abstained=False, debug=debug,
                    ttft_ms=ttft_ms, tokens_out=tokens_out, model_id=self.model_id,
                )),
            ]

        if unresolved and message_id and self.telemetry is not None:
            record_unresolved(
                self.telemetry, message_id, unresolved,
                [str(rc.ordinal_in_prompt) for rc in evidence if rc.ordinal_in_prompt],
            )
        self._event(
            "generation", started,
            meta={"strategy": strategy, "citations": len(citations),
                  "unresolved": len(unresolved), "documents": len({c.document_id for c in citations})},
        )
        return [
            AnswerChunk(kind="citations", payload={
                "citations": citations, "merged": merged, "unresolved": unresolved}),
            AnswerChunk(kind="done", payload=GenerationResult(
                text=cleaned, citations=citations, unresolved=unresolved,
                abstained=False, debug=debug, ttft_ms=ttft_ms,
                tokens_out=tokens_out, model_id=self.model_id,
            )),
        ]

    def _event(self, name: str, started: float, *, ok: bool = True,
               error_code: str | None = None, meta: dict[str, Any] | None = None) -> None:
        recorder = getattr(self.telemetry, "event", None)
        if recorder is None:
            return
        recorder(name, duration_ms=int((time.perf_counter() - started) * 1000),
                 ok=ok, error_code=error_code, meta=meta or {})


def _stream_text(text: str, *, chunk_words: int = 4) -> list[str]:
    """Порізати готовий текст на порції для SSE.

    Відмова теж має «друкуватись»: інакше UI показує стрибок від порожнього
    до повного тексту, і викладач сприймає це як збій, а не як відповідь.
    """
    words = text.split(" ")
    out: list[str] = []
    for i in range(0, len(words), chunk_words):
        piece = " ".join(words[i:i + chunk_words])
        out.append(piece if i == 0 else " " + piece)
    return out
