"""Гібридний пошук — три гілки, злиття, реранкінг, диверсифікація, порядок.

Конвеєр (план, §7):

    dense top-60  ∪  BM25-по-лемах top-60  ∪  char 3–5-грамний TF-IDF top-30
        → зважений RRF → дедуплікація → реранк top-50
        → утримання АБО відбір → диверсифікація джерел
        → auto-merge до L1-батька → дворівневе впорядкування
        → пакування під бюджет

ЧОМУ ТРИ ГІЛКИ, А НЕ ОДНА
-------------------------
Доказова база тут чесно суперечлива, і план фіксує це прямо: переможець
UNLP 2026 повідомляє, що RRF не дав приросту над чистим dense (їхній корпус —
перефразовані MCQ, тобто режим, де семантика вирішує все), але команда УКУ
виміряла, що СИМВОЛЬНИЙ TF-IDF перевершує словесний на +21% відносних саме
через українську морфологію. Наш корпус — артилерійські підручники: індекси,
калібри, коди ГРАУ/НАТО, позначення на кшталт «Д-30» і «2С1», номери таблиць
стрільби. Це режим, де sparse виграє. Тому будуємо всі три гілки, робимо ваги
конфігурованими (`AssistantConfig`) і даємо harness'у вирішити per-corpus.
Це вимірюваний результат НДР, а не здогад.

МЕТАДАННЕ ФІЛЬТРУВАННЯ — ПЕРШОКЛАСНЕ, А НЕ ДОДАТКОВЕ
----------------------------------------------------
Стаття автора (УДК 004.65:623.4) називає фільтрацію за метаданими
ОБОВ'ЯЗКОВОЮ. Трирівнева адаптивна стратегія:

  L0 — фізична партиція: одна колекція = один файл індексу. Асистенти
       ізольовані фізично, тож фільтр «свій/чужий» коштує нуль і не втрачає
       recall узагалі.
  L1 — ANN з overfetch, масштабованим за селективністю: HNSW не вміє
       фільтрувати всередині обходу графа, тож при фільтрі, що лишає 40%
       колекції, треба дістати ~2.5× більше й відкинути зайве.
  L2 — точний brute force над відфільтрованою множиною, коли селективність
       < 0.15: на вузькому фільтрі обхід графа вироджується в блукання серед
       відкинутих сусідів, і чесний скан по 3% колекції і швидший, і точніший.

ЕКРАНУВАННЯ FTS5 — НЕ СТИЛЬОВА ДРІБНИЦЯ
---------------------------------------
Кожен термін подається в MATCH як `'"' + t.replace('"','""') + '"'`.
Голий `Д-30` парситься FTS5 як `Д NOT 30` і кидає `no such column: 30`.
Запит викладача «ТТХ Д-30» без екранування — це 500-та помилка в чаті.
"""

from __future__ import annotations

import math
import re
import sqlite3
import time
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from app.db.repositories import ChunkRepo, CollectionRepo, blob_to_vector
from app.domain import AssistantConfig, Chunk, ChunkLevel, RetrievalDebug, RetrievedChunk
from app.retrieval import diversity, fusion, reorder
from app.retrieval.vector_index import CollectionIndex, IndexParams

__all__ = [
    "DEFAULT_CONTEXT_CHAR_BUDGET",
    "EXACT_SELECTIVITY_THRESHOLD",
    "HybridRetriever",
    "MetadataFilter",
    "QueryTerms",
    "RerankerLike",
    "analyze_query",
    "build_match_expression",
    "char_ngram_scores",
    "fts_term",
]

# Нижче цієї селективності ANN поступається точному скану (стратегія L2).
EXACT_SELECTIVITY_THRESHOLD = 0.15
# Стеля overfetch: без неї фільтр на 0.1% колекції попросив би в HNSW
# 60 000 сусідів і зробив «швидкий» шлях повільнішим за точний.
MAX_OVERFETCH = 20

# Бюджет контексту в СИМВОЛАХ — з тієї самої причини, що й бюджет чанка
# (контракт, §4): українська fertility коливається 2.16–3.90 ток/слово між
# токенайзерами, тож токен-номінований бюджет тихо змінюється вдвічі при
# зміні генератора. 8000 символів ≈ 3400 токенів Gemma — рівно та цифра, під
# яку план розраховував вікно 8k (§10).
DEFAULT_CONTEXT_CHAR_BUDGET = 8000

# Обмеження символьної n-грамної гілки. Вона рахується на чистому Python над
# КАНДИДАТАМИ, а не над корпусом, тож ці стелі — це її бюджет затримки.
NGRAM_SIZES = (3, 4, 5)
NGRAM_POOL_LIMIT = 200
NGRAM_TEXT_LIMIT = 800
NGRAM_QUERY_LIMIT = 120
NGRAM_LENGTH_B = 0.75

# Ваги колонок FTS5: леми 1.0, словоформи 0.4, коди 2.5 (план, §3).
BM25_WEIGHTS = (1.0, 0.4, 2.5)

_TOKEN_RE = re.compile(r"(?u)[^\W_]+(?:[''’ʼ-][^\W_]+)*")
_CODE_RE = re.compile(r"(?u)(?<![\w-])(?=[\w-]*[^\W\d_])(?=[\w-]*\d)[\w-]{2,}(?![\w-])")
_APOSTROPHES = {"’": "'", "ʼ": "'", "´": "'", "′": "'", "`": "'"}
_DASHES = {"–": "-", "—": "-", "−": "-", "‑": "-"}


# --------------------------------------------------------------- екранування FTS
def fts_term(term: str) -> str:
    """ЄДИНИЙ дозволений спосіб подати термін у FTS5 MATCH.

    Без подвійних лапок `Д-30` стає оператором NOT, а `:`, `*`, `^` і `NEAR`
    змінюють семантику запиту або кидають виняток.
    """
    return '"' + term.replace('"', '""') + '"'


def build_match_expression(terms: Sequence[str]) -> str:
    """OR по екранованих термінах. Порожній список → порожній рядок."""
    seen: list[str] = []
    for t in terms:
        t = t.strip()
        if len(t) < 2 or t in seen:
            continue
        seen.append(t)
    return " OR ".join(fts_term(t) for t in seen)


# ------------------------------------------------------------- аналіз запиту
@dataclass(slots=True)
class QueryTerms:
    """Розібраний запит. Нормалізація ІДЕНТИЧНА індексації — інакше леми не
    зійдуться і BM25-гілка тихо перестане знаходити що-небудь."""

    raw: str
    normalized: str
    forms: list[str] = field(default_factory=list)
    lemmas: list[str] = field(default_factory=list)
    codes: list[str] = field(default_factory=list)
    anchored: bool = False

    @property
    def match_terms(self) -> list[str]:
        """Усе, що подається в chunk_fts: леми + словоформи + коди."""
        out: list[str] = []
        for group in (self.lemmas, self.forms, self.codes):
            for t in group:
                if t not in out:
                    out.append(t)
        return out


def _fallback_normalize(text: str) -> str:
    """Резервна нормалізація на випадок, коли модуль приймання ще не готовий.

    Свідомо мінімальна підмножина `app.ingestion.normalize_uk`: NFC,
    уніфікація апострофів і тире, casefold. Гомогліфи й подвійну емісію
    складених слів робить справжня функція — дублювати її тут означало б
    завести другу версію правди, яка розійдеться з індексом.
    """
    text = unicodedata.normalize("NFC", text)
    for src, dst in {**_APOSTROPHES, **_DASHES}.items():
        text = text.replace(src, dst)
    return text.casefold()


@lru_cache(maxsize=1)
def _normalizer() -> Any:
    """Розв'язати функцію нормалізації один раз на процес.

    Імпорт лінивий навмисно: API-процес не має платити за модуль приймання на
    старті (контракт, правило 1 — холодний старт ~0.6 с). Але й платити за
    пошук у sys.modules на кожен із ~200 кандидатів n-грамної гілки теж не
    треба, тому результат кешується.
    """
    try:
        from app.ingestion.normalize_uk import normalize_uk  # type: ignore[attr-defined]

        return normalize_uk
    except Exception:
        return _fallback_normalize


def _normalize(text: str) -> str:
    try:
        return str(_normalizer()(text))
    except Exception:
        return _fallback_normalize(text)


def _coerce_lemmas(stream: Any, fallback: list[str]) -> list[str]:
    """Привести `LemmaStream` модуля приймання до списку рядків.

    Точна форма LemmaStream належить іншому модулю й може змінитись; ми не
    імпортуємо його типи (контракт: модулі не імпортують типи один в одного),
    тому приймаємо кілька правдоподібних форм і мовчки відступаємо до
    словоформ. Гірший наслідок помилки тут — BM25 по формах замість лем,
    тобто просто слабший, але робочий пошук.
    """
    if stream is None:
        return fallback
    for attr in ("lemmas", "tokens"):
        value = getattr(stream, attr, None)
        if isinstance(value, (list, tuple)):
            stream = value
            break
    if isinstance(stream, str):
        return stream.split()
    if not isinstance(stream, (list, tuple)):
        return fallback
    out: list[str] = []
    for item in stream:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, (list, tuple)) and item:
            out.append(str(item[0]))
        else:
            lemma = getattr(item, "lemma", None)
            if isinstance(lemma, str):
                out.append(lemma)
    return out or fallback


@lru_cache(maxsize=1)
def _lemmatizer() -> Any:
    """Те саме, що й `_normalizer`: лінивий імпорт, розв'язаний один раз."""
    try:
        from app.ingestion.lemmatize import lemmatize  # type: ignore[attr-defined]

        return lemmatize
    except Exception:
        return None


def _lemmatize(tokens: list[str], language: str) -> list[str]:
    lemmatize = _lemmatizer()
    if lemmatize is None:
        return tokens
    try:
        return _coerce_lemmas(lemmatize(tokens, language), tokens)
    except Exception:
        return tokens


def analyze_query(query: str, *, language: str = "uk") -> QueryTerms:
    """Нормалізувати, токенізувати, лематизувати, вибрати коди."""
    normalized = _normalize(query)
    forms = _TOKEN_RE.findall(normalized)
    # Коди беруться з СИРОГО запиту: casefold і нормалізація тире можуть
    # змінити позначення, а воно мусить лишитись побайтово тим, що написав
    # викладач.
    codes = _CODE_RE.findall(query)
    lemmas = _lemmatize(forms, language)
    return QueryTerms(
        raw=query,
        normalized=normalized,
        forms=forms,
        lemmas=lemmas,
        codes=codes,
        anchored=fusion.has_lexical_anchor(query),
    )


# ------------------------------------------------------- метаданий фільтр
@dataclass(frozen=True, slots=True)
class MetadataFilter:
    """Фасети з `chunk_facets`. Порожній фільтр = вся колекція."""

    document_ids: tuple[str, ...] = ()
    doc_types: tuple[str, ...] = ()
    languages: tuple[str, ...] = ()
    years: tuple[int, ...] = ()
    year_from: int | None = None
    year_to: int | None = None
    chapter_ids: tuple[int, ...] = ()

    def is_empty(self) -> bool:
        return not (
            self.document_ids
            or self.doc_types
            or self.languages
            or self.years
            or self.chapter_ids
            or self.year_from is not None
            or self.year_to is not None
        )

    def sql(self, alias: str = "f") -> tuple[str, list[Any]]:
        """Умови WHERE поверх `chunk_facets` (без ведучого AND)."""
        clauses: list[str] = []
        params: list[Any] = []

        def _in(column: str, values: Sequence[Any]) -> None:
            if not values:
                return
            marks = ",".join("?" * len(values))
            clauses.append(f"{alias}.{column} IN ({marks})")
            params.extend(values)

        _in("document_id", self.document_ids)
        _in("doc_type", self.doc_types)
        _in("language", self.languages)
        _in("year", self.years)
        _in("chapter_id", self.chapter_ids)
        if self.year_from is not None:
            clauses.append(f"{alias}.year >= ?")
            params.append(self.year_from)
        if self.year_to is not None:
            clauses.append(f"{alias}.year <= ?")
            params.append(self.year_to)
        return " AND ".join(clauses), params

    def as_dict(self) -> dict[str, Any]:
        return {
            "document_ids": list(self.document_ids),
            "doc_types": list(self.doc_types),
            "languages": list(self.languages),
            "years": list(self.years),
            "year_from": self.year_from,
            "year_to": self.year_to,
            "chapter_ids": list(self.chapter_ids),
        }


# ---------------------------------------------------------------- реранкер
@runtime_checkable
class RerankerLike(Protocol):
    """Контракт модуля реранкінгу (`app/rerank/`), відтворений структурно.

    Імпортувати сам модуль тут не можна: контракт забороняє модулям
    імпортувати типи один в одного, і крім того реранкер може бути відсутнім
    (немає ваг, stub-режим, CI без моделей). Тому — Protocol і ін'єкція.
    """

    def score(self, query: str, texts: list[str]) -> list[float]: ...


# ---------------------------------------------- символьна n-грамна гілка
def _char_ngrams(text: str, sizes: Sequence[int] = NGRAM_SIZES) -> list[str]:
    padded = f" {text} "
    out: list[str] = []
    for n in sizes:
        out.extend(padded[i : i + n] for i in range(len(padded) - n + 1))
    return out


def char_ngram_scores(
    query: str,
    documents: Mapping[int, str],
    *,
    top_k: int = 30,
    sizes: Sequence[int] = NGRAM_SIZES,
) -> list[tuple[int, float]]:
    """Символьний 3–5-грамний TF-IDF над пулом кандидатів.

    Чому символьні, а не словесні n-грами: українська морфологія. «Гаубиці»,
    «гаубицями», «гаубиць» — три різні словесні токени й майже той самий набір
    символьних 4-грам. Команда УКУ виміряла +21% відносних саме на цьому.
    Гілка додатково рятує на друкарських помилках і на OCR-шумі, де лема
    просто не побудується.

    Чому на чистому Python, а не scikit-learn: sklearn тягне scipy й ~90 МБ
    у офлайн-інсталятор заради однієї функції, яку тут видно у 20 рядках.
    Довжина нормується по-BM25 (`b=0.75`), а не через справжню косинусну
    норму: точна норма вимагає повного лічильника n-грам кожного документа —
    це найдорожча частина, і вона не окупається на пулі з кількох сотень
    кандидатів.
    """
    if not documents or top_k <= 0:
        return []
    q_norm = query[:NGRAM_QUERY_LIMIT]
    q_grams = sorted(set(_char_ngrams(q_norm, sizes)))
    if not q_grams:
        return []

    ids = list(documents)[:NGRAM_POOL_LIMIT]
    texts = {cid: documents[cid][:NGRAM_TEXT_LIMIT] for cid in ids}
    lengths = {cid: max(1, len(t)) for cid, t in texts.items()}
    avg_len = sum(lengths.values()) / len(lengths)
    n_docs = len(ids)

    counts: dict[str, dict[int, int]] = {}
    for gram in q_grams:
        row = {cid: texts[cid].count(gram) for cid in ids}
        row = {cid: c for cid, c in row.items() if c}
        if row:
            counts[gram] = row

    scores: dict[int, float] = {}
    for row in counts.values():
        idf = math.log((n_docs + 1) / (len(row) + 1)) + 1.0
        for cid, count in row.items():
            tf = 1.0 + math.log(count)
            norm = 1.0 - NGRAM_LENGTH_B + NGRAM_LENGTH_B * lengths[cid] / avg_len
            scores[cid] = scores.get(cid, 0.0) + idf * tf / norm

    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return ranked[:top_k]


# ------------------------------------------------------------------ ретривер
class HybridRetriever:
    """Публічний вхід модуля пошуку.

    Залежності ін'єктуються, а не імпортуються: провайдер ембедингів і
    реранкер живуть у сусідніх модулях, які можуть бути в stub-режимі або
    взагалі відсутні в CI. Пошук зобов'язаний працювати в обох випадках.
    """

    def __init__(
        self,
        db: Any,
        *,
        provider: Any,
        index_dir: Path | str,
        reranker: RerankerLike | None = None,
        params: IndexParams | None = None,
        prefer_usearch: bool = True,
    ) -> None:
        self.db = db
        self.provider = provider
        self.reranker = reranker
        self.index_dir = Path(index_dir)
        self.params = params
        self.prefer_usearch = prefer_usearch
        self._indexes: dict[str, CollectionIndex] = {}
        self._titles: dict[str, str] = {}

    # ------------------------------------------------------------ службове
    def index_for(self, collection_id: str) -> CollectionIndex:
        idx = self._indexes.get(collection_id)
        if idx is None:
            idx = CollectionIndex(
                self.db,
                collection_id,
                index_dir=self.index_dir,
                params=self.params,
                prefer_usearch=self.prefer_usearch,
            )
            self._indexes[collection_id] = idx
        return idx

    def close(self) -> None:
        """Закрити всі mmap-індекси. ОБОВ'ЯЗКОВО перед підміною файлів на Windows."""
        for idx in self._indexes.values():
            idx.close()
        self._indexes.clear()

    def _document_title(self, con: sqlite3.Connection, document_id: str) -> str:
        title = self._titles.get(document_id)
        if title is None:
            title = ChunkRepo(con).document_title(document_id)
            self._titles[document_id] = title
        return title

    # -------------------------------------------------------- фільтр і партиція
    def _resolve_filter(
        self, con: sqlite3.Connection, collection_id: str, flt: MetadataFilter
    ) -> tuple[set[int] | None, float, int]:
        """Повертає (дозволені id або None, селективність, розмір колекції)."""
        total = con.execute(
            "SELECT count(*) FROM chunk_facets WHERE collection_id=?", (collection_id,)
        ).fetchone()[0]
        if flt.is_empty() or not total:
            return None, 1.0, int(total)
        where, params = flt.sql()
        # Псевдонім `f` обов'язковий: `MetadataFilter.sql()` за замовчуванням
        # кваліфікує колонки саме ним, бо в BM25-гілці той самий фільтр
        # застосовується до з'єднання з `chunk_fts`.
        sql = "SELECT f.chunk_id FROM chunk_facets f WHERE f.collection_id=?"
        if where:
            sql += f" AND {where}"
        rows = con.execute(sql, (collection_id, *params)).fetchall()
        allowed = {int(r[0]) for r in rows}
        return allowed, (len(allowed) / total if total else 0.0), int(total)

    # ------------------------------------------------------------- гілки
    def _dense_branch(
        self,
        collection_id: str,
        terms: QueryTerms,
        top_k: int,
        allowed: set[int] | None,
        selectivity: float,
    ) -> tuple[list[tuple[int, float]], str]:
        vectors = self.provider.embed_queries([terms.raw])
        query_vec = np.asarray(vectors, dtype=np.float32).reshape(-1)
        index = self.index_for(collection_id)

        if allowed is None:
            hits = index.search(query_vec, top_k)
            strategy = "L0"
        elif selectivity < EXACT_SELECTIVITY_THRESHOLD:
            hits = index.exact_search(query_vec, top_k, allowed=allowed)
            strategy = "L2"
        else:
            # Overfetch, масштабований за селективністю: щоб після постфільтра
            # лишилось top_k, дістати треба приблизно top_k / селективність.
            overfetch = max(1, min(MAX_OVERFETCH, math.ceil(1.0 / max(selectivity, 1e-6))))
            hits = index.search(query_vec, top_k, allowed=allowed, overfetch=overfetch)
            strategy = "L1"
        return [(h.key, h.score) for h in hits], strategy

    def _sparse_branch(
        self,
        con: sqlite3.Connection,
        collection_id: str,
        terms: QueryTerms,
        top_k: int,
        flt: MetadataFilter,
    ) -> list[tuple[int, float]]:
        expression = build_match_expression(terms.match_terms)
        if not expression:
            return []
        where, params = flt.sql()
        extra = f" AND {where}" if where else ""
        # `-bm25(...)` робить «більше = краще»; ваги колонок — леми/форми/коди.
        sql = (
            "SELECT f.chunk_id AS chunk_id, -bm25(chunk_fts, ?, ?, ?) AS s"
            " FROM chunk_fts JOIN chunk_facets f ON f.chunk_id = chunk_fts.rowid"
            f" WHERE chunk_fts MATCH ? AND f.collection_id = ?{extra}"
            " ORDER BY s DESC LIMIT ?"
        )
        rows = con.execute(
            sql, (*BM25_WEIGHTS, expression, collection_id, *params, top_k)
        ).fetchall()
        scored = {int(r["chunk_id"]): float(r["s"]) for r in rows}

        # Мікро-таблиця позначень: `code_fts` індексується триграмами, тож
        # підрядкові збіги на кшталт «Д-3» усередині «Д-30» знаходяться там,
        # де звичайний токенайзер безсилий. Прозу так індексувати не можна —
        # індекс потроївся б, — а коди можна й треба.
        code_terms = [c for c in terms.codes if len(c) >= 3]
        if code_terms:
            code_expr = build_match_expression(code_terms)
            code_sql = (
                "SELECT f.chunk_id AS chunk_id, -bm25(code_fts) AS s"
                " FROM code_fts JOIN chunk_facets f ON f.chunk_id = code_fts.rowid"
                f" WHERE code_fts MATCH ? AND f.collection_id = ?{extra}"
                " ORDER BY s DESC LIMIT ?"
            )
            for r in con.execute(
                code_sql, (code_expr, collection_id, *params, top_k)
            ).fetchall():
                cid = int(r["chunk_id"])
                scored[cid] = max(scored.get(cid, float("-inf")), float(r["s"]))

        return sorted(scored.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]

    def _ngram_branch(
        self,
        con: sqlite3.Connection,
        pool_ids: Sequence[int],
        terms: QueryTerms,
        top_k: int,
    ) -> list[tuple[int, float]]:
        if not pool_ids:
            return []
        ids = list(pool_ids)[:NGRAM_POOL_LIMIT]
        marks = ",".join("?" * len(ids))
        rows = con.execute(
            f"SELECT id, display_text FROM chunks WHERE id IN ({marks})", tuple(ids)
        ).fetchall()
        # Нормалізація документів мусить бути ТОЮ САМОЮ, що й нормалізація
        # запиту (`terms.normalized`). Інакше виправлення гомогліфів
        # застосоване лише до одного боку — і символьні n-грами перестають
        # збігатися саме на OCR-тексті з латинськими `i`/`c`/`o` в кирилиці,
        # тобто рівно там, де ця гілка й мала рятувати.
        documents = {int(r["id"]): _normalize(r["display_text"] or "") for r in rows}
        return char_ngram_scores(terms.normalized, documents, top_k=top_k)

    # --------------------------------------------------------------- вектори
    @staticmethod
    def _vectors_for(
        con: sqlite3.Connection, chunk_ids: Sequence[int], dim: int
    ) -> dict[int, np.ndarray]:
        """Канонічні вектори кандидатів для косинусної перевірки дедуплікації.

        Прямий SQL, а не `ChunkRepo`: у репозиторії немає доступу до
        `embedding` за списком id (є лише `indexable`, що тягне ВСЮ колекцію),
        а `repositories.py` — спільний фундамент, який цей модуль не редагує.
        """
        if not chunk_ids:
            return {}
        marks = ",".join("?" * len(chunk_ids))
        rows = con.execute(
            f"SELECT id, embedding FROM chunks WHERE id IN ({marks}) AND embedding IS NOT NULL",
            tuple(chunk_ids),
        ).fetchall()
        return {int(r["id"]): blob_to_vector(r["embedding"], dim) for r in rows}

    # ------------------------------------------------------------ auto-merge
    def _automerge(
        self, con: sqlite3.Connection, items: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        """Замінити ≥2 листки спільної секції одним L1-батьком.

        «Retrieve small, show medium»: індексуються лише листки (L2), бо на
        них ембединг гострий, але коли до відповіді потрапили два сусідні
        листки однієї секції, показувати моделі шматки з діркою посередині
        гірше, ніж показати секцію цілком. Один фрагмент так НЕ розширюється:
        це роздуло б контекст удвічі без приросту.
        """
        by_parent: dict[int, list[RetrievedChunk]] = {}
        for item in items:
            pid = item.chunk.parent_id
            if pid is not None:
                by_parent.setdefault(pid, []).append(item)

        merged: list[RetrievedChunk] = []
        replaced: set[int] = set()
        chunks = ChunkRepo(con)
        for parent_id, group in by_parent.items():
            if len(group) < 2:
                continue
            parent = chunks.get(parent_id)
            if parent is None or parent.level is not ChunkLevel.SECTION:
                continue
            best = max(group, key=lambda r: r.score)
            merged.append(
                RetrievedChunk(
                    chunk=parent,
                    document_title=best.document_title,
                    dense_score=best.dense_score,
                    sparse_score=best.sparse_score,
                    ngram_score=best.ngram_score,
                    fused_score=best.fused_score,
                    rerank_score=best.rerank_score,
                )
            )
            replaced.update(id(r) for r in group)

        out = [r for r in items if id(r) not in replaced] + merged
        out.sort(key=lambda r: -r.score)
        return out

    # -------------------------------------------------------------- пакування
    @staticmethod
    def _pack(items: Sequence[RetrievedChunk], char_budget: int) -> list[RetrievedChunk]:
        """Обрізати під бюджет контексту ЗА СКОРОМ, до впорядкування.

        Порядок кроків тут навмисно відрізняється від схеми §7, і це важливо:
        U-подібне розміщення ставить найслабші блоки в СЕРЕДИНУ, а найкращий
        другий блок — у КІНЕЦЬ. Обрізати хвіст уже впорядкованого списку
        означало б викинути друге за якістю джерело й лишити найслабші.
        Тому пакуємо за скором, а U-форму будуємо з того, що влізло.
        """
        packed: list[RetrievedChunk] = []
        used = 0
        for item in items:
            size = len(item.chunk.display_text)
            if packed and used + size > char_budget:
                continue
            packed.append(item)
            used += size
        return packed

    # ---------------------------------------------------------------- головне
    def retrieve(
        self,
        query: str,
        config: AssistantConfig,
        collection_id: str,
        *,
        filters: MetadataFilter | None = None,
        char_budget: int | None = None,
        language: str = "uk",
    ) -> tuple[list[RetrievedChunk], RetrievalDebug]:
        """Повний конвеєр. Повертає фінальні фрагменти й діагностику.

        Фрагменти повертаються НАВІТЬ при спрацюванні утримання: екран «Чому
        ця відповідь» має показати викладачеві, що саме система знайшла і чому
        визнала це недостатнім. Рішення «відповідати чи ні» приймає модуль
        генерації за прапорцем `debug.abstained` — порогом У КОДІ, а не
        проханням у промпті (контракт, правило 8).
        """
        flt = filters or MetadataFilter()
        budget = char_budget if char_budget is not None else DEFAULT_CONTEXT_CHAR_BUDGET
        debug = RetrievalDebug(query=query)
        timings: dict[str, float] = {}
        clock = time.perf_counter

        t0 = clock()
        terms = analyze_query(query, language=language)
        timings["analyze"] = (clock() - t0) * 1000.0

        with self.db.connection() as con:
            collection = CollectionRepo(con).get(collection_id)
            if collection is None:
                raise KeyError(f"Колекцію {collection_id} не знайдено.")

            t0 = clock()
            allowed, selectivity, total = self._resolve_filter(con, collection_id, flt)
            timings["filter"] = (clock() - t0) * 1000.0

            # Фільтр не залишив нічого — це не помилка, це порожній результат.
            if allowed is not None and not allowed:
                debug.latency_ms = timings
                debug.abstained = True
                debug.abstain_confidence = 0.0
                debug.candidates = []
                return [], debug

            t0 = clock()
            dense, strategy = self._dense_branch(
                collection_id, terms, config.dense_top_k, allowed, selectivity
            )
            timings["dense"] = (clock() - t0) * 1000.0

            t0 = clock()
            sparse = self._sparse_branch(con, collection_id, terms, config.sparse_top_k, flt)
            timings["sparse"] = (clock() - t0) * 1000.0

            # Пул n-грамної гілки — кандидати перших двох гілок. Якщо
            # відфільтрована множина й так невелика, беремо її цілком: тоді
            # гілка може ЗНАЙТИ фрагмент, який dense і BM25 пропустили, а не
            # лише переставити знайдене.
            pool = [cid for cid, _ in dense] + [cid for cid, _ in sparse]
            if allowed is not None and len(allowed) <= NGRAM_POOL_LIMIT:
                pool = sorted(allowed)
            seen_pool: set[int] = set()
            pool = [c for c in pool if not (c in seen_pool or seen_pool.add(c))]

            t0 = clock()
            ngram = self._ngram_branch(con, pool, terms, max(1, config.dense_top_k // 2))
            timings["ngram"] = (clock() - t0) * 1000.0

            weights = fusion.resolve_weights(
                query,
                weight_dense=config.weight_dense,
                weight_sparse=config.weight_sparse,
                weight_ngram=config.weight_char_ngram,
                rrf_k=config.rrf_k,
            )
            t0 = clock()
            fused = fusion.weighted_rrf(
                dense=dense, sparse=sparse, ngram=ngram, weights=weights
            )
            timings["fuse"] = (clock() - t0) * 1000.0

            debug.dense_count = len(dense)
            debug.sparse_count = len(sparse)
            debug.ngram_count = len(ngram)
            debug.fused_count = len(fused)

            if not fused:
                debug.latency_ms = timings
                debug.abstained = True
                debug.abstain_confidence = 0.0
                return [], debug

            # Матеріалізуємо лише те, що піде в реранкер.
            head = fused[: max(config.rerank_top_k, config.final_top_k)]
            ids = [c.chunk_id for c in head]
            chunk_map = {c.id: c for c in ChunkRepo(con).by_ids(ids) if c.id is not None}
            candidates: list[RetrievedChunk] = []
            by_id: dict[int, fusion.FusedCandidate] = {c.chunk_id: c for c in head}
            for cid in ids:
                chunk = chunk_map.get(cid)
                if chunk is None:
                    continue  # чанк видалено між пошуком і читанням
                cand = by_id[cid]
                candidates.append(
                    RetrievedChunk(
                        chunk=chunk,
                        document_title=self._document_title(con, chunk.document_id),
                        dense_score=cand.dense_score,
                        sparse_score=cand.sparse_score,
                        ngram_score=cand.ngram_score,
                        fused_score=cand.rrf_score,
                    )
                )

            t0 = clock()
            vectors = self._vectors_for(con, [c.chunk.id or 0 for c in candidates], collection.dim)
            deduped, dropped = diversity.deduplicate(candidates, vectors=vectors)
            timings["dedup"] = (clock() - t0) * 1000.0

            t0 = clock()
            self._rerank(query, deduped)
            timings["rerank"] = (clock() - t0) * 1000.0
            debug.reranked_count = len(deduped)

            # Впевненість і лічильник опорних фрагментів МУСЯТЬ жити на одній
            # шкалі. Порівнювати `RetrievedChunk.score` (тобто RRF, коли
            # реранкера немає) з порогом реранкера — це гарантоване утримання
            # завжди: RRF-скор першого місця — це ~1/61 ≈ 0.016, тобто нижче
            # будь-якого з порогів 0.15 / 0.35 / 0.55.
            support = self._confidence_scores(deduped)
            confidence = max(support, default=0.0)
            threshold = config.confidence_threshold()
            supporting = sum(1 for s in support if s >= threshold)
            abstained = (
                confidence < threshold or supporting < config.min_supporting_chunks()
            )

            t0 = clock()
            selected = diversity.diversify(
                deduped,
                final_k=config.final_top_k,
                max_per_document=config.max_per_document,
                min_distinct_documents=config.min_distinct_documents,
                relative_score_floor=config.relative_score_floor,
            )
            selected = self._automerge(con, selected)
            selected = self._pack(selected, budget)
            timings["select"] = (clock() - t0) * 1000.0

            final = reorder.reorder(selected)

            debug.final_count = len(final)
            debug.distinct_documents = diversity.distinct_documents(final)
            debug.abstained = abstained
            debug.abstain_confidence = round(float(confidence), 6)
            debug.latency_ms = {k: round(v, 3) for k, v in timings.items()}
            debug.candidates = self._debug_candidates(
                head, deduped, final, dropped, weights, strategy, selectivity, total
            )
            return final, debug

    # -------------------------------------------------------------- реранкінг
    def _rerank(self, query: str, items: Sequence[RetrievedChunk]) -> None:
        """Проставити `rerank_score` кожному кандидату. Без реранкера — нічого.

        Реранкер ін'єктується, а не імпортується: у stub-режимі й у CI його
        може не бути зовсім, і пошук зобов'язаний працювати без нього.
        """
        if not items or self.reranker is None:
            return
        texts = [r.chunk.rerank_text or r.chunk.display_text for r in items]
        scores = list(self.reranker.score(query, texts))
        for item, score in zip(items, scores, strict=False):
            item.rerank_score = float(score)

    @staticmethod
    def _confidence_scores(items: Sequence[RetrievedChunk]) -> list[float]:
        """Скори кандидатів на шкалі, придатній для порога утримання.

        Пороги `AssistantConfig.confidence_threshold()` (0.15/0.35/0.55)
        відкалібровані під скор реранкера в [0, 1]. Коли реранкера немає,
        єдина ЗІСТАВНА величина — сирий косинус dense-гілки: він теж живе
        приблизно в [0, 1] для нормованих векторів. RRF-скор сюди не годиться
        за побудовою — він несе лише ранг, а не абсолютну релевантність
        (див. `fusion.py`), тому й переноситься окремим полем.

        Це тимчасовий замінник механізму `u_lin` (план, §7): гребенева регресія
        на відсортованому векторі скорів реранкера, відкалібрована на ~50
        золотих питаннях колекції. Константний поріг гірший, бо шкала
        крос-енкодера дрейфує між мовами — але він працює, а відсутність
        будь-якого порога не працює взагалі.
        """
        out: list[float] = []
        for item in items:
            if item.rerank_score is not None:
                out.append(float(item.rerank_score))
            elif item.dense_score is not None:
                out.append(float(item.dense_score))
        return out

    # ---------------------------------------------------------------- діагностика
    @staticmethod
    def _debug_candidates(
        fused: Sequence[fusion.FusedCandidate],
        reranked: Sequence[RetrievedChunk],
        final: Sequence[RetrievedChunk],
        dropped: Sequence[diversity.DroppedDuplicate],
        weights: fusion.FusionWeights,
        strategy: str,
        selectivity: float,
        total: int,
    ) -> list[dict[str, Any]]:
        """Рядки для екрана «Чому ця відповідь».

        Перший рядок — не кандидат, а параметри прогону: без них викладач
        бачить список фрагментів без жодного пояснення, чому саме вони.
        """
        final_ids = {r.chunk.id for r in final}
        # Після auto-merge у відповіді стоїть L1-батько, а не листки. Якби
        # «обраним» вважався лише збіг за id, екран «Чому ця відповідь» показав
        # би, що ЖОДЕН знайдений фрагмент не потрапив у відповідь — рівно на
        # тих запитах, де конвеєр спрацював найкраще.
        merged_parents = {
            r.chunk.id for r in final if r.chunk.level is ChunkLevel.SECTION
        }
        kept_ids = {r.chunk.id for r in reranked}
        rows: list[dict[str, Any]] = [
            {
                "kind": "run",
                "weights": weights.as_dict(),
                "filter_strategy": strategy,
                "selectivity": round(selectivity, 4),
                "collection_chunks": total,
                "duplicates_dropped": [d.as_dict() for d in dropped],
            }
        ]
        by_id = {r.chunk.id: r for r in reranked}
        for cand in fused:
            item = by_id.get(cand.chunk_id)
            row: dict[str, Any] = {"kind": "candidate", **cand.as_dict()}
            if item is not None:
                merged_into = (
                    item.chunk.parent_id if item.chunk.parent_id in merged_parents else None
                )
                row.update(
                    {
                        "chunk_uid": item.chunk.chunk_uid,
                        "document_title": item.document_title,
                        "pages": item.chunk.citation_label(),
                        "rerank": item.rerank_score,
                        "selected": item.chunk.id in final_ids or merged_into is not None,
                        "merged_into": merged_into,
                        "ordinal": item.ordinal_in_prompt,
                    }
                )
            else:
                row["selected"] = False
                row["dropped"] = "duplicate" if cand.chunk_id not in kept_ids else "truncated"
            rows.append(row)
        return rows


def build_retrieved(chunk: Chunk, title: str = "") -> RetrievedChunk:
    """Дрібний помічник для тестів і скриптів оцінювання."""
    return RetrievedChunk(chunk=chunk, document_title=title)
