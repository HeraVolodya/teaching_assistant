"""Крос-енкодерний реранкінг.

Найважливіший висновок дослідження (план, §6), який мусить лишитися в коді:
**попередні виміри автора 12-38 с на реранкінг були виною РАНТАЙМУ, а не
моделей.** FLOP-модель відтворює всі три ті виміри з точністю ~10% і показує
три множники, що перемножувалися:

  1. **CPU + fp32.** 0.26 TFLOP/s ефективних — це рівно те, що ноутбучний CPU
     дає на fp32 GEMM. Лікується квантованим ONNX (int8 / graph-опт O3) на CPU
     і float16 на GPU.
  2. **Паддінг КОЖНОЇ пари до `max_length`.** Пара «запит + чанк на 300
     токенів», допаднута до 8192, коштує у 27 разів більше за саму себе.
     Лікується динамічним паддінгом із бакетуванням за довжиною — тим самим
     прийомом, що й в ембедері.
  3. **Одна пара — один прохід.** Лікується батчуванням за бюджетом токенів.

Тобто «реранкер занадто повільний» — хибний висновок; правильний висновок —
«реранкер запускали найгіршим із можливих способів».

**FlashAttention для крос-енкодерів НЕ вмикати.** Для голів
sequence-classification він на 20-45% ПОВІЛЬНІШИЙ (керівництво супровідників
Sentence Transformers, переміряно 07/2026). Це заодно знімає з порядку денного
проблему відсутнього Windows-колеса `flash-attn`, на яку автор уже наступав.

**LM Studio реранкер хостити НЕ МОЖЕ.** Немає ні `/v1/rerank`, ні
`/api/v0/rerank` (запити на фічу: lms#167, lms#521, docs#162, bug-tracker#470),
а `logprobs` заглушено і повертає NULL — тобто pointwise yes/no-реранкінг за
логітами через LM Studio неможливий у принципі. Запасний варіант — покласти
поруч один бінарник `llama-server` (`--reranking`, `/v1/rerank`), а НЕ
`llama-cpp-python` (матриця коліс CUDA/Metal/CPU на 300-800 МБ).

Дефолт — `onnx-community/Qwen3-Reranker-0.6B-ONNX` (`model_quantized.onnx`,
Apache-2.0): єдиний варіант із ПРЯМИМ українським виміром (UNLP 2026,
фіксований ретривер: public 0.9099 → 0.9172, private 0.9243 → 0.9395).

УВАГА щодо решти записів реєстру. Станом на перевірку 09/2026 у їхніх
репозиторіях ONNX-файлів НЕМАЄ взагалі: `Alibaba-NLP/gte-multilingual-reranker-base`
і `BAAI/bge-reranker-v2-m3` несуть лише ваги PyTorch, а
`onnx-community/Qwen3-Reranker-4B-ONNX` не віддає жодного файлу. Вони лишаються
як намір (їх треба експортувати самотужки через `optimum`), але виставити їх
через ASISTENT_RERANK_MODEL зараз означає впасти на завантаженні. У постачанні
з `fetch_models.py` їх немає, тому на дефолтному шляху це не спрацьовує.
Уся родина `cross-encoder/ms-marco-*` виключена: підтверджений колапс на
українській (власні виміри автора — ранги 11-20).
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

__all__ = [
    "DEFAULT_MODEL_ID",
    "MODEL_ENV",
    "REGISTRY",
    "STUB_ENV",
    "OnnxReranker",
    "RerankModel",
    "RerankScoring",
    "Reranker",
    "StubReranker",
    "build_qwen_pair",
    "create_reranker",
    "get",
    "model_key",
    "plan_pair_batches",
]

log = logging.getLogger(__name__)

STUB_ENV = "ASISTENT_STUB"
MODEL_ENV = "ASISTENT_RERANK_MODEL"
MODEL_DIR_ENV = "ASISTENT_RERANK_MODEL_DIR"
TOKEN_BUDGET_ENV = "ASISTENT_RERANK_TOKEN_BUDGET"
MAX_TOKENS_ENV = "ASISTENT_RERANK_MAX_TOKENS"
PROVIDERS_ENV = "ASISTENT_RERANK_PROVIDERS"
DEVICE_ENV = "ASISTENT_RERANK_DEVICE"

# Пара «запит + чанк». Жорсткий максимум чанка — 3600 символів ≈ 1900 токенів
# Qwen, плюс запит і обгортка інструкції. 2048 покриває це з запасом і не дає
# одній аномальній парі роздути паддінг усього батча.
DEFAULT_MAX_PAIR_TOKENS = 2048
# Бюджет батча в токенах. Менший за ембедерний, бо крос-енкодер робить повну
# увагу по всій парі: вартість росте квадратично за довжиною, а не лінійно.
DEFAULT_TOKEN_BUDGET = 8192
MAX_BATCH_ROWS = 32

# Українська інструкція реранкера. Той самий текст, що і в ембедері, лишається
# у registry ембедингів; тут він свій, бо крос-енкодер бачить інший формат.
UK_RERANK_TASK = (
    "Given a question in Ukrainian about military or technical teaching materials, "
    "judge whether the passage from the textbook answers it"
)


class RerankScoring(str, Enum):
    """Як із виходу мережі дістається скор."""

    YES_NO_LOGITS = "yes_no_logits"                    # причинна LM: softmax по логітах yes/no
    SEQUENCE_CLASSIFICATION = "sequence_classification"  # голова класифікації: sigmoid/softmax


@dataclass(frozen=True, slots=True)
class RerankModel:
    id: str
    revision: str
    max_seq: int
    scoring: RerankScoring
    licence: str
    onnx_file: str
    instruction: str = UK_RERANK_TASK
    notes: str = ""
    aliases: tuple[str, ...] = field(default_factory=tuple)


REGISTRY: dict[str, RerankModel] = {
    "qwen3-reranker-0.6b": RerankModel(
        id="onnx-community/Qwen3-Reranker-0.6B-ONNX",
        revision="main",
        max_seq=8192,
        scoring=RerankScoring.YES_NO_LOGITS,
        licence="Apache-2.0",
        # Ім'я БЕЗ префікса `onnx/`: fetch_models.py кладе файл пласко, за
        # basename. І саме `model_quantized`, а не `model_int8` — останнього в
        # репозиторії немає взагалі, там лише model_quantized.onnx і model_q4.onnx.
        onnx_file="model_quantized.onnx",
        notes="Дефолт. Єдиний реранкер із прямим українським виміром (UNLP 2026: "
              "0.9099 → 0.9172 public). MMTEB-R 66.36.",
        aliases=("qwen3-reranker", "default"),
    ),
    "gte-multilingual-reranker-base": RerankModel(
        id="Alibaba-NLP/gte-multilingual-reranker-base",
        revision="main",
        max_seq=8192,
        scoring=RerankScoring.SEQUENCE_CLASSIFICATION,
        licence="Apache-2.0",
        onnx_file="onnx/model_O3.onnx",
        notes="Альтернатива для A/B першого дня: 306M, вікно 8192, власний вимір автора "
              "(UA 2,6 / ENG 2,1). MMTEB-R 59.44.",
        aliases=("gte-multilingual", "gte", "alt"),
    ),
    "qwen3-reranker-4b": RerankModel(
        id="onnx-community/Qwen3-Reranker-4B-ONNX",
        revision="main",
        max_seq=8192,
        scoring=RerankScoring.YES_NO_LOGITS,
        licence="Apache-2.0",
        onnx_file="onnx/model_int8.onnx",
        notes="Рівень A: лише при GPU >= 24 ГБ. UNLP 2026: 0.9429 public / 0.9562 private.",
        aliases=("qwen3-reranker-4b", "best"),
    ),
    "bge-reranker-v2-m3": RerankModel(
        id="BAAI/bge-reranker-v2-m3",
        revision="main",
        max_seq=8192,
        scoring=RerankScoring.SEQUENCE_CLASSIFICATION,
        licence="Apache-2.0",
        onnx_file="onnx/model.onnx",
        notes="НЕ дефолт: найгірший мультимовний реранкер набору на MMTEB-R (58.36), "
              "власний вимір автора — ранги UA 7,3. Саме на ньому видно крос-мовний "
              "дрейф шкали (англійська пара 0.9953 проти китайської 0.2093), через який "
              "константний поріг непридатний — див. abstention.py. Тримається як база "
              "для файн-тюнінгу: єдиний опублікований український реранкер-файнтюн, що "
              "взяв медаль, зроблено на ньому.",
        aliases=("bge-reranker",),
    ),
}

DEFAULT_MODEL_ID = "qwen3-reranker-0.6b"

_ALIASES: dict[str, str] = {
    alias: key for key, m in REGISTRY.items() for alias in (*m.aliases, key)
}


def get(name: str | None = None) -> RerankModel:
    key = _ALIASES.get((name or os.environ.get(MODEL_ENV) or DEFAULT_MODEL_ID).lower())
    if key is None:
        known = ", ".join(sorted(REGISTRY))
        raise KeyError(f"Невідома модель реранкера {name!r}. Відомі: {known}")
    return REGISTRY[key]


def model_key(model: RerankModel) -> str:
    """Відбиток реранкера.

    Потрібен НЕ для кешу векторів, а для калібрування утримання: скори різних
    реранкерів живуть у різних шкалах, і калібрування, зняте на одному, на
    іншому означає інше. Зміна цього ключа мусить інвалідувати калібрування
    (див. `abstention.CalibrationStore`).
    """
    payload = "|".join((model.id, model.revision, model.scoring.value, model.onnx_file, model.instruction))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


@runtime_checkable
class Reranker(Protocol):
    """Контракт, проти якого пишуть пошук і утримання від відповіді."""

    def score(self, query: str, docs: list[str]) -> list[float]:
        """Скор релевантності в [0, 1] для кожного документа, у порядку входу."""
        ...

    @property
    def model_key(self) -> str:
        ...


# ------------------------------------------------------------- шаблон Qwen3
# Побайтово з картки моделі Qwen3-Reranker. Не «причісувати»: порожній блок
# <think> обов'язковий — без нього модель у reasoning-режимі витрачає позицію
# останнього токена не на відповідь, і логіти yes/no читаються не там.
QWEN_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query "
    'and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
    "<|im_start|>user\n"
)
QWEN_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def build_qwen_pair(query: str, doc: str, instruction: str = UK_RERANK_TASK) -> str:
    """Повний рядок однієї пари у форматі Qwen3-Reranker."""
    return f"{QWEN_PREFIX}<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}{QWEN_SUFFIX}"


# ---------------------------------------------------------------- батчування
def plan_pair_batches(
    lengths: list[int],
    *,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
    max_rows: int = MAX_BATCH_ROWS,
) -> list[list[int]]:
    """Бакетування пар за довжиною — головне лікування «12-38 секунд».

    Пари сортуються за довжиною, батч ріжеться за `рядків × найдовший рядок`.
    Саме допаднення кожної пари до `max_length` і давало множник у 10-27 разів
    у попередніх вимірах.
    """
    order = sorted(range(len(lengths)), key=lambda i: (lengths[i], i))
    batches: list[list[int]] = []
    current: list[int] = []
    current_max = 0
    for idx in order:
        length = max(1, lengths[idx])
        new_max = max(current_max, length)
        if current and ((len(current) + 1) * new_max > token_budget or len(current) + 1 > max_rows):
            batches.append(current)
            current, current_max = [idx], length
        else:
            current.append(idx)
            current_max = new_max
    if current:
        batches.append(current)
    return batches


# ------------------------------------------------------------------ заглушка
_WORD_RE = re.compile(r"\w+", re.UNICODE)
_APOSTROPHES = str.maketrans({"’": "'", "ʼ": "'", "`": "'", "´": "'", "′": "'"})


def _normalize(text: str) -> str:
    """Мінімальна нормалізація, свідомо незалежна від `app.ingestion`."""
    return " ".join(unicodedata.normalize("NFC", text).translate(_APOSTROPHES).casefold().split())


def _tokens(text: str) -> list[str]:
    return _WORD_RE.findall(_normalize(text))


def _char_grams(text: str, n: int = 3) -> set[str]:
    padded = f"  {_normalize(text)} "
    return {padded[i:i + n] for i in range(max(0, len(padded) - n + 1))}


class StubReranker:
    """Детермінований реранкер без моделі (`ASISTENT_STUB=1`).

    Скор — лексичне перекриття запиту й документа. Не «майже випадкове число»:
    він мусить давати ПРАВИЛЬНИЙ порядок на очевидних прикладах, інакше тести
    пошуку, утримання й генерації перевіряли б нуль. Дві складові:
      * покриття словоформ запиту (точний сигнал на позначеннях `Д-30`, `2С1`);
      * покриття символьних 3-грам (морфологічна стійкість української, де
        точний збіг словоформ рідкісний: «гаубиці» проти «гаубиця»).
    """

    def __init__(self, model: RerankModel, *, instruction: str = UK_RERANK_TASK) -> None:
        self._model = model
        self._instruction = instruction

    @property
    def model(self) -> RerankModel:
        return self._model

    @property
    def model_key(self) -> str:
        return f"stub:{model_key(self._model)}"

    def score(self, query: str, docs: list[str]) -> list[float]:
        if not docs:
            return []
        q_tokens = set(_tokens(query))
        q_grams = _char_grams(query)
        out: list[float] = []
        for doc in docs:
            d_tokens = set(_tokens(doc))
            d_grams = _char_grams(doc)
            word_cover = len(q_tokens & d_tokens) / len(q_tokens) if q_tokens else 0.0
            gram_cover = len(q_grams & d_grams) / len(q_grams) if q_grams else 0.0
            base = 0.6 * word_cover + 0.4 * gram_cover
            # Детермінований мікрозсув розводить нічиї стабільно між запусками:
            # реранкер, що повертає однакові скори, робить порядок залежним від
            # порядку кандидатів, а той залежить від фьюжну — і тести стають
            # крихкими без жодної реальної причини.
            digest = hashlib.blake2b(f"{query}\x00{doc}".encode(), digest_size=4).digest()
            jitter = int.from_bytes(digest, "little") / 2**32 * 1e-4
            out.append(float(min(1.0, max(0.0, base + jitter))))
        return out


# ---------------------------------------------------------------- ONNX-версія
class OnnxReranker:
    """Крос-енкодер на прямій сесії ONNX Runtime (без torch)."""

    def __init__(
        self,
        model: RerankModel,
        *,
        model_dir: Path,
        instruction: str | None = None,
        max_pair_tokens: int | None = None,
        token_budget: int | None = None,
        session: Any | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        self._model = model
        self._dir = Path(model_dir)
        self._instruction = instruction or model.instruction

        env_max = os.environ.get(MAX_TOKENS_ENV, "").strip()
        env_budget = os.environ.get(TOKEN_BUDGET_ENV, "").strip()
        requested = max_pair_tokens or (int(env_max) if env_max.isdigit() else DEFAULT_MAX_PAIR_TOKENS)
        self._max_pair_tokens = max(16, min(requested, model.max_seq - 8))
        self._token_budget = token_budget or (
            int(env_budget) if env_budget.isdigit() else DEFAULT_TOKEN_BUDGET
        )

        self._tok = tokenizer if tokenizer is not None else self._load_tokenizer()
        self._pad_id = self._resolve_pad()
        self._yes_id, self._no_id = self._resolve_yes_no()
        self._session = session if session is not None else self._load_session()
        self._inputs = {i.name: i for i in self._session.get_inputs()}

        # Те саме, що й в ембедері (onnx_backend.py): Qwen3-Reranker експортовано
        # як ДЕКОДЕР, тож `_feeds` подає порожній KV-кеш `(batch, heads, 0, head_dim)`.
        # CoreML EP не приймає тензорів нульової довжини і повідомляє про це прямо:
        # «has a dynamic shape ({-1,8,-1,128}) but the runtime shape ({3,8,0,128})
        # has zero elements». Помилка вилазить не при складанні сесії, а аж на
        # першому `score()` — тобто вже посеред відповіді користувачу.
        # Наявність KV визначається з САМОЇ моделі, а не з її назви: наступний
        # експорт може називатися так само й не мати кешу.
        if (
            session is None
            and any(n.startswith("past_key_values") for n in self._inputs)
            and any("CoreML" in p for p in self._session.get_providers())
        ):
            log.info(
                "Реранкер %s має KV-кеш — виключаю CoreML EP (він не приймає "
                "тензори нульової довжини) і перескладаю сесію.",
                self._model.id,
            )
            self._session = self._load_session(exclude={"CoreMLExecutionProvider"})
            self._inputs = {i.name: i for i in self._session.get_inputs()}

        self._output_names = [o.name for o in self._session.get_outputs()]

    # ------------------------------------------------------------ завантаження
    def _load_tokenizer(self) -> Any:
        from tokenizers import Tokenizer

        path = self._dir / "tokenizer.json"
        if not path.is_file():
            raise FileNotFoundError(
                f"Не знайдено tokenizer.json для реранкера {self._model.id} у {self._dir}. "
                "Асістент не завантажує моделі з мережі: скопіюйте файли з інсталяційного "
                "носія або запустіть у режимі заглушки (ASISTENT_STUB=1)."
            )
        tok = Tokenizer.from_file(str(path))
        tok.no_truncation()
        tok.no_padding()
        return tok

    def _load_session(self, exclude: set[str] | None = None) -> Any:
        import onnxruntime as ort

        path = self._dir / self._model.onnx_file
        if not path.is_file():
            raise FileNotFoundError(
                f"Не знайдено файл моделі {self._model.onnx_file} для реранкера "
                f"{self._model.id} у {self._dir}."
            )
        options = ort.SessionOptions()
        # O3/int8 на CPU — саме той множник, якого бракувало в попередніх вимірах.
        # FlashAttention для крос-енкодерів НЕ вмикаємо свідомо (див. докстрінг модуля).
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = max(1, (os.cpu_count() or 2) - 1)
        options.inter_op_num_threads = 1

        # Логіка вибору EP спільна з ембедером, змінні середовища — власні:
        # ембедер і реранкер свідомо можуть жити на різних пристроях (на 8-12 ГБ
        # VRAM, зайнятих LM Studio, це типова конфігурація).
        from app.embeddings.onnx_backend import select_providers

        providers = select_providers(
            list(ort.get_available_providers()),
            device=os.environ.get(DEVICE_ENV, "auto"),
            explicit=os.environ.get(PROVIDERS_ENV, ""),
        )
        if exclude:
            providers = [p for p in providers if p not in exclude] or ["CPUExecutionProvider"]

        try:
            return ort.InferenceSession(str(path), sess_options=options, providers=providers)
        except Exception as exc:      # pragma: no cover — залежить від машини
            if providers == ["CPUExecutionProvider"]:
                raise
            log.warning("ORT-реранкер не стартував із %s (%s); переходжу на CPU.", providers, exc)
            return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])

    def _resolve_pad(self) -> int:
        for token in ("<pad>", "[PAD]", "<|endoftext|>", "</s>"):
            tid = self._tok.token_to_id(token)
            if tid is not None:
                return int(tid)
        return 0

    def _resolve_yes_no(self) -> tuple[int | None, int | None]:
        if self._model.scoring is not RerankScoring.YES_NO_LOGITS:
            return None, None
        yes = self._tok.encode("yes", add_special_tokens=False).ids
        no = self._tok.encode("no", add_special_tokens=False).ids
        if not yes or not no:
            raise RuntimeError(
                f"Токенайзер реранкера {self._model.id} не кодує слова 'yes'/'no' — "
                "pointwise-скоринг за логітами неможливий."
            )
        return int(yes[0]), int(no[0])

    # ------------------------------------------------------------ властивості
    @property
    def model(self) -> RerankModel:
        return self._model

    @property
    def model_key(self) -> str:
        return model_key(self._model)

    @property
    def providers(self) -> list[str]:
        return list(self._session.get_providers())

    # ------------------------------------------------------------- токенізація
    def _encode_pair(self, query: str, doc: str) -> list[int]:
        if self._model.scoring is RerankScoring.YES_NO_LOGITS:
            ids = list(self._tok.encode(build_qwen_pair(query, doc, self._instruction),
                                        add_special_tokens=False).ids)
        else:
            ids = list(self._tok.encode(query, doc, add_special_tokens=True).ids)
        if len(ids) <= self._max_pair_tokens:
            return ids
        # Ріжемо СЕРЕДИНУ (тіло документа), а не хвіст: у Qwen-форматі хвіст —
        # це обгортка assistant/<think>, без якої логіти читаються не з тієї
        # позиції, а в парному форматі хвіст несе [SEP].
        head = self._max_pair_tokens // 2
        tail = self._max_pair_tokens - head
        return ids[:head] + ids[-tail:]

    # ------------------------------------------------------------------ скор
    def _feeds(self, rows: list[list[int]]) -> tuple[dict[str, np.ndarray], np.ndarray]:
        width = max(len(r) for r in rows)
        ids = np.full((len(rows), width), self._pad_id, dtype=np.int64)
        mask = np.zeros((len(rows), width), dtype=np.int64)
        for i, row in enumerate(rows):
            ids[i, : len(row)] = row
            mask[i, : len(row)] = 1

        feeds: dict[str, np.ndarray] = {}
        if "input_ids" in self._inputs:
            feeds["input_ids"] = ids
        if "attention_mask" in self._inputs:
            feeds["attention_mask"] = mask
        if "token_type_ids" in self._inputs:
            feeds["token_type_ids"] = np.zeros_like(ids)
        if "position_ids" in self._inputs:
            feeds["position_ids"] = np.clip(np.cumsum(mask, axis=1) - 1, 0, None).astype(np.int64)
        if "use_cache_branch" in self._inputs:
            feeds["use_cache_branch"] = np.array([False], dtype=bool)

        # Декодерні експорти вимагають порожній KV-кеш на першому проході.
        # Форма: [batch, kv_heads, 0, head_dim] — динамічні виміри беремо з
        # батча й нуля, статичні читаємо просто з графа.
        for name, spec in self._inputs.items():
            if not name.startswith("past_key_values"):
                continue
            shape = []
            for axis, dim in enumerate(spec.shape):
                if isinstance(dim, int):
                    shape.append(dim)
                elif axis == 0:
                    shape.append(len(rows))
                else:
                    shape.append(0)
            feeds[name] = np.zeros(tuple(shape), dtype=np.float32)
        return feeds, mask

    def _scores_from_outputs(self, outputs: list[np.ndarray], mask: np.ndarray) -> np.ndarray:
        by_name = dict(zip(self._output_names, outputs, strict=False))
        logits = by_name.get("logits")
        if logits is None:
            logits = next((np.asarray(o) for o in outputs if np.asarray(o).ndim in (2, 3)), None)
        if logits is None:      # pragma: no cover — нетиповий експорт
            raise RuntimeError(f"Реранкер {self._model.id} не повернув логітів: {self._output_names}.")
        logits = np.asarray(logits, dtype=np.float32)

        if self._model.scoring is RerankScoring.YES_NO_LOGITS:
            last = np.clip(mask.sum(axis=1) - 1, 0, logits.shape[1] - 1).astype(np.int64)
            row = logits[np.arange(logits.shape[0]), last, :]
            pair = np.stack([row[:, self._no_id], row[:, self._yes_id]], axis=1)
            pair = pair - pair.max(axis=1, keepdims=True)
            exp = np.exp(pair)
            return exp[:, 1] / exp.sum(axis=1)

        if logits.ndim == 3:    # pragma: no cover — нетиповий експорт
            logits = logits[:, 0, :]
        if logits.shape[1] == 1:
            return 1.0 / (1.0 + np.exp(-logits[:, 0]))
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp[:, -1] / exp.sum(axis=1)

    def score(self, query: str, docs: list[str]) -> list[float]:
        if not docs:
            return []
        encoded = [self._encode_pair(query, d) for d in docs]
        out = np.zeros(len(docs), dtype=np.float64)
        for batch in plan_pair_batches(
            [len(e) for e in encoded], token_budget=self._token_budget, max_rows=MAX_BATCH_ROWS
        ):
            feeds, mask = self._feeds([encoded[i] for i in batch])
            raw = self._session.run(None, feeds)
            scores = self._scores_from_outputs(list(raw), mask)
            for position, source_index in enumerate(batch):
                out[source_index] = float(scores[position])
        return [float(x) for x in out]

    def close(self) -> None:
        self._session = None  # type: ignore[assignment]


# ------------------------------------------------------------------- фабрика
def _stub_enabled() -> bool:
    return os.environ.get(STUB_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def default_model_dir(model: RerankModel, models_dir: Path | None = None) -> Path:
    override = os.environ.get(MODEL_DIR_ENV, "").strip()
    if override:
        return Path(override).expanduser()
    if models_dir is None:
        from app.config import Paths

        models_dir = Paths.resolve().models_dir
    key = next((k for k, m in REGISTRY.items() if m is model), model.id.replace("/", "--"))
    return Path(models_dir) / "rerank" / key


def create_reranker(
    model_name: str | None = None,
    stub: bool | None = None,
    *,
    model_dir: Path | None = None,
    models_dir: Path | None = None,
    instruction: str | None = None,
    max_pair_tokens: int | None = None,
    token_budget: int | None = None,
) -> Reranker:
    """Створити реранкер. `stub=None` означає «дивись на ASISTENT_STUB»."""
    model = get(model_name)
    if _stub_enabled() if stub is None else stub:
        return StubReranker(model, instruction=instruction or model.instruction)
    path = model_dir if model_dir is not None else default_model_dir(model, models_dir)
    return OnnxReranker(
        model,
        model_dir=path,
        instruction=instruction,
        max_pair_tokens=max_pair_tokens,
        token_budget=token_budget,
    )
