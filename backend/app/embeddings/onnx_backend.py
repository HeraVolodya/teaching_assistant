"""Ембедер на ПРЯМИХ сесіях ONNX Runtime.

Чому не `sentence-transformers` і не `fastembed` (контракт, правило 2):
`sentence-transformers` 6.x жорстко вимагає `torch>=2.2` навіть із
`backend="onnx"`, а `import torch` коштує 1.5-4 с на Windows — тобто 6+ секунд
до чутливості API-процесу; `fastembed` не має Qwen3 у каталозі взагалі.
Прямий ORT — 14 МБ і холодний старт ~0.6 с.

Чотири пастки, задокументовані в плані (§4), яких тут навмисно уникнуто:

* **EOS при last-token pooling.** Якщо токенайзер не додає EOS, pooling бере
  НЕ ТОЙ токен. Вектори при цьому виглядають абсолютно правдоподібно —
  правильна розмірність, одинична норма, розумні косинуси, — а recall тихо
  стає випадковим. Жоден тест типів цього не побачить. Тому EOS тут
  додається явно й перевіряється, а `requires_eos` без знайденого EOS-токена
  є фатальною помилкою конструктора, а не попередженням.
* **Побайтові шаблони префіксів.** Форматування живе в `registry.py` і
  застосовується в `provider.BaseEmbeddingProvider.format_text`. Тут його
  немає взагалі — щоб не було другого місця, де його можна «виправити».
* **Обрізання.** Стеля входу — `min(бюджет, max_seq - 8)`, і кожне обрізання
  потрапляє в `EncodeStats.truncated`, звідки його бачить звіт індексації.
* **Паддінг.** Батчування — за бюджетом ТОКЕНІВ, не за кількістю рядків, із
  попереднім сортуванням за довжиною. Батч із 64 рядків по 1500 токенів і
  батч із 64 рядків по 20 токенів відрізняються за вартістю у 75 разів;
  сортування перед батчуванням зрізає 30-50% паддінгу на реальному корпусі.

Вибір Execution Provider. За замовчуванням CUDA → CoreML (macOS) → CPU.
УВАГА: план §0 нагадує, що на цільовій машині GPU вже зайнято LM Studio, і
паралельна індексація на тій самій карті дає CUDA OOM *на етапі завантаження
моделі в LM Studio* — викладач переживе це як «ШІ перестав працювати». Тому
воркер індексації має явно виставляти `ASISTENT_EMBED_DEVICE=cpu`; авто-вибір
лишається для машини розробника й для конфігурацій із вільним GPU.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from app.embeddings.provider import (
    DEFAULT_MAX_INPUT_TOKENS,
    DEFAULT_TOKEN_BUDGET,
    MAX_TOKENS_ENV,
    TOKEN_BUDGET_ENV,
    BaseEmbeddingProvider,
    Side,
)
from app.embeddings.registry import UK_RETRIEVAL_TASK, EmbeddingModel, Pooling

__all__ = [
    "OnnxEmbeddingProvider",
    "ModelFilesMissing",
    "TokenizerContractError",
    "plan_batches",
    "select_providers",
    "DEVICE_ENV",
    "PROVIDERS_ENV",
    "MAX_BATCH_ROWS",
]

log = logging.getLogger(__name__)

DEVICE_ENV = "ASISTENT_EMBED_DEVICE"        # auto | cpu | cuda | coreml
PROVIDERS_ENV = "ASISTENT_EMBED_PROVIDERS"  # явний список ORT EP через кому

# Стеля рядків у батчі понад бюджет токенів. Потрібна лише для дуже коротких
# рядків (запити), де бюджет токенів дозволив би тисячі рядків і батч почав би
# упиратись у накладні витрати самої сесії.
MAX_BATCH_ROWS = 128

# Кандидати на EOS, якщо конфіг токенайзера мовчить. Порядок має значення:
# Qwen кладе `<|endoftext|>`, sentencepiece-родини — `</s>`.
_EOS_CANDIDATES = ("<|endoftext|>", "</s>", "<|im_end|>", "[SEP]", "<eos>")
_PAD_CANDIDATES = ("<pad>", "[PAD]", "<|endoftext|>", "</s>")

# Імена виходів, які реально трапляються в експортах ембедерів.
_OUTPUT_PREFERENCE = (
    "last_hidden_state",
    "token_embeddings",
    "sentence_embedding",
    "embeddings",
    "text_embeds",
    "pooler_output",
)


class ModelFilesMissing(FileNotFoundError):
    """Ваг або токенайзера немає в каталозі моделей."""

    def __init__(self, model: EmbeddingModel, path: Path, missing: str) -> None:
        super().__init__(
            f"Не знайдено {missing} для embedding-моделі {model.id} у каталозі {path}. "
            "Асістент працює у закритому контурі й НЕ завантажує моделі з мережі: "
            "скопіюйте файли моделі з інсталяційного носія або запустіть застосунок "
            "у режимі заглушки (ASISTENT_STUB=1)."
        )
        self.model_id = model.id
        self.path = path


class TokenizerContractError(RuntimeError):
    """Токенайзер не здатен виконати контракт моделі (найчастіше — EOS)."""


# --------------------------------------------------------------- батчування
def plan_batches(
    lengths: list[int],
    *,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
    max_rows: int = MAX_BATCH_ROWS,
) -> list[list[int]]:
    """Розкласти індекси на батчі за бюджетом ТОКЕНІВ із сортуванням за довжиною.

    Вартість батча — це `рядків × довжина_найдовшого`, бо коротші рядки
    доводиться допаднути до найдовшого. Тому:
      * сортуємо за довжиною → сусіди в батчі мають близькі довжини;
      * ріжемо батч, коли `(рядків+1) × новий_максимум` перевищує бюджет.
    Рядок, довший за бюджет сам по собі, утворює власний батч — ми його не
    відкидаємо, бо обрізання вже зробив токенайзер.
    """
    order = sorted(range(len(lengths)), key=lambda i: (lengths[i], i))
    batches: list[list[int]] = []
    current: list[int] = []
    current_max = 0
    for idx in order:
        length = max(1, lengths[idx])
        new_max = max(current_max, length)
        too_many_tokens = (len(current) + 1) * new_max > token_budget
        too_many_rows = len(current) + 1 > max_rows
        if current and (too_many_tokens or too_many_rows):
            batches.append(current)
            current, current_max = [idx], length
        else:
            current.append(idx)
            current_max = new_max
    if current:
        batches.append(current)
    return batches


# ------------------------------------------------------- execution providers
def select_providers(
    available: list[str] | None = None,
    *,
    device: str | None = None,
    explicit: str | None = None,
) -> list[str]:
    """CUDA → CoreML (лише macOS) → CPU, з можливістю примусу через середовище.

    `explicit` (явний список EP) має пріоритет над `device`
    (auto|cpu|cuda|coreml). `None` в обох означає «взяти з середовища»:
    `ASISTENT_EMBED_PROVIDERS` і `ASISTENT_EMBED_DEVICE`. Реранкер передає сюди
    СВОЇ змінні — логіка вибору спільна, а налаштування роздільні, бо ембедер і
    реранкер цілком свідомо можуть жити на різних пристроях.

    CPU завжди лишається останнім елементом списку: ORT падає назад на нього
    сам, і без цього відсутній CUDA-рантайм означав би не деградацію, а виняток.
    """
    if available is None:                       # pragma: no cover — залежить від машини
        import onnxruntime as ort

        available = list(ort.get_available_providers())

    explicit = (explicit if explicit is not None else os.environ.get(PROVIDERS_ENV, "")).strip()
    if explicit:
        chosen = [p.strip() for p in explicit.split(",") if p.strip() and p.strip() in available]
        if "CPUExecutionProvider" not in chosen:
            chosen.append("CPUExecutionProvider")
        return chosen

    device = (device if device is not None else os.environ.get(DEVICE_ENV, "auto")).strip().lower() or "auto"
    chosen = []
    if device in {"auto", "cuda"} and "CUDAExecutionProvider" in available:
        chosen.append("CUDAExecutionProvider")
    elif device in {"auto", "coreml"} and sys.platform == "darwin" and "CoreMLExecutionProvider" in available:
        chosen.append("CoreMLExecutionProvider")
    chosen.append("CPUExecutionProvider")
    return chosen


# ------------------------------------------------------------------ провайдер
class OnnxEmbeddingProvider(BaseEmbeddingProvider):
    """Пряма сесія ORT + `tokenizers`. Без torch, без sentence-transformers."""

    def __init__(
        self,
        model: EmbeddingModel,
        *,
        model_dir: Path,
        task: str = UK_RETRIEVAL_TASK,
        max_input_tokens: int | None = None,
        token_budget: int | None = None,
        session: Any | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        super().__init__(model, task=task)
        self._dir = Path(model_dir)

        env_max = os.environ.get(MAX_TOKENS_ENV, "").strip()
        env_budget = os.environ.get(TOKEN_BUDGET_ENV, "").strip()
        requested = max_input_tokens or (int(env_max) if env_max.isdigit() else DEFAULT_MAX_INPUT_TOKENS)
        # -8 — це не запас «на всякий випадок», а безпосередньо твердження §4.4
        # самотесту: довжина будь-якого входу мусить лишатися <= max_seq - 8.
        self._max_input_tokens = max(8, min(requested, model.max_seq - 8))
        self._token_budget = token_budget or (
            int(env_budget) if env_budget.isdigit() else DEFAULT_TOKEN_BUDGET
        )

        self._tok = tokenizer if tokenizer is not None else self._load_tokenizer()
        self._eos_id, self._eos_token = self._resolve_eos()
        self._pad_id = self._resolve_pad()
        self.tokenizer_appends_eos = self._probe_tokenizer_eos()

        if model.requires_eos and self._eos_id is None:
            raise TokenizerContractError(
                f"Модель {model.id} використовує last-token pooling і вимагає EOS, але в "
                f"токенайзері з {self._dir} не вдалося визначити EOS-токен. Без EOS pooling "
                "бере не той токен: вектори виглядають правдоподібно, а якість пошуку тихо "
                "стає випадковою. Перевірте tokenizer_config.json у каталозі моделі."
            )

        self._session = session if session is not None else self._load_session()
        self._input_names = {i.name for i in self._session.get_inputs()}
        # Специфікації KV-кешу читаються з САМОЇ сесії: кількість шарів і
        # розміри голів різні у 0.6B, 4B і 8B, а помилка тут проявляється лише
        # в рантаймі, після повного розбору документа.
        self._past_kv_specs: list[tuple[str, int, int, Any]] = []
        for spec in self._session.get_inputs():
            if "past_key_values" not in spec.name:
                continue
            shape = list(spec.shape)
            if len(shape) != 4:
                continue
            heads = shape[1] if isinstance(shape[1], int) else 8
            head_dim = shape[3] if isinstance(shape[3], int) else 128
            dtype = np.float16 if "float16" in (spec.type or "") else np.float32
            self._past_kv_specs.append((spec.name, int(heads), int(head_dim), dtype))

        # CoreML НЕ ВМІЄ тензорів із нульовою кількістю елементів, а порожній
        # KV-кеш — це саме `(batch, heads, 0, head_dim)`. ORT повідомляє про це
        # прямо: «has a dynamic shape ({-1,8,-1,128}) but the runtime shape
        # ({1,8,0,128}) has zero elements. This is not supported by the CoreML EP».
        # Тому для декодерних ембедерів на macOS перескладаємо сесію без CoreML.
        # Робимо це ПІСЛЯ інспекції входів, бо дізнатися про наявність KV-кешу
        # інакше як із самої моделі неможливо — а покладатися на її назву
        # означало б зламатися на наступному експорті.
        if self._past_kv_specs and session is None:
            used = [p for p in self._session.get_providers() if "CoreML" in p]
            if used:
                log.info(
                    "Модель %s має KV-кеш — виключаю CoreML EP (він не приймає "
                    "тензори нульової довжини) і перескладаю сесію на CPU.",
                    self._model.id,
                )
                self._session = self._load_session(exclude={"CoreMLExecutionProvider"})
                self._input_names = {i.name for i in self._session.get_inputs()}

        self._output_names = [o.name for o in self._session.get_outputs()]

    # ------------------------------------------------------------ завантаження
    def _load_tokenizer(self) -> Any:
        from tokenizers import Tokenizer

        path = self._dir / "tokenizer.json"
        if not path.is_file():
            raise ModelFilesMissing(self._model, self._dir, "tokenizer.json")
        tok = Tokenizer.from_file(str(path))
        # Обрізання й паддінг робимо самі: обрізання токенайзера зрізало б EOS,
        # а його паддінг не знає про наше бакетування за довжиною.
        tok.no_truncation()
        tok.no_padding()
        return tok

    def _load_session(self, exclude: set[str] | None = None) -> Any:
        import onnxruntime as ort

        path = self._dir / self._model.onnx_file
        if not path.is_file():
            raise ModelFilesMissing(self._model, self._dir, self._model.onnx_file)

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Фізичні ядра мінус одне — щоб індексація не «з'їдала» чутливість UI.
        options.intra_op_num_threads = max(1, (os.cpu_count() or 2) - 1)
        options.inter_op_num_threads = 1

        providers = select_providers()
        if exclude:
            providers = [p for p in providers if p not in exclude] or ["CPUExecutionProvider"]
        try:
            return ort.InferenceSession(str(path), sess_options=options, providers=providers)
        except Exception as exc:      # pragma: no cover — залежить від машини
            if providers == ["CPUExecutionProvider"]:
                raise
            log.warning(
                "ORT не зміг стартувати з провайдерами %s (%s); переходжу на CPU.", providers, exc
            )
            return ort.InferenceSession(
                str(path), sess_options=options, providers=["CPUExecutionProvider"]
            )

    def _read_json(self, name: str) -> dict[str, Any]:
        path = self._dir / name
        if not path.is_file():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):   # pragma: no cover — пошкоджений конфіг
            return {}

    def _resolve_eos(self) -> tuple[int | None, str | None]:
        candidates: list[str] = []
        for source in (self._read_json("tokenizer_config.json"), self._read_json("special_tokens_map.json")):
            token = source.get("eos_token")
            if isinstance(token, str):
                candidates.append(token)
            elif isinstance(token, dict) and isinstance(token.get("content"), str):
                candidates.append(token["content"])
        candidates.extend(_EOS_CANDIDATES)
        for token in candidates:
            tid = self._tok.token_to_id(token)
            if tid is not None:
                return int(tid), token
        return None, None

    def _resolve_pad(self) -> int:
        for source in (self._read_json("tokenizer_config.json"), self._read_json("special_tokens_map.json")):
            token = source.get("pad_token")
            content = token if isinstance(token, str) else (token or {}).get("content")
            if isinstance(content, str):
                tid = self._tok.token_to_id(content)
                if tid is not None:
                    return int(tid)
        for token in _PAD_CANDIDATES:
            tid = self._tok.token_to_id(token)
            if tid is not None:
                return int(tid)
        # Паддінг маскується attention_mask=0, тож конкретний id не впливає на
        # результат — але 0 має бути валідним, інакше сесія впаде на вибірці.
        return 0

    def _probe_tokenizer_eos(self) -> bool:
        """Чи токенайзер САМ дописує EOS. Діагностика для самотесту."""
        if self._eos_id is None:
            return False
        ids = self._tok.encode("перевірка", add_special_tokens=True).ids
        return bool(ids) and ids[-1] == self._eos_id

    # ------------------------------------------------------------ властивості
    @property
    def max_input_tokens(self) -> int:
        return self._max_input_tokens

    @property
    def token_budget(self) -> int:
        return self._token_budget

    @property
    def providers(self) -> list[str]:
        return list(self._session.get_providers())

    def count_tokens(self, text: str) -> int:
        return len(self._encode_ids(text)[0])

    # ------------------------------------------------------------- токенізація
    def _encode_ids(self, text: str) -> tuple[list[int], bool]:
        """Токени одного тексту з гарантованим EOS і чесним прапорцем обрізання."""
        ids = list(self._tok.encode(text, add_special_tokens=True).ids)
        limit = self._max_input_tokens
        truncated = False

        if self._model.requires_eos and self._eos_id is not None:
            if ids and ids[-1] == self._eos_id:
                ids = ids[:-1]
            if len(ids) > limit - 1:
                ids = ids[: limit - 1]
                truncated = True
            ids.append(self._eos_id)
        elif len(ids) > limit:
            ids = ids[:limit]
            truncated = True

        if not ids:
            ids = [self._eos_id if self._eos_id is not None else self._pad_id]
        return ids, truncated

    # ---------------------------------------------------------------- pooling
    @staticmethod
    def _pool(hidden: np.ndarray, mask: np.ndarray, pooling: Pooling) -> np.ndarray:
        """Три види pooling із реєстру. Паддінг — ПРАВОСТОРОННІЙ."""
        if pooling is Pooling.CLS:
            return hidden[:, 0, :]
        if pooling is Pooling.MEAN:
            m = mask[:, :, None].astype(np.float32)
            summed = (hidden * m).sum(axis=1)
            counts = np.clip(m.sum(axis=1), 1e-9, None)
            return summed / counts
        # LAST_TOKEN: останній НЕпаддінговий токен, тобто EOS. Саме тому EOS
        # обов'язковий: без нього тут читається останній токен тексту.
        last = np.clip(mask.sum(axis=1) - 1, 0, hidden.shape[1] - 1).astype(np.int64)
        return hidden[np.arange(hidden.shape[0]), last, :]

    def _empty_past_key_values(self, batch: int) -> dict[str, np.ndarray]:
        """Порожній KV-кеш для декодерних ембедерів.

        Qwen3-Embedding експортовано як ДЕКОДЕР із KV-кешем: `model_int8.onnx`
        вимагає 56 входів `past_key_values.<шар>.<key|value>` (28 шарів × 2)
        поруч із `input_ids`, `attention_mask` і `position_ids`. Без них
        ONNX Runtime валить сесію повідомленням «Required inputs
        (['past_key_values.0.key', …]) are missing», і індексація падає ПІСЛЯ
        успішного розбору всіх 140 сторінок — найдорожчий момент для відмови.

        Нам потрібен один прямий прохід, тож минуле має нульову довжину:
        форма `(batch, n_kv_heads, 0, head_dim)`. Кількість шарів і розміри
        читаємо з самої сесії, а не з конфігу — так це працює і для 0.6B,
        і для 4B, і для будь-якого іншого експорту.
        """
        if not self._past_kv_specs:
            return {}
        feeds: dict[str, np.ndarray] = {}
        for name, heads, head_dim, dtype in self._past_kv_specs:
            feeds[name] = np.zeros((batch, heads, 0, head_dim), dtype=dtype)
        return feeds

    def _pick_output(self, outputs: list[np.ndarray]) -> np.ndarray:
        by_name = dict(zip(self._output_names, outputs, strict=False))
        for name in _OUTPUT_PREFERENCE:
            if name in by_name:
                return np.asarray(by_name[name])
        for arr in outputs:                       # pragma: no cover — нетиповий експорт
            if np.asarray(arr).ndim in (2, 3):
                return np.asarray(arr)
        raise RuntimeError(
            f"Сесія ONNX для {self._model.id} не повернула жодного виходу, придатного для "
            f"pooling'у. Доступні виходи: {self._output_names}."
        )

    # ------------------------------------------------------------- кодування
    def _run_batch(self, rows: list[list[int]]) -> np.ndarray:
        width = max(len(r) for r in rows)
        ids = np.full((len(rows), width), self._pad_id, dtype=np.int64)
        mask = np.zeros((len(rows), width), dtype=np.int64)
        for i, row in enumerate(rows):
            ids[i, : len(row)] = row
            mask[i, : len(row)] = 1

        feeds: dict[str, np.ndarray] = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._input_names:
            feeds["token_type_ids"] = np.zeros_like(ids)
        if "position_ids" in self._input_names:
            feeds["position_ids"] = np.clip(np.cumsum(mask, axis=1) - 1, 0, None).astype(np.int64)
        feeds.update(self._empty_past_key_values(len(rows)))

        raw = self._session.run(None, {k: v for k, v in feeds.items() if k in self._input_names})
        out = self._pick_output(list(raw)).astype(np.float32)
        pooled = self._pool(out, mask, self._model.pooling) if out.ndim == 3 else out

        if pooled.shape[1] < self._model.dim:
            raise RuntimeError(
                f"Модель {self._model.id} повернула вектор розмірності {pooled.shape[1]}, "
                f"а реєстр очікує {self._model.dim}. Перевірте, що у каталозі лежить саме "
                f"файл {self._model.onnx_file}."
            )
        # MRL (Qwen3): матрьошкові виміри зрізаються з ХВОСТА, перші `dim`
        # координат самодостатні. Нормалізацію робить BaseEmbeddingProvider.
        return pooled[:, : self._model.dim]

    def _encode_formatted(self, texts: list[str], *, side: Side) -> np.ndarray:
        encoded: list[list[int]] = []
        for text in texts:
            ids, truncated = self._encode_ids(text)
            self.stats.note(tokens=len(ids), truncated=truncated)
            if truncated:
                log.warning(
                    "Обрізано вхід ембедера до %d токенів (модель %s, бік %s). "
                    "Частина чанка не потрапить у вектор.",
                    self._max_input_tokens, self._model.id, side,
                )
            encoded.append(ids)

        out = np.zeros((len(texts), self._model.dim), dtype=np.float32)
        batches = plan_batches(
            [len(e) for e in encoded], token_budget=self._token_budget, max_rows=MAX_BATCH_ROWS
        )
        for batch in batches:
            vectors = self._run_batch([encoded[i] for i in batch])
            for position, source_index in enumerate(batch):
                out[source_index] = vectors[position]
            self.stats.batches += 1
        return out

    def close(self) -> None:
        self._session = None  # type: ignore[assignment]
