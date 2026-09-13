"""Векторний індекс: USearch HNSW як перебудовуваний сайдкар.

ГОЛОВНА ВЛАСТИВІСТЬ — ІНДЕКС ВІДКРИВАЄТЬСЯ MEMORY-MAPPED
--------------------------------------------------------
`Index.restore(path, view=True)` не завантажує граф у RAM: сторінки підтягує
ядро на вимогу і викидає під тиском пам'яті. Це не оптимізація, а умова
працездатності. LM Studio з моделлю 12B Q4_K_M і KV-кешем на 16k уже тримає
8–12 ГБ; резидентний індекс на 100 тис. чанків × 1024 × f16 — це ще 200 МБ
графа плюс піки перебудови, і саме вони виштовхують LLM у своп посеред
відповіді. Викладач переживе це як «асистент завис».

SQLite — ДЖЕРЕЛО ІСТИНИ, ЦЕЙ ФАЙЛ — КЕШ
---------------------------------------
Канонічні вектори лежать у `chunks.embedding` (float16 LE BLOB). Порахувати їх
заново — години локального CPU; побудувати з них HNSW — 1–3 хвилини на 100 тис.
Тому граф має право будь-якої миті виявитись застарілим або відсутнім, і
система мусить це пережити БЕЗ помилки: розбіжність `build_id` → перебудова у
фоні, а поки що обслуговуємо точним brute force прямо з SQLite. Повільніше,
але правильно; тихо неправильних відповідей не буває.

ПАСТКА WINDOWS, ЯКУ ТРЕБА ЗНАТИ ЗАЗДАЛЕГІДЬ
-------------------------------------------
`view=True` тримає відкритий дескриптор файлу. На POSIX це нікому не заважає:
`os.replace` над відкритим файлом працює, читач далі бачить стару інодну копію.
На Windows відкритий дескриптор БЛОКУЄ і видалення, і заміну — `os.replace`
падає з `PermissionError` (WinError 5/32). Тому будь-який читач мусить
ЯВНО викликати `close()` перед підміною файлу індексу, а не покладатися на
збирач сміття. У нашій топології читач (API-процес) і будівник (воркер) — різні
процеси, тож `publish_index()` ловить `PermissionError` і повідомляє
українською, що саме сталося, замість того щоб впасти незрозумілим стеком.

USEARCH — ОПЦІЙНИЙ ІМПОРТ
-------------------------
Без пакета модуль працює на власному mmap-контейнері з точним пошуком. Це
потрібно не заради екзотики, а щоб CI й тести не залежали від бінарного колеса
й щоб «індекс ще не побудований» був звичайним станом, а не аварією.
"""

from __future__ import annotations

import hashlib
import os
import struct
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from app.db.repositories import ChunkRepo, CollectionRepo, blob_to_vector

__all__ = [
    "SCHEMA_VERSION",
    "INDEX_LAYOUT_VERSION",
    "IndexParams",
    "SearchHit",
    "BuildReport",
    "VectorIndex",
    "CollectionIndex",
    "ExactVectors",
    "usearch_available",
    "compute_build_id",
    "index_path_for",
    "build_collection_index",
]

# Версія схеми БД, що входить у build_id: зміна схеми чанків робить старий граф
# несумісним навіть за тих самих векторів.
SCHEMA_VERSION = 1
# Версія РОЗКЛАДКИ каталогу індексів. Дозволяє двом embedding-моделям
# співіснувати і робить міграцію просто зміною вказівника (план, §14).
INDEX_LAYOUT_VERSION = 1

Backend = Literal["usearch", "mmap", "exact", "empty"]


def usearch_available() -> bool:
    """Чи є пакет `usearch`. Опційність навмисна — див. докстрінг модуля."""
    try:
        import usearch.index  # noqa: F401
    except Exception:  # pragma: no cover — залежить від середовища
        return False
    return True


@dataclass(frozen=True, slots=True)
class IndexParams:
    """Параметри HNSW. Ці числа — з плану, §2, і їх не можна міняти наосліп.

    `connectivity=32` (M) — удвічі більше за типовий дефолт 16: українські
    підручники дають щільні кластери майже-синонімічних абзаців, і на M=16
    recall@10 просідає саме на них. `expansion_add=200` — вартість збірки,
    платиться один раз. `expansion_search=128` — вартість запиту; він НЕ
    зберігається у файлі індексу, тож його треба виставляти після кожного
    `restore` (інакше мовчки отримаєте дефолтні 64 і нижчий recall).
    """

    dim: int
    metric: str = "cos"
    dtype: str = "f16"
    connectivity: int = 32
    expansion_add: int = 200
    expansion_search: int = 128


@dataclass(frozen=True, slots=True)
class SearchHit:
    """Ключ = `chunks.id`; скор = косинусна подібність у [-1, 1]."""

    key: int
    score: float


@dataclass(slots=True)
class BuildReport:
    build_id: str
    count: int
    path: Path
    backend: Backend
    index_generation: int
    dim: int


def compute_build_id(
    *,
    embedding_model_key: str,
    metric: str,
    connectivity: int,
    expansion_add: int,
    max_chunk_id: int,
    count: int,
    schema_version: int = SCHEMA_VERSION,
) -> str:
    """Відбиток збірки індексу (план, §2).

    Пара `(max_chunk_id, count)` — навмисно дешевий замінник повного хешу
    вмісту: два COUNT/MAX по індексованій колонці замість читання 100 тис.
    BLOB'ів. Вона ловить усі реальні сценарії — додали документ (обидва
    зросли), видалили документ (count упав), переіндексували іншою моделлю
    (змінився model_key). Теоретично можлива колізія «видалили N чанків і
    додали рівно N нових із меншими id» неможлива в SQLite: `chunks.id` —
    зростаючий INTEGER PRIMARY KEY.
    """
    payload = "|".join(
        (
            embedding_model_key,
            metric,
            str(connectivity),
            str(expansion_add),
            str(max_chunk_id),
            str(count),
            str(schema_version),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def index_path_for(index_dir: Path | str, collection_id: str, embedding_model_key: str) -> Path:
    """Версійований шлях `{layout}/{колекція}/{модель}/index.usearch`.

    Один файл на колекцію → ізоляція асистентів ФІЗИЧНА, а не через метаданий
    фільтр: нуль втрати recall і нуль ризику, що фрагмент чужого асистента
    протече у відповідь. Це рівень L0 адаптивної стратегії фільтрування.
    """
    return (
        Path(index_dir)
        / f"v{INDEX_LAYOUT_VERSION}"
        / collection_id
        / embedding_model_key[:16]
        / "index.usearch"
    )


# ------------------------------------------------------ точний пошук (brute force)
@dataclass(slots=True)
class ExactVectors:
    """Точний пошук над матрицею нормованих векторів.

    Використовується у трьох випадках: індексу ще немає, `build_id` розійшовся,
    або метаданий фільтр настільки вузький, що ANN втрачає сенс (L2 стратегії
    з §7 плану). На 20 тис. чанків × 1024 це ~40 МБ і ~15 мс на запит — цілком
    прийнятно як тимчасовий режим.
    """

    keys: np.ndarray
    vectors: np.ndarray

    @classmethod
    def from_rows(cls, rows: Sequence[tuple[int, bytes]], dim: int) -> ExactVectors:
        if not rows:
            return cls(np.zeros(0, dtype=np.uint64), np.zeros((0, dim), dtype=np.float32))
        keys = np.fromiter((int(r[0]) for r in rows), dtype=np.uint64, count=len(rows))
        mat = np.empty((len(rows), dim), dtype=np.float32)
        for i, (_, blob) in enumerate(rows):
            mat[i] = blob_to_vector(blob, dim)
        return cls(keys, mat)

    def __len__(self) -> int:
        return int(self.keys.shape[0])

    def search(
        self, query: np.ndarray, k: int, *, allowed: set[int] | None = None
    ) -> list[SearchHit]:
        if len(self) == 0 or k <= 0:
            return []
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        mat = self.vectors
        keys = self.keys
        if allowed is not None:
            mask = np.fromiter((int(x) in allowed for x in keys), dtype=bool, count=len(keys))
            mat = mat[mask]
            keys = keys[mask]
            if keys.shape[0] == 0:
                return []
        scores = mat @ q
        take = min(k, scores.shape[0])
        # argpartition — O(n) проти O(n log n) повного сортування; на 100 тис.
        # кандидатів різниця вимірна, і саме цей шлях працює, доки граф
        # перебудовується.
        idx = np.argpartition(-scores, take - 1)[:take]
        idx = idx[np.argsort(-scores[idx])]
        return [SearchHit(key=int(keys[i]), score=float(scores[i])) for i in idx]


# --------------------------------------------- власний mmap-контейнер (без usearch)
_MAGIC = b"ASISTIDX"
_HEADER_SIZE = 64
_HEADER_FMT = "<8sIIQ"


def _write_fallback(path: Path, keys: np.ndarray, vectors: np.ndarray) -> None:
    """Один файл: заголовок, ключі uint64, вектори float16.

    Один файл, а не два, саме заради атомарності: `os.replace` атомарний для
    ОДНОГО файлу, а підміна пари файлів має вікно, в якому ключі вже нові,
    а вектори ще старі — і пошук тихо повертає чужі фрагменти.
    """
    dim = int(vectors.shape[1]) if vectors.ndim == 2 else 0
    header = struct.pack(_HEADER_FMT, _MAGIC, 1, dim, int(keys.shape[0]))
    with open(path, "wb") as fh:
        fh.write(header.ljust(_HEADER_SIZE, b"\0"))
        fh.write(np.ascontiguousarray(keys, dtype="<u8").tobytes())
        fh.write(np.ascontiguousarray(vectors, dtype="<f2").tobytes())


def _read_fallback_header(path: Path) -> tuple[int, int]:
    with open(path, "rb") as fh:
        raw = fh.read(_HEADER_SIZE)
    magic, version, dim, count = struct.unpack(_HEADER_FMT, raw[: struct.calcsize(_HEADER_FMT)])
    if magic != _MAGIC or version != 1:
        raise ValueError(f"Файл індексу {path} має невідомий формат — перебудуйте індекс.")
    return int(dim), int(count)


# --------------------------------------------------------------------- сам індекс
class VectorIndex:
    """Один файл індексу однієї колекції.

    Життєвий цикл читача: `open()` → багато `search()` → `close()`.
    Життєвий цикл будівника: `write(...)` у темп-файл → `publish(...)`.
    """

    def __init__(
        self, path: Path | str, params: IndexParams, *, prefer_usearch: bool = True
    ) -> None:
        self.path = Path(path)
        self.params = params
        self.prefer_usearch = prefer_usearch and usearch_available()
        self._index: Any | None = None
        self._keys: np.ndarray | None = None
        self._vectors: np.ndarray | None = None
        self._count = 0

    # ------------------------------------------------------------- запис
    @staticmethod
    def write(
        path: Path | str,
        params: IndexParams,
        keys: Sequence[int] | np.ndarray,
        vectors: np.ndarray,
        *,
        prefer_usearch: bool = True,
    ) -> Backend:
        """Записати індекс за вказаним шляхом. Каталог створюється сам."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        key_arr = np.asarray(keys, dtype=np.uint64).reshape(-1)
        vec = np.asarray(vectors, dtype=np.float32)
        if vec.ndim == 1:
            vec = vec.reshape(1, -1) if key_arr.size else vec.reshape(0, params.dim)
        if vec.shape[0] != key_arr.shape[0]:
            raise ValueError(
                f"Кількість ключів ({key_arr.shape[0]}) не збігається з кількістю "
                f"векторів ({vec.shape[0]})."
            )
        if key_arr.size and vec.shape[1] != params.dim:
            raise ValueError(
                f"Розмірність вектора {vec.shape[1]} не збігається з розмірністю "
                f"колекції ({params.dim})."
            )

        if prefer_usearch and usearch_available():
            from usearch.index import Index

            index = Index(
                ndim=params.dim,
                metric=params.metric,
                dtype=params.dtype,
                connectivity=params.connectivity,
                expansion_add=params.expansion_add,
                expansion_search=params.expansion_search,
            )
            # `reserve` прибрали в usearch 2.26 (індекс росте сам), але в
            # старіших версіях без нього масове додавання перевиділяє пам'ять
            # десятки разів. Тому викликаємо, ЯКЩО він є.
            reserve = getattr(index, "reserve", None)
            if callable(reserve) and key_arr.size:
                reserve(int(key_arr.shape[0]))
            if key_arr.size:
                index.add(key_arr, vec)
            index.save(str(path))
            index.reset()
            return "usearch"

        _write_fallback(path, key_arr, vec)
        return "mmap"

    @staticmethod
    def publish(tmp_path: Path | str, final_path: Path | str) -> None:
        """Атомарна підміна файлу індексу.

        На Windows це впаде, якщо якийсь процес тримає індекс відкритим із
        `view=True`. Повідомлення має пояснювати саме це, бо голий
        `PermissionError: [WinError 5]` виглядає як проблема з правами доступу
        і надійно відправляє в неправильний бік.
        """
        final = Path(final_path)
        final.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.replace(tmp_path, final)
        except PermissionError as exc:  # pragma: no cover — відтворюється лише на Windows
            raise PermissionError(
                f"Не вдалося замінити файл індексу {final}: його тримає відкритим інший "
                "процес. На Windows memory-mapped індекс (view=True) блокує заміну — "
                "закрийте читача (VectorIndex.close()) перед публікацією нової збірки."
            ) from exc

    # ------------------------------------------------------------ читання
    def open(self) -> VectorIndex:
        """Відкрити індекс memory-mapped. Ідемпотентно."""
        if self.is_open:
            return self
        if not self.path.exists():
            raise FileNotFoundError(
                f"Файл індексу {self.path} не знайдено. Індекс ще не побудовано або "
                "його видалили — перебудуйте його або скористайтесь точним пошуком."
            )

        if self.prefer_usearch:
            from usearch.index import Index

            try:
                index = Index.restore(str(self.path), view=True)
            except (ValueError, RuntimeError, OSError):
                # `Index.restore` на чужому файлі кидає ValueError
                # («Not a dense USearch index!»), а не повертає None. Це не
                # аварія: індекс міг бути зібраний на машині без колеса
                # `usearch`, і власний mmap-контейнер нижче прочитає його
                # коректно. Падати тут означало б вимагати переіндексації
                # корпусу лише через різницю в середовищах.
                index = None
            if index is not None:
                # expansion_search НЕ зберігається у файлі: після restore
                # повертається дефолт 64. Мовчазна втрата recall.
                index.expansion_search = self.params.expansion_search
                self._index = index
                self._count = int(index.size)
                return self
            # Файл є, але це не USearch — імовірно, збірка фолбеком.

        dim, count = _read_fallback_header(self.path)
        self._keys = np.memmap(
            self.path, dtype="<u8", mode="r", offset=_HEADER_SIZE, shape=(count,)
        )
        self._vectors = np.memmap(
            self.path,
            dtype="<f2",
            mode="r",
            offset=_HEADER_SIZE + 8 * count,
            shape=(count, dim),
        )
        self._count = count
        return self

    @property
    def is_open(self) -> bool:
        return self._index is not None or self._keys is not None

    @property
    def backend(self) -> Backend:
        if self._index is not None:
            return "usearch"
        if self._keys is not None:
            return "mmap"
        return "empty"

    def __len__(self) -> int:
        return self._count

    def close(self) -> None:
        """ЗАВЖДИ викликати перед підміною файлу. Див. пастку Windows вище."""
        if self._index is not None:
            self._index.reset()
            self._index = None
        for name in ("_vectors", "_keys"):
            arr = getattr(self, name)
            if arr is not None:
                mm = getattr(arr, "_mmap", None)
                if mm is not None:
                    mm.close()
                setattr(self, name, None)
        self._count = 0

    def __enter__(self) -> VectorIndex:
        return self.open()

    def __exit__(self, *exc: object) -> None:
        self.close()

    def search(
        self,
        query: np.ndarray,
        k: int,
        *,
        allowed: set[int] | None = None,
        overfetch: int = 1,
    ) -> list[SearchHit]:
        """Top-k за косинусом. `allowed` — постфільтр за `chunks.id`.

        `overfetch` існує саме через постфільтр: ANN не вміє фільтрувати
        всередині обходу графа, тож при вузькому фільтрі доводиться діставати
        більше й відкидати. Масштаб overfetch обирає викликач за селективністю
        фільтра (стратегія L1 з §7 плану) — тут ми лише виконуємо.
        """
        if not self.is_open or k <= 0 or self._count == 0:
            return []
        want = max(k, min(self._count, k * max(1, overfetch)))
        q = np.asarray(query, dtype=np.float32).reshape(-1)

        if self._index is not None:
            matches = self._index.search(q, want)
            hits = [
                SearchHit(key=int(key), score=float(1.0 - dist))
                for key, dist in zip(
                    np.atleast_1d(matches.keys), np.atleast_1d(matches.distances), strict=False
                )
            ]
        else:
            assert self._keys is not None and self._vectors is not None
            scores = np.asarray(self._vectors, dtype=np.float32) @ q
            take = min(want, scores.shape[0])
            idx = np.argpartition(-scores, take - 1)[:take]
            idx = idx[np.argsort(-scores[idx])]
            hits = [SearchHit(key=int(self._keys[i]), score=float(scores[i])) for i in idx]

        if allowed is not None:
            hits = [h for h in hits if h.key in allowed]
        return hits[:k]


# --------------------------------------------------------------- збірка з БД
def build_collection_index(
    db: Any,
    collection_id: str,
    *,
    index_dir: Path | str,
    params: IndexParams | None = None,
    prefer_usearch: bool = True,
) -> BuildReport:
    """Зібрати індекс колекції з канонічних векторів SQLite і опублікувати.

    Дві критичні деталі, обидві з плану, §2:

    1. Збірка йде в ТЕМП-файл поруч із цільовим (той самий том — інакше
       `os.replace` перестає бути атомарним і перетворюється на копіювання),
       і лише потім підміняється.
    2. `index_generation` збільшується в ТІЙ САМІЙ транзакції, що й підміна
       файлу. Читачі порівнюють свій generation зі збереженим і перевідкривають
       індекс, коли він змінився.

    Чесна межа надійності: `os.replace` — не частина транзакції SQLite. Якщо
    коміт упаде вже після заміни файлу, на диску лишиться новіший граф зі
    старим `build_id` у БД. Це БЕЗПЕЧНИЙ бік відмови: розбіжність `build_id`
    переводить читача в точний пошук із SQLite, тобто у повільний, але
    правильний режим, і наступна збірка все виправляє.
    """
    with db.connection() as con:
        collection = CollectionRepo(con).get(collection_id)
        if collection is None:
            raise KeyError(f"Колекцію {collection_id} не знайдено.")
        chunks = ChunkRepo(con)
        rows = chunks.indexable(collection_id, collection.embedding_model_key)
        max_chunk_id = chunks.max_id(collection_id)

    p = params or IndexParams(dim=collection.dim, metric=collection.metric)
    if p.dim != collection.dim:
        raise ValueError(
            f"Розмірність параметрів індексу ({p.dim}) не збігається з розмірністю "
            f"колекції {collection_id} ({collection.dim})."
        )

    exact = ExactVectors.from_rows(rows, collection.dim)
    build_id = compute_build_id(
        embedding_model_key=collection.embedding_model_key,
        metric=p.metric,
        connectivity=p.connectivity,
        expansion_add=p.expansion_add,
        max_chunk_id=max_chunk_id,
        count=len(exact),
    )

    final_path = index_path_for(index_dir, collection_id, collection.embedding_model_key)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=".build-", suffix=".usearch", dir=str(final_path.parent)
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        backend = VectorIndex.write(
            tmp_path, p, exact.keys, exact.vectors, prefer_usearch=prefer_usearch
        )
        with db.transaction() as con:
            VectorIndex.publish(tmp_path, final_path)
            generation = CollectionRepo(con).bump_index_generation(collection_id, build_id)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)

    return BuildReport(
        build_id=build_id,
        count=len(exact),
        path=final_path,
        backend=backend,
        index_generation=generation,
        dim=collection.dim,
    )


# ------------------------------------------------------------- читацька сторона
class CollectionIndex:
    """Читач, що сам стежить за актуальністю індексу.

    Перед кожним запитом звіряє `build_id` і `index_generation` з БД:
      * збіг — шукаємо по mmap-графу (швидко);
      * розбіжність або відсутній файл — точний brute force із SQLite
        (повільніше, але завжди правильно), а перебудова — справа воркера.

    Саме тому «індекс застарів» ніколи не перетворюється на помилку в чаті.
    """

    def __init__(
        self,
        db: Any,
        collection_id: str,
        *,
        index_dir: Path | str,
        params: IndexParams | None = None,
        prefer_usearch: bool = True,
    ) -> None:
        self.db = db
        self.collection_id = collection_id
        self.index_dir = Path(index_dir)
        self.prefer_usearch = prefer_usearch
        self._params = params
        self._index: VectorIndex | None = None
        self._exact: ExactVectors | None = None
        self._generation = -1
        self._fingerprint: tuple[str, int, int] | None = None
        self._stale = False

    # ----------------------------------------------------------- внутрішнє
    def _state(self) -> tuple[Any, str, int, int]:
        with self.db.connection() as con:
            collection = CollectionRepo(con).get(self.collection_id)
            if collection is None:
                raise KeyError(f"Колекцію {self.collection_id} не знайдено.")
            chunks = ChunkRepo(con)
            count = con.execute(
                "SELECT count(*) FROM chunks WHERE collection_id=? AND level='L2'"
                " AND embedding IS NOT NULL AND embedding_model_key=?",
                (self.collection_id, collection.embedding_model_key),
            ).fetchone()[0]
            max_id = chunks.max_id(self.collection_id)
        return collection, collection.embedding_model_key, int(count), int(max_id)

    def _load_exact(self, collection: Any) -> None:
        with self.db.connection() as con:
            rows = ChunkRepo(con).indexable(self.collection_id, collection.embedding_model_key)
        self._exact = ExactVectors.from_rows(rows, collection.dim)

    def refresh(self) -> None:
        """Звірити стан із БД і, за потреби, перевідкрити індекс."""
        collection, model_key, count, max_id = self._state()
        params = self._params or IndexParams(dim=collection.dim, metric=collection.metric)
        expected = compute_build_id(
            embedding_model_key=model_key,
            metric=params.metric,
            connectivity=params.connectivity,
            expansion_add=params.expansion_add,
            max_chunk_id=max_id,
            count=count,
        )
        fingerprint = (expected, collection.index_generation, count)
        if fingerprint == self._fingerprint and (self._index is not None or self._exact is not None):
            return

        self.close()
        self._fingerprint = fingerprint
        self._generation = collection.index_generation
        self._stale = collection.build_id != expected

        path = index_path_for(self.index_dir, self.collection_id, model_key)
        if not self._stale and path.exists():
            try:
                self._index = VectorIndex(
                    path, params, prefer_usearch=self.prefer_usearch
                ).open()
                return
            except (OSError, ValueError):
                # Пошкоджений або обірваний файл індексу — не привід падати.
                self._index = None
                self._stale = True

        self._stale = True
        self._load_exact(collection)

    # ------------------------------------------------------------ публічне
    @property
    def backend(self) -> Backend:
        if self._index is not None:
            return self._index.backend
        if self._exact is not None:
            return "exact"
        return "empty"

    @property
    def is_stale(self) -> bool:
        """True → граф застарів і зараз обслуговує точний пошук."""
        return self._stale

    @property
    def index_generation(self) -> int:
        return self._generation

    def __len__(self) -> int:
        if self._index is not None:
            return len(self._index)
        return len(self._exact) if self._exact is not None else 0

    def search(
        self,
        query: np.ndarray,
        k: int,
        *,
        allowed: set[int] | None = None,
        overfetch: int = 1,
    ) -> list[SearchHit]:
        self.refresh()
        if self._index is not None:
            return self._index.search(query, k, allowed=allowed, overfetch=overfetch)
        if self._exact is not None:
            return self._exact.search(query, k, allowed=allowed)
        return []

    def exact_search(
        self, query: np.ndarray, k: int, *, allowed: set[int] | None = None
    ) -> list[SearchHit]:
        """Примусовий точний пошук — рівень L2 адаптивної стратегії (§7).

        Коли фільтр лишає менше ~15% колекції, обхід HNSW-графа перетворюється
        на блукання серед відкинутих сусідів: overfetch росте швидше, ніж
        економія від графа, і точний скан по вузькій множині виявляється і
        швидшим, і точнішим.
        """
        self.refresh()
        if self._exact is None:
            collection, _, _, _ = self._state()
            self._load_exact(collection)
        assert self._exact is not None
        return self._exact.search(query, k, allowed=allowed)

    def close(self) -> None:
        if self._index is not None:
            self._index.close()
            self._index = None
        self._exact = None

    def __enter__(self) -> CollectionIndex:
        self.refresh()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def iter_vectors(rows: Iterable[tuple[int, bytes]], dim: int) -> Iterable[tuple[int, np.ndarray]]:
    """Дрібний помічник для скриптів оцінювання: BLOB → float32-вектор."""
    for chunk_id, blob in rows:
        yield int(chunk_id), blob_to_vector(blob, dim)
