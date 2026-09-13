"""Шляхи, каталоги даних і платформні особливості.

Правила, що коштували б дорого, якби їх виявили пізніше:
  * БД ніколи не лежить на UNC / мережевому диску / у теці хмарної синхронізації —
    SQLite поверх SMB і OneDrive є задокументованим джерелом пошкодження файлів.
  * Користувацькі файли зберігаються під UUID, а не під оригінальною назвою:
    кириличне ім'я підручника плюс глибокий шлях тривіально пробиває MAX_PATH 260.
    Оригінальна назва живе в SQLite як метадані.
  * Дані — у LOCALAPPDATA (Windows) / Application Support (macOS), НЕ в
    ProgramData: той вимагає прав адміністратора й ламає per-user встановлення.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

APP_NAME = "Asistent"

_FORBIDDEN_PATH_MARKERS = ("onedrive", "dropbox", "google drive", "yandex", "icloud")


class UnsafeDataDirectory(RuntimeError):
    pass


def _default_data_dir() -> Path:
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local")
        return Path(base) / APP_NAME
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_NAME
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / APP_NAME


def _default_log_dir(data_dir: Path) -> Path:
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Logs" / APP_NAME
    return data_dir / "logs"


def validate_data_dir(path: Path) -> None:
    """Відхилити каталоги, у яких SQLite не можна тримати безпечно."""
    text = str(path)
    lowered = text.lower()

    if sys.platform == "win32" and (text.startswith("\\\\") or text.startswith("//")):
        raise UnsafeDataDirectory(
            f"Каталог даних не може бути мережевим шляхом UNC: {text}. "
            "SQLite поверх SMB пошкоджується. Оберіть локальний диск."
        )
    for marker in _FORBIDDEN_PATH_MARKERS:
        if marker in lowered:
            raise UnsafeDataDirectory(
                f"Каталог даних не може бути в теці хмарної синхронізації ({marker}): {text}. "
                "Синхронізація пошкоджує файл бази даних. Оберіть звичайну локальну теку."
            )


@dataclass(frozen=True, slots=True)
class Paths:
    data_dir: Path
    db_path: Path
    documents_dir: Path   # оригінали, збережені під UUID
    artifacts_dir: Path   # doc.json + витягнуті растри від Docling
    index_dir: Path       # сайдкари USearch, по одному каталогу на колекцію
    models_dir: Path      # ВСІ ML-моделі; ніколи не кеш HuggingFace
    cache_dir: Path       # рендери сторінок, мініатюри — можна видаляти
    logs_dir: Path

    @classmethod
    def resolve(cls, data_dir: str | os.PathLike[str] | None = None) -> "Paths":
        # Перевіряється РЯДОК зі змінної середовища, а не зібраний із нього
        # `Path`. `Path("")` дорівнює `Path(".")` і є ІСТИННИМ, тому колишнє
        # `Path(os.environ.get(...)) or _default_data_dir()` ніколи не доходило
        # до фолбеку: при незаданій змінній каталогом даних тихо ставав
        # поточний робочий каталог, тобто БД і моделі лягали в репозиторій.
        env = os.environ.get("ASISTENT_DATA_DIR", "").strip()
        if data_dir:
            base = Path(data_dir)
        elif env:
            base = Path(env)
        else:
            base = _default_data_dir()
        base = base.expanduser().resolve()
        validate_data_dir(base)
        return cls(
            data_dir=base,
            db_path=base / "assistant.db",
            documents_dir=base / "docs",
            artifacts_dir=base / "artifacts",
            index_dir=base / "idx",
            models_dir=base / "models",
            cache_dir=base / "cache",
            logs_dir=_default_log_dir(base),
        )

    def ensure(self) -> "Paths":
        for p in (
            self.data_dir, self.documents_dir, self.artifacts_dir,
            self.index_dir, self.models_dir, self.cache_dir, self.logs_dir,
        ):
            p.mkdir(parents=True, exist_ok=True)
        return self


def export_model_env(paths: Paths) -> dict[str, str]:
    """Змінні середовища, які мусить нести кожен процес-воркер.

    `DOCLING_ARTIFACTS_PATH` має вказувати на БАТЬКІВСЬКИЙ каталог, що містить
    теки `<org>--<repo>`, а не на теку конкретної моделі. Якщо теки немає,
    Docling мовчки викликає snapshot_download — тобто йде в мережу. Офлайн-
    змінні HuggingFace плюс net_guard роблять цей шлях гучною помилкою.
    """
    return {
        "DOCLING_ARTIFACTS_PATH": str(paths.models_dir / "docling"),
        "HF_HOME": str(paths.models_dir / "hf"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_SYMLINKS": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
        # Фізичні ядра, не логічні: Docling сам бере cpu_count()-1, щоб машина
        # лишалась чутливою.
        "OMP_NUM_THREADS": str(max(1, (os.cpu_count() or 2) - 1)),
    }


# PRAGMA виставляються на КОЖНОМУ з'єднанні, саме в цьому порядку.
SQLITE_PRAGMAS: tuple[tuple[str, str], ...] = (
    ("journal_mode", "WAL"),
    ("synchronous", "NORMAL"),
    ("foreign_keys", "ON"),
    ("busy_timeout", "5000"),
    ("cache_size", "-262144"),      # 256 МБ сторінкового кешу
    ("mmap_size", "536870912"),     # 512 МБ
    ("temp_store", "MEMORY"),
    ("auto_vacuum", "INCREMENTAL"),
    # 256 сторінок × 8 КБ = 2 МБ, а не 16 МБ.
    # Поріг 2000 сторінок звичайна робота викладача не досягає взагалі:
    # на живій машині основний файл лишався 8 КБ і change-counter 1, тоді як
    # уся база разом зі схемою жила у WAL. Копія `assistant.db` була б порожня.
    ("wal_autocheckpoint", "256"),
)
# page_size можна змінити лише до першого запису — застосовується окремо.
SQLITE_PAGE_SIZE = 8192
