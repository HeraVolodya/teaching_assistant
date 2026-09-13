"""Налаштування процесу: шляхи, порт, stub-режим, LM Studio, воркери.

Один клас `Settings` читає середовище з префіксом `ASISTENT_`, тож усе, що
керує поведінкою застосунку, видно в одному місці й перевизначається без
редагування коду — це вимога закритого контуру, де змінити код на машині
викладача неможливо, а змінну середовища в ярлику — можна.

Дві змінні мають особливий статус:

* `ASISTENT_STUB=1` — детерміновані заглушки ембедера, реранкера й LLM. Це не
  режим розробника, а вимога: увесь застосунок мусить проходити наскрізь без
  жодної завантаженої моделі, інакше CI і UI-робота стають заручниками 1.5 ГБ
  ваг (патерн LLAMA_CPP_STUB з on-device POC NeoLens).
* `ASISTENT_DATA_DIR` — той самий ключ, що читає `app.config.Paths.resolve()`.
  Тримати два різні імена для однієї речі — найдешевший спосіб отримати БД у
  двох каталогах одночасно.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.config import Paths
from app.domain import IngestMode

__all__ = ["Settings", "WorkerMode", "get_settings"]

WorkerMode = Literal["process", "inline", "off"]


class Settings(BaseSettings):
    """Конфігурація API-процесу і воркерів."""

    model_config = SettingsConfigDict(
        env_prefix="ASISTENT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- режим ---------------------------------------------------------
    stub: bool = Field(
        default=False,
        description="Детерміновані заглушки ембедера, реранкера й LLM.",
    )

    # --- мережа --------------------------------------------------------
    # Слухати ТІЛЬКИ петлю назад. 0.0.0.0 у закритому контурі — це відкритий
    # доступ до чужих навчальних матеріалів з локальної мережі академії.
    host: str = "127.0.0.1"
    port: int = 8765

    # --- шляхи ---------------------------------------------------------
    data_dir: Path | None = None

    # --- LM Studio -----------------------------------------------------
    # None → автовиявлення через ~/.lmstudio/.internal/http-server-config.json.
    lmstudio_url: str | None = None
    lmstudio_model: str | None = None
    # Реальний n_ctx завантаженої моделі. 8k достатньо для одноразового
    # заземленого RAG (план, §10: ~3300 токенів на запит).
    context_tokens: int = 8192

    # --- моделі --------------------------------------------------------
    embedding_model: str | None = None      # None → дефолт реєстру
    rerank_model: str | None = None
    use_reranker: bool = True

    # --- воркери -------------------------------------------------------
    # "process" — постачання: важкі імпорти живуть у окремому процесі.
    # "inline"  — розробка, CI і stub-режим: конвеєр крутиться в пулі потоків
    #             API-процесу. Дозволено лише тому, що в цих режимах у конвеєрі
    #             немає ні torch, ні docling; у постачанні це заборонено.
    # "off"     — API без індексації (наприклад, діагностика).
    worker_mode: WorkerMode = "process"
    worker_count: int = 1
    # Алокатор torch не повертає пам'ять ОС: воркер після 40 документів тримає
    # ~3 ГБ. Переробка процесу після N завдань — єдиний надійний спосіб її
    # повернути (для потоків він недоступний у принципі).
    worker_recycle_after: int = 20
    worker_poll_interval_s: float = 1.0
    default_ingest_mode: IngestMode = IngestMode.FAST
    # Індексація за замовчуванням на CPU: GPU вже зайнято LM Studio, і
    # конкуренція за VRAM дає OOM саме на завантаженні LLM.
    ingest_device: Literal["cpu", "cuda", "mps"] = "cpu"

    # --- SSE -----------------------------------------------------------
    # Кільцевий буфер подій для Last-Event-ID. 512 подій ≈ хвилина активної
    # індексації — достатньо, щоб пережити перепідключення WebView.
    event_buffer: int = 512
    event_keepalive_s: float = 15.0
    # Період опитування таблиці jobs, з якого народжуються job.progress.
    # Воркер — окремий процес, тож він не може писати в шину API напряму;
    # єдине спільне сховище — SQLite, і саме воно є джерелом істини прогресу.
    job_watch_interval_s: float = 0.5

    # --- завантаження файлів -------------------------------------------
    max_upload_bytes: int = 4 * 1024 * 1024 * 1024   # 4 ГіБ
    upload_chunk_bytes: int = 1024 * 1024

    # --- діагностика ---------------------------------------------------
    # ВИМКНЕНО за замовчуванням: це військова академія, і назви матеріалів
    # самі по собі є чутливими. Вміст документів у пакет не потрапляє НІКОЛИ.
    diagnostics_include_titles: bool = False

    def paths(self) -> Paths:
        """Каталоги даних, уже створені на диску."""
        return Paths.resolve(self.data_dir).ensure()

    def describe(self) -> dict[str, object]:
        """Знімок для /health і діагностичного пакета (без секретів)."""
        return {
            "stub": self.stub,
            "host": self.host,
            "port": self.port,
            "worker_mode": self.worker_mode,
            "worker_count": self.worker_count,
            "worker_recycle_after": self.worker_recycle_after,
            "default_ingest_mode": self.default_ingest_mode.value,
            "ingest_device": self.ingest_device,
            "context_tokens": self.context_tokens,
            "embedding_model": self.embedding_model,
            "rerank_model": self.rerank_model,
            "use_reranker": self.use_reranker,
            "lmstudio_url": self.lmstudio_url,
        }


def get_settings() -> Settings:
    """Прочитати середовище наново.

    Свідомо БЕЗ кешу: тест і воркер-підпроцес мають бачити те середовище, яке
    їм щойно виставили, а не те, що прочитав перший імпорт модуля.
    """
    return Settings()
