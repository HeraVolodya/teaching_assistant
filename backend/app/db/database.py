"""З'єднання з SQLite, PRAGMA й міграції.

SQLite — джерело істини для ВСЬОГО. ANN-індекс USearch — перебудовуваний
сайдкар: вектори дорого перераховувати (години локального CPU), тож вони
транзакційно довговічні; HNSW-граф дешевий (хвилини), тож він кеш.

Топологія WAL: один письменник + N читачів — це рівно топологія
«воркер індексації + потік запитів».
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from app.config import SQLITE_PAGE_SIZE, SQLITE_PRAGMAS

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"

_local = threading.local()


def _apply_pragmas(con: sqlite3.Connection, *, first_write: bool = False) -> None:
    if first_write:
        # page_size можна змінити лише до першого запису.
        con.execute(f"PRAGMA page_size = {SQLITE_PAGE_SIZE}")
    for name, value in SQLITE_PRAGMAS:
        con.execute(f"PRAGMA {name} = {value}")


def connect(db_path: Path | str, *, read_only: bool = False) -> sqlite3.Connection:
    path = Path(db_path)
    fresh = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)

    if read_only:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
    else:
        con = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
    con.row_factory = sqlite3.Row
    if not read_only:
        _apply_pragmas(con, first_write=fresh)
    return con


def migrate(con: sqlite3.Connection) -> int:
    """Застосувати нові міграції. Повертає підсумкову версію схеми."""
    current = 0
    try:
        row = con.execute("SELECT max(version) FROM schema_version").fetchone()
        current = int(row[0]) if row and row[0] is not None else 0
    except sqlite3.OperationalError:
        current = 0

    files = sorted(MIGRATIONS_DIR.glob("*.sql"))
    for f in files:
        version = int(f.name.split("_", 1)[0])
        if version <= current:
            continue
        con.executescript(f.read_text(encoding="utf-8"))
        # 0001 сам вставляє свій рядок; наступні мають робити це через runner.
        row = con.execute("SELECT max(version) FROM schema_version").fetchone()
        if not row or row[0] is None or int(row[0]) < version:
            con.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))
        current = version
    return current


class Database:
    """Тонка обгортка. Одне з'єднання на потік — SQLite-з'єднання не
    потокобезпечні для одночасного використання, а WAL дає паралельних читачів."""

    def __init__(self, db_path: Path | str) -> None:
        self.path = Path(db_path)
        with self.connection() as con:
            migrate(con)

    def _thread_connection(self) -> sqlite3.Connection:
        con = getattr(_local, "connections", {}).get(str(self.path))
        if con is None:
            con = connect(self.path)
            if not hasattr(_local, "connections"):
                _local.connections = {}
            _local.connections[str(self.path)] = con
        return con

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        yield self._thread_connection()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Явна транзакція. Індексація має комітити НЕВЕЛИКИМИ пачками:
        довга транзакція блокує інших письменників."""
        con = self._thread_connection()
        con.execute("BEGIN IMMEDIATE")
        try:
            yield con
        except Exception:
            con.execute("ROLLBACK")
            raise
        else:
            con.execute("COMMIT")

    def checkpoint(self, mode: str = "TRUNCATE") -> tuple[int, int, int]:
        """Перелити WAL в основний файл БД.

        Без цього `assistant.db` лишається порожньою шкаралупою: спостережено
        на живій машині — основний файл 8 КБ і change-counter 1 (жодного
        запису після створення), тоді як УСЯ база разом зі схемою жила у
        WAL на 1.3 МБ. `wal_autocheckpoint = 2000` сторінок при page_size
        8192 — це поріг у 16 МБ, якого звичайна робота викладача просто
        не досягає.

        Наслідки, поки цього не було:
          * копіювання `assistant.db` як резервної копії давало ПОРОЖНЮ базу;
          * втрата чи розсинхронізація WAL знищувала все;
          * `PRAGMA integrity_check` перевіряв шкаралупу, а не дані.

        TRUNCATE, а не PASSIVE: він ще й обнуляє файл WAL, тож після
        коректного завершення на диску лишається один самодостатній файл —
        рівно те, що обіцяє інструкція з резервного копіювання.
        """
        con = self._thread_connection()
        row = con.execute(f"PRAGMA wal_checkpoint({mode})").fetchone()
        return (int(row[0]), int(row[1]), int(row[2])) if row else (0, 0, 0)

    def close(self) -> None:
        conns = getattr(_local, "connections", {})
        con = conns.pop(str(self.path), None)
        if con is not None:
            try:
                con.execute("PRAGMA optimize")
                # Перед закриттям — обов'язково, інакше дані лишаться у WAL.
                con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except Exception:
                pass
            finally:
                con.close()

    def integrity_check(self) -> str:
        """Доказова база для ДСТУ ISO/IEC 27001:2023 (контроль цілісності даних)."""
        with self.connection() as con:
            return con.execute("PRAGMA integrity_check").fetchone()[0]
