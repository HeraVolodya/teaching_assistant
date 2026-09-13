"""Гейт довговічності: основний файл БД мусить містити дані.

Спостережено на живій машині: `assistant.db` — 8192 байти, change-counter 1
(жодного запису після створення), тоді як `assistant.db-wal` — 1.3 МБ, і
саме там жила УСЯ база разом зі схемою. Причина: `wal_autocheckpoint = 2000`
сторінок при page_size 8192 — це поріг 16 МБ, якого робота викладача не
досягає.

Наслідок, який робить це не косметикою: інструкція обіцяє резервну копію
одним файлом, а копія `assistant.db` була б ПОРОЖНЯ.
"""

from __future__ import annotations

import sqlite3
import struct
from pathlib import Path

from app.db.database import Database
from app.db.repositories import AssistantRepo
from app.domain import Assistant, new_id


def _change_counter(db_path: Path) -> int:
    """Лічильник змін із заголовка БД. 1 = файл не писався після створення."""
    with open(db_path, "rb") as f:
        return struct.unpack(">I", f.read(100)[24:28])[0]


def test_close_leaves_all_data_in_the_main_file(tmp_path: Path) -> None:
    """Після close() копія одного файлу — повноцінна база."""
    path = tmp_path / "a.db"
    db = Database(path)
    with db.transaction() as con:
        for i in range(50):
            AssistantRepo(con).create(Assistant(id=new_id(), name=f"Асистент {i}"))
    db.close()

    wal = path.with_name(path.name + "-wal")
    assert not wal.exists() or wal.stat().st_size == 0, (
        "після close() WAL має бути обнулений — інакше дані лишаються поза основним файлом"
    )
    assert _change_counter(path) > 1, "основний файл БД так і не був записаний"

    # Найголовніше: копія ОДНОГО файлу має читатися самостійно.
    copy = tmp_path / "backup.db"
    copy.write_bytes(path.read_bytes())
    con = sqlite3.connect(copy)
    assert con.execute("SELECT count(*) FROM assistants").fetchone()[0] == 50
    con.close()


def test_explicit_checkpoint_flushes_wal(tmp_path: Path) -> None:
    path = tmp_path / "b.db"
    db = Database(path)
    with db.transaction() as con:
        AssistantRepo(con).create(Assistant(id=new_id(), name="Балістика"))

    busy, log_pages, checkpointed = db.checkpoint()
    assert busy == 0, "контрольна точка не змогла виконатись"
    assert checkpointed >= 0
    assert _change_counter(path) > 1
    db.close()


def test_autocheckpoint_threshold_is_reachable(tmp_path: Path) -> None:
    """Поріг має бути таким, щоб звичайна робота його досягала.

    2000 сторінок × 8 КБ = 16 МБ — база викладача може роками до цього не
    дорости, і весь цей час основний файл лишатиметься порожнім.
    """
    db = Database(tmp_path / "c.db")
    with db.connection() as con:
        pages = con.execute("PRAGMA wal_autocheckpoint").fetchone()[0]
        page_size = con.execute("PRAGMA page_size").fetchone()[0]
    db.close()
    threshold_mb = pages * page_size / 2**20
    assert threshold_mb <= 4, (
        f"поріг контрольної точки {threshold_mb:.0f} МБ завеликий: "
        "основний файл БД лишатиметься порожнім місяцями"
    )


def test_integrity_check_runs_against_real_data(tmp_path: Path) -> None:
    """PRAGMA integrity_check має перевіряти дані, а не порожню шкаралупу.

    Це доказова база для ДСТУ ISO/IEC 27001:2023 (контроль цілісності).
    """
    db = Database(tmp_path / "d.db")
    with db.transaction() as con:
        AssistantRepo(con).create(Assistant(id=new_id(), name="Тест"))
    db.checkpoint()
    assert db.integrity_check() == "ok"
    with db.connection() as con:
        assert con.execute("SELECT count(*) FROM assistants").fetchone()[0] == 1
    db.close()
