"""Налаштування процесу: читання середовища, шляхи, знімок для діагностики."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.domain import IngestMode
from app.settings import Settings, get_settings


def test_env_prefix_matches_the_contract(monkeypatch, tmp_path) -> None:
    """`ASISTENT_STUB=1` — це рядок із контракту, а не деталь реалізації:
    саме його виставляє ярлик, CI і майстер першого запуску."""
    monkeypatch.setenv("ASISTENT_STUB", "1")
    monkeypatch.setenv("ASISTENT_PORT", "9000")
    monkeypatch.setenv("ASISTENT_WORKER_COUNT", "3")
    settings = get_settings()
    assert settings.stub is True
    assert settings.port == 9000
    assert settings.worker_count == 3


def test_data_dir_env_is_shared_with_config(monkeypatch, tmp_path) -> None:
    """`ASISTENT_DATA_DIR` читає і `Settings`, і `app.config.Paths`.

    Два різні імена для однієї речі — найдешевший спосіб отримати БД у двох
    каталогах одночасно, тож ім'я має бути рівно одне.
    """
    monkeypatch.setenv("ASISTENT_DATA_DIR", str(tmp_path / "дані"))
    settings = get_settings()
    paths = settings.paths()
    assert paths.data_dir == (tmp_path / "дані").resolve()
    assert paths.db_path.parent == paths.data_dir
    # Каталоги створюються одразу: воркер не має падати на відсутній теці.
    for directory in (paths.documents_dir, paths.index_dir, paths.models_dir):
        assert directory.is_dir()


@pytest.mark.parametrize("value", [None, "", "   "])
def test_unset_data_dir_falls_back_to_the_platform_default(monkeypatch, value) -> None:
    """Незадана `ASISTENT_DATA_DIR` мусить дати платформний каталог, а НЕ CWD.

    Регресія: `Path(os.environ.get(...)) or _default_data_dir()` виглядає як
    фолбек, але `Path("")` дорівнює `Path(".")` і є істинним, тож фолбек був
    недосяжним, а каталогом даних тихо ставав поточний робочий каталог. При
    запуску з `backend/` це клало `assistant.db` і мультигігабайтні моделі
    просто в репозиторій — де їх не ловить жоден запис у `.gitignore`.

    Усі інші тести виставляють змінну явно, тому вада й пережила весь набір.
    """
    from app.config import Paths, _default_data_dir

    if value is None:
        monkeypatch.delenv("ASISTENT_DATA_DIR", raising=False)
    else:
        monkeypatch.setenv("ASISTENT_DATA_DIR", value)

    base = Paths.resolve().data_dir
    assert base == _default_data_dir().expanduser().resolve()
    assert base != Path.cwd()


def test_get_settings_is_not_cached(monkeypatch) -> None:
    """Тест і воркер-підпроцес мусять бачити те середовище, яке їм щойно
    виставили, а не те, що прочитав перший імпорт модуля."""
    monkeypatch.setenv("ASISTENT_PORT", "8001")
    assert get_settings().port == 8001
    monkeypatch.setenv("ASISTENT_PORT", "8002")
    assert get_settings().port == 8002


def test_defaults_are_the_safe_ones(tmp_path) -> None:
    settings = Settings(data_dir=tmp_path)
    # Слухати ТІЛЬКИ петлю назад: 0.0.0.0 у закритому контурі — це відкритий
    # доступ до чужих навчальних матеріалів із локальної мережі академії.
    assert settings.host == "127.0.0.1"
    # Індексація на CPU: GPU вже зайнято LM Studio.
    assert settings.ingest_device == "cpu"
    # У постачанні воркер — окремий процес (GIL, ізоляція, скасовуваність).
    assert settings.worker_mode == "process"
    assert settings.default_ingest_mode is IngestMode.FAST
    # Назви документів у діагностиці вимкнені: це військова академія.
    assert settings.diagnostics_include_titles is False


def test_describe_hides_nothing_it_should_not_and_shows_the_mode(tmp_path) -> None:
    settings = Settings(data_dir=tmp_path, stub=True, worker_mode="inline")
    snapshot = settings.describe()
    assert snapshot["stub"] is True
    assert snapshot["worker_mode"] == "inline"
    assert snapshot["default_ingest_mode"] == "FAST"
    assert "data_dir" not in snapshot   # шлях може містити ім'я користувача


@pytest.mark.parametrize("value", ["1", "true", "on", "yes"])
def test_stub_accepts_the_usual_truthy_spellings(monkeypatch, value: str) -> None:
    monkeypatch.setenv("ASISTENT_STUB", value)
    assert get_settings().stub is True


def test_unknown_env_keys_do_not_break_startup(monkeypatch) -> None:
    """`extra="ignore"`: чужа змінна з тим самим префіксом не має валити
    старт застосунку на машині викладача."""
    monkeypatch.setenv("ASISTENT_НЕВІДОМЕ_ПОЛЕ", "42")
    assert get_settings() is not None
