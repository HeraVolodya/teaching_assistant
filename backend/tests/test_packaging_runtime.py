"""Збірка Python-рантайму: чисті функції, які можна перевірити без uv.

Сама збірка потребує мережі й uv, тому вона перевіряється в CI. Тут
фіксуються рішення, помилка в яких коштує найдорожче:

  * розкладка інтерпретатора мусить збігатися з тим, що шукає оболонка —
    інакше застосунок не запускається взагалі;
  * `install_name_tool` мусить чіпати лише те, що справді зламається після
    перенесення в `.app` (системні бібліотеки абсолютні законно);
  * бюджет розміру мусить ловити випадкове CUDA-колесо torch.
"""

from __future__ import annotations

import inspect
import re
import tomllib
from pathlib import Path

from helpers_packaging import REPO_ROOT, TAURI_DIR, read, script

runtime = script("build_runtime")
budget = script("check_size_budget")

BACKEND_DIR = REPO_ROOT / "backend"
PYPROJECT = tomllib.loads(read(BACKEND_DIR / "pyproject.toml"))


def _declared(dependencies: list[str]) -> set[str]:
    """Імена дистрибутивів із рядків PEP 508, нормалізовані за PEP 503."""
    return {
        re.split(r"[<>=!~\[;\s]", spec, maxsplit=1)[0].strip().lower().replace("_", "-")
        for spec in dependencies
    }


# ------------------------------------------------------------- розкладка
def test_шлях_інтерпретатора_збігається_з_оболонкою() -> None:
    root = Path("/rt")
    assert runtime.python_executable(root, "Windows") == root / "python.exe"
    assert runtime.python_executable(root, "Darwin") == root / "bin" / "python3"
    assert runtime.python_executable(root, "Linux") == root / "bin" / "python3"

    # Дзеркало в Rust. Розбіжність = «Python-рантайм не знайдено» на старті.
    sidecar = read(TAURI_DIR / "src" / "sidecar.rs")
    assert 'runtime_dir.join("python.exe")' in sidecar
    assert 'join("bin").join("python3")' in sidecar


# --------------------------------------------------------------- залежності
def test_multipart_оголошено_бо_api_приймає_форми() -> None:
    """FastAPI вимагає `python-multipart` на РЕЄСТРАЦІЇ маршруту, не на запиті.

    Тобто будь-який `Form(...)`/`UploadFile` без цієї залежності — не помилка
    завантаження файлу, а RuntimeError на імпорті `app.api`, тобто застосунок,
    який не стартує. У `.venv` розробника пакет опинився транзитивно (fastapi
    та starlette згадують його в опційних extras), тому локально все працювало,
    а в чистому рантаймі `uv pip install backend[worker]` його не було — і
    падіння вилізло аж у запакованому інсталяторі.
    """
    api_dir = BACKEND_DIR / "app" / "api"
    users = [p.name for p in api_dir.glob("*.py") if re.search(r"\bUploadFile\b|\bForm\(", read(p))]
    assert users, "очікували, що API приймає форми — якщо ні, тест треба переписати"
    assert "python-multipart" in _declared(PYPROJECT["project"]["dependencies"]), (
        f"{', '.join(users)} використовують форми, але python-multipart не оголошено "
        "в backend/pyproject.toml — запакований застосунок помре на старті."
    )


def test_self_check_імпортує_весь_застосунок() -> None:
    """Гейт, якого бракувало: відсутня залежність роутера мусить валити ЗБІРКУ.

    Поки `self_check` імпортував лише `app.config` і `app.net_guard`, рантайм
    без `python-multipart` спокійно доїжджав до інсталятора, і причина
    з'ясовувалась аж у димовому тесті — цілий цикл CI (~30 хв) замість двох
    секунд тут. `app.main` тягне всі роутери, тому саме він і мусить бути в
    перевірці, разом із `uvicorn`, якого вимагає `python -m app.main`.

    Звіряємо КОНСТАНТУ, а не текст функції: перевірка підрядка у
    `inspect.getsource` проходила б і тоді, коли `app.main` згадано лише в
    коментарі.
    """
    assert "app.main" in runtime.SELF_CHECK_CODE
    assert "uvicorn" in runtime.SELF_CHECK_CODE
    # `len(app.routes)` під FastAPI 0.141 дорівнює 8 незалежно від кількості
    # ендпоїнтів — число, яке має вигляд перевірки й не перевіряє нічого.
    assert "openapi()" in runtime.SELF_CHECK_CODE
    assert inspect.getsource(runtime.self_check).count("SELF_CHECK_CODE") == 1


def test_self_check_звіряє_контракт_димового_тесту() -> None:
    """Перейменований маршрут мусить валити ЗБІРКУ, а не димовий тест.

    `smoke_installer.py` стукає у фіксовані шляхи; якщо роутер перейменують,
    зараз це видно лише через пів години, вже на запакованому інсталяторі, як
    «Ендпоїнт не знайдено — контракт API змінився».
    """
    smoke = script("smoke_installer")
    expected = {
        smoke.HEALTH_PATH,
        smoke.ASSISTANTS_PATH,
        smoke.UPLOAD_PATH,
        smoke.SESSIONS_PATH,
        smoke.CHAT_PATH,
    }
    assert expected <= set(runtime.SMOKE_ENDPOINTS), (
        "self_check не звіряє всі шляхи, якими користується димовий тест: "
        f"бракує {sorted(expected - set(runtime.SMOKE_ENDPOINTS))}"
    )


def test_плейсхолдер_моделей_створюється(tmp_path: Path) -> None:
    # bundle.resources посилається на теку моделей; порожня тека ламає пакування,
    # а варіант «моделі на окремому носії» цілком законний.
    runtime.ensure_models_placeholder(tmp_path)
    readme = tmp_path / "models" / "README.txt"
    assert readme.exists()
    assert "manifest.json" in readme.read_text(encoding="utf-8")


# ---------------------------------------------------------------- прибирання
def test_прибираємо_кеш_байткоду_і_idle(tmp_path: Path) -> None:
    assert runtime.is_prunable(tmp_path / "__pycache__")
    assert runtime.is_prunable(tmp_path / "idlelib")
    (tmp_path / "x.pyc").write_bytes(b"")
    assert runtime.is_prunable(tmp_path / "x.pyc")


def test_тести_вирізаються_лише_зі_stdlib(tmp_path: Path) -> None:
    stdlib = tmp_path / "lib" / "python3.12"
    site = stdlib / "site-packages" / "numpy"
    site.mkdir(parents=True)
    (stdlib / "test").mkdir(parents=True, exist_ok=True)
    (site / "tests").mkdir()

    assert runtime.is_prunable(stdlib / "test", stdlib_root=stdlib)
    # У site-packages `tests` бувають частиною публічного API пакета.
    assert not runtime.is_prunable(site / "tests", stdlib_root=stdlib)


def test_prune_звільняє_місце(tmp_path: Path) -> None:
    cache = tmp_path / "lib" / "python3.12" / "__pycache__"
    cache.mkdir(parents=True)
    (cache / "a.cpython-312.pyc").write_bytes(b"x" * 1024)
    freed = runtime.prune(tmp_path)
    assert freed >= 1024
    assert not cache.exists()


def test_пошук_кореня_stdlib(tmp_path: Path) -> None:
    (tmp_path / "lib" / "python3.12").mkdir(parents=True)
    assert runtime.find_stdlib_root(tmp_path) == tmp_path / "lib" / "python3.12"
    assert runtime.find_stdlib_root(tmp_path / "порожньо") is None


# -------------------------------------------------------------------- macOS
OTOOL_SAMPLE = """/rt/bin/python3.12:
\t/install/lib/libpython3.12.dylib (compatibility version 3.12.0, current version 3.12.0)
\t@rpath/libssl.3.dylib (compatibility version 3.0.0, current version 3.0.0)
\t/usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 1345.0.0)
"""


def test_розбір_otool() -> None:
    paths = runtime.parse_otool_load_paths(OTOOL_SAMPLE)
    assert paths == [
        "/install/lib/libpython3.12.dylib",
        "@rpath/libssl.3.dylib",
        "/usr/lib/libSystem.B.dylib",
    ]


def test_релокації_потребує_лише_абсолютний_нессистемний_шлях() -> None:
    # Саме він ламається після перенесення в Asistent.app/Contents/Resources.
    assert runtime.needs_relocation("/install/lib/libpython3.12.dylib")
    # Системні бібліотеки справді лежать за абсолютними шляхами.
    assert not runtime.needs_relocation("/usr/lib/libSystem.B.dylib")
    assert not runtime.needs_relocation("/System/Library/Frameworks/CoreFoundation")
    assert not runtime.needs_relocation("@rpath/libssl.3.dylib")
    assert not runtime.needs_relocation("@executable_path/../lib/libpython3.12.dylib")


def test_індекс_бібліотек_віддає_найближчу_до_кореня(tmp_path: Path) -> None:
    shallow = tmp_path / "lib" / "libssl.3.dylib"
    deep = tmp_path / "lib" / "python3.12" / "site-packages" / "w" / "libssl.3.dylib"
    for path in (shallow, deep):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
    # Копія з колеса не має перемагати системну бібліотеку рантайму.
    assert runtime.library_index(tmp_path)["libssl.3.dylib"] == shallow


def test_переписаний_шлях_рахується_від_самого_бінарника(tmp_path: Path) -> None:
    """`@loader_path`, бо `@executable_path` залежить від того, ХТО вантажить.

    Стара арифметика (`'../' * (глибина - 1)` від `@executable_path`) була
    правильною лише для глибини 2. Для `bin/python3` і `lib/libpython3.12.dylib`
    — тобто для двох файлів, заради яких функція й існує, — вона давала
    `@executable_path/lib/…`, тобто `runtime/bin/lib/…`, шлях у порожнечу.
    """
    lib = tmp_path / "lib" / "libpython3.12.dylib"
    lib.parent.mkdir(parents=True)
    lib.write_bytes(b"")
    index = runtime.library_index(tmp_path)
    absolute = "/install/lib/libpython3.12.dylib"

    # Інтерпретатор: runtime/bin/python3 → ../lib/libpython3.12.dylib
    assert runtime.relocated_load_path(absolute, tmp_path / "bin" / "python3", index, tmp_path) == (
        "@loader_path/../lib/libpython3.12.dylib"
    )
    # Сусід у тій самій теці — без жодного `..`.
    assert runtime.relocated_load_path(absolute, lib.parent / "libssl.3.dylib", index, tmp_path) == (
        "@loader_path/libpython3.12.dylib"
    )
    # Модуль, закопаний у site-packages.
    deep = tmp_path / "lib" / "python3.12" / "site-packages" / "numpy" / "core.so"
    assert runtime.relocated_load_path(absolute, deep, index, tmp_path) == (
        "@loader_path/../../../libpython3.12.dylib"
    )


def test_невідома_бібліотека_шукається_в_lib(tmp_path: Path) -> None:
    # Нічого не знайшли — лишається єдине розумне припущення, `runtime/lib`.
    assert runtime.relocated_load_path(
        "/nowhere/libfoo.dylib", tmp_path / "bin" / "python3", {}, tmp_path
    ) == "@loader_path/../lib/libfoo.dylib"


def test_фікс_install_names_ігнорується_поза_macos(tmp_path: Path) -> None:
    import platform

    if platform.system() != "Darwin":
        assert runtime.fix_macos_install_names(tmp_path) == 0


# -------------------------------------------------------------------- бюджет
def test_бюджет_ловить_роздутий_артефакт(tmp_path: Path, capsys) -> None:
    big = tmp_path / "runtime"
    big.mkdir()
    (big / "torch_cuda.so").write_bytes(b"0" * (3 * 1024 * 1024))

    assert budget.main([str(big), "--max-mb", "10"]) == 0
    assert budget.main([str(big), "--max-mb", "1"]) == 1
    assert "torch" in capsys.readouterr().err


def test_бюджет_рантайму_консервативний() -> None:
    # CPU-torch — ~124 МБ (win) / ~127 МБ (macOS arm64). Бюджет мусить лишати
    # місце для нього і не лишати — для CUDA-варіанта в кілька гігабайтів.
    assert 800 <= runtime.DEFAULT_MAX_MB <= 2000
    assert runtime.DEFAULT_MAX_MB_NO_WORKER < runtime.DEFAULT_MAX_MB


def test_розмір_каталогу_рахує_лише_файли(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "a").write_bytes(b"x" * 10)
    assert runtime.dir_size(tmp_path) == 10
    assert budget.size_of(tmp_path) == 10
    assert runtime.human(10) == "10.0 Б"
