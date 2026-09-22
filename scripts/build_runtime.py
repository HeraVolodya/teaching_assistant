#!/usr/bin/env python3
"""Збірка релокованого Python-рантайму для бандла Tauri (uv + python-build-standalone).

## Чому НЕ PyInstaller

1. **Хибні спрацювання антивірусу.** Самозакорковуваний бутстрап PyInstaller —
   рівно той механізм, який евристики Defender/Avast позначають як пакувальник
   шкідливого ПЗ. Ми постачаємо державній установі підписаний бінарник; одне
   спрацювання на доменній машині зупиняє пілот на тижні.
2. **Розпакування в temp при кожному старті** — 3–10 с до першого байта, щоразу.
3. **Дерево з двох процесів.** Бутстрап породжує дитину, і оболонка не може
   надійно завершити обох — це задокументовано в tauri#11686 як «tauri-plugin-shell
   не може повністю завершити виконуваний файл, згенерований PyInstaller».

python-build-standalone дає звичайний `python.exe` і звичайні DLL/.dylib: один
процес, передбачуваний `codesign`, нуль розпакування при старті.

## Пастка macOS

Бінарники python-build-standalone несуть **абсолютні** шляхи інсталяції у
LC_LOAD_DYLIB. Після перенесення в `Asistent.app/Contents/Resources/runtime`
такий шлях указує в порожнечу, і застосунок падає на старті з
`Library not loaded`. `install_name_tool` переписує їх на `@loader_path/…`
ДО підписання — після підписання будь-яка зміна бінарника ламає підпис.
Саме `@loader_path`, а не `@executable_path`: див. `relocated_load_path`.

Приклад:
    python scripts/build_runtime.py --out src-tauri/resources/runtime
    python scripts/build_runtime.py --out /tmp/rt --no-worker --max-size-mb 400
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PYTHON_VERSION = "3.12"
REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "backend"

# Бюджет розміру. Наявний, щоб CUDA-колесо torch не пролізло непоміченим:
# CPU-torch — ~124 МБ (win_amd64) / ~127 МБ (macOS arm64), CUDA-варіант —
# кілька гігабайтів. Перевищення бюджету валить збірку.
DEFAULT_MAX_MB = 1400
DEFAULT_MAX_MB_NO_WORKER = 320

# Що викидаємо з рантайму. Тести стандартної бібліотеки, IDLE і Tk — це
# ~40 МБ, які ніколи не виконуються в застосунку без GUI на Python.
PRUNE_DIR_NAMES = frozenset(
    {
        "__pycache__",
        "idlelib",
        "turtledemo",
        "tkinter",
        "ensurepip",
        "lib2to3",
        "pydoc_data",
    }
)
PRUNE_SUFFIXES = (".pyc", ".pyo", ".a", ".whl")
# Каталоги тестів у stdlib. Пакети сторонніх бібліотек не чіпаємо: у деяких
# (наприклад, numpy) тестові модулі імпортуються з рантайму.
STDLIB_TEST_DIRS = ("test", "tests")


def log(message: str) -> None:
    print(f"[build_runtime] {message}", file=sys.stderr)


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    log("$ " + " ".join(str(c) for c in cmd))
    return subprocess.run([str(c) for c in cmd], check=True, **kwargs)


# ---------------------------------------------------------------- розкладка
def python_executable(runtime_dir: Path, system: str | None = None) -> Path:
    """Шлях інтерпретатора всередині рантайму.

    ДЗЕРКАЛО `src-tauri/src/sidecar.rs::python_executable`. Якщо розійдуться —
    оболонка не знайде інтерпретатор, і застосунок не запуститься взагалі.
    """
    system = system or platform.system()
    if system == "Windows":
        return runtime_dir / "python.exe"
    return runtime_dir / "bin" / "python3"


def is_prunable(path: Path, *, stdlib_root: Path | None = None) -> bool:
    """Чи прибирати цей шлях із рантайму."""
    if path.name in PRUNE_DIR_NAMES:
        return True
    if path.is_file() and path.suffix in PRUNE_SUFFIXES:
        return True
    # `test`/`tests` вирізаємо ЛИШЕ безпосередньо у stdlib: у site-packages
    # такі теки бувають частиною публічного API пакета.
    return stdlib_root is not None and path.name in STDLIB_TEST_DIRS and path.parent == stdlib_root


def find_stdlib_root(runtime_dir: Path) -> Path | None:
    """Корінь стандартної бібліотеки: `lib/python3.12` на Unix, `Lib` на Windows.

    Порядок кандидатів важливий: файлові системи macOS за замовчуванням
    нечутливі до регістру, тому `Lib` там «існує» щоразу, коли існує `lib`, і
    перевірка в зворотному порядку повертала б каталог на рівень вище.
    """
    candidates = [runtime_dir / "lib" / f"python{PYTHON_VERSION}", runtime_dir / "Lib"]
    return next((c for c in candidates if c.is_dir()), None)


def prune(runtime_dir: Path) -> int:
    """Прибрати непотрібне. Повертає кількість звільнених байтів."""
    stdlib_root = find_stdlib_root(runtime_dir)
    freed = 0
    for path in sorted(runtime_dir.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if not path.exists():
            continue
        if not is_prunable(path, stdlib_root=stdlib_root):
            continue
        if path.is_dir():
            freed += dir_size(path)
            shutil.rmtree(path, ignore_errors=True)
        else:
            try:
                freed += path.stat().st_size
                path.unlink()
            except OSError:
                pass
    return freed


def dir_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file() and not p.is_symlink())


def human(n: int) -> str:
    value = float(n)
    for unit in ("Б", "КБ", "МБ", "ГБ"):
        if value < 1024 or unit == "ГБ":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} ГБ"


# ------------------------------------------------------------------- macOS
_OTOOL_LINE = re.compile(r"^\s+(\S+)\s+\(compatibility version")


def parse_otool_load_paths(output: str) -> list[str]:
    """Витягти шляхи LC_LOAD_DYLIB з виводу `otool -L`."""
    return [m.group(1) for line in output.splitlines() if (m := _OTOOL_LINE.match(line))]


def needs_relocation(load_path: str) -> bool:
    """Чи є цей шлях абсолютним посиланням, яке зламається після перенесення.

    Системні бібліотеки (`/usr/lib/…`, `/System/…`) абсолютні законно — вони
    справді там і є. Все інше абсолютне — це залишок машини, на якій збирали
    python-build-standalone.
    """
    if load_path.startswith(("@rpath", "@executable_path", "@loader_path")):
        return False
    if load_path.startswith(("/usr/lib/", "/System/")):
        return False
    return load_path.startswith("/")


def library_index(runtime_dir: Path) -> dict[str, Path]:
    """Індекс «ім'я файлу → шлях» для всіх бібліотек рантайму.

    Будується ОДИН раз: `rglob` на кожен із тисяч бінарників перетворив би
    збірку на хвилини. Якщо ім'я трапляється кілька разів, виграє найближче до
    кореня — тобто `lib/libssl.3.dylib`, а не його копія, закопана в колесі.
    """
    index: dict[str, Path] = {}
    for path in runtime_dir.rglob("*"):
        if path.suffix not in {".dylib", ".so"} or path.is_symlink() or not path.is_file():
            continue
        known = index.get(path.name)
        if known is None or len(path.parts) < len(known.parts):
            index[path.name] = path
    return index


def relocated_load_path(
    load_path: str, binary: Path, index: dict[str, Path], runtime_dir: Path
) -> str:
    """Чим замінити абсолютний `load_path` у `binary`.

    ЧОМУ `@loader_path`, А НЕ `@executable_path`.
    `@executable_path` — тека ВИКОНУВАНОГО ФАЙЛУ ПРОЦЕСУ, тобто `runtime/bin`
    для `python3`. Тому відносний шлях від нього залежить не від бінарника,
    який ми правимо, а від того, хто його вантажить, — і воркер, запущений
    іншим шляхом, дістав би інший корінь. `@loader_path` — тека самого
    бінарника, тож шлях лишається правильним завжди.

    ЩО ТУТ БУЛО ЗЛАМАНО. Попередня арифметика —
    `'@executable_path/' + '../' * (len(parts) - 2) + f'lib/{name}'` — давала
    правильний результат рівно для глибини 2 і мовчки промахувалась для решти.
    Зокрема для `bin/python3` і для `lib/libpython3.12.dylib` вона повертала
    `@executable_path/lib/…`, тобто `runtime/bin/lib/…`, — шлях у порожнечу, і
    саме для тих двох файлів, заради яких уся ця функція й написана. Помилка
    не проявилась лише тому, що сучасний python-build-standalone уже приїздить
    із `@rpath`-іменами й переписувати не було чого; наступна зміна апстріму
    зробила б її фатальною на машині викладача, а не в CI.

    Бібліотеку шукаємо в індексі, а не припускаємо `lib/<ім'я>`: колеса PyPI
    возять власні `.dylib` поруч зі своїми `.so`, і для них `lib/` — хибна
    адреса.
    """
    name = Path(load_path).name
    target = index.get(name, runtime_dir / "lib" / name)
    relative = os.path.relpath(target, binary.parent).replace(os.sep, "/")
    return f"@loader_path/{relative}"


def fix_macos_install_names(runtime_dir: Path) -> int:
    """Переписати абсолютні install names на `@loader_path`. Повертає кількість правок."""
    if platform.system() != "Darwin":
        return 0

    binaries = [p for p in runtime_dir.rglob("*") if p.is_file() and not p.is_symlink()]
    targets = [p for p in binaries if p.suffix in {".dylib", ".so"} or (p.parent.name == "bin" and os.access(p, os.X_OK))]
    index = library_index(runtime_dir)
    fixed = 0

    for binary in targets:
        try:
            out = subprocess.run(
                ["otool", "-L", str(binary)], capture_output=True, text=True, check=True
            ).stdout
        except (subprocess.CalledProcessError, OSError):
            continue  # не Mach-O — otool просто не має що сказати

        for load_path in parse_otool_load_paths(out):
            if not needs_relocation(load_path):
                continue
            new = relocated_load_path(load_path, binary, index, runtime_dir)
            try:
                run(["install_name_tool", "-change", load_path, new, str(binary)])
                fixed += 1
            except subprocess.CalledProcessError:
                log(f"УВАГА: не вдалося переписати {load_path} у {binary}")

        if binary.suffix == ".dylib":
            try:
                run(["install_name_tool", "-id", f"@rpath/{binary.name}", str(binary)])
            except subprocess.CalledProcessError:
                pass

    if fixed:
        log(f"Переписано абсолютних install names: {fixed}. Підписувати ЛИШЕ після цього.")
    return fixed


# ---------------------------------------------------------------- складання
def install_python(out_dir: Path, uv: str) -> Path:
    """Поставити python-build-standalone через uv і розкласти його в `out_dir`."""
    staging = out_dir.parent / f".{out_dir.name}-pbs"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    # uv тягне python-build-standalone і кладе його у версійовану теку
    # виду `cpython-3.12.x-macos-aarch64-none`.
    run([uv, "python", "install", "--install-dir", str(staging), PYTHON_VERSION])

    # ДЕДУПЛІКАЦІЯ ЗА РЕАЛЬНИМ ШЛЯХОМ, А НЕ ПІДРАХУНОК КАТАЛОГІВ.
    # uv кладе поруч зі справжнім каталогом ще й аліас мінорної версії:
    # `cpython-3.12-windows-x86_64-none` → `cpython-3.12.14-windows-x86_64-none`.
    # Для `Path.is_dir()` симлінк на каталог — теж каталог, тож наївний підрахунок
    # бачив два рантайми й валив збірку на обох ОС одразу. Рахувати треба
    # РЕАЛЬНІ каталоги: `resolve()` зводить аліас і ціль в один шлях, і це
    # однаково працює для симлінка (macOS, Git for Windows) і для junction
    # (NTFS), а не покладається на `is_symlink()`, який junction не бачить.
    candidates = [p for p in staging.iterdir() if p.is_dir() and p.name.startswith("cpython-")]
    installed = sorted({p.resolve() for p in candidates})
    if len(installed) != 1:
        raise SystemExit(
            f"Очікували рівно один рантайм у {staging}, знайшли: {installed} "
            f"(кандидати до зведення аліасів: {[p.name for p in candidates]})"
        )

    if out_dir.exists():
        shutil.rmtree(out_dir)
    # Переносимо саме ціль, а не аліас: інакше на місце рантайму ліг би
    # симлінк, і в інсталятор поїхало б порожнє посилання замість Python.
    shutil.move(str(installed[0]), str(out_dir))
    # Аліас лишився висіти в staging — прибираємо разом із нею. `ignore_errors`
    # саме через нього: видалення обірваного посилання на Windows буває гучним.
    shutil.rmtree(staging, ignore_errors=True)
    _drop_externally_managed(out_dir)
    return out_dir


def _drop_externally_managed(runtime_dir: Path) -> None:
    """Прибрати маркер PEP 668, який `uv` лишає у власних рантаймах.

    ЧОМУ ЦЕ НЕ ОБХІД ЗАХИСТУ, А ВИПРАВЛЕННЯ ХИБНИХ МЕТАДАНИХ.
    `uv python install` кладе у рантайм файл `EXTERNALLY-MANAGED` з текстом
    «This Python installation is managed by uv and should not be modified»,
    і наступний `uv pip install --python …` через нього відмовляє:

        error: The interpreter at … is externally managed

    Твердження маркера правдиве, доки рантайм лежить у сховищі `uv`. Ми ж
    щойно ВИНЕСЛИ його в `src-tauri/resources/runtime`: це вже не керована
    копія, а постачальний рантайм застосунку, і встановлення бекенду в нього
    — весь сенс цього скрипта.

    Тому саме видалення, а не `--break-system-packages`: нічого не ламається,
    просто метадані перестали відповідати дійсності після переносу. Інакше
    хибний маркер поїхав би ще й в інсталятор і блокував би будь-яке майбутнє
    обслуговування рантайму на машині викладача.

    Шлях різний за платформами (`Lib/` на Windows, `lib/pythonX.Y/` на Unix),
    тож шукаємо глобом, а не складаємо вручну.
    """
    for marker in runtime_dir.rglob("EXTERNALLY-MANAGED"):
        marker.unlink()
        log(f"Прибрано маркер PEP 668: {marker.relative_to(runtime_dir)}")


def install_packages(runtime_dir: Path, uv: str, *, extras: list[str]) -> None:
    py = python_executable(runtime_dir)
    if not py.exists():
        raise SystemExit(f"Інтерпретатор не знайдено після встановлення: {py}")
    spec = str(BACKEND_DIR) + (f"[{','.join(extras)}]" if extras else "")
    # --no-cache: колесо з кешу може бути зібране під інший ABI; у нас цільова
    # машина інша за побудовою.
    run([uv, "pip", "install", "--python", str(py), "--no-cache", spec])


# Ендпоїнти, на які спирається димовий тест інсталятора
# (`scripts/smoke_installer.py`). Звіряються ТУТ, за дві секунди, а не через
# пів години після збірки: перейменований маршрут інакше проявився б як
# «Ендпоїнт не знайдено — контракт API змінився» вже на запакованому .exe.
SMOKE_ENDPOINTS = (
    "/api/health",
    "/api/assistants",
    "/api/collections/{collection_id}/documents",
    "/api/sessions",
    "/api/chat",
)

# Код самоперевірки — окремою константою, щоб тест пакування звіряв саме те,
# що виконається, а не підрядок вихідного тексту функції.
#
# `app.main.app.openapi()`, а НЕ `len(app.routes)`: під FastAPI 0.141 роутери
# лишаються вкладеними, тому `len(app.routes)` дорівнює 8 незалежно від того,
# скільки ендпоїнтів зареєстровано, — число, яке має вигляд перевірки, але не
# перевіряє нічого. Побудова схеми заодно матеріалізує кожен `response_model`,
# тобто ловить ще й помилки pydantic-моделей.
#
# `import uvicorn` тут тому, що `python -m app.main` імпортує його при старті:
# без цього рядка рантайм зі зламаним колесом uvicorn проходив би збірку і
# помирав аж у димовому тесті.
SELF_CHECK_CODE = (
    "import sys, uvicorn, app.main; from app import config; "
    f"expected = {SMOKE_ENDPOINTS!r}; "
    "paths = app.main.app.openapi()['paths']; "
    "missing = [p for p in expected if p not in paths]; "
    "assert not missing, ('немає ендпоїнтів ' + repr(missing) + '; є: ' + repr(sorted(paths))); "
    "print(sys.version.split()[0], config.APP_NAME, 'uvicorn', uvicorn.__version__, "
    "len(paths), 'ендпоїнтів')"
)


def self_check(runtime_dir: Path) -> None:
    """Перевірити, що рантайм запускається і що застосунок ЗБИРАЄТЬСЯ ЦІЛКОМ.

    ІМПОРТУЄМО `app.main`, А НЕ ЛИШЕ `app.config`. Попередня версія перевіряла
    два найлегші модулі й тому пропускала цілий клас відмов: відсутню
    залежність, потрібну комусь із роутерів. Саме так у постачання поїхав
    рантайм без `python-multipart` — FastAPI перевіряє його в момент
    РЕЄСТРАЦІЇ маршруту з `Form(...)`, тобто на імпорті `app.api`, і
    запакований застосунок помирав на старті. Помилка коштувала повного циклу
    CI (збірка інсталятора + димовий тест, ~30 хв) замість двох секунд тут.

    `import app.main` безпечний як перевірка: `create_app()` на рівні модуля
    лише реєструє роутери, а все, що торкається диска, бази й мережі, живе у
    `lifespan`, який виконується тільки під сервером. Порт не займається,
    uvicorn не стартує.

    `python -I`: ізольований режим викидає PYTHONPATH і site-packages
    користувача, тож перевіряється САМЕ вміст рантайму, а не те, що випадково
    лежить у середовищі машини збірки. Разом із цим `-I` включає `-E`, тобто
    ігнорує ВСІ змінні `PYTHON*`, — тому UTF-8 задається прапорцем `-X utf8`, а
    не `PYTHONUTF8`: інакше `APP_NAME` українською вбив би перевірку
    UnicodeEncodeError на windows-раннері. `ASISTENT_STUB` не починається з
    `PYTHON`, тож `-E` його не чіпає.
    """
    py = python_executable(runtime_dir)
    run([str(py), "-I", "-X", "utf8", "-c", SELF_CHECK_CODE], env={**os.environ, "ASISTENT_STUB": "1"})


def write_manifest(runtime_dir: Path, *, extras: list[str], size: int) -> dict:
    manifest = {
        "schema": 1,
        "python": PYTHON_VERSION,
        "platform": platform.system(),
        "machine": platform.machine(),
        "extras": extras,
        "bytes": size,
        "built_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
    }
    (runtime_dir / "runtime_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def ensure_models_placeholder(resources_dir: Path) -> None:
    """`bundle.resources` посилається на теку моделей — вона мусить існувати завжди.

    Порожня тека ламає пакування, а варіант постачання «моделі з USB» законний,
    тому кладемо пояснювальний файл.
    """
    models_dir = resources_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    readme = models_dir / "README.txt"
    if not readme.exists():
        readme.write_text(
            "Тека моделей Асістента.\n\n"
            "Якщо тут немає manifest.json — постачання виконано у варіанті\n"
            "«моделі на окремому носії». Майстер першого запуску попросить\n"
            "вказати теку моделей і перевірить її контрольні суми.\n"
            "Заповнити локально: python scripts/fetch_models.py --models-dir <ця тека>\n",
            encoding="utf-8",
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Зібрати релокований Python-рантайм.")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "src-tauri" / "resources" / "runtime")
    parser.add_argument("--uv", default=os.environ.get("UV", "uv"))
    parser.add_argument("--no-worker", action="store_true",
                        help="Без docling/torch/rapidocr — легка збірка для UI-тестів у CI.")
    parser.add_argument("--max-size-mb", type=int, default=None)
    parser.add_argument("--skip-self-check", action="store_true")
    args = parser.parse_args(argv)

    if shutil.which(args.uv) is None:
        raise SystemExit(
            f"Не знайдено {args.uv!r}. Встановіть uv: https://docs.astral.sh/uv/ "
            "(крос-компіляція неможлива — рантайм збирається на рідній ОС)."
        )

    extras: list[str] = []
    if not args.no_worker:
        extras.append("worker")
        if platform.system() == "Darwin":
            # Apple Vision через ocrmac: uk-UA підтримується, моделі возити не треба.
            extras.append("worker-macos")

    out_dir = args.out.expanduser().resolve()
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    log(f"Рантайм: {out_dir}  extras={extras or '—'}")
    install_python(out_dir, args.uv)
    install_packages(out_dir, args.uv, extras=extras)

    freed = prune(out_dir)
    log(f"Прибрано зайвого: {human(freed)}")
    fix_macos_install_names(out_dir)

    if not args.skip_self_check:
        self_check(out_dir)

    size = dir_size(out_dir)
    manifest = write_manifest(out_dir, extras=extras, size=size)
    ensure_models_placeholder(out_dir.parent)

    budget_mb = args.max_size_mb or (DEFAULT_MAX_MB_NO_WORKER if args.no_worker else DEFAULT_MAX_MB)
    log(f"Розмір рантайму: {human(size)} (бюджет {budget_mb} МБ)")
    if size > budget_mb * 1024 * 1024:
        raise SystemExit(
            f"Рантайм {human(size)} перевищує бюджет {budget_mb} МБ. "
            "Найімовірніша причина — колесо torch із CUDA-залежностями: "
            "вони мають маркер platform_system == 'Linux' і на Windows/macOS "
            "з'являються ЛИШЕ якщо навмисно тягнути колеса з download.pytorch.org."
        )

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
