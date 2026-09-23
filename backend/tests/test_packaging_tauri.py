"""Конфігурація оболонки: рішення, які дорого відкочувати.

Ці тести охороняють не стиль, а конкретні наслідки:
  * увімкнений оновлювач = мережевий виклик у закритому контурі;
  * `perMachine` = застосунок неможливо встановити без адміністратора;
  * `downloadBootstrapper` WebView2 = зависання встановлення без мережі;
  * зайва команда `invoke` з логікою = міграція у веб-платформу стає
    переписуванням, а не видаленням оболонки (інваріант §0 плану).
"""

from __future__ import annotations

import json
import plistlib
import re
import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from helpers_packaging import REPO_ROOT, TAURI_DIR, WORKFLOWS_DIR, read, rust_code_lines

CONF_PATH = TAURI_DIR / "tauri.conf.json"
CONF = json.loads(read(CONF_PATH))

# Оболонці дозволено рівно стільки команд, скільки потрібно, щоб фронтенд
# знайшов API і показав лог. Будь-яка нова — привід для архітектурної розмови.
ALLOWED_COMMANDS = {"api_base_url", "sidecar_log_path", "reveal_log"}

# Бюджет обсягу. Міряється КОДОМ (без коментарів): коментарі тут несуть
# обґрунтування пасток і є цінністю, а не шумом. Ціль плану — «~300 рядків»,
# тобто оболонка без бізнес-логіки; поріг лишає запас на платформні гілки
# (Job Object на Windows + група процесів і обробник сигналів на Unix живуть в
# одному файлі). Піднято з 420 заради `on_terminating_signal`: без нього SIGTERM
# убивав оболонку в обхід `RunEvent::Exit` і лишав живий python — це не окраса,
# а виконання обіцянки «вбити дерево на будь-якому шляху виходу».
RUST_CODE_BUDGET = 460


# ------------------------------------------------------------------ tauri.conf
def test_конфіг_валідний_json_і_має_ідентифікатор() -> None:
    assert CONF["identifier"] == "ua.gov.nasv.asistent"
    assert CONF["productName"] == "Asistent"


def test_оновлювач_вимкнено() -> None:
    # Закритий контур: оновлення = новий інсталятор з носія.
    assert CONF["bundle"]["createUpdaterArtifacts"] is False
    assert "updater" not in CONF.get("plugins", {})
    assert "updater" not in read(TAURI_DIR / "Cargo.toml")


def test_nsis_per_user_і_офлайн_webview2() -> None:
    nsis = CONF["bundle"]["windows"]["nsis"]
    # Прав адміністратора немає: викладач на доменній машині академії.
    # Саме `currentUser`: у схемі Tauri допустимі лише currentUser/perMachine/both,
    # і тест довго закріплював неіснуюче `perUser` — через що `tauri build` падав
    # на валідації конфігу ще до компіляції, а тест цього не бачив.
    assert nsis["installMode"] == "currentUser"
    assert "Ukrainian" in nsis["languages"]
    assert nsis["languages"][0] == "Ukrainian"
    # downloadBootstrapper у закритому контурі просто зависає.
    assert CONF["bundle"]["windows"]["webviewInstallMode"]["type"] == "offlineInstaller"
    assert (TAURI_DIR / nsis["installerHooks"]).exists()


def test_ресурси_бандла_містять_рантайм_і_моделі() -> None:
    resources = CONF["bundle"]["resources"]
    assert resources["resources/runtime"] == "runtime"
    assert resources["resources/models"] == "models"


def test_macos_підписуваний() -> None:
    mac = CONF["bundle"]["macOS"]
    assert mac["hardenedRuntime"] is True   # без цього нотаризація неможлива
    assert mac["minimumSystemVersion"] == "13.0"
    assert (TAURI_DIR / mac["entitlements"]).exists()
    assert CONF["bundle"]["category"] == "Education"


def test_головне_вікно_приховане_до_готовності_api() -> None:
    windows = {w["label"]: w for w in CONF["app"]["windows"]}
    assert windows["main"]["visible"] is False
    assert windows["splash"]["visible"] is True


def test_csp_дозволяє_лише_петлю_назад() -> None:
    csp = CONF["app"]["security"]["csp"]
    assert "127.0.0.1" in csp
    assert "https://" not in csp   # жодного зовнішнього джерела
    assert CONF["app"]["security"]["assetProtocol"]["enable"] is False
    assert CONF["app"]["withGlobalTauri"] is False


# ------------------------------------------------------------------- маніфест
def test_маніфест_windows_вмикає_довгі_шляхи_і_utf8() -> None:
    path = TAURI_DIR / "windows-app-manifest.xml"
    ET.parse(path)  # валідний XML
    text = read(path)
    # Кириличне ім'я підручника пробиває MAX_PATH 260 тривіально.
    assert re.search(r"<longPathAware[^>]*>true</longPathAware>", text)
    # Інакше кирилиця в аргументах sidecar-а стає cp1251-сміттям.
    assert re.search(r"<activeCodePage[^>]*>UTF-8</activeCodePage>", text)
    assert 'level="asInvoker"' in text


def test_маніфест_оголошує_common_controls_v6() -> None:
    """Без цього блоку .exe не стартує взагалі — і без жодної діагностики.

    `WindowsAttributes::app_manifest` ЗАМІНЮЄ типовий маніфест tauri-build, а
    той складається рівно з однієї залежності — на Common-Controls 6.0.0.0.
    `tauri-plugin-dialog` вмикає у `rfd` фічу `common-controls-v6`, тобто
    `TaskDialogIndirect` імпортується з `comctl32.dll` СТАТИЧНО (windows-sys,
    raw-dylib). Версія 5.82 із System32 цього символу не експортує, а версію 6
    підключає лише ця SxS-залежність — тож завантажувач убиває процес ще до
    `main()` з кодом 0xC0000139 (3221225785). Саме так падав димовий тест:
    порожній sidecar.log, відсутній api-port.json, нуль підказок про причину.
    """
    ns = {"asm": "urn:schemas-microsoft-com:asm.v1"}
    tree = ET.parse(TAURI_DIR / "windows-app-manifest.xml")
    identities = [
        element.attrib
        for element in tree.iterfind(
            "asm:dependency/asm:dependentAssembly/asm:assemblyIdentity", ns
        )
    ]
    common = [a for a in identities if a.get("name") == "Microsoft.Windows.Common-Controls"]
    assert common, (
        "У маніфесті немає залежності на Common-Controls. Поки в оболонці є "
        "tauri-plugin-dialog, її видалення = застосунок, який не запускається."
    )
    assert common[0].get("version") == "6.0.0.0"
    assert common[0].get("publicKeyToken") == "6595b64144ccf1df"


def test_build_rs_підключає_маніфест() -> None:
    assert "windows-app-manifest.xml" in read(TAURI_DIR / "build.rs")


def test_entitlements_дозволяють_завантажити_dylib_з_коліс() -> None:
    data = plistlib.loads((TAURI_DIR / "entitlements.plist").read_bytes())
    # Без цього dyld відмовиться вантажити .so з onnxruntime/numpy у
    # hardened runtime, і застосунок помре на першому import.
    assert data["com.apple.security.cs.disable-library-validation"] is True
    assert data["com.apple.security.cs.allow-jit"] is True


# ---------------------------------------------------------------- залежності
def test_cargo_без_шелу_оновлювача_і_http() -> None:
    cargo = tomllib.loads(read(TAURI_DIR / "Cargo.toml"))
    deps = cargo["dependencies"]
    assert "tauri-plugin-dialog" in deps
    assert "tauri-plugin-opener" in deps
    # tauri-plugin-shell задокументовано не вбиває дерево процесів (tauri#11686);
    # ми керуємо sidecar-ом самі. http/fs з фронтенду не потрібні: усе через API.
    for forbidden in ("tauri-plugin-shell", "tauri-plugin-updater", "tauri-plugin-http", "tauri-plugin-fs"):
        assert forbidden not in deps, f"{forbidden} не має бути залежністю оболонки"


def test_дозволи_фронтенду_мінімальні() -> None:
    caps = json.loads(read(TAURI_DIR / "capabilities" / "default.json"))
    for permission in caps["permissions"]:
        assert not permission.startswith(("shell:", "fs:", "http:")), permission
    assert set(caps["windows"]) == {"main", "splash"}


# --------------------------------------------------------------------- Rust
def test_команди_invoke_не_містять_бізнес_логіки() -> None:
    """Інваріант §0: оболонка володіє лише вікном, діалогами, sidecar і логами."""
    main_rs = read(TAURI_DIR / "src" / "main.rs")
    commands = re.findall(r"#\[tauri::command\]\s*(?:pub\s+)?fn\s+(\w+)", main_rs)
    assert set(commands) == ALLOWED_COMMANDS, (
        "Змінився набір команд invoke. Кожна нова команда з бізнес-логікою "
        "перетворює міграцію у веб-платформу з видалення оболонки на переписування."
    )
    assert "generate_handler!" in main_rs


def test_обсяг_оболонки_в_межах_бюджету() -> None:
    total = sum(rust_code_lines(TAURI_DIR / "src" / name) for name in ("main.rs", "sidecar.rs"))
    assert total <= RUST_CODE_BUDGET, f"Оболонка розрослася до {total} рядків коду"


def test_sidecar_інжектує_офлайн_змінні() -> None:
    text = read(TAURI_DIR / "src" / "sidecar.rs")
    for key in (
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "HF_HUB_DISABLE_SYMLINKS",
        "PYTHONUTF8",
        "PYTHONIOENCODING",
        "DOCLING_ARTIFACTS_PATH",
        "ASISTENT_DATA_DIR",
    ):
        assert f'"{key}"' in text, f"sidecar не виставляє {key}"
    # Саме DISABLE_SYMLINKS, а не DISABLE_SYMLINKS_WARNING: на Windows без
    # Developer Mode симлінки дають WinError 1314.
    assert "HF_HUB_DISABLE_SYMLINKS_WARNING" not in text


def test_sidecar_вбиває_дерево_процесів() -> None:
    text = read(TAURI_DIR / "src" / "sidecar.rs")
    assert "JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE" in text     # Windows
    assert "process_group(0)" in text and "killpg" in text  # Unix
    assert "TerminateJobObject" in text


def test_sidecar_переживає_sigterm_не_лишаючи_сиріт() -> None:
    """SIGTERM на Unix убиває процес в обхід деструкторів Rust.

    Ані tauri, ані tao обробників сигналів не ставлять, тому `RunEvent::Exit`
    не настає, `Drop for Sidecar` не викликається — і python лишається живим у
    власній групі процесів із зайнятим портом. Димовий тест інсталятора
    завершує застосунок саме `proc.terminate()`, тобто SIGTERM, і перевіряє
    відсутність сиріт; без обробника він падав би вже ПІСЛЯ успішного сценарію.
    """
    text = read(TAURI_DIR / "src" / "sidecar.rs")
    assert "arm_terminating_signals" in text
    for signal in ("SIGTERM", "SIGINT", "SIGHUP"):
        assert f"libc::{signal}" in text, f"обробник не перехоплює {signal}"
    # Обробник виконується в контексті сигналу, тому чекати на дитину він мусить
    # async-signal-safe засобами (waitpid + nanosleep), а виходити — через _exit,
    # а не через звичайне завершення з деструкторами.
    for call in ("libc::killpg", "libc::waitpid", "libc::nanosleep", "libc::_exit"):
        assert call in text, f"обробник сигналу не використовує {call}"


def test_sidecar_ловить_вивід_у_лог() -> None:
    text = read(TAURI_DIR / "src" / "sidecar.rs")
    assert "sidecar.log" in text
    assert "PYTHONUNBUFFERED" in text  # інакше traceback лишиться в буфері


def test_health_шлях_узгоджений_зі_скриптом_димового_тесту() -> None:
    rust = read(TAURI_DIR / "src" / "sidecar.rs")
    smoke = read(REPO_ROOT / "scripts" / "smoke_installer.py")
    match = re.search(r'HEALTH_PATH:\s*&str\s*=\s*"([^"]+)"', rust)
    assert match, "у sidecar.rs немає константи HEALTH_PATH"
    assert f'HEALTH_PATH = "{match.group(1)}"' in smoke


def test_фронтенд_чекає_на_sidecar_не_менше_за_оболонку() -> None:
    """SPA не має права здатися раніше, ніж оболонка визнає старт провальним.

    Вікно `main` створюється разом із процесом і вантажить SPA одразу — воно
    лише приховане. Тобто проба sidecar-а змагається з холодним стартом Python,
    і поки дедлайн у `client.ts` був 1.5 с, SPA програвала цю гонку ЩОРАЗУ:
    мовчки вмикала демонстраційний транспорт, і викладач бачив вигаданий корпус
    із вигаданими цитатами замість власних матеріалів. Дві константи мусять
    лишатися однією величиною.
    """
    rust = read(TAURI_DIR / "src" / "sidecar.rs")
    client = read(REPO_ROOT / "frontend" / "src" / "api" / "client.ts")
    shell = re.search(r"STARTUP_TIMEOUT:\s*Duration\s*=\s*Duration::from_secs\((\d+)\)", rust)
    spa = re.search(r"STARTUP_DEADLINE_MS\s*=\s*([\d_]+)", client)
    assert shell, "у sidecar.rs немає константи STARTUP_TIMEOUT"
    assert spa, "у client.ts немає константи STARTUP_DEADLINE_MS"
    assert int(spa.group(1).replace("_", "")) == int(shell.group(1)) * 1000


# ------------------------------------------------------------------ workflows
yaml = pytest.importorskip("yaml", reason="PyYAML потрібен лише для перевірки CI-конфігів")


def _load(path: Path) -> dict:
    # `on:` у YAML 1.1 парситься як булеве True — тому ключі шукаємо обережно.
    return yaml.safe_load(read(path))


def test_ci_має_швидкий_гейт_і_тест_офлайну() -> None:
    ci = _load(WORKFLOWS_DIR / "ci.yml")
    assert ci["env"]["ASISTENT_STUB"] == "1"
    fast = ci["jobs"]["fast"]
    runs = " ".join(str(step.get("run", "")) for step in fast["steps"])
    assert "pytest" in runs
    assert "offline_check.py" in runs
    assert "ruff" in runs
    # Свідомо без extras `worker`: torch не має права потрапити в гарячий шлях.
    installs = [
        line.strip()
        for line in runs.splitlines()
        if "pip install" in line and not line.strip().startswith("#")
    ]
    assert any("[dev]" in line for line in installs)
    assert not any("worker" in line for line in installs)


def test_ci_інтеграція_на_рідних_ос() -> None:
    ci = _load(WORKFLOWS_DIR / "ci.yml")
    integration = ci["jobs"]["integration"]
    assert set(integration["strategy"]["matrix"]["os"]) == {"windows-latest", "macos-26"}
    runs = " ".join(str(step.get("run", "")) for step in integration["steps"])
    assert "build_runtime.py" in runs   # крос-компіляція неможлива
    assert "cargo check" in runs
    uses = [step.get("uses", "") for step in integration["steps"]]
    assert "./.github/actions/fetch-models" in uses
    profiles = [step.get("with", {}).get("profile") for step in integration["steps"]]
    assert "tiny" in profiles           # PR-прогони не тягнуть 1.8 ГБ


def test_реліз_робить_димовий_тест_і_перевіряє_бюджет() -> None:
    release = _load(WORKFLOWS_DIR / "release.yml")
    job = release["jobs"]["build"]
    oses = {entry["os"] for entry in job["strategy"]["matrix"]["include"]}
    assert oses == {"windows-latest", "macos-26"}
    runs = " ".join(str(step.get("run", "")) for step in job["steps"])
    assert "smoke_installer.py" in runs
    assert "check_size_budget.py" in runs
    assert "SHA256" in runs.upper()
