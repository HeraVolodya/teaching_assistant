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
# (Job Object на Windows + група процесів на Unix живуть в одному файлі).
RUST_CODE_BUDGET = 420


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
    assert nsis["installMode"] == "perUser"
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
