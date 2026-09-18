# Оболонка Tauri

`tauri.conf.json` — строгий JSON, коментарі в ньому неможливі. Обґрунтування
кожного нетривіального рішення записане тут.

| Рішення в конфізі | Чому саме так |
|---|---|
| `bundle.createUpdaterArtifacts: false`, плагіна `updater` немає в `Cargo.toml` | Закритий контур. Оновлювач, що ходить у мережу, — це саме те, чого не має існувати в продукті, який ставлять з USB. Оновлення = новий інсталятор з носія. |
| `windows.nsis.installMode: "currentUser"` | Прав адміністратора немає. Викладач на доменній машині академії майже напевно не локальний адмін; `perMachine` зробив би застосунок невстановлюваним. Наслідок: ставиться у `%LOCALAPPDATA%\Programs\Asistent`. |
| `windows.webviewInstallMode.type: "offlineInstaller"` | +127 МБ до інсталятора, зате нуль мережевих викликів. `downloadBootstrapper` (дефолт) у закритому контурі просто зависає. Windows 11 має WebView2 передвстановлено, але LTSC-редакції — ні, і саме на них це рятує. |
| `nsis.languages: ["Ukrainian", "English"]` | Мова інтерфейсу інсталятора. Українська — перша, тому дефолтна. |
| `nsis.installerHooks: "nsis/hooks.nsh"` | Завершення живого sidecar-а перед перезаписом файлів рантайму. Без цього повторне встановлення падає на заблокованих DLL. |
| `longPathAware` у `windows-app-manifest.xml` | Кириличне ім'я підручника + `%LOCALAPPDATA%\Asistent\artifacts\<uuid>\…` пробиває MAX_PATH 260 тривіально. |
| `activeCodePage: UTF-8` у маніфесті | Інакше кирилиця в аргументах командного рядка до sidecar-а стає cp1251-сміттям. |
| `macOS.hardenedRuntime: true` + `entitlements.plist` | Нотаризація без hardened runtime неможлива. `disable-library-validation` обов'язковий: інакше dyld не завантажить .dylib з коліс PyPI (див. коментар у самому plist). |
| `macOS.minimumSystemVersion: "13.0"` | Ventura — перша версія з передбачуваною поведінкою `WKWebView` для наших потреб; нижче Apple вже не постачає оновлення безпеки. |
| `app.security.assetProtocol.enable: false` | PDF-и й растри фронтенд бере через локальний HTTP API, а не через `asset://`. Менша поверхня — і той самий код працює у веб-браузері в режимі розробки. |
| `withGlobalTauri: false` | Фронтенд не має «випадково» отримати доступ до API оболонки. Три легальні команди імпортуються явно через `@tauri-apps/api`. |
| Вікно `splash` окремим вікном | Головне вікно створюється прихованим і показується лише після успішного `GET /api/health`. Інакше викладач бачить порожній білий прямокутник і білу сторінку «не вдалося підключитися». |

## Що фронтенд мусить надати

* `frontend/dist/index.html` — SPA;
* `frontend/dist/splash.html` — сплеш-екран (статичний, без запитів до API).

## Три команди `invoke`, і жодної більше

```ts
import { invoke } from "@tauri-apps/api/core";

const base = await invoke<string>("api_base_url");     // http://127.0.0.1:<порт>
const log  = await invoke<string | null>("sidecar_log_path");
await invoke("reveal_log");                            // показати в Провіднику/Finder
```

Порт **динамічний** — оболонка бере вільний у ядра при старті. У браузерному
режимі розробки (`vite dev` + `uvicorn`) `invoke` недоступний; фронтенд має
відкочуватись на `http://127.0.0.1:8765`.

Будь-яка нова команда `invoke` з бізнес-логікою — порушення інваріанта §0 плану
і робить міграцію у веб-платформу переписуванням, а не видаленням оболонки.

## Локальна збірка

```bash
npm --prefix ../frontend install
cargo tauri dev          # оболонка + sidecar
ASISTENT_EXTERNAL_API=http://127.0.0.1:8765 cargo tauri dev   # без sidecar
```

Іконки (`icons/`) генеруються один раз із логотипу:
`npx @tauri-apps/cli icon ../assets/logo.png`.
