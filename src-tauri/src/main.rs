// Без консольного вікна в release-збірці на Windows.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

//! Оболонка «Асістента».
//!
//! **Архітектурний інваріант, який тут захищається:** оболонка володіє лише
//! вікном, нативними діалогами, «показати в Провіднику/Finder», життєвим циклом
//! sidecar-а і логами. Уся логіка — за локальним HTTP API, увесь UI — React SPA,
//! що ходить у нього через `fetch` + SSE. Жодної команди `invoke()` з бізнес-
//! логікою. Тримаючи цю лінію, міграція у веб-платформу = видалити оболонку.
//! Порушивши її — переписувати все (див. план, §0).
//!
//! Тому нижче є рівно три команди, і всі три — суто оболонкові: «де API»,
//! «де лог», «відкрий теку логів».

mod sidecar;

use std::sync::Arc;

use sidecar::{Layout, Sidecar};
use tauri::{Emitter, Manager};
use tauri_plugin_dialog::{DialogExt, MessageDialogButtons, MessageDialogKind};

struct AppState {
    sidecar: Option<Arc<Sidecar>>,
    /// Базовий URL API. При зовнішньому сервері (режим розробки) sidecar-а немає.
    base_url: String,
}

/// Куди фронтенду слати `fetch` і SSE. Порт обирається динамічно при старті.
#[tauri::command]
fn api_base_url(state: tauri::State<'_, AppState>) -> String {
    state.base_url.clone()
}

/// Шлях до `sidecar.log` — для екрана «Діагностика».
#[tauri::command]
fn sidecar_log_path(state: tauri::State<'_, AppState>) -> Option<String> {
    state
        .sidecar
        .as_ref()
        .map(|s| s.log_path.to_string_lossy().to_string())
}

/// Показати лог у Провіднику/Finder. Нативна дія — законна для оболонки.
#[tauri::command]
fn reveal_log(app: tauri::AppHandle, state: tauri::State<'_, AppState>) -> Result<(), String> {
    use tauri_plugin_opener::OpenerExt;
    let Some(s) = state.sidecar.as_ref() else {
        return Err("Sidecar не запущено (зовнішній режим розробки).".into());
    };
    app.opener()
        .reveal_item_in_dir(&s.log_path)
        .map_err(|e| format!("Не вдалося показати файл лога: {e}"))
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![
            api_base_url,
            sidecar_log_path,
            reveal_log
        ])
        .setup(|app| {
            let handle = app.handle().clone();

            // Режим розробки: `vite dev` + `uvicorn --reload` у сусідньому
            // терміналі. Оболонка тоді не запускає нічого — саме так проходять
            // 95% часу розробки (див. план, §0).
            if let Ok(external) = std::env::var("ASISTENT_EXTERNAL_API") {
                app.manage(AppState {
                    sidecar: None,
                    base_url: external,
                });
                show_main(&handle);
                return Ok(());
            }

            let resource_dir = app.path().resource_dir()?;
            let layout = Layout::resolve(&resource_dir);

            let sidecar = match Sidecar::spawn(&layout) {
                Ok(s) => Arc::new(s),
                Err(message) => {
                    fatal(&handle, &message);
                    return Ok(());
                }
            };

            app.manage(AppState {
                base_url: sidecar.base_url(),
                sidecar: Some(sidecar.clone()),
            });

            // Health-polling у окремому потоці: блокувати `setup` означало б
            // тримати вікно сплеша нефарбованим до 60 с.
            std::thread::spawn(move || match sidecar.wait_until_ready() {
                Ok(()) => {
                    let _ = handle.emit("asistent://ready", ());
                    show_main(&handle);
                }
                Err(message) => fatal(&handle, &message),
            });

            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("не вдалося ініціалізувати Асістент")
        .run(|app_handle, event| {
            // Вбити дерево процесів на будь-якому шляху виходу, включно з
            // закриттям останнього вікна й Cmd+Q.
            if let tauri::RunEvent::Exit = event {
                if let Some(state) = app_handle.try_state::<AppState>() {
                    if let Some(sidecar) = state.sidecar.as_ref() {
                        sidecar.shutdown();
                    }
                }
            }
        });
}

/// Показати головне вікно і прибрати сплеш.
fn show_main(handle: &tauri::AppHandle) {
    if let Some(main) = handle.get_webview_window("main") {
        let _ = main.show();
        let _ = main.set_focus();
    }
    if let Some(splash) = handle.get_webview_window("splash") {
        let _ = splash.close();
    }
}

/// Фатальна помилка старту: показати текст (із хвостом лога) і вийти.
///
/// Саме цей діалог перетворює «застосунок не запускається» на конкретний
/// Python-traceback, який можна переслати розробнику.
fn fatal(handle: &tauri::AppHandle, message: &str) {
    let handle_for_exit = handle.clone();
    handle
        .dialog()
        .message(message)
        .kind(MessageDialogKind::Error)
        .title("Асістент не зміг запуститися")
        .buttons(MessageDialogButtons::Ok)
        .show(move |_| {
            handle_for_exit.exit(1);
        });
}
