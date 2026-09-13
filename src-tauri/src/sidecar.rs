//! Життєвий цикл Python-sidecar: запуск, лог, health-polling, гарантоване вбивство.
//!
//! Оболонка НЕ знає нічого про бізнес-логіку. Вона вміє рівно чотири речі:
//! запустити рантайм, зловити його вивід у файл, дочекатися готовності HTTP API
//! і вбити все дерево процесів при виході.
//!
//! Три рішення, кожне з яких лікує конкретну задокументовану поломку:
//!
//! 1. **Ми не використовуємо tauri-plugin-shell і не використовуємо PyInstaller.**
//!    tauri#11686: плагін не може повністю завершити виконуваний файл PyInstaller,
//!    бо той створює два процеси (бутстрап + розпакований дитячий). Ми постачаємо
//!    python-build-standalone — один процес і звичайні DLL. Але воркери — це теж
//!    процеси (див. план, §0), тож гарантія вбивства дерева потрібна однаково.
//!    Windows → Job Object з KILL_ON_JOB_CLOSE, Unix → група процесів + killpg.
//!
//! 2. **stdout/stderr перенаправляються у файл `sidecar.log`.** Саме там з'явиться
//!    Python-traceback, що вбиває старт. Це різниця між 10-хвилинною і 3-денною
//!    діагностикою на машині викладача, до якої немає ssh.
//!
//! 3. **Порт обирається оболонкою, а не зашитий.** 8765 може бути зайнятий чим
//!    завгодно на доменній машині; ми беремо вільний порт у ядра й передаємо його
//!    аргументом. Вікно гонки між закриттям слухача і стартом Python існує, але
//!    воно мікросекундне й падіння тут гучне (порт зайнято → traceback у логу).
//!    Обраний порт записується в `api-port.json` поруч із логом — без цього
//!    димовий тест запакованого інсталятора не знав би, куди стукати.
//!    `ASISTENT_PORT` перекриває вибір (саме так CI фіксує порт).

use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Шлях health-ендпоїнта API. Мусить збігатися з `backend/app/api/`.
const HEALTH_PATH: &str = "/api/health";
/// Скільки чекаємо на готовність до того, як визнати старт провальним.
/// 60 с — це холодний старт на HDD доменної машини з увімкненим Defender.
const STARTUP_TIMEOUT: Duration = Duration::from_secs(60);
const POLL_INTERVAL: Duration = Duration::from_millis(250);

/// Стан запущеного sidecar-а. Живе в `tauri::State`.
pub struct Sidecar {
    pub port: u16,
    pub log_path: PathBuf,
    child: Mutex<Option<Child>>,
    #[cfg(windows)]
    job: Mutex<Option<isize>>,
}

/// Розкладка каталогів, обчислена оболонкою до старту Python.
///
/// Дублює логіку `backend/app/config.py::_default_data_dir`. Дублювання свідоме:
/// оболонці треба знати шлях до лога ДО того, як існує Python-процес, який міг би
/// його повідомити. Обидві реалізації тестуються проти одних і тих самих правил.
pub struct Layout {
    pub data_dir: PathBuf,
    pub logs_dir: PathBuf,
    pub runtime_dir: PathBuf,
    pub models_dir: Option<PathBuf>,
}

impl Layout {
    pub fn resolve(resource_dir: &Path) -> Self {
        let data_dir = std::env::var_os("ASISTENT_DATA_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(default_data_dir);

        let logs_dir = if cfg!(target_os = "macos") {
            home_dir().join("Library").join("Logs").join("Asistent")
        } else {
            data_dir.join("logs")
        };

        // Моделі в бандлі — варіант «усе в інсталяторі». Якщо їх там немає
        // (варіант «моделі з USB»), Python візьме їх з каталогу даних.
        let bundled_models = resource_dir.join("models");
        let models_dir = bundled_models
            .join("manifest.json")
            .exists()
            .then_some(bundled_models);

        Self {
            data_dir,
            logs_dir,
            runtime_dir: resource_dir.join("runtime"),
            models_dir,
        }
    }
}

fn home_dir() -> PathBuf {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map_or_else(|| PathBuf::from("."), PathBuf::from)
}

fn default_data_dir() -> PathBuf {
    let base = if cfg!(windows) {
        std::env::var_os("LOCALAPPDATA")
            .map_or_else(|| home_dir().join("AppData").join("Local"), PathBuf::from)
    } else if cfg!(target_os = "macos") {
        home_dir().join("Library").join("Application Support")
    } else {
        home_dir().join(".local").join("share")
    };
    base.join("Asistent")
}

/// Шлях до інтерпретатора всередині релокованого рантайму.
fn python_executable(runtime_dir: &Path) -> PathBuf {
    if cfg!(windows) {
        runtime_dir.join("python.exe")
    } else {
        runtime_dir.join("bin").join("python3")
    }
}

fn free_port() -> std::io::Result<u16> {
    if let Some(forced) = std::env::var("ASISTENT_PORT").ok().and_then(|v| v.parse().ok()) {
        return Ok(forced);
    }
    let listener = TcpListener::bind("127.0.0.1:0")?;
    let port = listener.local_addr()?.port();
    drop(listener);
    Ok(port)
}

impl Sidecar {
    /// Запустити рантайм. Помилка тут — це фатальна помилка старту застосунку.
    pub fn spawn(layout: &Layout) -> Result<Self, String> {
        for dir in [&layout.data_dir, &layout.logs_dir] {
            std::fs::create_dir_all(dir)
                .map_err(|e| format!("Не вдалося створити каталог {dir:?}: {e}"))?;
        }

        let python = python_executable(&layout.runtime_dir);
        if !python.exists() {
            return Err(format!(
                "Python-рантайм не знайдено: {python:?}. Інсталяція пошкоджена — \
                 перевстановіть Асістент з носія."
            ));
        }

        let port = free_port().map_err(|e| format!("Не вдалося зайняти локальний порт: {e}"))?;
        let log_path = layout.logs_dir.join("sidecar.log");
        let log = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)
            .map_err(|e| format!("Не вдалося відкрити {log_path:?}: {e}"))?;
        let log_err = log
            .try_clone()
            .map_err(|e| format!("Не вдалося продублювати дескриптор лога: {e}"))?;

        // Порт і хост передаються ЗМІННИМИ СЕРЕДОВИЩА, а не аргументами:
        // `app.main` читає їх через pydantic-settings із префіксом ASISTENT_
        // і не має розбору командного рядка.
        let mut cmd = Command::new(&python);
        cmd.arg("-m")
            .arg("app.main")
            .env("ASISTENT_HOST", "127.0.0.1")
            .env("ASISTENT_PORT", port.to_string())
            .current_dir(&layout.runtime_dir)
            .stdin(Stdio::null())
            .stdout(Stdio::from(log))
            .stderr(Stdio::from(log_err));

        inject_env(&mut cmd, layout);

        #[cfg(unix)]
        {
            // Власна група процесів: killpg накриє і воркерів, породжених Python.
            use std::os::unix::process::CommandExt;
            cmd.process_group(0);
        }
        #[cfg(windows)]
        {
            // CREATE_NO_WINDOW: інакше на кожен старт блимає консоль.
            // CREATE_SUSPENDED не потрібен — Job Object призначається одразу після
            // spawn, а вікно між spawn і призначенням вимірюється мікросекундами.
            use std::os::windows::process::CommandExt;
            const CREATE_NO_WINDOW: u32 = 0x0800_0000;
            cmd.creation_flags(CREATE_NO_WINDOW);
        }

        let child = cmd
            .spawn()
            .map_err(|e| format!("Не вдалося запустити Python-рантайм {python:?}: {e}"))?;

        #[cfg(windows)]
        let job = assign_to_kill_on_close_job(&child);

        // Хто слухає й на чому — для димового тесту й для екрана «Діагностика».
        let _ = std::fs::write(
            layout.logs_dir.join("api-port.json"),
            format!("{{\"port\":{},\"pid\":{}}}\n", port, child.id()),
        );

        Ok(Self {
            port,
            log_path,
            child: Mutex::new(Some(child)),
            #[cfg(windows)]
            job: Mutex::new(job),
        })
    }

    pub fn base_url(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }

    /// Опитувати health до готовності. Повертає помилку з хвостом лога —
    /// саме він містить traceback, що вбив старт.
    pub fn wait_until_ready(&self) -> Result<(), String> {
        let deadline = Instant::now() + STARTUP_TIMEOUT;
        let url_path = HEALTH_PATH;
        loop {
            if let Some(status) = self.exited() {
                return Err(format!(
                    "Сервер Асістента завершився з кодом {status} під час запуску.\n\n{}",
                    self.log_tail(60)
                ));
            }
            if http_ok(self.port, url_path) {
                return Ok(());
            }
            if Instant::now() >= deadline {
                return Err(format!(
                    "Сервер Асістента не відповів за {} с на {}{}.\n\n{}",
                    STARTUP_TIMEOUT.as_secs(),
                    self.base_url(),
                    url_path,
                    self.log_tail(60)
                ));
            }
            std::thread::sleep(POLL_INTERVAL);
        }
    }

    /// Код завершення, якщо процес уже помер; `None` — якщо ще живий.
    fn exited(&self) -> Option<i32> {
        match self.child.lock().ok()?.as_mut()?.try_wait() {
            Ok(Some(status)) => Some(status.code().unwrap_or(-1)),
            _ => None,
        }
    }

    /// Останні `lines` рядків лога — це те, що бачить викладач у діалозі помилки.
    pub fn log_tail(&self, lines: usize) -> String {
        let mut buf = String::new();
        match File::open(&self.log_path).map(|mut f| f.read_to_string(&mut buf)) {
            Ok(Ok(_)) => {}
            _ => return format!("(лог {:?} недоступний)", self.log_path),
        }
        let tail: Vec<&str> = buf.lines().rev().take(lines).collect();
        tail.into_iter().rev().collect::<Vec<_>>().join("\n")
    }

    /// Вбити дерево процесів. Ідемпотентно; викликається на виході застосунку.
    pub fn shutdown(&self) {
        let Ok(mut guard) = self.child.lock() else {
            return;
        };
        let Some(mut child) = guard.take() else {
            return;
        };

        #[cfg(unix)]
        {
            // Спершу ввічливо всій групі, потім жорстко. SIGTERM дає uvicorn
            // шанс закрити SQLite (PRAGMA optimize + чистий чекпойнт WAL).
            let pgid = child.id() as i32;
            unsafe { libc::killpg(pgid, libc::SIGTERM) };
            let deadline = Instant::now() + Duration::from_secs(5);
            while Instant::now() < deadline {
                if matches!(child.try_wait(), Ok(Some(_))) {
                    break;
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            unsafe { libc::killpg(pgid, libc::SIGKILL) };
        }

        #[cfg(windows)]
        {
            // Закриття хендла Job Object з KILL_ON_JOB_CLOSE вбиває ВСІ процеси
            // в джобі, включно з воркерами, — це і є гарантія «без осиротілого
            // python.exe», яку перевіряє димовий тест інсталятора.
            if let Ok(mut job_guard) = self.job.lock() {
                if let Some(handle) = job_guard.take() {
                    unsafe {
                        windows_sys::Win32::System::JobObjects::TerminateJobObject(handle as _, 0);
                        windows_sys::Win32::Foundation::CloseHandle(handle as _);
                    }
                }
            }
            let _ = child.kill();
        }

        let _ = child.wait();
    }
}

impl Drop for Sidecar {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Змінні середовища, без яких офлайн-режим тихо перетворюється на мережевий.
///
/// `HF_HUB_DISABLE_SYMLINKS` — саме так, а НЕ `*_WARNING`: на Windows без
/// Developer Mode створення симлінка дає WinError 1314 і падіння посеред
/// завантаження моделі.
fn inject_env(cmd: &mut Command, layout: &Layout) {
    cmd.env("PYTHONUTF8", "1")
        .env("PYTHONIOENCODING", "utf-8")
        // Небуферизований вивід: інакше traceback залишиться в буфері процесу,
        // який щойно помер, і лог буде порожній саме тоді, коли він потрібен.
        .env("PYTHONUNBUFFERED", "1")
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .env("HF_HUB_OFFLINE", "1")
        .env("TRANSFORMERS_OFFLINE", "1")
        .env("HF_HUB_DISABLE_SYMLINKS", "1")
        .env("HF_HUB_DISABLE_TELEMETRY", "1")
        .env("ASISTENT_DATA_DIR", &layout.data_dir)
        .env("ASISTENT_LOG_DIR", &layout.logs_dir);

    // DOCLING_ARTIFACTS_PATH має вказувати на БАТЬКІВСЬКИЙ каталог із теками
    // `<org>--<repo>`. Якщо його немає — Docling мовчки викличе snapshot_download,
    // тобто піде в мережу; net_guard зробить це гучною помилкою, але краще
    // взагалі не дійти до цього шляху.
    let models_root = layout
        .models_dir
        .clone()
        .unwrap_or_else(|| layout.data_dir.join("models"));
    cmd.env("ASISTENT_MODELS_DIR", &models_root)
        .env("DOCLING_ARTIFACTS_PATH", models_root.join("docling"))
        .env("HF_HOME", models_root.join("hf"));
}

/// Мінімальний HTTP GET по петлі назад.
///
/// Свідомо без HTTP-клієнтської бібліотеки: єдиний запит, який робить оболонка, —
/// це `GET 127.0.0.1/api/health`, і тягнути заради нього reqwest+tokio означає
/// додати десятки залежностей у продукт закритого контуру.
fn http_ok(port: u16, path: &str) -> bool {
    let Ok(mut stream) = TcpStream::connect(("127.0.0.1", port)) else {
        return false;
    };
    let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
    let _ = stream.set_write_timeout(Some(Duration::from_secs(2)));
    let req = format!(
        "GET {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\nUser-Agent: asistent-shell\r\n\r\n"
    );
    if stream.write_all(req.as_bytes()).is_err() {
        return false;
    }
    let mut buf = [0u8; 256];
    match stream.read(&mut buf) {
        Ok(n) if n > 12 => buf[..n].starts_with(b"HTTP/1.1 200") || buf[..n].starts_with(b"HTTP/1.0 200"),
        _ => false,
    }
}

/// Створити Job Object із KILL_ON_JOB_CLOSE і призначити в нього дитину.
#[cfg(windows)]
fn assign_to_kill_on_close_job(child: &Child) -> Option<isize> {
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, SetInformationJobObject,
        JobObjectExtendedLimitInformation, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
    };

    unsafe {
        let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
        if job.is_null() {
            return None;
        }
        let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        let configured = SetInformationJobObject(
            job,
            JobObjectExtendedLimitInformation,
            &info as *const _ as *const std::ffi::c_void,
            std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
        ) != 0
            && AssignProcessToJobObject(job, child.as_raw_handle() as _) != 0;
        if !configured {
            windows_sys::Win32::Foundation::CloseHandle(job);
            return None;
        }
        Some(job as isize)
    }
}
