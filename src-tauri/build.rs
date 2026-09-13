//! Скрипт збірки оболонки.
//!
//! Єдина нетипова річ — власний маніфест Windows. `longPathAware` неможливо
//! ввімкнути з `tauri.conf.json`: це властивість вбудованого в .exe маніфесту,
//! тому вона подається сюди через `WindowsAttributes::app_manifest`.

fn main() {
    let mut attributes = tauri_build::Attributes::new();

    #[cfg(windows)]
    {
        let windows = tauri_build::WindowsAttributes::new()
            .app_manifest(include_str!("windows-app-manifest.xml"));
        attributes = attributes.windows_attributes(windows);
    }

    tauri_build::try_build(attributes).expect("не вдалося зібрати конфігурацію Tauri");
}
