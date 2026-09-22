//! Скрипт збірки оболонки.
//!
//! Єдина нетипова річ — власний маніфест Windows. `longPathAware` і
//! `activeCodePage` неможливо ввімкнути з `tauri.conf.json`: це властивості
//! вбудованого в .exe маніфесту, тому вони подаються сюди через
//! `WindowsAttributes::app_manifest`.
//!
//! `app_manifest` ЗАМІНЮЄ типовий маніфест tauri-build ЦІЛКОМ, а не доповнює
//! його. Типовий складається рівно з однієї речі — залежності на
//! Common-Controls 6.0.0.0 — і без неї застосунок із `tauri-plugin-dialog`
//! падає в завантажувачі з 0xC0000139 ще до `main()`. Тому вона перенесена у
//! `windows-app-manifest.xml`, де пояснена детально; тест
//! `test_packaging_tauri.py` не дає її звідти зникнути.

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
