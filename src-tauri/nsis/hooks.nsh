; Хуки інсталятора NSIS.
;
; Одна проблема, яку вони лікують: повторне встановлення поверх запущеного
; застосунку. Tauri вміє закрити своє вікно, але Python-sidecar — окремий
; процес; якщо він живий, файли рантайму заблоковані, і встановлення падає з
; «error opening file for writing» посеред розпакування ~1 ГБ.

!macro NSIS_HOOK_PREINSTALL
  DetailPrint "Завершення процесів попередньої версії Асістента..."
  ; Саме вікно застосунку.
  nsExec::ExecToLog `taskkill /F /T /IM "Asistent.exe"`
  Pop $0
  ; Будь-який процес, чий виконуваний файл лежить у теці встановлення —
  ; це наш python.exe і його воркери. Порівняння за шляхом, а не за іменем:
  ; убивати всі python.exe на машині викладача неприпустимо.
  nsExec::ExecToLog `powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-CimInstance Win32_Process | Where-Object { $$_.ExecutablePath -like '$INSTDIR\*' } | ForEach-Object { Stop-Process -Id $$_.ProcessId -Force -ErrorAction SilentlyContinue }"`
  Pop $0
!macroend

!macro NSIS_HOOK_POSTINSTALL
  DetailPrint "Асістент встановлено. Моделі перевіряються при першому запуску."
!macroend

!macro NSIS_HOOK_PREUNINSTALL
  nsExec::ExecToLog `taskkill /F /T /IM "Asistent.exe"`
  Pop $0
  nsExec::ExecToLog `powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-CimInstance Win32_Process | Where-Object { $$_.ExecutablePath -like '$INSTDIR\*' } | ForEach-Object { Stop-Process -Id $$_.ProcessId -Force -ErrorAction SilentlyContinue }"`
  Pop $0
!macroend

!macro NSIS_HOOK_POSTUNINSTALL
  ; Каталог даних (%LOCALAPPDATA%\Asistent) НЕ видаляється: там база знань
  ; викладача, яку переіндексовувати — години. Видалення — свідома дія
  ; користувача, описана в docs/DEPLOY.md.
  DetailPrint "Матеріали й базу знань збережено у %LOCALAPPDATA%\Asistent"
!macroend
