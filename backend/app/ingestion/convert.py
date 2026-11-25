"""Копія для читання: Word / PowerPoint / DjVu → PDF, таблиці й XML → Markdown.

Оригінал НЕ змінюється й лишається тим, що викладач завантажив і може
завантажити назад. Поруч кладеться копія `<id>.view.pdf` або `<id>.view.md`:
саме її читає конвеєр і саме її показує переглядач джерела. Опис копії
(`<id>.view.json`) дозволяє не перетворювати документ повторно під час
переіндексації — Word на 300 сторінках думає десятки секунд.

ЧИМ ПЕРЕТВОРЮЄТЬСЯ, у порядку переваги:

  1. Microsoft Office через COM. На машинах кафедр він є майже завжди, і
     PDF від самого Word має рівно ті сторінки, що викладач бачить при друці,
     — тобто «с. 12» у цитаті збігається з «с. 12» на папері.
     Сценарій PowerShell передається через `-EncodedCommand`, а НЕ файлом
     `.ps1`: політика виконання (`ExecutionPolicy`, у військових мережах її
     часто ставлять групою на AllSigned) стосується лише файлів сценаріїв.
  2. LibreOffice (`soffice --headless`), якщо встановлено.
  3. DjVuLibre (`ddjvu`) — єдиний шлях для DjVu.
  4. Для DOCX / PPTX без жодної програми — власні читачі Docling у Markdown.
     Сторінок тоді немає: для DOCX вони умовні («≈ с. 3»), для PPTX — слайди.

ПАСТКИ ОФІСНОЇ АВТОМАТИЗАЦІЇ, які тут обійдено:

  * діалог пароля чекав би людину вічно → у `Open` передається свідомо
    хибний пароль: незахищений файл його ігнорує, захищений одразу падає
    з помилкою замість діалогу;
  * макроси → `AutomationSecurity = 3` (вимкнено примусово);
  * завислий Word → тайм-аут, після якого вбивається САМЕ той процес, який
    створив наш сценарій (PID записується у файл), а не Word викладача;
  * PowerPoint, на відміну від Word і Excel, — один процес на сеанс:
    `Quit()` закрив би відкриті презентації викладача. Тому `Quit()`
    викликається, лише якщо PowerPoint не працював до нас.
"""

from __future__ import annotations

import base64
import contextlib
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from app.ingestion.formats import SourceKind, source_kind

__all__ = [
    "RENDITION_VERSION",
    "ConversionCancelled",
    "ConversionError",
    "Rendition",
    "available_tools",
    "prepare_rendition",
    "remove_renditions",
    "view_file",
]

log = logging.getLogger(__name__)

# Збільшувати, коли змінюється вигляд копії: це інвалідовує і копії, і кеш
# парсингу (версія входить у профіль).
RENDITION_VERSION = 1

ConverterChoice = Literal["auto", "office", "libreoffice", "none"]
CancelFn = Callable[[], bool]

# Не довше 15 символів: це межа пароля Excel, і з довшим «пустушкою» Excel
# відмовляється відкривати навіть незахищені ODS / XML («Unable to get the
# Open property of the Workbooks class»).
_PASSWORD = "asist-nopw-0000"
_OOXML_NATIVE = {".docx", ".docm", ".dotx", ".pptx", ".pptm", ".ppsx"}
_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


# ----------------------------------------------------------------- помилки
class ConversionError(RuntimeError):
    """Перетворення неможливе. `code` і `hint` ідуть в UI як є."""

    def __init__(self, code: str, message: str, hint: str = "") -> None:
        super().__init__(message)
        self.code = code
        self.hint = hint


class ConversionCancelled(RuntimeError):
    """Викладач скасував обробку посеред перетворення."""


class _ToolFailed(RuntimeError):
    def __init__(self, message: str, *, encrypted: bool = False) -> None:
        super().__init__(message)
        self.encrypted = encrypted


# ------------------------------------------------------------------ копія
@dataclass(slots=True)
class Rendition:
    """Те, що читатиме конвеєр замість оригіналу."""

    path: Path
    kind: SourceKind
    tool: str = "none"
    page_labels: dict[int, str] = field(default_factory=dict)
    approximate_pages: bool = False
    warnings: list[str] = field(default_factory=list)
    duration_ms: int = 0
    reused: bool = False

    @property
    def is_copy(self) -> bool:
        return self.tool not in ("none", "stub")

    def profile(self) -> dict[str, Any]:
        """Частина профілю парсингу: інша програма — інші сторінки."""
        return {"kind": self.kind, "tool": self.tool, "version": RENDITION_VERSION}


def _view_paths(documents_dir: Path, stem: str) -> tuple[Path, Path, Path]:
    return (
        documents_dir / f"{stem}.view.pdf",
        documents_dir / f"{stem}.view.md",
        documents_dir / f"{stem}.view.json",
    )


def view_file(documents_dir: Path, original: Path) -> tuple[Path, Literal["pdf", "text"]] | None:
    """Файл для переглядача: PDF (оригінал чи копія) або текст.

    None — показати нічого (наприклад, Word, який ще не перетворено).
    """
    suffix = original.suffix.lower()
    if suffix == ".pdf":
        return original, "pdf"
    pdf, md, _meta = _view_paths(documents_dir, original.stem)
    if pdf.is_file():
        return pdf, "pdf"
    if md.is_file():
        return md, "text"
    if source_kind(original) == "text":
        return original, "text"
    return None


def remove_renditions(documents_dir: Path, stem: str) -> None:
    for path in _view_paths(documents_dir, stem):
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:  # pragma: no cover — файл зайнятий переглядачем
            log.warning("Не вдалося видалити копію %s: %s", path, exc)


def _read_meta(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_meta(path: Path, rendition: Rendition, original: Path) -> None:
    payload = {
        "version": RENDITION_VERSION,
        "kind": rendition.kind,
        "tool": rendition.tool,
        "sourceSize": original.stat().st_size,
        "labels": {str(k): v for k, v in rendition.page_labels.items()},
        "approximatePages": rendition.approximate_pages,
        "warnings": rendition.warnings,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")


# ------------------------------------------------------------- інструменти
def _com_registered(progid: str) -> bool:
    if sys.platform != "win32":
        return False
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, f"{progid}\\CLSID"):
            return True
    except OSError:
        return False


def _powershell() -> str | None:
    root = os.environ.get("SYSTEMROOT", r"C:\Windows")   # у Windows змінні без регістру
    candidate = Path(root) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
    if candidate.is_file():
        return str(candidate)
    return shutil.which("powershell") or shutil.which("pwsh")


def _find_program(env: str, names: tuple[str, ...], dirs: tuple[str, ...]) -> str | None:
    explicit = os.environ.get(env, "").strip().strip('"')
    if explicit and Path(explicit).is_file():
        return explicit
    for name in names:
        found = shutil.which(name)
        if found:
            return found
    for base in dirs:
        for name in names:
            candidate = Path(os.path.expandvars(base)) / name
            if candidate.is_file():
                return str(candidate)
    return None


def _soffice() -> str | None:
    return _find_program(
        "ASISTENT_SOFFICE",
        ("soffice.exe", "soffice") if sys.platform == "win32" else ("soffice", "libreoffice"),
        (r"%ProgramFiles%\LibreOffice\program", r"%ProgramFiles(x86)%\LibreOffice\program",
         "/Applications/LibreOffice.app/Contents/MacOS", "/usr/bin", "/usr/local/bin"),
    )


def _ddjvu() -> str | None:
    return _find_program(
        "ASISTENT_DDJVU",
        ("ddjvu.exe", "ddjvu") if sys.platform == "win32" else ("ddjvu",),
        (r"%ProgramFiles(x86)%\DjVuLibre", r"%ProgramFiles%\DjVuLibre",
         "/opt/homebrew/bin", "/usr/local/bin", "/usr/bin"),
    )


def available_tools() -> dict[str, bool]:
    """Що є на цій машині — для діагностики й підказок у помилках."""
    has_ps = _powershell() is not None
    return {
        "word": has_ps and _com_registered("Word.Application"),
        "excel": has_ps and _com_registered("Excel.Application"),
        "powerpoint": has_ps and _com_registered("PowerPoint.Application"),
        "libreoffice": _soffice() is not None,
        "djvulibre": _ddjvu() is not None,
    }


# ------------------------------------------------------------- процеси
def _timeout_for(path: Path) -> float:
    """Базові 2 хвилини + 15 с на мегабайт, але не більше пів години."""
    megabytes = path.stat().st_size / (1024 * 1024)
    return min(1800.0, 120.0 + 15.0 * megabytes)


def _kill_tree(pid: int) -> None:
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True,
                       creationflags=_CREATE_NO_WINDOW, check=False)
    else:  # pragma: no cover — Windows є цільовою платформою
        with contextlib.suppress(OSError):
            os.kill(pid, 9)


def _run(cmd: list[str], *, timeout: float, cancel: CancelFn | None,
         on_abort: Callable[[], None] | None = None) -> subprocess.CompletedProcess[bytes]:
    """Запустити програму з тайм-аутом і кооперативним скасуванням."""
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            stdin=subprocess.DEVNULL, creationflags=_CREATE_NO_WINDOW)
    deadline = time.monotonic() + timeout
    while True:
        try:
            out, err = proc.communicate(timeout=0.5)
            return subprocess.CompletedProcess(cmd, proc.returncode, out, err)
        except subprocess.TimeoutExpired:
            cancelled = cancel is not None and cancel()
            if not cancelled and time.monotonic() < deadline:
                continue
            _kill_tree(proc.pid)
            if on_abort is not None:
                on_abort()
            proc.communicate()
            if cancelled:
                raise ConversionCancelled() from None
            raise _ToolFailed(f"програма не відповіла за {int(timeout)} с") from None


def _decode(data: bytes) -> str:
    for encoding in ("utf-8", "cp1251", "cp866"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")  # pragma: no cover


# ------------------------------------------------------- Microsoft Office
def _ps_literal(value: str | Path) -> str:
    """Рядок PowerShell в одинарних лапках: єдиний спецсимвол там — сама лапка."""
    return "'" + str(value).replace("'", "''") + "'"


_PS_PROLOGUE = """
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$src = __SRC__
$dst = __DST__
$pidFile = __PID__
$pw = __PW__
$m = [System.Reflection.Missing]::Value
$code = 0
$app = $null
$item = $null
function Save-NewPids([string]$name, [int[]]$before) {
    $new = @(Get-Process -Name $name -ErrorAction SilentlyContinue |
             Where-Object { $before -notcontains $_.Id } | ForEach-Object { $_.Id })
    [IO.File]::WriteAllText($pidFile, ($new -join ','))
}
"""

_PS_WORD = """
try {
    $before = @(Get-Process -Name WINWORD -ErrorAction SilentlyContinue | ForEach-Object { $_.Id })
    $app = New-Object -ComObject Word.Application
    Save-NewPids 'WINWORD' $before
    $app.Visible = $false
    $app.DisplayAlerts = 0
    try { $app.AutomationSecurity = 3 } catch {}
    $item = $app.Documents.Open($src, $false, $true, $false, $pw, $pw, $true, $pw, $pw, `
                                0, $m, $false, $false, $m, $true)
    $item.ExportAsFixedFormat($dst, 17, $false, 0, 0, 1, 1, 0, $true, $true, 1, $true, $true, $false)
    [Console]::Out.WriteLine('ASISTENT-OK')
} catch {
    [Console]::Out.WriteLine('ASISTENT-ERROR ' + $_.Exception.Message)
    $code = 3
} finally {
    # У Word параметри Close/Quit — ByRef: голий `Quit(0)` PowerShell відхиляє
    # («should be a PSReference»), catch його ковтає, і WINWORD.EXE лишається
    # жити у фоні після кожного документа. Звідси [ref].
    if ($item -ne $null) { try { $item.Close([ref]0) } catch {} }
    if ($app -ne $null) {
        try { $app.Quit([ref]0) } catch { try { $app.Quit() } catch {} }
        try { [void][Runtime.InteropServices.Marshal]::ReleaseComObject($app) } catch {}
    }
}
exit $code
"""

_PS_POWERPOINT = """
try {
    $running = @(Get-Process -Name POWERPNT -ErrorAction SilentlyContinue).Count -gt 0
    $before = @(Get-Process -Name POWERPNT -ErrorAction SilentlyContinue | ForEach-Object { $_.Id })
    $app = New-Object -ComObject PowerPoint.Application
    if (-not $running) { Save-NewPids 'POWERPNT' $before }
    try { $app.DisplayAlerts = 1 } catch {}
    try { $app.AutomationSecurity = 3 } catch {}
    $item = $app.Presentations.Open($src, -1, 0, 0)
    $shown = @()
    foreach ($slide in $item.Slides) {
        if ($slide.SlideShowTransition.Hidden -eq 0) { $shown += $slide.SlideIndex }
    }
    $item.SaveAs($dst, 32)
    [Console]::Out.WriteLine('ASISTENT-OK slides=' + ($shown -join ','))
} catch {
    [Console]::Out.WriteLine('ASISTENT-ERROR ' + $_.Exception.Message)
    $code = 3
} finally {
    if ($item -ne $null) { try { $item.Close() } catch {} }
    if ($app -ne $null) {
        if (-not $running) { try { $app.Quit() } catch {} }
        try { [void][Runtime.InteropServices.Marshal]::ReleaseComObject($app) } catch {}
    }
}
exit $code
"""

_PS_EXCEL = """
try {
    $before = @(Get-Process -Name EXCEL -ErrorAction SilentlyContinue | ForEach-Object { $_.Id })
    $app = New-Object -ComObject Excel.Application
    Save-NewPids 'EXCEL' $before
    $app.Visible = $false
    $app.DisplayAlerts = $false
    $app.ScreenUpdating = $false
    $app.EnableEvents = $false
    $app.AskToUpdateLinks = $false
    try { $app.AutomationSecurity = 3 } catch {}
    # Format (4-й параметр) — лише для текстових файлів; з явним значенням
    # Excel відмовляється відкривати ODS і XML 2003 («Unable to get the Open
    # property»), тому Missing.
    $item = $app.Workbooks.Open($src, 0, $true, $m, $pw, $pw, $true)
    $item.SaveAs($dst, 51)
    [Console]::Out.WriteLine('ASISTENT-OK')
} catch {
    [Console]::Out.WriteLine('ASISTENT-ERROR ' + $_.Exception.Message)
    $code = 3
} finally {
    if ($item -ne $null) { try { $item.Close($false) } catch {} }
    if ($app -ne $null) {
        try { $app.Quit() } catch {}
        try { [void][Runtime.InteropServices.Marshal]::ReleaseComObject($app) } catch {}
    }
}
exit $code
"""

_OFFICE_SCRIPTS = {"word": _PS_WORD, "powerpoint": _PS_POWERPOINT, "excel": _PS_EXCEL}
_OFFICE_PROGIDS = {
    "word": "Word.Application",
    "powerpoint": "PowerPoint.Application",
    "excel": "Excel.Application",
}


def office_script(app: str, src: Path, dst: Path, pid_file: Path) -> str:
    """Повний текст сценарію PowerShell (окремо — щоб його можна було тестувати)."""
    prologue = (
        _PS_PROLOGUE.replace("__SRC__", _ps_literal(src))
        .replace("__DST__", _ps_literal(dst))
        .replace("__PID__", _ps_literal(pid_file))
        .replace("__PW__", _ps_literal(_PASSWORD))
    )
    return prologue + _OFFICE_SCRIPTS[app]


def _looks_encrypted(message: str) -> bool:
    lowered = message.lower()
    return any(mark in lowered for mark in ("password", "парол", "encrypt", "зашифр"))


def _listed_pids(pid_file: Path) -> list[int]:
    try:
        raw = pid_file.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    return [int(t) for t in (s.strip() for s in raw.replace(";", ",").split(",")) if t.isdigit()]


def _kill_listed(pid_file: Path) -> None:
    for pid in _listed_pids(pid_file):
        _kill_tree(pid)


def _alive(pid: int) -> bool:
    if sys.platform != "win32":  # pragma: no cover
        return False
    import ctypes

    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(0x1000, False, pid)   # QUERY_LIMITED_INFORMATION
    if not handle:
        return False
    try:
        code = ctypes.c_ulong()
        kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        return code.value == 259                        # STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


def _reap(pid_file: Path, grace_s: float = 5.0) -> None:
    """Добити НАШ екземпляр Office, якщо він не вийшов сам після Quit().

    Фоновий WINWORD.EXE без вікна викладач не бачить і не закриє — а кожен
    тримає 100+ МБ і блокує файл. Після сценарію даємо кілька секунд на
    штатний вихід, потім знімаємо.
    """
    pids = _listed_pids(pid_file)
    deadline = time.monotonic() + grace_s
    while pids and time.monotonic() < deadline:
        pids = [p for p in pids if _alive(p)]
        if pids:
            time.sleep(0.25)
    for pid in pids:
        log.warning("Office (PID %s) не завершився сам — завершую примусово.", pid)
        _kill_tree(pid)


def _office(app: str, src: Path, dst: Path, *, timeout: float,
            cancel: CancelFn | None) -> str:
    """Перетворити програмою Office. Повертає рядок «ASISTENT-OK …»."""
    exe = _powershell()
    if exe is None:  # pragma: no cover — PowerShell є в кожній Windows
        raise _ToolFailed("PowerShell не знайдено")
    pid_file = dst.with_name(dst.name + ".pid")
    script = office_script(app, src, dst, pid_file)
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    cmd = [exe, "-NoProfile", "-NonInteractive", "-NoLogo", "-EncodedCommand", encoded]
    try:
        result = _run(cmd, timeout=timeout, cancel=cancel, on_abort=lambda: _kill_listed(pid_file))
        _reap(pid_file)
    finally:
        pid_file.unlink(missing_ok=True)
    output = _decode(result.stdout).strip()
    marker = next((line for line in output.splitlines() if line.startswith("ASISTENT-")), "")
    if result.returncode != 0 or not marker.startswith("ASISTENT-OK"):
        message = marker.removeprefix("ASISTENT-ERROR").strip() or _decode(result.stderr).strip()
        raise _ToolFailed(message or f"код виходу {result.returncode}",
                          encrypted=_looks_encrypted(message))
    if not dst.is_file() or dst.stat().st_size == 0:
        raise _ToolFailed("програма завершилась, але файл не створено")
    return marker


# ------------------------------------------------------------- LibreOffice
def _libreoffice(src: Path, dst: Path, target: str, *, timeout: float,
                 cancel: CancelFn | None, profile_dir: Path) -> None:
    exe = _soffice()
    if exe is None:
        raise _ToolFailed("LibreOffice не знайдено")
    out_dir = dst.parent / f".convert-{dst.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        cmd = [
            exe, f"-env:UserInstallation={profile_dir.resolve().as_uri()}",
            "--headless", "--invisible", "--norestore", "--nologo", "--nodefault",
            "--nolockcheck", "--convert-to", target, "--outdir", str(out_dir), str(src),
        ]
        result = _run(cmd, timeout=timeout, cancel=cancel)
        produced = out_dir / f"{src.stem}.{target}"
        if not produced.is_file() or produced.stat().st_size == 0:
            message = _decode(result.stderr).strip() or _decode(result.stdout).strip()
            raise _ToolFailed(message or "LibreOffice не створив файл",
                              encrypted=_looks_encrypted(message))
        shutil.move(str(produced), str(dst))
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


# ------------------------------------------------------------------ DjVu
def _djvu(src: Path, dst: Path, *, timeout: float, cancel: CancelFn | None) -> None:
    exe = _ddjvu()
    if exe is None:
        raise _ToolFailed("DjVuLibre не знайдено")
    # `-quality` стискає растр у JPEG: без нього сторінка А4 300 dpi лягає в
    # PDF нестиснутою (~25 МБ). Старі збірки ключа не знають — тоді вдруге без.
    attempts = ([exe, "-format=pdf", "-quality=85", str(src), str(dst)],
                [exe, "-format=pdf", str(src), str(dst)])
    message = ""
    for cmd in attempts:
        result = _run(cmd, timeout=timeout, cancel=cancel)
        if result.returncode == 0 and dst.is_file() and dst.stat().st_size > 0:
            return
        message = _decode(result.stderr).strip() or f"код виходу {result.returncode}"
    raise _ToolFailed(message)


# ------------------------------------------------------- власні читачі
def _docling_markdown(src: Path, kind: SourceKind) -> tuple[str, dict[int, str], bool]:
    """DOCX / PPTX без Office і LibreOffice — читачами самого Docling."""
    from docling.document_converter import DocumentConverter

    document = DocumentConverter().convert(str(src), raises_on_error=True).document
    if kind == "slides":
        # PPTX-читач Docling знає слайди як сторінки — їх і зберігаємо.
        numbers = sorted(int(n) for n in (document.pages or {}))
        parts = [document.export_to_markdown(page_no=n, image_placeholder="") for n in numbers]
        markdown = "".join(part.strip() + "\n\f\n" for part in parts)
        return markdown, {i: f"слайд {n}" for i, n in enumerate(numbers, start=1)}, False
    markdown = document.export_to_markdown(image_placeholder="")
    return markdown.strip() + "\n", {}, True


# ------------------------------------------------------------------ вибір
def _chain(kind: SourceKind, converters: ConverterChoice) -> list[str]:
    """Програми для перетворення у порядку переваги."""
    if converters == "none":
        return []
    office = converters in ("auto", "office")
    libre = converters in ("auto", "libreoffice")
    tools = available_tools()
    chain: list[str] = []
    if kind == "djvu":
        return ["djvulibre"] if tools["djvulibre"] else []
    app = {"word": "word", "slides": "powerpoint", "sheet": "excel"}.get(kind)
    if app and office and tools[app]:
        chain.append(app)
    if libre and tools["libreoffice"]:
        chain.append("libreoffice")
    return chain


def _missing_hint(kind: SourceKind, suffix: str) -> tuple[str, str]:
    if kind == "djvu":
        return (
            "Для DjVu на цьому комп'ютері немає перетворювача (DjVuLibre).",
            "Встановіть безкоштовну програму DjVuLibre (з неї потрібен ddjvu.exe) "
            "або збережіть файл як PDF у WinDjView і завантажте ще раз.",
        )
    what = {"word": "Word", "slides": "PowerPoint", "sheet": "Excel"}.get(kind, "Office")
    return (
        f"Файл {suffix} нічим перетворити: на комп'ютері немає ні Microsoft {what}, "
        f"ні LibreOffice.",
        f"Встановіть Microsoft Office або безкоштовний LibreOffice — або збережіть файл "
        f"у {what} як PDF{' чи XLSX' if kind == 'sheet' else ''} і завантажте ще раз.",
    )


def _to_pdf(original: Path, dst: Path, kind: SourceKind, chain: list[str], *,
            cancel: CancelFn | None, profile_dir: Path) -> tuple[str, dict[int, str], list[str]]:
    """Перепробувати програми ланцюжка. Повертає (програма, мітки, помилки)."""
    timeout = _timeout_for(original)
    errors: list[str] = []
    for tool in chain:
        dst.unlink(missing_ok=True)
        try:
            if tool in _OFFICE_SCRIPTS:
                marker = _office(tool, original, dst, timeout=timeout, cancel=cancel)
                labels: dict[int, str] = {}
                if tool == "powerpoint" and "slides=" in marker:
                    shown = [s for s in marker.split("slides=", 1)[1].split(",") if s.strip()]
                    labels = {i: f"слайд {s.strip()}" for i, s in enumerate(shown, start=1)}
                return tool, labels, errors
            if tool == "libreoffice":
                _libreoffice(original, dst, "pdf", timeout=timeout, cancel=cancel,
                             profile_dir=profile_dir)
                return tool, {}, errors
            if tool == "djvulibre":
                _djvu(original, dst, timeout=timeout, cancel=cancel)
                return tool, {}, errors
        except _ToolFailed as exc:
            if exc.encrypted:
                raise ConversionError(
                    "OFFICE_ENCRYPTED",
                    f"Файл захищено паролем ({tool}: {exc}).",
                    "Зніміть пароль у Word / Excel / PowerPoint, збережіть і завантажте ще раз.",
                ) from exc
            log.warning("Перетворення %s через %s не вдалося: %s", original.name, tool, exc)
            errors.append(f"{tool}: {exc}")
    return "", {}, errors


def prepare_rendition(
    original: Path,
    *,
    documents_dir: Path,
    converters: ConverterChoice = "auto",
    stub: bool = False,
    cancel: CancelFn | None = None,
    profile_dir: Path | None = None,
    title: str = "",
) -> Rendition:
    """Підготувати файл, який читатиме конвеєр.

    Для PDF і тексту це сам оригінал. Для решти — копія поруч із ним; уже
    готова копія тієї самої версії перевикористовується.
    """
    kind = source_kind(original)
    if kind is None:
        raise ConversionError(
            "UNSUPPORTED_FORMAT",
            f"Формат {original.suffix or original.name} не підтримується.",
            "Збережіть файл як PDF, DOCX або XLSX і завантажте ще раз.",
        )
    if kind in ("pdf", "text"):
        return Rendition(path=original, kind=kind)

    started = time.perf_counter()
    stem = original.stem
    pdf_path, md_path, meta_path = _view_paths(documents_dir, stem)
    profile_dir = profile_dir or (documents_dir.parent / "cache" / "libreoffice-profile")

    if kind in ("sheet", "xml"):
        return _text_rendition(original, kind, md_path, meta_path, converters=converters,
                               stub=stub, cancel=cancel, profile_dir=profile_dir,
                               started=started, title=title)

    if stub:
        # Stub-режим не запускає жодної сторонньої програми: CI і тести не
        # мають залежати від того, чи стоїть на машині Word.
        return Rendition(path=original, kind=kind, tool="stub")

    meta = _read_meta(meta_path)
    if (meta.get("version") == RENDITION_VERSION and meta.get("sourceSize") == original.stat().st_size
            and meta.get("tool")):
        target = pdf_path if pdf_path.is_file() else md_path if md_path.is_file() else None
        if target is not None and target.stat().st_size > 0:
            return Rendition(
                path=target, kind=kind, tool=str(meta["tool"]),
                page_labels={int(k): str(v) for k, v in (meta.get("labels") or {}).items()},
                approximate_pages=bool(meta.get("approximatePages")),
                warnings=list(meta.get("warnings") or []), reused=True,
            )

    remove_renditions(documents_dir, stem)
    chain = _chain(kind, converters)
    tool, labels, errors = _to_pdf(original, pdf_path, kind, chain, cancel=cancel,
                                   profile_dir=profile_dir)
    if tool:
        rendition = Rendition(path=pdf_path, kind=kind, tool=tool, page_labels=labels)
    elif original.suffix.lower() in _OOXML_NATIVE:
        rendition = _native_rendition(original, kind, md_path, errors)
    elif errors:
        raise ConversionError(
            "CONVERT_FAILED",
            "Не вдалося перетворити файл: " + "; ".join(errors),
            "Відкрийте файл у Word / PowerPoint, збережіть як PDF і завантажте PDF.",
        )
    else:
        message, hint = _missing_hint(kind, original.suffix.lower())
        raise ConversionError("CONVERTER_MISSING", message, hint)

    rendition.duration_ms = int((time.perf_counter() - started) * 1000)
    _write_meta(meta_path, rendition, original)
    log.info("Копія %s → %s (%s, %d мс)", original.name, rendition.path.name,
             rendition.tool, rendition.duration_ms)
    return rendition


def _native_rendition(original: Path, kind: SourceKind, md_path: Path,
                      errors: list[str]) -> Rendition:
    try:
        markdown, labels, approximate = _docling_markdown(original, kind)
    except ImportError as exc:
        message, hint = _missing_hint(kind, original.suffix.lower())
        raise ConversionError("CONVERTER_MISSING", f"{message} ({exc})", hint) from exc
    except Exception as exc:
        raise ConversionError(
            "CONVERT_FAILED",
            f"Не вдалося прочитати файл: {exc}",
            "Перевірте, чи файл не пошкоджений: відкрийте й перезбережіть його.",
        ) from exc
    if not markdown.strip():
        raise ConversionError("NO_TEXT", "У файлі немає тексту.", "Перевірте вміст файлу.")
    md_path.write_text(markdown, encoding="utf-8")
    warnings = [f"Не вдалося перетворити на PDF ({'; '.join(errors)})."] if errors else []
    if approximate:
        warnings.append(
            "Word / LibreOffice не знайдено: документ прочитано без верстки, тому "
            "номери сторінок у цитатах приблизні (≈ 1800 символів на сторінку)."
        )
    return Rendition(path=md_path, kind=kind, tool="docling", page_labels=labels,
                     approximate_pages=approximate, warnings=warnings)


def _text_rendition(original: Path, kind: SourceKind, md_path: Path, meta_path: Path, *,
                    converters: ConverterChoice, stub: bool, cancel: CancelFn | None,
                    profile_dir: Path, started: float, title: str) -> Rendition:
    """Таблиці й XML → Markdown. Дешево (мілісекунди), тож без кешу копії."""
    warnings: list[str] = []
    labels: dict[int, str] = {}
    tool = kind
    if kind == "xml":
        from app.ingestion.xml_text import xml_to_markdown

        try:
            markdown = xml_to_markdown(original)
        except Exception as exc:
            raise ConversionError(
                "PARSE_FAILED", f"Не вдалося прочитати XML: {exc}",
                "Перевірте, чи файл — коректний XML (відкривається в браузері).",
            ) from exc
    else:
        from app.ingestion.sheets import spreadsheet_to_markdown

        source = original
        suffix = original.suffix.lower()
        needs_excel = suffix in (".ods", ".xml") or (suffix == ".xls" and not _has_xlrd())
        if needs_excel:
            if stub:
                raise ConversionError(
                    "CONVERTER_MISSING", f"Формат {suffix} потребує Excel або LibreOffice.",
                    "Збережіть таблицю як XLSX і завантажте ще раз.",
                )
            source = md_path.with_suffix(".xlsx")
            chain = _chain("sheet", converters)
            tool, errors = _to_xlsx(original, source, chain, cancel=cancel, profile_dir=profile_dir)
            if not tool:
                if errors:
                    raise ConversionError(
                        "CONVERT_FAILED", "Не вдалося перетворити таблицю: " + "; ".join(errors),
                        "Відкрийте файл в Excel, збережіть як XLSX і завантажте ще раз.",
                    )
                message, hint = _missing_hint("sheet", suffix)
                raise ConversionError("CONVERTER_MISSING", message, hint)
        try:
            result = spreadsheet_to_markdown(source, csv_name=title)
        except Exception as exc:
            raise ConversionError(
                "PARSE_FAILED", f"Не вдалося прочитати таблицю: {exc}",
                "Перевірте, чи файл не пошкоджений і не захищений паролем.",
            ) from exc
        finally:
            if source != original:
                source.unlink(missing_ok=True)
        markdown, labels = result.markdown, result.page_labels
        warnings.extend(result.warnings)
        tool = "sheet" if source == original else f"{tool}+sheet"
    if not markdown.strip():
        raise ConversionError(
            "NO_TEXT", "У файлі не знайдено жодної заповненої клітинки чи тексту.",
            "Перевірте, чи це потрібний файл.",
        )
    md_path.write_text(markdown, encoding="utf-8")
    rendition = Rendition(path=md_path, kind=kind, tool=tool, page_labels=labels,
                          warnings=warnings,
                          duration_ms=int((time.perf_counter() - started) * 1000))
    _write_meta(meta_path, rendition, original)
    return rendition


def _to_xlsx(original: Path, dst: Path, chain: list[str], *, cancel: CancelFn | None,
             profile_dir: Path) -> tuple[str, list[str]]:
    timeout = _timeout_for(original)
    errors: list[str] = []
    for tool in chain:
        dst.unlink(missing_ok=True)
        try:
            if tool == "excel":
                _office("excel", original, dst, timeout=timeout, cancel=cancel)
            else:
                _libreoffice(original, dst, "xlsx", timeout=timeout, cancel=cancel,
                             profile_dir=profile_dir)
            return tool, errors
        except _ToolFailed as exc:
            if exc.encrypted:
                raise ConversionError(
                    "OFFICE_ENCRYPTED", f"Таблицю захищено паролем ({exc}).",
                    "Зніміть пароль в Excel, збережіть і завантажте ще раз.",
                ) from exc
            errors.append(f"{tool}: {exc}")
    return "", errors


def _has_xlrd() -> bool:
    try:
        import xlrd  # noqa: F401
    except ImportError:
        return False
    return True
