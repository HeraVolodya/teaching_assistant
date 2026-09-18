#!/usr/bin/env python3
"""Димовий тест ЗАПАКОВАНОГО інсталятора.

Це рівень, який більшість проєктів пропускає і в якому ховається більшість
регресій: усе, що ламається між «тести зелені» і «на машині викладача не
запускається», ламається саме тут — відсутній ресурс у бандлі, незапечений
підпис, забутий DLL, осиротілий процес.

Послідовність (однакова на обох ОС):
    встановити → запустити → дочекатися /api/health → завантажити 3-сторінковий
    PDF → дочекатися готовності документа → поставити питання → **ПЕРЕВІРИТИ,
    ЩО ПОВЕРНУЛАСЯ ЦИТАТА З НОМЕРОМ СТОРІНКИ** → завершити → переконатися, що
    не лишилось осиротілого python.

Стороннього LLM тут немає: запускаємо з `ASISTENT_STUB=1`, тобто відповідає
детермінована заглушка. Перевіряється конвеєр і пакування, а не якість моделі.

ПРИМІТКА ПРО ЕНДПОЇНТИ. Шляхи API нижче — контракт із модулем `backend/app/api/`.
Якщо вони розійдуться, тест впаде з явним повідомленням «ендпоїнт не знайдено»,
а не мовчки пройде. Перекриваються прапорцями, щоб не блокувати релізи через
перейменування маршруту.

Приклад:
    python scripts/smoke_installer.py --installer dist/Asistent_0.1.0_x64-setup.exe
    python scripts/smoke_installer.py --app "/Applications/Asistent.app" --skip-install
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

DEFAULT_PORT = 8765
HEALTH_PATH = "/api/health"
ASSISTANTS_PATH = "/api/assistants"
UPLOAD_PATH = "/api/collections/{collection_id}/documents"
DOCUMENT_PATH = "/api/collections/{collection_id}/documents"   # список; фільтруємо за id
SESSIONS_PATH = "/api/sessions"
CHAT_PATH = "/api/chat"

STARTUP_TIMEOUT_S = 60      # падіння через 60 с — вимога плану
INGEST_TIMEOUT_S = 300      # 3 сторінки в режимі «Швидко» + холодний старт воркера
ANSWER_TIMEOUT_S = 120

READY_STATES = {"READY"}
FAILED_STATES = {"FAILED", "CANCELLED"}

MARKER = "DERYVATSIA"       # унікальний токен, який шукаємо у відповіді


class SmokeFailure(RuntimeError):
    """Провал димового тесту — завжди з конкретною причиною."""


def log(msg: str) -> None:
    print(f"[smoke] {msg}", flush=True)


# --------------------------------------------------------------- фікстура PDF
def make_test_pdf(path: Path) -> Path:
    """Створити валідний 3-сторінковий PDF лише на стандартній бібліотеці.

    Текст латиницею навмисно: вбудувати кириличний шрифт без сторонніх
    бібліотек неможливо (base-14 Helvetica не має кириличних гліфів), а
    завдання цього тесту — довести, що працює КОНВЕЄР пакування, а не якість
    українського OCR. Українські фікстури живуть у регресійному наборі пошуку.
    """
    pages_text = [
        [
            "ARTYLERIYSKA PIDHOTOVKA - TEST DOCUMENT",
            "Storinka persha. Zahalni polozhennya.",
            f"Unikalnyi marker: {MARKER}-01",
        ],
        [
            "Storinka druha. Popravka na deryvatsiyu.",
            "Deryvatsiya - vidhylennya snaryada vbik obertannya.",
            f"Unikalnyi marker: {MARKER}-02",
        ],
        [
            "Storinka tretya. Tablytsi strilby.",
            "Dalnist 4000 m, popravka 0-12.",
            f"Unikalnyi marker: {MARKER}-03",
        ],
    ]

    objects: list[bytes] = []

    def add(body: bytes) -> int:
        objects.append(body)
        return len(objects)  # номери об'єктів починаються з 1

    font_id = None
    page_ids: list[int] = []
    content_ids: list[int] = []

    # Резервуємо 1 = Catalog, 2 = Pages; заповнимо після того, як дізнаємось id.
    objects.append(b"")  # 1
    objects.append(b"")  # 2

    for lines in pages_text:
        stream_lines = ["BT", "/F1 14 Tf", "72 770 Td", "18 TL"]
        stream_lines += [f"({line}) Tj T*" for line in lines]
        stream_lines.append("ET")
        stream = "\n".join(stream_lines).encode("ascii")
        content_ids.append(add(b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream)))

    font_id = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    for content_id in content_ids:
        page_ids.append(
            add(
                b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
                b"/Resources << /Font << /F1 %d 0 R >> >> /Contents %d 0 R >>"
                % (font_id, content_id)
            )
        )

    kids = b" ".join(b"%d 0 R" % pid for pid in page_ids)
    objects[0] = b"<< /Type /Catalog /Pages 2 0 R >>"
    objects[1] = b"<< /Type /Pages /Kids [%s] /Count %d >>" % (kids, len(page_ids))

    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets: list[int] = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"

    xref_at = len(out)
    out += b"xref\n0 %d\n" % (len(objects) + 1)
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref_at,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(out))
    return path


# ------------------------------------------------------------------- HTTP
def request(
    url: str,
    *,
    method: str = "GET",
    body: bytes | None = None,
    content_type: str | None = None,
    timeout: float = 30.0,
) -> tuple[int, bytes]:
    req = urllib.request.Request(url, data=body, method=method)
    if content_type:
        req.add_header("Content-Type", content_type)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise SmokeFailure(f"{method} {url}: {exc}") from exc


def multipart(field: str, filename: str, data: bytes, extra: dict[str, str]) -> tuple[bytes, str]:
    boundary = f"----asistent{uuid.uuid4().hex}"
    parts: list[bytes] = []
    for key, value in extra.items():
        parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{key}\"\r\n\r\n{value}\r\n".encode()
        )
    parts.append(
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"{field}\"; "
        f"filename=\"{filename}\"\r\nContent-Type: application/pdf\r\n\r\n".encode()
    )
    parts.append(data)
    parts.append(f"\r\n--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def wait_health(base: str, timeout_s: int) -> None:
    deadline = time.monotonic() + timeout_s
    last = "з'єднання не встановлено"
    while time.monotonic() < deadline:
        try:
            status, payload = request(base + HEALTH_PATH, timeout=3)
            if status == 200:
                log(f"health OK за {timeout_s - int(deadline - time.monotonic())} с: {payload[:200]!r}")
                return
            last = f"HTTP {status}"
        except SmokeFailure as exc:
            last = str(exc)
        time.sleep(0.5)
    raise SmokeFailure(f"Сервер не відповів на {base}{HEALTH_PATH} за {timeout_s} с ({last})")


# --------------------------------------------------------------- встановлення
def install_windows(installer: Path) -> Path:
    """`/S` — тихе встановлення NSIS. Режим currentUser → %LOCALAPPDATA%\\Programs."""
    run = subprocess.run([str(installer), "/S"], check=False)
    if run.returncode != 0:
        raise SmokeFailure(f"Інсталятор завершився з кодом {run.returncode}")
    local = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    exe = local / "Programs" / "Asistent" / "Asistent.exe"
    deadline = time.monotonic() + 60
    while not exe.exists() and time.monotonic() < deadline:
        time.sleep(1)
    if not exe.exists():
        raise SmokeFailure(f"Після встановлення не знайдено {exe}")
    return exe


def install_macos(installer: Path) -> Path:
    """hdiutil attach → /Applications → spctl.

    `spctl --assess` тут не формальність: він перевіряє, що нотаризація зі
    стейплом реально спрацювала. Ловити це треба в CI, а не на Mac викладача,
    де воно проявиться як «застосунок пошкоджено».
    """
    out = subprocess.run(
        ["hdiutil", "attach", "-nobrowse", "-readonly", str(installer)],
        capture_output=True, text=True, check=True,
    ).stdout
    mount = next((line.split("\t")[-1].strip() for line in out.splitlines() if "/Volumes/" in line), None)
    if not mount:
        raise SmokeFailure(f"Не вдалося змонтувати {installer}: {out}")

    try:
        app_src = next(Path(mount).glob("*.app"))
        app_dst = Path("/Applications") / app_src.name
        if app_dst.exists():
            shutil.rmtree(app_dst)
        shutil.copytree(app_src, app_dst, symlinks=True)
    finally:
        subprocess.run(["hdiutil", "detach", mount], check=False, capture_output=True)

    assess = subprocess.run(
        ["spctl", "--assess", "--type", "execute", "--verbose=4", str(app_dst)],
        capture_output=True, text=True, check=False,
    )
    log(f"spctl: rc={assess.returncode} {assess.stderr.strip()}")
    if assess.returncode != 0 and not os.environ.get("ASISTENT_ALLOW_UNNOTARIZED"):
        raise SmokeFailure(
            "spctl відхилив застосунок — нотаризація або стейпл не спрацювали:\n"
            + assess.stderr
        )
    return app_dst / "Contents" / "MacOS" / "Asistent"


# ----------------------------------------------------------------- процеси
def orphan_processes(install_root: Path) -> list[str]:
    """Процеси, чий виконуваний файл лежить у теці встановлення.

    Порівняння за ШЛЯХОМ, а не за іменем: `python.exe` викладача до нас
    стосунку не має.
    """
    root = str(install_root).rstrip("\\/")
    if platform.system() == "Windows":
        ps = (
            "Get-CimInstance Win32_Process | "
            f"Where-Object {{ $_.ExecutablePath -like '{root}*' }} | "
            "ForEach-Object { \"$($_.ProcessId) $($_.ExecutablePath)\" }"
        )
        out = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps],
            capture_output=True, text=True, check=False,
        ).stdout
    else:
        out = subprocess.run(
            ["/bin/sh", "-c", f"ps -Ao pid=,comm= | grep -F '{root}' || true"],
            capture_output=True, text=True, check=False,
        ).stdout
    return [line.strip() for line in out.splitlines() if line.strip()]


# ------------------------------------------------------------------ сценарій
def post_json(base: str, path: str, payload: dict, timeout: float = 60.0) -> dict:
    status, body = request(
        base + path,
        method="POST",
        body=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        content_type="application/json",
        timeout=timeout,
    )
    if status == 404:
        raise SmokeFailure(f"Ендпоїнт {path} не знайдено — контракт API змінився.")
    if status >= 300:
        raise SmokeFailure(f"POST {path}: HTTP {status} {body[:400]!r}")
    return json.loads(body)


def sse_events(base: str, path: str, payload: dict, timeout: float) -> list[tuple[str, dict]]:
    """Прочитати SSE-потік до кінця. Відповідь стрімиться — іншого шляху немає."""
    req = urllib.request.Request(
        base + path,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        method="POST",
    )
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "text/event-stream")
    events: list[tuple[str, dict]] = []
    event = ""
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8").rstrip("\n")
                if line.startswith("event: "):
                    event = line[7:]
                elif line.startswith("data: "):
                    try:
                        events.append((event, json.loads(line[6:])))
                    except json.JSONDecodeError:
                        events.append((event, {"raw": line[6:]}))
    except urllib.error.HTTPError as exc:
        raise SmokeFailure(f"POST {path}: HTTP {exc.code} {exc.read()[:400]!r}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise SmokeFailure(f"POST {path}: {exc}") from exc
    return events


def run_scenario(base: str) -> None:
    # 1. Асистент (він же створює власну ізольовану колекцію).
    #    `confidenceRequired: low` — димовий тест перевіряє КОНВЕЄР, а не
    #    якість пошуку: гейт якості живе в регресійному наборі, і зав'язувати
    #    на нього реліз означало б валити збірку через дрейф скорів.
    assistant = post_json(
        base,
        ASSISTANTS_PATH,
        {"name": "Димовий тест", "emoji": "🧪", "config": {"confidenceRequired": "low"}},
    )
    collections = assistant.get("collections") or []
    if not collections:
        raise SmokeFailure(f"Асистент створився без колекції: {assistant!r}")
    collection_id = collections[0]["id"]
    log(f"асистент {assistant['id']}, колекція {collection_id}")

    # 2. Завантаження 3-сторінкового PDF.
    pdf = make_test_pdf(Path("smoke-fixture.pdf"))
    body, content_type = multipart("files", pdf.name, pdf.read_bytes(), {"docType": "textbook"})
    status, payload = request(
        base + UPLOAD_PATH.format(collection_id=collection_id),
        method="POST", body=body, content_type=content_type, timeout=120,
    )
    if status >= 300:
        raise SmokeFailure(f"Завантаження PDF: HTTP {status} {payload[:400]!r}")
    uploaded = json.loads(payload)
    if not uploaded:
        raise SmokeFailure("Сервер прийняв 0 документів")
    doc_id = uploaded[0]["id"]
    log(f"документ {doc_id} прийнято")

    # 3. Чекаємо на doc.ready.
    deadline = time.monotonic() + INGEST_TIMEOUT_S
    state = "?"
    while time.monotonic() < deadline:
        status, payload = request(base + DOCUMENT_PATH.format(collection_id=collection_id), timeout=15)
        if status >= 300:
            raise SmokeFailure(f"Список документів: HTTP {status} {payload[:400]!r}")
        row = next((d for d in json.loads(payload) if d["id"] == doc_id), None)
        state = (row or {}).get("status", "?")
        if state in READY_STATES:
            break
        if state in FAILED_STATES:
            raise SmokeFailure(f"Індексація документа провалилася: {row!r}")
        time.sleep(2)
    else:
        raise SmokeFailure(f"Документ не проіндексувався за {INGEST_TIMEOUT_S} с (стан {state!r})")
    log(f"документ готовий: {state}")

    # 4. Питання до заглушки LLM.
    session = post_json(base, SESSIONS_PATH, {"assistantId": assistant["id"], "title": "Дим"})
    # Питання цитує сторінку дослівно: заглушка ембедера детермінована, але
    # не «розумна», тож семантичну близькість треба дати лексично.
    question = f"Popravka na deryvatsiyu: shcho oznachaye marker {MARKER}-02?"
    events = sse_events(
        base, CHAT_PATH, {"sessionId": session["id"], "message": question}, timeout=ANSWER_TIMEOUT_S
    )
    kinds = [name for name, _ in events]
    if "chat.error" in kinds:
        payload = next(data for name, data in events if name == "chat.error")
        raise SmokeFailure(f"Відповідь із помилкою: {payload!r}")

    citations = [c for name, data in events if name == "chat.citations" for c in data.get("citations", [])]
    if not citations:
        done = next((data for name, data in events if name == "chat.done"), {})
        debug = next((data for name, data in events if name == "chat.debug"), {})
        if done.get("abstained"):
            raise SmokeFailure(
                "Асистент утримався від відповіді, тому цитат немає. "
                f"confidence={debug.get('confidence')!r}, знайдено фрагментів={debug.get('found')!r}. "
                "Це не проблема пакування: поріг утримання спрацьовує ще до генерації "
                "(див. app/rerank/abstention.py)."
            )
        raise SmokeFailure(f"Відповідь без жодної цитати — це провал вимоги НДР. Кадри: {kinds}")

    # 5. ГОЛОВНЕ: цитата мусить нести номер сторінки.
    with_pages = [c for c in citations if c.get("pageFrom") or c.get("pageLabel")]
    if not with_pages:
        raise SmokeFailure(f"Жодна цитата не несе номера сторінки: {citations!r}")
    log(f"цитат: {len(citations)}, з номером сторінки: {len(with_pages)} — {with_pages[0]}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Димовий тест запакованого інсталятора.")
    parser.add_argument("--installer", type=Path, help="app-setup.exe або .dmg")
    parser.add_argument("--app", type=Path, help="Уже встановлений виконуваний файл (із --skip-install).")
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--make-pdf", type=Path, help="Лише створити фікстуру й вийти.")
    args = parser.parse_args(argv)

    if args.make_pdf:
        print(make_test_pdf(args.make_pdf))
        return 0

    if args.skip_install:
        if not args.app:
            parser.error("--skip-install вимагає --app")
        exe = args.app
    elif args.installer:
        # `Path("")` нормалізується в `Path(".")` і є ІСТИННИМ: у pathlib немає
        # `__bool__`. Тому порожній `--installer ""` (наприклад, коли глоб у CI
        # нічого не зіставив) проходив гілку вище й діставався `hdiutil attach .`
        # або `start /wait .` — і падав сирим CalledProcessError замість того,
        # щоб назвати причину. Перевірка існування коштує рядок, а економить
        # розбір traceback на чужій машині.
        if not args.installer.is_file():
            parser.error(f"інсталятор не знайдено: {args.installer}")
        exe = install_windows(args.installer) if platform.system() == "Windows" else install_macos(args.installer)
    else:
        parser.error("потрібен --installer або --skip-install --app")

    log(f"виконуваний файл: {exe}")
    install_root = exe.parent
    if platform.system() == "Darwin" and ".app/" in str(exe):
        install_root = Path(str(exe).split(".app/")[0] + ".app")

    env = {
        **os.environ,
        # Фіксований порт: інакше димовий тест не знав би, куди стукати.
        # Оболонка поважає ASISTENT_PORT (див. sidecar.rs::free_port).
        "ASISTENT_PORT": str(args.port),
        # Без завантажених моделей і без LM Studio: заглушки детерміновані.
        "ASISTENT_STUB": "1",
    }
    proc = subprocess.Popen([str(exe)], env=env)
    base = f"http://127.0.0.1:{args.port}"

    failure: str | None = None
    try:
        wait_health(base, STARTUP_TIMEOUT_S)
        run_scenario(base)
    except SmokeFailure as exc:
        failure = str(exc)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            failure = failure or "Застосунок не завершився за 30 с після terminate"

    # Головна перевірка kill-on-exit: після виходу оболонки не має лишитись
    # ЖОДНОГО процесу з теки встановлення. Саме це ламається в tauri#11686.
    time.sleep(3)
    orphans = orphan_processes(install_root)
    if orphans:
        failure = failure or "Осиротілі процеси після виходу:\n  " + "\n  ".join(orphans)

    if failure:
        print(f"\nДИМОВИЙ ТЕСТ ПРОВАЛЕНО: {failure}", file=sys.stderr)
        return 1

    log("димовий тест пройдено")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
