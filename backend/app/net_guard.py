"""Жорсткий мережевий гард.

Довіра до продукту тримається на твердженні «жоден навчальний матеріал не
залишає цей комп'ютер». Це твердження має бути виконуваним кодом і тестом,
а не обіцянкою в README.

Гард перехоплює httpx і requests на рівні транспорту й кидає виняток на
будь-якому хості, крім явного allowlist. Docling вимагає
`enable_remote_services=True` навіть для 127.0.0.1 (він не відрізняє localhost
від хмари), тому саме цей гард — те, що робить прапорець безпечним.

Викликати `install()` ОДИН раз, першим ділом, у кожному процесі
(api і worker), до будь-яких імпортів, що створюють клієнти.
"""

from __future__ import annotations

import ipaddress
import os
import socket
from typing import Iterable
from urllib.parse import urlsplit

__all__ = ["OutboundNetworkBlocked", "install", "is_allowed", "allowed_hosts"]


class OutboundNetworkBlocked(RuntimeError):
    """Спроба вихідного з'єднання за межі локальної машини."""

    def __init__(self, host: str, port: int | None = None) -> None:
        target = f"{host}:{port}" if port is not None else host
        super().__init__(
            f"Заблоковано вихідне мережеве з'єднання до {target}. "
            f"Асістент працює у закритому контурі: дозволені лише {sorted(allowed_hosts())}. "
            f"Якщо це LM Studio — перевірте, що він слухає на 127.0.0.1."
        )
        self.host = host
        self.port = port


_LOOPBACK_NAMES = frozenset({"localhost", "localhost.localdomain", "ip6-localhost"})
_extra_hosts: set[str] = set()


def allowed_hosts() -> set[str]:
    return _LOOPBACK_NAMES | {"127.0.0.1", "::1"} | _extra_hosts


def is_allowed(host: str | None) -> bool:
    """True, якщо host — петля назад (loopback) або в явному allowlist."""
    if not host:
        return False
    h = host.strip().strip("[]").lower()
    if h in _LOOPBACK_NAMES or h in _extra_hosts:
        return True
    try:
        return ipaddress.ip_address(h).is_loopback
    except ValueError:
        return False


def _check_url(url: str) -> None:
    host = urlsplit(url).hostname
    if not is_allowed(host):
        port = urlsplit(url).port
        raise OutboundNetworkBlocked(host or url, port)


# --------------------------------------------------------------------- httpx
def _is_in_process_transport(client: Any) -> bool:
    """True, якщо клієнт НЕ використовує справжній мережевий транспорт httpx.

    У httpx реальний ввід-вивід іде рівно через `HTTPTransport` і
    `AsyncHTTPTransport`. Усе інше — ASGI/WSGI-транспорти, `_TestClientTransport`
    Starlette, саморобні mock-транспорти — працює всередині процесу й сокета
    не відкриває, тож байти фізично не залишають машину.

    Перевірка за ТИПОМ, а не за іменем хоста: `testserver` можна підробити,
    наявність сокета — ні.

    Це послаблення безпечне, бо гарантію дає не цей патч, а патч сокета нижче:
    будь-який транспорт, що справді відкриє з'єднання, буде спинений там.
    Патч httpx існує лише щоб дати зрозумілу помилку раніше й ближче до причини.
    """
    transport = getattr(client, "_transport", None)
    if transport is None:
        return False  # дефолтний транспорт httpx — мережевий
    try:
        import httpx

        return not isinstance(transport, (httpx.HTTPTransport, httpx.AsyncHTTPTransport))
    except ImportError:  # pragma: no cover
        return False


def _patch_httpx() -> bool:
    try:
        import httpx
    except ImportError:
        return False

    def _guarded(send):
        def wrapper(self, request, *args, **kwargs):
            if not _is_in_process_transport(self):
                _check_url(str(request.url))
            return send(self, request, *args, **kwargs)

        return wrapper

    def _guarded_async(send):
        async def wrapper(self, request, *args, **kwargs):
            if not _is_in_process_transport(self):
                _check_url(str(request.url))
            return await send(self, request, *args, **kwargs)

        return wrapper

    if not getattr(httpx.Client.send, "_asistent_guarded", False):
        httpx.Client.send = _guarded(httpx.Client.send)  # type: ignore[method-assign]
        httpx.Client.send._asistent_guarded = True  # type: ignore[attr-defined]
    if not getattr(httpx.AsyncClient.send, "_asistent_guarded", False):
        httpx.AsyncClient.send = _guarded_async(httpx.AsyncClient.send)  # type: ignore[method-assign]
        httpx.AsyncClient.send._asistent_guarded = True  # type: ignore[attr-defined]
    return True


# ------------------------------------------------------------------ requests
def _patch_requests() -> bool:
    try:
        from requests.adapters import HTTPAdapter
    except ImportError:
        return False

    if getattr(HTTPAdapter.send, "_asistent_guarded", False):
        return True

    original = HTTPAdapter.send

    def wrapper(self, request, *args, **kwargs):
        _check_url(request.url)
        return original(self, request, *args, **kwargs)

    wrapper._asistent_guarded = True  # type: ignore[attr-defined]
    HTTPAdapter.send = wrapper  # type: ignore[method-assign]
    return True


# -------------------------------------------------------------------- socket
def _patch_socket() -> None:
    """Останній рубіж: ловить бібліотеки, що ходять у мережу повз httpx/requests
    (huggingface_hub, urllib, aiohttp, ...)."""
    if getattr(socket.socket.connect, "_asistent_guarded", False):
        return

    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    def _guard_address(address) -> None:
        if not isinstance(address, tuple) or not address:
            return  # AF_UNIX тощо — локальні за побудовою
        host = address[0]
        port = address[1] if len(address) > 1 else None
        if not is_allowed(str(host)):
            raise OutboundNetworkBlocked(str(host), port if isinstance(port, int) else None)

    def connect(self, address):
        _guard_address(address)
        return original_connect(self, address)

    def connect_ex(self, address):
        _guard_address(address)
        return original_connect_ex(self, address)

    connect._asistent_guarded = True  # type: ignore[attr-defined]
    socket.socket.connect = connect  # type: ignore[method-assign]
    socket.socket.connect_ex = connect_ex  # type: ignore[method-assign]


def install(extra_allowed: Iterable[str] = ()) -> None:
    """Увімкнути гард. Ідемпотентно.

    `extra_allowed` існує лише для тестів (наприклад, локальна заглушка LLM на
    іншому інтерфейсі). У постачанні НЕ використовувати.

    Заодно виставляє офлайн-змінні HuggingFace, щоб відсутній артефакт падав
    голосно, а не висів на DNS.
    """
    _extra_hosts.update(h.strip().lower() for h in extra_allowed if h.strip())

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    # Не *_WARNING: на Windows без Developer Mode симлінки дають WinError 1314.
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    # Кириличні імена файлів у subprocess-межах на Windows інакше падають на cp1251.
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    _patch_httpx()
    _patch_requests()
    _patch_socket()
