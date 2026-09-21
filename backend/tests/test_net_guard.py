"""Гейт «нічого не залишає цей комп'ютер».

Це не гігієнічний тест, а перевірка головного твердження продукту. Він мусить
валити збірку. Docling вимагає enable_remote_services=True навіть для
127.0.0.1 — саме цей гард робить той прапорець безпечним.
"""

from __future__ import annotations

import socket

import pytest

from app import net_guard


@pytest.fixture(autouse=True)
def _guard() -> None:
    net_guard.install()


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1", "127.0.0.53"])
def test_loopback_is_allowed(host: str) -> None:
    assert net_guard.is_allowed(host)


@pytest.mark.parametrize(
    "host",
    [
        "huggingface.co",
        "api.openai.com",
        "generativelanguage.googleapis.com",
        "api.cohere.ai",
        "8.8.8.8",
        "192.168.1.10",   # навіть LAN: матеріали не мають лишати машину
        "10.0.0.5",
        "",
        None,
    ],
)
def test_everything_else_is_blocked(host: str | None) -> None:
    assert not net_guard.is_allowed(host)


def test_socket_connect_to_external_host_raises() -> None:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(net_guard.OutboundNetworkBlocked):
            s.connect(("huggingface.co", 443))
    finally:
        s.close()


def test_socket_connect_to_loopback_is_not_blocked_by_the_guard() -> None:
    """Гард не має заважати LM Studio. Відмова з'єднання — це нормально
    (сервер може бути вимкнений); заборона гардом — ні."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.2)
    try:
        s.connect(("127.0.0.1", 1234))
    except net_guard.OutboundNetworkBlocked:
        pytest.fail("гард заблокував localhost — LM Studio став би недосяжним")
    except OSError:
        pass  # LM Studio не запущений — очікувано
    finally:
        s.close()


def test_httpx_is_blocked_if_installed() -> None:
    httpx = pytest.importorskip("httpx")
    with httpx.Client(timeout=1.0) as c, pytest.raises(net_guard.OutboundNetworkBlocked):
        c.get("https://huggingface.co/api/models")


def test_install_sets_offline_env_vars() -> None:
    import os

    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
    # Саме HF_HUB_DISABLE_SYMLINKS, а не *_WARNING: на Windows без Developer Mode
    # симлінки дають WinError 1314.
    assert os.environ["HF_HUB_DISABLE_SYMLINKS"] == "1"


def test_error_message_is_actionable_in_ukrainian() -> None:
    err = net_guard.OutboundNetworkBlocked("huggingface.co", 443)
    text = str(err)
    assert "huggingface.co:443" in text
    assert "LM Studio" in text, "повідомлення має підказувати найімовірнішу причину"
