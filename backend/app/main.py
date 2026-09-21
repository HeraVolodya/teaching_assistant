"""FastAPI-застосунок «Асістент»: локальний HTTP API для UI й оболонки.

ПЕРШИМ ДІЛОМ — `net_guard.install()`.
Довіра до продукту тримається на твердженні «жоден навчальний матеріал не
залишає цей комп'ютер», і це твердження мусить бути виконуваним кодом, а не
обіцянкою в README. Гард ставиться до будь-якого імпорту, що створює
HTTP-клієнти, — інакше клієнт, збудований на імпорті, лишиться незалатаним.

У ЦЬОМУ ПРОЦЕСІ НЕМАЄ TORCH, DOCLING І RAPIDOCR. НІКОЛИ.
`import torch` коштує 1.5–4 с на Windows. В API це означає 6+ секунд до того,
як вікно стане чутливим, і викладач вирішить, що застосунок зламався. Уся
важка робота живе у воркер-процесі (`app/worker.py`), і на це є ще три
причини, крім швидкості: GIL, ізоляція збоїв і скасовуваність.

Ембедер і реранкер тут МОЖНА: це прямі сесії `onnxruntime` (14 МБ на Windows),
вони потрібні на гарячому шляху запиту, і саме відмова від
`sentence-transformers` (який жорстко вимагає torch навіть із `backend="onnx"`)
робить це можливим.

УСЕ ЗА HTTP, НІЧОГО ЗА IPC ОБОЛОНКИ.
Оболонка володіє лише вікном, нативними діалогами й життєвим циклом sidecar.
Жодних `invoke()` Tauri, жодного Electron IPC. Тримаючи цю лінію, міграція в
майбутню веб-платформу = видалити оболонку. Порушивши — переписати все.
"""

from __future__ import annotations

from app import net_guard

net_guard.install()

import logging  # noqa: E402
from collections.abc import AsyncIterator  # noqa: E402
from contextlib import asynccontextmanager  # noqa: E402

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402

from app.api import (  # noqa: E402
    routes_assistants,
    routes_chat,
    routes_documents,
    routes_system,
)
from app.api.state import Services, build_services  # noqa: E402
from app.api.watcher import JobWatcher  # noqa: E402
from app.config import UnsafeDataDirectory  # noqa: E402
from app.jobs.supervisor import create_supervisor  # noqa: E402
from app.net_guard import OutboundNetworkBlocked  # noqa: E402
from app.settings import Settings, get_settings  # noqa: E402

__all__ = ["app", "create_app"]

log = logging.getLogger("asistent.api")

API_PREFIX = "/api"

# Джерела режиму розробки: `vite dev` живе на 5173, Tauri — на `tauri://`.
# Список закритий навмисно: `allow_origins=["*"]` у застосунку, що слухає
# петлю назад, робить локальний API доступним будь-якій відкритій сторінці.
DEV_ORIGINS = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:1420",
    "http://127.0.0.1:1420",
    "tauri://localhost",
    "http://tauri.localhost",
)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Зібрати застосунок.

    `settings=None` означає «прочитати середовище на СТАРТІ, а не на імпорті»:
    інакше `app = create_app()` на рівні модуля зафіксував би конфігурацію
    того процесу, який першим імпортував модуль, і тест уже не міг би її
    змінити.
    """

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        resolved = settings or get_settings()
        services = build_services(resolved)
        services.events.attach_loop()
        application.state.services = services
        application.state.settings = resolved

        services.supervisor = create_supervisor(
            resolved, db=services.db, emit=services.emit
        )
        await services.supervisor.start()
        # Спостерігач потрібен лише для ПРОЦЕСІВ: inline-воркер публікує події
        # напряму, і другий їх постачальник дав би подвійний прогрес.
        if resolved.worker_mode == "process":
            services.watcher = JobWatcher(services)
            await services.watcher.start()

        log.info(
            "Асістент API готовий: %s:%s, stub=%s, воркер=%s",
            resolved.host, resolved.port, resolved.stub, resolved.worker_mode,
        )
        try:
            yield
        finally:
            if services.watcher is not None:
                await services.watcher.stop()
            if services.supervisor is not None:
                await services.supervisor.stop()
            services.close()

    application = FastAPI(
        title="Асістент — локальний AI-асистент викладача",
        version=routes_system.APP_VERSION,
        description=(
            "Локальний API бази знань: приймання документів, гібридний пошук, "
            "генерація відповіді з покликанням на сторінки. Жодного мережевого "
            "виклику за межі 127.0.0.1."
        ),
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(DEV_ORIGINS),
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Message-Id", "Content-Range", "Accept-Ranges"],
    )

    application.include_router(routes_system.router, prefix=API_PREFIX)
    application.include_router(routes_assistants.router, prefix=API_PREFIX)
    application.include_router(routes_documents.router, prefix=API_PREFIX)
    application.include_router(routes_chat.router, prefix=API_PREFIX)

    _install_error_handlers(application)
    return application


def _install_error_handlers(application: FastAPI) -> None:
    @application.exception_handler(OutboundNetworkBlocked)
    async def _blocked(_request: Request, exc: OutboundNetworkBlocked) -> JSONResponse:
        """Гард спрацював — це подія рівня довіри, а не звичайна 500.

        Повертаємо 502 з повним текстом: якщо якась бібліотека спробувала піти
        в мережу, викладач і аудитор мусять побачити це негайно й дослівно.
        """
        log.error("Заблоковано вихідне з'єднання: %s", exc)
        return JSONResponse(
            status_code=502,
            content={"detail": str(exc), "errorCode": "OUTBOUND_BLOCKED"},
        )

    @application.exception_handler(UnsafeDataDirectory)
    async def _unsafe_dir(_request: Request, exc: UnsafeDataDirectory) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={"detail": str(exc), "errorCode": "UNSAFE_DATA_DIR"},
        )

    @application.exception_handler(KeyError)
    async def _key_error(_request: Request, exc: KeyError) -> JSONResponse:
        # Ретривер кидає KeyError на невідомій колекції; для HTTP це 404.
        return JSONResponse(
            status_code=404,
            content={"detail": str(exc.args[0] if exc.args else exc), "errorCode": "NOT_FOUND"},
        )


def services_of(application: FastAPI) -> Services:
    """Доступ до стану поза запитом — для скриптів і тестів."""
    return application.state.services


app = create_app()


def main() -> int:  # pragma: no cover — точка входу процесу
    """`python -m app.main` — запуск без uvicorn у командному рядку."""
    import uvicorn

    settings = get_settings()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        # Один воркер: SQLite у режимі WAL допускає одного письменника, і
        # другий процес API конкурував би з воркером індексації за нього.
        workers=1,
        log_level="info",
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
