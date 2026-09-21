"""HTTP-шар: роутери, схеми, шина подій, стан застосунку.

Публічний інтерфейс модуля (див. docs/CONTRACT.md):
    create_app(settings) -> FastAPI   — у `app/main.py`
    EventBus                          — єдиний SSE-канал
    Services                          — БД, черга, моделі, наглядач

Роутери НЕ піднімаються сюди: `routes_chat` тягне генерацію, `routes_system`
— пробу заліза, і сам факт `import app.api` не має за них платити.
"""

from __future__ import annotations

from app.api.events import EventBus, sse_frame
from app.api.state import Services, build_services

__all__ = ["EventBus", "Services", "build_services", "sse_frame"]
