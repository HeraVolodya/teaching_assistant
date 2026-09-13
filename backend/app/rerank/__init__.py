"""Реранкінг і утримання від відповіді.

Публічний контракт модуля (docs/CONTRACT.md):
    `Reranker.score(query, list[str]) -> list[float]`

Імпорти навмисно НЕ підняті сюди: `reranker` тягне numpy і (за потреби)
onnxruntime, а `abstention` потрібен API-процесу навіть тоді, коли реранкер
вимкнено. Імпортуйте підмодулі явно.
"""

from __future__ import annotations

__all__: list[str] = []
