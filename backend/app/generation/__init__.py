"""Шар генерації: промпт, цитати, стрімінг відповіді.

Публічний інтерфейс модуля (див. docs/CONTRACT.md):
    build_prompt(...) -> Messages
    parse_citations(text, mapping) -> tuple[str, list[Citation], list[str]]
    Generator.answer(...) -> AsyncIterator[AnswerChunk]
"""

from __future__ import annotations

from app.generation.citations import (
    citations_from_chunks,
    merge_page_ranges,
    parse_citations,
    record_unresolved,
)
from app.generation.generator import CancelToken, Generator
from app.generation.prompt_builder import (
    SYSTEM_RULES_UK,
    PromptBundle,
    abstain_text,
    build_map_prompt,
    build_prompt,
    build_prompt_bundle,
    build_reduce_prompt,
    citation_map,
    estimate_tokens_uk,
    fit_evidence,
    generation_params,
)

__all__ = [
    "SYSTEM_RULES_UK",
    "CancelToken",
    "Generator",
    "PromptBundle",
    "abstain_text",
    "build_map_prompt",
    "build_prompt",
    "build_prompt_bundle",
    "build_reduce_prompt",
    "citation_map",
    "citations_from_chunks",
    "estimate_tokens_uk",
    "fit_evidence",
    "generation_params",
    "merge_page_ranges",
    "parse_citations",
    "record_unresolved",
]
