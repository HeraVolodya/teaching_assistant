# Контракт модулів

Спільний фундамент уже написано й перевірено. Кожен модуль кодується **проти
цього контракту** й не імпортує типи інших модулів.

## Уже існує — читати, не переписувати

| Файл | Що дає |
|---|---|
| `backend/app/domain.py` | Усі доменні типи: `Chunk`, `Document`, `Collection`, `Assistant`, `AssistantConfig`, `PageInfo`, `BBox`, `RetrievedChunk`, `RetrievalDebug`, `Citation`, `GenerationResult`, енуми |
| `backend/app/db/database.py` | `Database` (`.connection()`, `.transaction()`), runner міграцій |
| `backend/app/db/repositories.py` | `AssistantRepo`, `CollectionRepo`, `DocumentRepo`, `PageRepo`, `ChapterRepo`, `ChunkRepo`, `JobRepo`, `ChatRepo`, `TelemetryRepo` |
| `backend/app/db/migrations/0001_initial.sql` | 29 таблиць. **Схему не міняти** — додавати `0002_*.sql`, якщо треба |
| `backend/app/config.py` | `Paths.resolve().ensure()`, `export_model_env()`, `SQLITE_PRAGMAS` |
| `backend/app/net_guard.py` | `install()`, `OutboundNetworkBlocked`. **Викликати `install()` першим ділом у кожному процесі** |
| `backend/app/embeddings/registry.py` | `get(name)`, `model_key(model)`, `EmbeddingModel` з точними шаблонами префіксів |

## Межі модулів — писати ТІЛЬКИ у свій каталог

| Модуль | Каталог | Публічний інтерфейс |
|---|---|---|
| приймання | `backend/app/ingestion/` | `parse_document(path, opts) -> ParsedDocument`, `chunk_document(parsed, cfg) -> list[Chunk]`, `normalize_uk(text) -> str`, `lemmatize(tokens, lang) -> LemmaStream` |
| ембединги | `backend/app/embeddings/` | `EmbeddingProvider.embed_queries(list[str]) -> np.ndarray`, `.embed_documents(list[str]) -> np.ndarray`, `run_selftest() -> SelfTestReport` |
| реранкінг | `backend/app/rerank/` | `Reranker.score(query, list[str]) -> list[float]` |
| пошук | `backend/app/retrieval/` | `VectorIndex` (build/search/mmap), `HybridRetriever.retrieve(query, cfg) -> tuple[list[RetrievedChunk], RetrievalDebug]` |
| LLM | `backend/app/backends/` | `LlmBackend` (`chat_stream`, `health`, `load_model`, `list_models`), `probe_hardware() -> Hardware`, `recommend_model(hw) -> ModelSpec` |
| генерація | `backend/app/generation/` | `build_prompt(...) -> Messages`, `parse_citations(text, mapping) -> tuple[str, list[Citation], list[str]]`, `Generator.answer(...) -> AsyncIterator[AnswerChunk]` |
| API + воркер | `backend/app/api/`, `backend/app/jobs/`, `backend/app/main.py` | FastAPI-застосунок, SSE, цикл воркера |
| оцінювання | `backend/app/eval/`, `backend/scripts/` | `run_retrieval_eval(...) -> Metrics`, скрипт OCR-бейк-офу |

## Незламні правила

1. **API-процес НІКОЛИ не імпортує `torch`, `docling`, `rapidocr`.** `import torch` коштує
   1.5–4 с на Windows; в API це означає 6+ секунд до чутливості. Тільки воркер.
2. **Ембедер і реранкер — прямі сесії ONNX Runtime.** Не `sentence-transformers`
   (жорстко вимагає torch навіть із `backend="onnx"`), не `fastembed` (немає Qwen3).
3. **Усе має працювати без моделей.** Обов'язковий stub-режим (`ASISTENT_STUB=1`):
   детерміновані псевдовектори з хешу, ехо-LLM. Це патерн `LLAMA_CPP_STUB` із
   on-device POC NeoLens, і саме він дозволяє ганяти весь UI і CI без 1.5 ГБ моделей.
4. **Бюджет чанка — у СИМВОЛАХ.** L2: ціль 1400, м'який максимум 2200, жорсткий 3600,
   мінімум 250. L1: стеля 6000. Перекриття 0 на справжніх межах, 300 символів лише
   на рекурсивному фолбеку.
5. **Ніколи `OcrAutoOptions`.** Windows → `RapidOcrOptions(lang=["eslav"])`,
   macOS → `OcrMacOptions(lang=["uk-UA","ru-RU","en-US"])`.
   Плюс `HeadingHierarchyOptions(enabled=True)` **і** `generate_parsed_pages=True`.
6. **Кожен термін у FTS5 MATCH екранується** через `'"' + t.replace('"','""') + '"'`.
   Голий `Д-30` парситься як NOT і кидає `no such column: 30`.
7. **Модель бачить `[n]`, не UUID.** Порядкові номери запиту; мапа `[n] → chunk_uid`
   зберігається на кожне повідомлення асистента.
8. **Пороги виконуються в коді, не в промпті.** Промптом неможливо надійно змусити
   12B-модель відмовитись; порогом реранкера — можна.
9. Українські рядки в UI й повідомленнях помилок. Коментарі в коді — українською.
10. Тести кладуться в `backend/tests/`, іменуються `test_<модуль>_*.py`.

## Запуск

```bash
cd backend && .venv/bin/python -m pytest tests/ -q      # тести
ASISTENT_STUB=1 .venv/bin/python -m uvicorn app.main:app --port 8765   # API
```
