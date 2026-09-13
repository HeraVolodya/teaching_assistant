-- Асістент — початкова схема бази знань.
--
-- Принцип: SQLite — джерело істини для ВСЬОГО (асистенти, документи, чанки,
-- канонічні вектори, історія чату, черга завдань, прогони оцінювання).
-- ANN-індекс USearch — перебудовуваний сайдкар на диску, а не джерело істини.
--
-- КРИТИЧНО: chunks.bbox_json і chunks.page_from існують із першого дня.
-- Без них підсвітка цитованого фрагмента в PDF неможлива, а доробка потім
-- вимагає повної переіндексації всього корпусу.

PRAGMA foreign_keys = ON;

-- ---------------------------------------------------------------- версіювання
CREATE TABLE schema_version (
    version     INTEGER NOT NULL,
    applied_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);
INSERT INTO schema_version (version) VALUES (1);

-- ---------------------------------------------------------------- асистенти
-- config_json — версійований конфіг із дискримінованим об'єднанням (як в
-- NeoLens), щоб змінювати поведінку асистента без релізу коду. Завдання 4 НДР.
CREATE TABLE assistants (
    id              TEXT    PRIMARY KEY,
    name            TEXT    NOT NULL,
    description     TEXT    NOT NULL DEFAULT '',
    colour          TEXT    NOT NULL DEFAULT '#4f46e5',
    emoji           TEXT    NOT NULL DEFAULT '📘',
    config_json     TEXT    NOT NULL,
    config_version  INTEGER NOT NULL DEFAULT 1,
    created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- ---------------------------------------------------------------- колекції
-- Одна колекція = один фізичний файл ANN-індексу. Ізоляція асистентів
-- фізична, а не через метаданий фільтр — нуль втрати recall.
CREATE TABLE collections (
    id                  TEXT    PRIMARY KEY,
    assistant_id        TEXT    NOT NULL REFERENCES assistants(id) ON DELETE CASCADE,
    name                TEXT    NOT NULL,
    embedding_model_id  TEXT    NOT NULL,
    -- sha256(model_id|revision|dim|pooling|normalize|query_tpl|doc_tpl|text_preproc_version)
    embedding_model_key TEXT    NOT NULL,
    dim                 INTEGER NOT NULL,
    metric              TEXT    NOT NULL DEFAULT 'cos',
    -- sha256(embedding_model_key|metric|connectivity|expansion_add|max_chunk_id|count|schema_version)
    build_id            TEXT,
    index_generation    INTEGER NOT NULL DEFAULT 0,
    dirty               INTEGER NOT NULL DEFAULT 0,
    created_at          TEXT    NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX ix_collections_assistant ON collections(assistant_id);

-- ---------------------------------------------------------------- документи
-- stored_path — файл під UUID, НЕ під оригінальною назвою: кириличні імена
-- плюс глибокі шляхи тривіально пробивають MAX_PATH 260 на Windows.
CREATE TABLE documents (
    id                  TEXT    PRIMARY KEY,
    collection_id       TEXT    NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
    title               TEXT    NOT NULL,
    original_name       TEXT    NOT NULL,
    stored_path         TEXT    NOT NULL,
    content_sha256      TEXT    NOT NULL,
    -- стабільний хеш від версії Docling, ревізій моделей, OCR-рушія+мови, опцій конвеєра
    parse_profile_hash  TEXT,
    doc_type            TEXT    NOT NULL DEFAULT 'textbook',
    language            TEXT    NOT NULL DEFAULT 'uk',
    year                INTEGER,
    author              TEXT,
    page_count          INTEGER NOT NULL DEFAULT 0,
    -- FAST | DEEP — режим індексації, обраний користувачем для цього документа
    ingest_mode         TEXT    NOT NULL DEFAULT 'FAST',
    status              TEXT    NOT NULL DEFAULT 'QUEUED',
    error_code          TEXT,
    error_detail        TEXT,
    -- агрегат ConfidenceReport: mean_grade найгіршої сторінки
    quality_grade       TEXT,
    created_at          TEXT    NOT NULL DEFAULT (datetime('now')),
    ingested_at         TEXT
);
CREATE INDEX ix_documents_collection ON documents(collection_id);
-- Кеш рівня документа: той самий підручник у другого асистента — миттєво.
CREATE INDEX ix_documents_cache ON documents(content_sha256, parse_profile_hash);

-- ---------------------------------------------------------------- сторінки
-- page_number  — ФІЗИЧНИЙ індекс у файлі (1-based)
-- page_label   — ДРУКОВАНА мітка ("2-1", "147", "xii"); резолвиться з
--                /PageLabels, далі регресією по колонтитулах, далі = physical
CREATE TABLE pages (
    id              INTEGER PRIMARY KEY,
    document_id     TEXT    NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_number     INTEGER NOT NULL,
    page_label      TEXT,
    char_from       INTEGER,
    char_to         INTEGER,
    width           REAL,
    height          REAL,
    -- тріаж рівня 0
    page_class      TEXT,     -- DIGITAL_CLEAN | DIGITAL_BROKEN | SCANNED | MIXED
    ocr_mode        TEXT,     -- DEFAULT | FULL_PAGE | LAYOUT_REGIONS
    cost_weight     REAL    NOT NULL DEFAULT 1.0,
    lexicon_hit_rate REAL,
    cyrillic_ratio  REAL,
    mojibake_ratio  REAL,
    -- ConfidenceReport від Docling
    parse_score     REAL,
    layout_score    REAL,
    table_score     REAL,
    ocr_score       REAL,
    UNIQUE (document_id, page_number)
);
CREATE INDEX ix_pages_document ON pages(document_id);

-- ---------------------------------------------------------------- розділи
-- Вкладені множини (lft/rgt), щоб "усе під Розділом 2" було range scan
-- по індексу, а не LIKE по header_path.
CREATE TABLE chapters (
    id          INTEGER PRIMARY KEY,
    document_id TEXT    NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    parent_id   INTEGER REFERENCES chapters(id) ON DELETE CASCADE,
    level       INTEGER NOT NULL,
    title       TEXT    NOT NULL,
    lft         INTEGER NOT NULL,
    rgt         INTEGER NOT NULL
);
CREATE INDEX ix_chapters_doc_lft ON chapters(document_id, lft, rgt);

-- ---------------------------------------------------------------- чанки
-- Три рівні: L0 картка документа (маршрутизація, не цитується),
--            L1 батьківська секція (показується генератору, НЕ індексується),
--            L2 листок (ЄДИНИЙ, що індексується).
--
-- Бюджет у СИМВОЛАХ, не в токенах: українська fertility коливається
-- 2.16–3.90 ток/слово між токенайзерами, тож токен-номінований чанк тихо
-- змінює обсяг тексту вдвічі при зміні embedding-моделі.
CREATE TABLE chunks (
    id                  INTEGER PRIMARY KEY,      -- == ключ у USearch
    chunk_uid           TEXT    NOT NULL UNIQUE,  -- 32 hex, контентний, стабільний
    collection_id       TEXT    NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
    document_id         TEXT    NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    parent_id           INTEGER REFERENCES chunks(id) ON DELETE CASCADE,
    level               TEXT    NOT NULL,         -- L0 | L1 | L2
    ordinal             INTEGER NOT NULL,
    header_path         TEXT    NOT NULL DEFAULT '//',
    chapter_id          INTEGER REFERENCES chapters(id) ON DELETE SET NULL,
    language            TEXT    NOT NULL DEFAULT 'uk',

    -- позиція: діапазон символів + діапазон сторінок (фізичних і друкованих)
    page_from           INTEGER,
    page_to             INTEGER,
    page_label_from     TEXT,
    page_label_to       TEXT,
    char_from           INTEGER,
    char_to             INTEGER,
    -- [{page, l, t, r, b}] у координатах PDF; ПОТРІБНО для підсвітки цитати
    bbox_json           TEXT,

    siblings_index      INTEGER NOT NULL DEFAULT 0,
    siblings_count      INTEGER NOT NULL DEFAULT 1,

    -- Чотири похідні текстові варіанти (покращення проти двох у NeoLens).
    -- embed_text   = context_note + header_path + body  → індексується
    -- rerank_text  = короткий шлях + body (без картки документа: крос-енкодери
    --                чутливі до шуму в префіксі)
    -- display_text = санітизоване тіло, яке бачить генератор
    -- body_lemmas  → у chunk_fts, не тут
    embed_text          TEXT    NOT NULL,
    rerank_text         TEXT    NOT NULL,
    display_text        TEXT    NOT NULL,
    -- Структурний контекст. Сирий, НЕ згенерований конспект: абляція UNLP 2026
    -- показала, що заміна сирого префікса на LLM-конспект погіршує (0.9346→0.9177).
    context_note        TEXT    NOT NULL DEFAULT '',

    pictures_json       TEXT    NOT NULL DEFAULT '{}',
    tables_json         TEXT    NOT NULL DEFAULT '[]',
    formulas_json       TEXT    NOT NULL DEFAULT '[]',
    -- SimHash-64 для дедуплікації майже-дублікатів між підручниками
    simhash             INTEGER,

    -- канонічний вектор: float16 little-endian, довжина 2*dim
    embedding           BLOB,
    embedding_model_key TEXT,

    created_at          TEXT    NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX ix_chunks_doc_ord   ON chunks(collection_id, document_id, ordinal);
CREATE INDEX ix_chunks_parent    ON chunks(parent_id);
CREATE INDEX ix_chunks_level     ON chunks(collection_id, level);

-- Вузька, повністю індексована таблиця фасетів для метаданого фільтрування.
-- Стаття автора називає фільтрацію за метаданими ОБОВ'ЯЗКОВОЮ, не додатковою.
CREATE TABLE chunk_facets (
    chunk_id      INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
    collection_id TEXT    NOT NULL,
    document_id   TEXT    NOT NULL,
    doc_type      TEXT    NOT NULL,
    language      TEXT    NOT NULL,
    year          INTEGER,
    chapter_id    INTEGER
);
CREATE INDEX ix_facets_coll_doc  ON chunk_facets(collection_id, document_id);
CREATE INDEX ix_facets_coll_type ON chunk_facets(collection_id, doc_type);
CREATE INDEX ix_facets_coll_lang ON chunk_facets(collection_id, language);
CREATE INDEX ix_facets_coll_year ON chunk_facets(collection_id, year);
CREATE INDEX ix_facets_coll_chap ON chunk_facets(collection_id, chapter_id);

-- ---------------------------------------------------------------- FTS5
-- Українська морфологія вирішується ДО рушія: індексується потік лем.
-- Жоден пошуковий рушій не має українського стемера (Snowball має 28 мов,
-- української серед них немає), тож пре-лематизація — єдина робоча відповідь.
--
-- tokenchars включає апостроф (п'ять, об'єкт) — інакше слово розривається.
-- remove_diacritics 2 зачіпає лише латиницю: й/ї/ґ/є НЕ згортаються.
-- Колонка codes (вага 2.5) — позначення Д-30, 2С1, ДСТУ 3008:2015; НЕ лематизуються.
CREATE VIRTUAL TABLE chunk_fts USING fts5(
    lemmas,
    forms,
    codes,
    content='',
    contentless_delete=1,
    tokenize = "unicode61 remove_diacritics 2 tokenchars '''-_.’ʼ' separators '«»„“”…'",
    prefix = '3 4'
);

-- Окрема мікро-таблиця лише для позначень: "Д30" має знаходити "Д-30".
-- Прозу триграмами НЕ індексувати — індекс потроїться.
CREATE VIRTUAL TABLE code_fts USING fts5(
    codes,
    content='',
    contentless_delete=1,
    tokenize = "trigram case_sensitive 0"
);

-- Кеш словоформ: аналізатор має викликатись раз на унікальну словоформу,
-- а не раз на токен. На 100k чанків це ~10^6 викликів (30–90 с) замість ~10^8.
CREATE TABLE lemma_cache (
    surface TEXT NOT NULL,
    lang    TEXT NOT NULL,
    lemma   TEXT NOT NULL,
    lemma2  TEXT,
    PRIMARY KEY (surface, lang)
) WITHOUT ROWID;

-- ---------------------------------------------------------------- черга завдань
-- Лізинг + heartbeat роблять чергу стійкою до падіння: на старті будь-який
-- рядок із lease_until < now скидається в QUEUED і attempts += 1.
CREATE TABLE jobs (
    id                TEXT    PRIMARY KEY,
    type              TEXT    NOT NULL,   -- PROBE | PARSE | ENRICH_FIGURES | VLM_REPAIR | CHUNK | INDEX
    document_id       TEXT    REFERENCES documents(id) ON DELETE CASCADE,
    collection_id     TEXT    REFERENCES collections(id) ON DELETE CASCADE,
    state             TEXT    NOT NULL DEFAULT 'QUEUED',
    priority          INTEGER NOT NULL DEFAULT 100,
    attempts          INTEGER NOT NULL DEFAULT 0,
    max_attempts      INTEGER NOT NULL DEFAULT 3,
    lease_until       TEXT,
    heartbeat_at      TEXT,
    cancel_requested  INTEGER NOT NULL DEFAULT 0,
    stage             TEXT,
    -- прогрес зважений за вартістю сторінки, а не лінійний лічильник:
    -- сканована таблична сторінка коштує в ~20 разів більше за порожню
    progress_weight_done  REAL NOT NULL DEFAULT 0,
    progress_weight_total REAL NOT NULL DEFAULT 0,
    created_at        TEXT    NOT NULL DEFAULT (datetime('now')),
    started_at        TEXT,
    finished_at       TEXT,
    error_code        TEXT,
    error_detail      TEXT
);
CREATE INDEX ix_jobs_claim ON jobs(state, priority, created_at);
CREATE INDEX ix_jobs_doc   ON jobs(document_id);

-- Відновлюваність живе ТУТ, а не в jobs. 1000-сторінковий підручник — 1000
-- рядків; після перезапуску відновлюємось із першої сторінки, що не DONE.
-- Без цього падіння на 940-й сторінці коштує 40 хвилин.
CREATE TABLE document_pages_state (
    document_id  TEXT    NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_number  INTEGER NOT NULL,
    state        TEXT    NOT NULL DEFAULT 'PENDING',
    -- хеш растру сторінки на 72 dpi: після оновлення моделі переобробляти
    -- лише змінені сторінки
    page_hash    TEXT,
    error_code   TEXT,
    updated_at   TEXT    NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (document_id, page_number)
) WITHOUT ROWID;

-- ---------------------------------------------------------------- чат
CREATE TABLE chat_sessions (
    id           TEXT PRIMARY KEY,
    assistant_id TEXT NOT NULL REFERENCES assistants(id) ON DELETE CASCADE,
    title        TEXT NOT NULL DEFAULT '',
    created_at   TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at   TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX ix_sessions_assistant ON chat_sessions(assistant_id);

CREATE TABLE chat_messages (
    id            TEXT    PRIMARY KEY,
    session_id    TEXT    NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role          TEXT    NOT NULL,           -- user | assistant
    content       TEXT    NOT NULL,
    -- мапа порядковий_номер_[n] -> chunk_uid, збережена НА КОЖНЕ повідомлення
    -- й накопичувана між ходами, щоб цитати зі старих ходів лишались розв'язними
    citation_map_json TEXT NOT NULL DEFAULT '{}',
    -- діагностика для екрана "Чому ця відповідь"
    retrieval_debug_json TEXT,
    abstained     INTEGER NOT NULL DEFAULT 0,
    model_id      TEXT,
    ttft_ms       INTEGER,
    tokens_out    INTEGER,
    created_at    TEXT    NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX ix_messages_session ON chat_messages(session_id, created_at);

-- ---------------------------------------------------------------- оцінювання
-- Золотий набір будується з власних матеріалів викладача; він же слугує
-- калібрувальним набором для u_lin-утримання (окупність ~38 запитів).
CREATE TABLE eval_questions (
    id             TEXT PRIMARY KEY,
    collection_id  TEXT NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
    question       TEXT NOT NULL,
    gold_chunk_uids TEXT NOT NULL DEFAULT '[]',
    gold_pages     TEXT NOT NULL DEFAULT '[]',
    note           TEXT NOT NULL DEFAULT '',
    created_at     TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE eval_runs (
    id            TEXT PRIMARY KEY,
    collection_id TEXT NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
    config_json   TEXT NOT NULL,
    started_at    TEXT NOT NULL DEFAULT (datetime('now')),
    finished_at   TEXT,
    metrics_json  TEXT
);

CREATE TABLE eval_results (
    run_id       TEXT NOT NULL REFERENCES eval_runs(id) ON DELETE CASCADE,
    question_id  TEXT NOT NULL REFERENCES eval_questions(id) ON DELETE CASCADE,
    ranked_uids  TEXT NOT NULL,
    scores_json  TEXT NOT NULL,
    latency_ms   INTEGER,
    PRIMARY KEY (run_id, question_id)
) WITHOUT ROWID;

-- ---------------------------------------------------------------- зворотний зв'язок
-- Це і є безкоштовний набір даних для оцінювання НДР.
CREATE TABLE feedback (
    id          TEXT    PRIMARY KEY,
    message_id  TEXT    REFERENCES chat_messages(id) ON DELETE CASCADE,
    chunk_uid   TEXT,
    verdict     TEXT    NOT NULL,   -- up | down
    note        TEXT    NOT NULL DEFAULT '',
    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- ---------------------------------------------------------------- телеметрія
-- ЛИШЕ локальна. Жодного мережевого виклику ніколи.
CREATE TABLE events (
    id          INTEGER PRIMARY KEY,
    ts          TEXT    NOT NULL DEFAULT (datetime('now')),
    name        TEXT    NOT NULL,
    duration_ms INTEGER,
    ok          INTEGER NOT NULL DEFAULT 1,
    error_code  TEXT,
    meta_json   TEXT    NOT NULL DEFAULT '{}'
);
CREATE INDEX ix_events_name_ts ON events(name, ts);

-- Нерозв'язані ідентифікатори цитат: безкоштовна телеметрія галюцинацій,
-- яку можна побудувати графіком у звіті з НДР.
CREATE TABLE unresolved_citations (
    id          INTEGER PRIMARY KEY,
    message_id  TEXT    REFERENCES chat_messages(id) ON DELETE CASCADE,
    emitted     TEXT    NOT NULL,
    available   TEXT    NOT NULL,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);
