/**
 * Типи HTTP-контракту з локальним sidecar.
 *
 * ДЖЕРЕЛО ПРАВДИ — `backend/app/api/schemas.py` і роутери поруч, а НЕ
 * `domain.py`. Різниця принципова: на дроті імена полів camelCase
 * (`alias_generator=to_camel`), а в Python — snake_case. Тому тут описано
 * саме те, що реально приходить із мережі; жодного шару перейменувань між
 * fetch і компонентом немає, бо кожен такий шар — це місце, де контракт
 * тихо розходиться з бекендом.
 *
 * Два винятки, які легко проґавити (перевірено по коду роутерів):
 *   • `GET /sessions?assistant_id=…` — параметр ЗАПИТУ лишається snake_case,
 *     бо `to_camel` перейменовує поля моделей, а не аргументи функцій;
 *   • тіла помилок FastAPI — `{detail}`, і лише власні обробники додають
 *     `errorCode`.
 *
 * Правило, важливіше за стиль: фронтенд НЕ рахує нічого, що вирішує бекенд.
 * Пороги впевненості, утримання, вибір фрагментів — усе приходить готовим.
 */

// --------------------------------------------------------------------- енуми
export type ChunkLevel = "L0" | "L1" | "L2";

export type DocStatus =
  | "QUEUED"
  | "PROBING"
  | "PARSING"
  | "ENRICHING"
  | "CHUNKING"
  | "INDEXING"
  | "READY"
  | "FAILED"
  | "CANCELLED";

export type IngestMode = "FAST" | "DEEP";

export type QualityGrade = "POOR" | "FAIR" | "GOOD" | "EXCELLENT";

export type PageClass = "DIGITAL_CLEAN" | "DIGITAL_BROKEN" | "SCANNED" | "MIXED";

export type OcrModeName = "DEFAULT" | "FULL_PAGE" | "LAYOUT_REGIONS" | "NONE";

/** Етапи конвеєра — рівно ті рядки, що їх шле `lease.checkpoint(stage=…)`. */
export type JobStage =
  | "PROBE"
  | "PARSE"
  | "ENRICH_FIGURES"
  | "VLM_REPAIR"
  | "CHUNK"
  | "INDEX"
  | "READY";

export type JobState = "QUEUED" | "LEASED" | "RUNNING" | "DONE" | "FAILED" | "CANCELLED";

/**
 * Коди помилок індексації. Перші сім реально виставляє `app/jobs/runner.py`;
 * решта — коди рівня API й транспорту. Кожен має власний український текст і
 * власну КНОПКУ ДІЇ (див. `lib/errors.ts`): «індексація не вдалась» без дії —
 * це те саме, що не сказати нічого.
 */
export type IngestErrorCode =
  | "FILE_MISSING"
  | "DOC_NOT_FOUND"
  | "NO_DOCUMENT"
  | "NO_TEXT"
  | "PARSE_FAILED"
  | "OUT_OF_MEMORY"
  | "UNSUPPORTED_FORMAT"
  | "CANCELLED"
  | "UNEXPECTED"
  | "EMBEDDER_MISSING"
  | "LMSTUDIO_UNAVAILABLE"
  | "OUTBOUND_BLOCKED"
  | "UNSAFE_DATA_DIR"
  | "SIDECAR_UNREACHABLE"
  | "UNKNOWN";

// ------------------------------------------------------------------ асистент
/** Дзеркало `AssistantConfig` із `domain.py`; на дроті — під ключем `config`. */
export interface AssistantConfig {
  instructions: string;
  topics_covered: string[];
  topics_refused: string[];
  on_missing_information: "no_information" | "general_knowledge_marked" | "refuse";
  confidence_required: "low" | "medium" | "high";
  answer_language: "match_question" | "always_uk";
  always_cite_pages: boolean;

  dense_top_k: number;
  sparse_top_k: number;
  rerank_top_k: number;
  final_top_k: number;
  max_per_document: number;
  min_distinct_documents: number;
  relative_score_floor: number;
  rrf_k: number;
  weight_dense: number;
  weight_sparse: number;
  weight_char_ngram: number;

  temperature: number;
  max_tokens: number;
  repeat_penalty: number;
  prompt_tier: "compact" | "full";
}

/**
 * `config` приходить як `asdict(AssistantConfig)`, тобто зі snake_case
 * ключами ВСЕРЕДИНІ camelCase-конверта. Це не недогляд бекенда: конфіг —
 * версійовані дані НДР (Завдання 4), і перейменування його ключів на дроті
 * зробило б збережені конфіги нечитними для наступної версії.
 */
export interface Assistant {
  id: string;
  name: string;
  description: string;
  colour: string;
  emoji: string;
  config: AssistantConfig;
  configVersion: number;
  collections: CollectionInfo[];
}

export interface CollectionInfo {
  id: string;
  name: string;
  embeddingModelId: string;
  dim: number;
  indexGeneration: number;
  /** Індекс відстає від бази: пошук працює, але не бачить свіжих документів. */
  dirty: boolean;
  documents: number;
  chunks: number;
}

export interface AssistantIn {
  name: string;
  description?: string;
  colour?: string;
  emoji?: string;
  config?: Partial<AssistantConfig>;
}

/** `GET /assistants/{id}/prompt-preview` — промпт І пороги на одному екрані. */
export interface PromptPreview {
  system: string;
  persona: string;
  rulesTokens: number;
  personaTokens: number;
  confidenceThreshold: number;
  minSupportingChunks: number;
  finalTopK: number;
  maxPerDocument: number;
  onMissingInformation: string;
  sample: string;
}

// ------------------------------------------------------------------ документ
export interface DocumentDto {
  id: string;
  collectionId: string;
  title: string;
  originalName: string;
  docType: string;
  language: string;
  year: number | null;
  author: string | null;
  pageCount: number;
  ingestMode: IngestMode;
  status: DocStatus;
  errorCode: string | null;
  errorDetail: string | null;
  qualityGrade: QualityGrade | null;
  sizeBytes: number | null;
  chunks: number;
  job: JobSummary | null;
}

export interface JobSummary {
  id: string;
  type: string;
  state: JobState;
  stage: JobStage | string | null;
  attempts: number;
  /** 0…1, ЗВАЖЕНА за вартістю сторінки, а не лінійний лічильник сторінок. */
  fraction: number;
  errorCode: string | null;
}

/** `GET /documents/{id}/pages` — посторінкові оцінки якості розбору. */
export interface PageDto {
  pageNumber: number;
  pageLabel: string | null;
  width: number;
  height: number;
  pageClass: PageClass;
  ocrMode: OcrModeName;
  costWeight: number;
  lexiconHitRate: number | null;
  cyrillicRatio: number | null;
  mojibakeRatio: number | null;
  parseScore: number | null;
  layoutScore: number | null;
  tableScore: number | null;
  ocrScore: number | null;
  needsRepair: boolean;
}

export interface UploadMeta {
  docType?: string;
  language?: string;
  year?: number | null;
  author?: string | null;
  ingestMode?: IngestMode;
}

// ------------------------------------------------------------------- чат
export interface ChatSession {
  id: string;
  assistantId: string;
  title: string;
  createdAt: string;
  updatedAt: string;
  messages: number;
}

/** Прямокутник у координатах PDF: початок — НИЖНІЙ лівий кут. */
export interface BBoxDto {
  page: number;
  l: number;
  t: number;
  r: number;
  b: number;
}

export interface CitationDto {
  ordinal: number;
  chunkUid: string;
  documentId: string;
  documentTitle: string;
  pageFrom: number | null;
  pageTo: number | null;
  pageLabel: string;
  quote: string;
  bboxes: BBoxDto[];
  language: string;
}

/**
 * Історія. `citations` тут — лише `{ordinal, chunkUid}`: повні цитати живуть
 * у кадрі `chat.citations` під час стріму й у `/messages/{id}/why` після
 * нього. Це навмисно: збережена мапа `[n] → chunk_uid` мусить лишатись
 * розв'язною після переіндексації, а копія тексту цитати — ні.
 */
export interface ChatMessage {
  id: string;
  sessionId: string;
  role: "user" | "assistant";
  content: string;
  abstained: boolean;
  modelId: string | null;
  ttftMs: number | null;
  tokensOut: number | null;
  createdAt: string;
  citations: { ordinal: number; chunkUid: string }[];
}

export interface ChatIn {
  sessionId: string;
  message: string;
  detailed?: boolean;
  documentIds?: string[];
  docTypes?: string[];
  languages?: string[];
  yearFrom?: number | null;
  yearTo?: number | null;
}

// ------------------------------------------------------- кадри SSE-каналу
export interface ChatTokenEvent {
  messageId: string;
  delta: string;
}

export interface ChatCitationsEvent {
  messageId: string;
  citations: CitationDto[];
  /** Зі злитими діапазонами сторінок — для смужки «Джерела» під відповіддю. */
  merged: CitationDto[];
  unresolved: string[];
}

export interface ChatDebugEvent {
  messageId: string;
  stage?: string;
  found?: number;
  documents?: number;
  abstained?: boolean;
  confidence?: number | null;
  latencyMs?: Record<string, number>;
  [key: string]: unknown;
}

export interface ChatDoneEvent {
  messageId: string;
  abstained: boolean;
  tokensOut: number;
  ttftMs: number | null;
  citations: number;
  elapsedMs: number;
}

export interface ChatErrorEvent {
  messageId: string;
  errorCode: string;
  message: string;
  hint: string;
}

export interface JobProgressEvent {
  jobId: string;
  docId: string | null;
  collectionId: string | null;
  stage: JobStage | string;
  current: number;
  total: number;
  fraction: number;
  etaSeconds: number | null;
}

export interface JobFailedEvent {
  jobId: string;
  docId: string | null;
  errorCode: string;
  message: string;
  hint: string;
}

export interface DocReadyEvent {
  docId: string;
  chunks: number;
  tables: number;
  formulas: number;
  pictures: number;
  pages: number;
  quality: QualityGrade | null;
  reusedFrom: string | null;
  warnings: string[];
}

export interface ModelDownloadEvent {
  downloadId: string;
  model: string;
  percent?: number | null;
  state?: string | null;
  message?: string;
}

export interface WorkerStatusEvent {
  mode: string;
  running: number;
  active?: number;
}

/**
 * Дискримінований союз усіх подій каналу `/api/events`.
 * Імена — точно ті, що в `app/api/events.py`; розходження тут означає тихо
 * загублену подію, тому вони зібрані в одному місці й перевіряються тестом.
 */
export type AppEvent =
  | { type: "job.progress"; id: number; data: JobProgressEvent }
  | { type: "job.failed"; id: number; data: JobFailedEvent }
  | { type: "doc.ready"; id: number; data: DocReadyEvent }
  | { type: "chat.token"; id: number; data: ChatTokenEvent }
  | { type: "chat.citations"; id: number; data: ChatCitationsEvent }
  | { type: "chat.debug"; id: number; data: ChatDebugEvent }
  | { type: "chat.done"; id: number; data: ChatDoneEvent }
  | { type: "chat.error"; id: number; data: ChatErrorEvent }
  | { type: "model.download"; id: number; data: ModelDownloadEvent }
  | { type: "worker.status"; id: number; data: WorkerStatusEvent };

export type AppEventType = AppEvent["type"];

// ------------------------------------------------------- «Чому ця відповідь»
export interface WhyFragment {
  ordinal: number;
  chunkUid: string;
  /** true → чанк зник при переіндексації; решта полів відсутня. */
  missing: boolean;
  documentId?: string;
  documentTitle?: string;
  headerPath?: string;
  pageFrom?: number | null;
  pageTo?: number | null;
  pageLabel?: string;
  level?: ChunkLevel;
  language?: string;
  text?: string;
  bboxes?: BBoxDto[];
}

export interface RetrievalDebugDto {
  query: string;
  dense_count: number;
  sparse_count: number;
  ngram_count: number;
  fused_count: number;
  reranked_count: number;
  final_count: number;
  distinct_documents: number;
  abstained: boolean;
  abstain_confidence: number | null;
  latency_ms: Record<string, number>;
  candidates: RetrievalCandidate[];
}

/**
 * Рядок `RetrievalDebug.candidates`. Він серіалізується `asdict()` як є,
 * тобто snake_case усередині — той самий випадок, що й `config`.
 */
export interface RetrievalCandidate {
  kind?: string;
  chunk_uid?: string;
  document_title?: string;
  pages?: string;
  rrf_score?: number;
  dense_score?: number | null;
  sparse_score?: number | null;
  ngram_score?: number | null;
  rerank?: number | null;
  selected?: boolean;
  merged_into?: string | null;
  [key: string]: unknown;
}

export interface WhyThisAnswer {
  messageId: string;
  abstained: boolean;
  modelId: string | null;
  ttftMs: number | null;
  tokensOut: number | null;
  fragments: WhyFragment[];
  unresolved: { emitted: string; available: string; created_at: string }[];
  debug: RetrievalDebugDto | null;
}

// -------------------------------------------------------------- система
export interface HealthStatus {
  ok: boolean;
  version: string;
  stub: boolean;
  uptimeSeconds: number;
  python: string;
  platform: string;
  database: { ok: boolean; detail: string; schemaVersion: number; path: string };
  lmstudio: LmStudioHealth;
  worker: { mode: string; running?: number; detail?: string; pids?: number[] };
  jobs: { active: number; queued: number };
  problems: Problem[];
  eventSubscribers: number;
  /** Заповнює КЛІЄНТ, не сервер: чи ми зараз на демонстраційних даних. */
  demo?: boolean;
}

export interface Problem {
  code: string;
  message: string;
  hint: string;
}

export interface LmStudioModel {
  key: string;
  loaded: boolean;
  kind: string;
  engine: string;
  maxContext: number | null;
  loadedContext: number | null;
  quantization: string | null;
  sizeBytes: number | null;
}

export interface LmStudioHealth {
  ok: boolean;
  /** Сервер відповідає І модель піднята в пам'ять. `ok` сам по собі цього не означає. */
  chatReady: boolean;
  baseUrl?: string;
  apiVersion: string;
  detail: string;
  models: LmStudioModel[];
}

// -------------------------------------------------------- перший запуск
export interface HardwareInfo {
  platform?: string;
  machine?: string;
  cpuCount?: number;
  ramBytes?: number;
  vramBytes?: number;
  isAppleSilicon?: boolean;
  unifiedMemory?: boolean;
  gpu?: { name: string; totalBytes: number } | null;
  describe?: string;
  error?: string;
}

export interface ModelRecommendation {
  modelKey?: string;
  title?: string;
  quant?: string;
  contextLength?: number;
  gpuOffload?: number | null;
  fileBytes?: number | null;
  /** false → параметри архітектури оцінкові, формулу VRAM ще не звірено. */
  archVerified?: boolean;
  error?: string;
}

export interface ModelFilesStatus {
  embeddings: { id: string; dim: number; dir: string; present: boolean };
  rerank: { id: string; dir: string; present: boolean };
  docling: { dir: string; present: boolean };
}

export interface SetupStatus {
  hardware: HardwareInfo;
  recommendation: ModelRecommendation;
  lmstudio: LmStudioHealth & { cliAvailable: boolean; configuredUrl: string };
  models: ModelFilesStatus;
  stub: boolean;
  dataDir: string;
  ready: boolean;
  problems: Problem[];
}

export interface LmStudioStartResult {
  ok: boolean;
  returncode: number;
  stdout: string;
  stderr: string;
  lmstudio: LmStudioHealth;
}

// -------------------------------------------------------------- playground
export interface PlaygroundBeforeRow {
  chunkUid: string | null;
  documentTitle: string | null;
  pages: string | null;
  rrf: number | null;
  dense: number | null;
  sparse: number | null;
  ngram: number | null;
  rerank: number | null;
  selected: boolean | null;
}

export interface PlaygroundAfterRow {
  ordinal: number;
  chunkUid: string;
  documentTitle: string;
  pages: string;
  level: ChunkLevel;
  score: number;
  rerank: number | null;
  fused: number;
  text: string;
}

export interface PlaygroundResult {
  query: string;
  collectionId: string;
  reranked: boolean;
  before: PlaygroundBeforeRow[];
  after: PlaygroundAfterRow[];
  debug: {
    denseCount: number;
    sparseCount: number;
    ngramCount: number;
    fusedCount: number;
    rerankedCount: number;
    finalCount: number;
    distinctDocuments: number;
    abstained: boolean;
    confidence: number | null;
    latencyMs: Record<string, number>;
  };
}
