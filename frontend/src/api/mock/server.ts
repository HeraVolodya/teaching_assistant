/**
 * Демонстраційний транспорт: увесь застосунок без жодного запущеного процесу.
 *
 * Це фронтендний двійник `ASISTENT_STUB=1`. Правило те саме, що на бекенді:
 * інтерфейс мусить підніматися й проходитися наскрізь без моделей, без
 * sidecar і без мережі. Різниця лише в тому, ЩО саме підмінено — там
 * ембедер і LLM, тут увесь HTTP.
 *
 * ГОЛОВНЕ ОБМЕЖЕННЯ, ЯКОГО ТУТ ДОТРИМАНО: цей файл підробляє РЕАЛЬНИЙ API,
 * а не зручний. Ті самі шляхи, ті самі імена полів (camelCase на дроті,
 * snake_case усередині `config` і `candidates`), той самий формат кадрів
 * SSE. Мок, що описує вигаданий контракт, шкідливіший за відсутність мока:
 * на ньому UI виглядає готовим і не працює з першої ж хвилини проти
 * справжнього бекенда.
 *
 * Режим ЧЕСНО позначається в шапці застосунку. Мовчазна підміна даних була б
 * гіршою за їх відсутність — особливо в застосунку, де вся цінність
 * тримається на довірі до цитат.
 */

import { ApiError, useTransport, type RequestOptions, type Transport } from "../client";
import type {
  Assistant,
  AssistantConfig,
  ChatMessage,
  ChatSession,
  DocumentDto,
  HealthStatus,
  PageDto,
  PlaygroundResult,
  PromptPreview,
  SetupStatus,
} from "../types";
import {
  DEFAULT_CONFIG,
  DEMO_ASSISTANTS,
  DEMO_CHUNKS,
  DEMO_DOCUMENTS,
  DEMO_SESSIONS,
  demoAnswer,
  demoCitation,
  demoSearch,
  type DemoChunk,
} from "./fixtures";
import { buildDemoPdf } from "./pdf";

// ------------------------------------------------------------------- стан
interface DemoState {
  assistants: Assistant[];
  documents: DocumentDto[];
  sessions: ChatSession[];
  messages: ChatMessage[];
  nextEventId: number;
  lmstudioRunning: boolean;
}

const state: DemoState = {
  assistants: structuredClone(DEMO_ASSISTANTS),
  documents: structuredClone(DEMO_DOCUMENTS),
  sessions: structuredClone(DEMO_SESSIONS),
  messages: [],
  nextEventId: 1,
  lmstudioRunning: false,
};

type Listener = (event: { id: number; type: string; data: unknown }) => void;
const listeners = new Set<Listener>();

function emit(type: string, data: unknown): void {
  const event = { id: state.nextEventId++, type, data };
  for (const listener of [...listeners]) listener(event);
}

let uid = 0;
const newId = (prefix: string) => `${prefix}-${Date.now().toString(36)}-${(uid += 1)}`;

const nowIso = () => new Date().toISOString().slice(0, 19).replace("T", " ");

// --------------------------------------------------- імітація індексації
/**
 * Прогрес не лінійний і не рівномірний.
 *
 * Рівний прогрес — найгірше, що можна показати: він виглядає як анімація
 * очікування й нічого не перевіряє. Тут крок «сторінки» навмисно
 * стрибкоподібний (сканована сторінка коштує вп'ятеро дорожче за цифрову),
 * бо саме на такому сигналі має бути видно, що ковзна медіана ETA
 * стабільніша за наївне середнє. Якщо ETA стрибає — це помилка в
 * `lib/eta.ts`, і демонстраційний режим має її показувати, а не ховати.
 */
function simulate(document: DocumentDto): void {
  const jobId = `job-${document.id}`;
  const total = Math.max(1, document.pageCount || 40);
  let done = 0;
  let stageIndex = 0;
  const stages = ["PROBE", "PARSE", "PARSE", "PARSE", "CHUNK", "INDEX"] as const;

  const timer = window.setInterval(() => {
    // Сторінки різної вартості: кожна п'ята «сканована».
    const step = done % 5 === 0 ? total * 0.012 : total * 0.045;
    done = Math.min(total, done + step);
    const fraction = done / total;
    stageIndex = Math.min(stages.length - 1, Math.floor(fraction * stages.length));
    const stage = stages[stageIndex];

    const target = state.documents.find((d) => d.id === document.id);
    if (!target) {
      window.clearInterval(timer);
      return;
    }
    target.status = stage === "CHUNK" ? "CHUNKING" : stage === "INDEX" ? "INDEXING" : "PARSING";
    target.job = {
      id: jobId,
      type: "PARSE",
      state: "RUNNING",
      stage,
      attempts: 1,
      fraction: Number(fraction.toFixed(4)),
      errorCode: null,
    };
    emit("job.progress", {
      jobId,
      docId: document.id,
      collectionId: document.collectionId,
      stage,
      current: Number(done.toFixed(3)),
      total,
      fraction: Number(fraction.toFixed(4)),
      etaSeconds: null,
    });

    if (done >= total) {
      window.clearInterval(timer);
      target.status = "READY";
      target.job = null;
      target.chunks = Math.round(total * 3.4);
      target.qualityGrade = "GOOD";
      emit("doc.ready", {
        docId: document.id,
        chunks: target.chunks,
        tables: Math.round(total / 18),
        formulas: Math.round(total / 7),
        pictures: Math.round(total / 22),
        pages: total,
        quality: "GOOD",
        reusedFrom: null,
        warnings: [],
      });
    }
  }, 700);
}

// Документ, що вже «в роботі» у фікстурах, має рухатися з першої секунди —
// інакше банер черги й ETA неможливо побачити, не завантаживши файл.
const pending = state.documents.find((d) => d.status === "INDEXING");
if (pending) window.setTimeout(() => simulate(pending), 800);

// ----------------------------------------------------------------- SSE
function sseResponse(
  produce: (send: (event: string, data: unknown, id?: number) => void, close: () => void) => () => void,
): Response {
  let cleanup: () => void = () => undefined;
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      const encoder = new TextEncoder();
      let closed = false;
      const send = (event: string, data: unknown, id?: number) => {
        if (closed) return;
        const head = id != null ? `id: ${id}\n` : "";
        controller.enqueue(
          encoder.encode(`${head}event: ${event}\ndata: ${JSON.stringify(data)}\n\n`),
        );
      };
      const close = () => {
        if (closed) return;
        closed = true;
        try {
          controller.close();
        } catch {
          /* потік уже закритий читачем */
        }
      };
      cleanup = produce(send, close);
    },
    cancel() {
      cleanup();
    },
  });
  return new Response(stream, {
    status: 200,
    headers: { "Content-Type": "text/event-stream" },
  });
}

// ------------------------------------------------------------- маршрути
type Handler = (ctx: {
  match: RegExpMatchArray;
  query: URLSearchParams;
  options: RequestOptions;
  body: Record<string, unknown>;
}) => unknown | Promise<unknown>;

interface Route {
  method: string;
  pattern: RegExp;
  handler: Handler;
}

const routes: Route[] = [];
const route = (method: string, pattern: RegExp, handler: Handler) =>
  routes.push({ method, pattern, handler });

const assistantById = (id: string) => state.assistants.find((a) => a.id === id);
const documentById = (id: string) => state.documents.find((d) => d.id === id);

function notFound(what: string): never {
  throw new ApiError(`${what} не знайдено.`, { status: 404, code: "NOT_FOUND" });
}

// --- система -------------------------------------------------------------
route("GET", /^\/health$/, () => {
  const health: HealthStatus = {
    ok: true,
    version: "0.1.0-demo",
    stub: true,
    uptimeSeconds: Math.round(performance.now() / 1000),
    python: "—",
    platform: "демонстраційний режим",
    database: { ok: true, detail: "", schemaVersion: 1, path: "(пам'ять)" },
    lmstudio: {
      ok: state.lmstudioRunning,
      chatReady: state.lmstudioRunning,
      apiVersion: state.lmstudioRunning ? "v1" : "none",
      detail: state.lmstudioRunning ? "" : "Демонстраційний режим: мовної моделі немає.",
      models: state.lmstudioRunning
        ? [
            {
              key: "google/gemma-3-12b",
              loaded: true,
              kind: "llm",
              engine: "llama.cpp",
              maxContext: 32768,
              loadedContext: 16384,
              quantization: "Q4_K_M",
              sizeBytes: 7_800_000_000,
            },
          ]
        : [],
    },
    worker: { mode: "demo", running: 1 },
    jobs: {
      active: state.documents.filter((d) => d.job?.state === "RUNNING").length,
      queued: state.documents.filter((d) => d.status === "QUEUED").length,
    },
    problems: [],
    eventSubscribers: listeners.size,
    demo: true,
  };
  return health;
});

route("GET", /^\/setup\/status$/, () => {
  const status: SetupStatus = {
    hardware: {
      platform: "demo",
      machine: "x86_64",
      cpuCount: 8,
      ramBytes: 34_359_738_368,
      vramBytes: 12_884_901_888,
      isAppleSilicon: false,
      unifiedMemory: false,
      gpu: { name: "NVIDIA GeForce RTX 4070 (демо)", totalBytes: 12_884_901_888 },
      describe: "Демонстраційне залізо: 12 ГБ відеопам'яті, 32 ГБ оперативної",
    },
    recommendation: {
      modelKey: "google/gemma-3-12b",
      title: "Gemma 3 12B",
      quant: "Q4_K_M",
      contextLength: 16384,
      gpuOffload: 1,
      fileBytes: 7_800_000_000,
      archVerified: true,
    },
    lmstudio: {
      ok: state.lmstudioRunning,
      chatReady: state.lmstudioRunning,
      apiVersion: state.lmstudioRunning ? "v1" : "none",
      detail: state.lmstudioRunning ? "" : "Сервер LM Studio вимкнено.",
      models: [],
      cliAvailable: true,
      configuredUrl: "http://127.0.0.1:1234",
    },
    models: {
      embeddings: { id: "Qwen3-Embedding-0.6B", dim: 1024, dir: "(демо)", present: true },
      rerank: { id: "Qwen3-Reranker-0.6B", dir: "(демо)", present: true },
      docling: { dir: "(демо)", present: true },
    },
    stub: true,
    dataDir: "(демонстраційний режим — дані не зберігаються)",
    ready: true,
    problems: [],
  };
  return status;
});

route("POST", /^\/setup\/lmstudio\/start$/, () => {
  state.lmstudioRunning = true;
  return {
    ok: true,
    returncode: 0,
    stdout: "Демонстраційний режим: сервер «запущено».",
    stderr: "",
    lmstudio: { ok: true, chatReady: true, apiVersion: "v1", detail: "", models: [] },
  };
});

route("POST", /^\/setup\/models\/download$/, ({ body }) => {
  const model = String(body.model ?? "demo-model");
  const downloadId = newId("dl");
  let percent = 0;
  const timer = window.setInterval(() => {
    percent = Math.min(100, percent + 7);
    emit("model.download", { downloadId, model, percent, state: percent >= 100 ? "completed" : "downloading" });
    if (percent >= 100) window.clearInterval(timer);
  }, 400);
  return { downloadId, model };
});

route("GET", /^\/events\/history$/, () => []);

// --- асистенти -----------------------------------------------------------
route("GET", /^\/assistants$/, () => state.assistants);

route("POST", /^\/assistants$/, ({ body }) => {
  const assistant: Assistant = {
    id: newId("as"),
    name: String(body.name ?? "Новий асистент"),
    description: String(body.description ?? ""),
    colour: String(body.colour ?? "#4f46e5"),
    emoji: String(body.emoji ?? "📘"),
    config: { ...DEFAULT_CONFIG, ...((body.config as Partial<AssistantConfig>) ?? {}) },
    configVersion: 1,
    collections: [
      {
        id: newId("col"),
        name: "Основна",
        embeddingModelId: "Qwen3-Embedding-0.6B",
        dim: 1024,
        indexGeneration: 0,
        dirty: false,
        documents: 0,
        chunks: 0,
      },
    ],
  };
  state.assistants.push(assistant);
  return assistant;
});

route("GET", /^\/assistants\/([^/]+)$/, ({ match }) => assistantById(match[1]) ?? notFound("Асистента"));

route("PUT", /^\/assistants\/([^/]+)$/, ({ match, body }) => {
  const assistant = assistantById(match[1]) ?? notFound("Асистента");
  assistant.name = String(body.name ?? assistant.name);
  assistant.description = String(body.description ?? "");
  assistant.colour = String(body.colour ?? assistant.colour);
  assistant.emoji = String(body.emoji ?? assistant.emoji);
  if (body.config) {
    assistant.config = { ...assistant.config, ...(body.config as Partial<AssistantConfig>) };
    assistant.configVersion += 1;
  }
  return assistant;
});

route("DELETE", /^\/assistants\/([^/]+)$/, ({ match }) => {
  state.assistants = state.assistants.filter((a) => a.id !== match[1]);
  return undefined;
});

route("GET", /^\/assistants\/([^/]+)\/prompt-preview$/, async ({ match }) => {
  const assistant = assistantById(match[1]) ?? notFound("Асистента");
  const { SYSTEM_RULES_UK, compilePromptPreview, estimateTokensUk } = await import("@/lib/prompt");
  const compiled = compilePromptPreview(assistant.config.instructions, assistant.config);
  const preview: PromptPreview = {
    system: compiled.system,
    persona: assistant.config.instructions,
    rulesTokens: estimateTokensUk(SYSTEM_RULES_UK),
    personaTokens: compiled.personaTokens,
    confidenceThreshold: compiled.policy.threshold,
    minSupportingChunks: compiled.policy.minSupporting,
    finalTopK: assistant.config.final_top_k,
    maxPerDocument: assistant.config.max_per_document,
    onMissingInformation: assistant.config.on_missing_information,
    sample: `[system]\n${compiled.system}\n\n[user]\nПриклад питання викладача`,
  };
  return preview;
});

// --- документи -----------------------------------------------------------
route("GET", /^\/collections\/([^/]+)\/documents$/, ({ match }) =>
  state.documents.filter((d) => d.collectionId === match[1]),
);

route("POST", /^\/collections\/([^/]+)\/documents$/, ({ match, options }) => {
  const form = options.form;
  const files = form ? (form.getAll("files") as File[]) : [];
  const created: DocumentDto[] = files.map((file) => ({
    id: newId("doc"),
    collectionId: match[1],
    title: file.name.replace(/\.[^.]+$/, ""),
    originalName: file.name,
    docType: String(form?.get("docType") ?? "textbook"),
    language: String(form?.get("language") ?? "uk"),
    year: form?.get("year") ? Number(form.get("year")) : null,
    author: (form?.get("author") as string) || null,
    // Кількість сторінок реально відома лише після PROBE; до того UI мусить
    // уміти малювати рядок узагалі без числа — тому тут навмисно 0.
    pageCount: 0,
    ingestMode: (String(form?.get("ingestMode") ?? "FAST") as DocumentDto["ingestMode"]) || "FAST",
    status: "QUEUED",
    errorCode: null,
    errorDetail: null,
    qualityGrade: null,
    sizeBytes: file.size,
    chunks: 0,
    job: null,
  }));
  state.documents.push(...created);
  for (const document of created) {
    window.setTimeout(() => {
      const target = documentById(document.id);
      if (!target) return;
      target.pageCount = Math.max(12, Math.round(file_pages(target.sizeBytes)));
      simulate(target);
    }, 1200);
  }
  return created;
});

/** Груба оцінка сторінок за розміром — лише щоб демо мало правдоподібні числа. */
function file_pages(sizeBytes: number | null): number {
  return Math.min(600, Math.max(12, Math.round((sizeBytes ?? 2_000_000) / 120_000)));
}

route("DELETE", /^\/documents\/([^/]+)$/, ({ match }) => {
  state.documents = state.documents.filter((d) => d.id !== match[1]);
  return undefined;
});

route("POST", /^\/documents\/([^/]+)\/reingest$/, ({ match }) => {
  const document = documentById(match[1]) ?? notFound("Документ");
  document.status = "QUEUED";
  document.errorCode = null;
  document.errorDetail = null;
  document.chunks = 0;
  window.setTimeout(() => simulate(document), 600);
  return document;
});

route("POST", /^\/documents\/([^/]+)\/cancel$/, ({ match }) => {
  const document = documentById(match[1]) ?? notFound("Документ");
  document.status = "CANCELLED";
  document.errorCode = "CANCELLED";
  document.job = null;
  return { cancelled: true, detail: "Скасування запитано." };
});

route("GET", /^\/documents\/([^/]+)\/pages$/, ({ match }) => {
  const document = documentById(match[1]) ?? notFound("Документ");
  const pages: PageDto[] = Array.from({ length: Math.min(60, document.pageCount) }, (_, i) => {
    const scanned = i % 7 === 3;
    return {
      pageNumber: i + 1,
      pageLabel: String(i + 1),
      width: 595,
      height: 842,
      pageClass: scanned ? "SCANNED" : "DIGITAL_CLEAN",
      ocrMode: scanned ? "FULL_PAGE" : "DEFAULT",
      costWeight: scanned ? 5 : 1,
      lexiconHitRate: scanned ? 0.52 : 0.86,
      cyrillicRatio: 0.91,
      mojibakeRatio: 0.01,
      parseScore: scanned ? 0.62 : 0.94,
      layoutScore: scanned ? 0.58 : 0.92,
      tableScore: i % 11 === 0 ? 0.47 : 0.88,
      ocrScore: scanned ? 0.61 : null,
      needsRepair: scanned && i % 14 === 3,
    };
  });
  return pages;
});

route("GET", /^\/documents\/([^/]+)\/jobs$/, () => []);

// --- чат -----------------------------------------------------------------
route("GET", /^\/sessions$/, ({ query }) => {
  const assistantId = query.get("assistant_id");
  return state.sessions.filter((s) => !assistantId || s.assistantId === assistantId);
});

route("POST", /^\/sessions$/, ({ body }) => {
  const session: ChatSession = {
    id: newId("sess"),
    assistantId: String(body.assistantId ?? ""),
    title: String(body.title ?? ""),
    createdAt: nowIso(),
    updatedAt: nowIso(),
    messages: 0,
  };
  state.sessions.push(session);
  return session;
});

route("GET", /^\/sessions\/([^/]+)\/messages$/, ({ match }) =>
  state.messages.filter((m) => m.sessionId === match[1]),
);

route("DELETE", /^\/sessions\/([^/]+)\/messages$/, ({ match }) => {
  state.messages = state.messages.filter((m) => m.sessionId !== match[1]);
  return undefined;
});

route("GET", /^\/messages\/([^/]+)\/why$/, ({ match }) => {
  const message = state.messages.find((m) => m.id === match[1]);
  if (!message) notFound("Повідомлення");
  const chunks = message.citations
    .map((c) => DEMO_CHUNKS.find((chunk) => chunk.chunkUid === c.chunkUid))
    .filter((c): c is DemoChunk => Boolean(c));
  return {
    messageId: message.id,
    abstained: message.abstained,
    modelId: message.modelId,
    ttftMs: message.ttftMs,
    tokensOut: message.tokensOut,
    fragments: chunks.map((chunk, index) => {
      const citation = demoCitation(chunk, index + 1);
      return {
        ordinal: index + 1,
        chunkUid: chunk.chunkUid,
        missing: false,
        documentId: chunk.documentId,
        documentTitle: citation.documentTitle,
        headerPath: chunk.headerPath,
        pageFrom: chunk.page,
        pageTo: chunk.page,
        pageLabel: citation.pageLabel,
        level: "L2" as const,
        language: "uk",
        text: chunk.text,
        bboxes: citation.bboxes,
      };
    }),
    unresolved: [],
    debug: {
      query: "",
      dense_count: 60,
      sparse_count: 60,
      ngram_count: 24,
      fused_count: 92,
      reranked_count: 50,
      final_count: chunks.length,
      distinct_documents: new Set(chunks.map((c) => c.documentId)).size,
      abstained: message.abstained,
      abstain_confidence: chunks.length ? 0.71 : 0.08,
      latency_ms: { dense: 41, sparse: 12, rerank: 186, total: 254 },
      candidates: chunks.map((chunk, index) => ({
        kind: "candidate",
        chunk_uid: chunk.chunkUid,
        document_title: demoCitation(chunk, index + 1).documentTitle,
        pages: `с. ${chunk.page}`,
        rrf_score: 0.031 - index * 0.002,
        dense_score: 0.74 - index * 0.05,
        sparse_score: 0.51 - index * 0.04,
        ngram_score: null,
        rerank: 0.82 - index * 0.09,
        selected: true,
      })),
    },
  };
});

route("POST", /^\/feedback$/, () => ({ ok: true }));

route("GET", /^\/dev\/retrieval-playground$/, ({ query }) => {
  const collectionId = query.get("collectionId") ?? "";
  const q = query.get("q") ?? "";
  const chunks = demoSearch(q, collectionId, Number(query.get("topK") ?? 20));
  const result: PlaygroundResult = {
    query: q,
    collectionId,
    reranked: query.get("rerank") !== "false",
    before: chunks.map((chunk, index) => ({
      chunkUid: chunk.chunkUid,
      documentTitle: demoCitation(chunk, index + 1).documentTitle,
      pages: `с. ${chunk.page}`,
      rrf: 0.031 - index * 0.002,
      dense: 0.74 - index * 0.05,
      sparse: 0.51 - index * 0.04,
      ngram: null,
      rerank: 0.82 - index * 0.09,
      selected: true,
    })),
    after: chunks.map((chunk, index) => ({
      ordinal: index + 1,
      chunkUid: chunk.chunkUid,
      documentTitle: demoCitation(chunk, index + 1).documentTitle,
      pages: `с. ${chunk.page}`,
      level: "L2",
      score: 0.82 - index * 0.09,
      rerank: 0.82 - index * 0.09,
      fused: 0.031 - index * 0.002,
      text: chunk.text,
    })),
    debug: {
      denseCount: 60,
      sparseCount: 60,
      ngramCount: 24,
      fusedCount: 92,
      rerankedCount: 50,
      finalCount: chunks.length,
      distinctDocuments: new Set(chunks.map((c) => c.documentId)).size,
      abstained: chunks.length === 0,
      confidence: chunks.length ? 0.71 : 0.08,
      latencyMs: { dense: 41, sparse: 12, rerank: 186, total: 254 },
    },
  };
  return result;
});

// ------------------------------------------------------- потокові маршрути
/**
 * Стрім відповіді. Затримки — не косметика.
 *
 * `RETRIEVAL_MS` і `TTFT_MS` підібрані під реальні виміри плану: пошук
 * укладається в частку секунди, а перший токен 12-мільярдної моделі йде
 * 2–6 с. Саме ця пауза й вимагає двофазного статусу; якщо в демо зробити її
 * миттєвою, то помилка «тиша читається як зависання» стане невидимою рівно
 * там, де її мали б ловити.
 */
const RETRIEVAL_MS = 550;
const TTFT_MS = 1900;
const TOKEN_MS = 22;

function chatStream(options: RequestOptions): Response {
  const body = (options.body ?? {}) as { sessionId?: string; message?: string };
  const sessionId = String(body.sessionId ?? "");
  const question = String(body.message ?? "");
  const session = state.sessions.find((s) => s.id === sessionId);
  const assistant = state.assistants.find((a) => a.id === session?.assistantId);
  const collectionId = assistant?.collections[0]?.id ?? "col-artillery";

  const messageId = newId("msg");
  state.messages.push({
    id: newId("msg-user"),
    sessionId,
    role: "user",
    content: question,
    abstained: false,
    modelId: null,
    ttftMs: null,
    tokensOut: null,
    createdAt: nowIso(),
    citations: [],
  });

  return sseResponse((send, close) => {
    const timers: number[] = [];
    let cancelled = false;
    const chunks = demoSearch(question, collectionId);
    const abstained = chunks.length === 0;
    const answer = demoAnswer(question, chunks);
    // Токенізація по «словах плюс пробіл»: рівно так виглядає стрім Gemma,
    // і саме на ньому видно, чи не смикається розмітка цитат при склеюванні.
    const tokens = answer.match(/\S+\s*/g) ?? [];

    const at = (delay: number, fn: () => void) => timers.push(window.setTimeout(fn, delay));

    at(RETRIEVAL_MS, () => {
      if (cancelled) return;
      send("chat.debug", {
        messageId,
        stage: "retrieval",
        found: chunks.length,
        documents: new Set(chunks.map((c) => c.documentId)).size,
        abstained,
        confidence: abstained ? 0.08 : 0.71,
        latencyMs: { dense: 41, sparse: 12, rerank: 186 },
      });
    });

    const start = abstained ? RETRIEVAL_MS + 120 : RETRIEVAL_MS + TTFT_MS;
    tokens.forEach((token, index) => {
      at(start + index * TOKEN_MS, () => {
        if (cancelled) return;
        send("chat.token", { messageId, delta: token });
      });
    });

    const end = start + tokens.length * TOKEN_MS + 60;
    at(end, () => {
      if (cancelled) return;
      const citations = chunks.map((chunk, index) => demoCitation(chunk, index + 1));
      if (citations.length) {
        send("chat.citations", { messageId, citations, merged: citations, unresolved: [] });
      }
      state.messages.push({
        id: messageId,
        sessionId,
        role: "assistant",
        content: answer,
        abstained,
        modelId: "google/gemma-3-12b (демо)",
        ttftMs: TTFT_MS,
        tokensOut: tokens.length,
        createdAt: nowIso(),
        citations: chunks.map((chunk, index) => ({ ordinal: index + 1, chunkUid: chunk.chunkUid })),
      });
      if (session && !session.title) session.title = question.slice(0, 80);
      send("chat.done", {
        messageId,
        abstained,
        tokensOut: tokens.length,
        ttftMs: TTFT_MS,
        citations: citations.length,
        elapsedMs: end,
      });
      close();
    });

    return () => {
      cancelled = true;
      for (const timer of timers) window.clearTimeout(timer);
    };
  });
}

function eventsStream(): Response {
  return sseResponse((send, _close) => {
    const listener: Listener = (event) => send(event.type, event.data, event.id);
    listeners.add(listener);
    const ping = window.setInterval(() => send("worker.status", { mode: "demo", running: 1 }), 15000);
    return () => {
      listeners.delete(listener);
      window.clearInterval(ping);
    };
  });
}

// ------------------------------------------------------------- транспорт
/** Кеш blob-URL синтетичних PDF: pdf.js вимагає адресу, яку може завантажити. */
const pdfUrls = new Map<string, string>();

function demoPdfUrl(documentId: string): string {
  const cached = pdfUrls.get(documentId);
  if (cached) return cached;
  const document = documentById(documentId);
  // Обмеження зверху навмисне: 512-сторінковий синтетичний PDF будувався б
  // помітну частку секунди в головному потоці, а для перевірки підсвітки
  // достатньо сторінок навколо цитованої.
  const pages = Math.min(64, Math.max(4, document?.pageCount ?? 24));
  const blob = new Blob([buildDemoPdf(pages) as unknown as BlobPart], { type: "application/pdf" });
  const url = URL.createObjectURL(blob);
  pdfUrls.set(documentId, url);
  return url;
}

const demoTransport: Transport = {
  kind: "demo",

  url(path) {
    const file = /^\/documents\/([^/]+)\/file/.exec(path);
    if (file) return demoPdfUrl(file[1]);
    return `demo://${path}`;
  },

  async raw(path, options = {}) {
    const rawPath = path.split("?")[0];
    if (rawPath === "/chat") return chatStream(options);
    if (rawPath === "/events") return eventsStream();
    if (/^\/documents\/([^/]+)\/file$/.test(rawPath)) {
      const id = rawPath.split("/")[2];
      return fetch(demoPdfUrl(id));
    }
    if (rawPath === "/diagnostics/bundle") {
      const text = "Демонстраційний режим: діагностичний пакет недоступний.";
      return new Response(new Blob([text], { type: "text/plain" }), { status: 200 });
    }
    const value = await this.request<unknown>(path, { ...options, headers: undefined });
    return new Response(JSON.stringify(value ?? null), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  },

  async request<T>(path: string, options: RequestOptions = {}): Promise<T> {
    const [rawPath, rawQuery = ""] = path.split("?");
    const query = new URLSearchParams(rawQuery);
    const method = (options.method ?? "GET").toUpperCase();
    // Затримка, порівнянна з локальним HTTP: без неї стани завантаження
    // ніколи не встигають з'явитися, і зламані скелетони проходять непоміченими.
    await new Promise((resolve) => window.setTimeout(resolve, 90));

    for (const entry of routes) {
      if (entry.method !== method) continue;
      const match = rawPath.match(entry.pattern);
      if (!match) continue;
      const body = (options.body ?? {}) as Record<string, unknown>;
      const result = await entry.handler({ match, query, options, body });
      return structuredClone(result) as T;
    }
    throw new ApiError(`Демонстраційний режим не підтримує ${method} ${rawPath}.`, {
      status: 404,
      code: "DEMO_ROUTE_MISSING",
    });
  },
};

export function installDemoTransport(): void {
  useTransport(demoTransport);
}

/** Лише для тестів: чистий стан між прогонами. */
export function __demoTransport(): Transport {
  return demoTransport;
}
