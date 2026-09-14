/**
 * Повний перелік викликів до sidecar — в одному місці.
 *
 * Компоненти НІКОЛИ не звертаються до fetch напряму. Це не стиль: коли
 * з'явиться друга оболонка (веб-платформа кафедр), змінюється рівно цей файл.
 *
 * Кожен рядок нижче звірено з роутером у `backend/app/api/`. Там, де шлях
 * виглядає непослідовним, це насправді контракт бекенда, і коментар пояснює,
 * який саме.
 */

import { streamEvents, transport } from "./client";
import type {
  AppEvent,
  Assistant,
  AssistantIn,
  ChatIn,
  ChatMessage,
  ChatSession,
  DocumentDto,
  HealthStatus,
  IngestMode,
  LmStudioStartResult,
  PageDto,
  PlaygroundResult,
  PromptPreview,
  SetupStatus,
  UploadMeta,
  WhyThisAnswer,
} from "./types";

const api = {
  get: <T>(path: string, signal?: AbortSignal) => transport().request<T>(path, { signal }),
  post: <T>(path: string, body?: unknown, signal?: AbortSignal) =>
    transport().request<T>(path, { method: "POST", body, signal }),
  put: <T>(path: string, body?: unknown) => transport().request<T>(path, { method: "PUT", body }),
  del: <T>(path: string) => transport().request<T>(path, { method: "DELETE" }),
};

// ------------------------------------------------------------------ система
export const health = () => api.get<HealthStatus>("/health");
export const setupStatus = () => api.get<SetupStatus>("/setup/status");
export const startLmStudio = () => api.post<LmStudioStartResult>("/setup/lmstudio/start");
export const downloadModel = (model: string) =>
  api.post<{ downloadId: string; model: string }>("/setup/models/download", { model });

/** Діагностичний пакет — ZIP. Назви документів вимкнені за замовчуванням. */
export const diagnosticsBundleUrl = (includeTitles: boolean, integrity = false) =>
  transport().url(
    `/diagnostics/bundle?includeTitles=${includeTitles ? "true" : "false"}` +
      `&integrity=${integrity ? "true" : "false"}`,
  );

export const fetchDiagnosticsBundle = () =>
  transport().raw("/diagnostics/bundle?includeTitles=false&integrity=false");

export const eventsHistory = (limit = 100) =>
  api.get<{ id: number; type: string; data: unknown }[]>(`/events/history?limit=${limit}`);

// --------------------------------------------------------------- асистенти
export const listAssistants = () => api.get<Assistant[]>("/assistants");
export const getAssistant = (id: string) => api.get<Assistant>(`/assistants/${id}`);
export const createAssistant = (body: AssistantIn) => api.post<Assistant>("/assistants", body);

/**
 * PUT, а не PATCH: бекенд віддає лише PUT. Він робить часткове оновлення
 * сам — тіло БЕЗ `config` не скидає налаштування пошуку до дефолтів. Тому
 * перейменування асистента не мусить тягти за собою весь конфіг.
 */
export const updateAssistant = (id: string, body: AssistantIn) =>
  api.put<Assistant>(`/assistants/${id}`, body);
export const deleteAssistant = (id: string) => api.del<void>(`/assistants/${id}`);
export const promptPreview = (id: string) =>
  api.get<PromptPreview>(`/assistants/${id}/prompt-preview`);

// --------------------------------------------------------------- документи
/** Документи належать КОЛЕКЦІЇ, а не асистентові: колекція — фізична партиція. */
export const listDocuments = (collectionId: string) =>
  api.get<DocumentDto[]>(`/collections/${collectionId}/documents`);

/**
 * Завантаження файлів. Рядки в таблиці мають з'явитися МИТТЄВО, тому UI
 * малює оптимістичні рядки ще до відповіді, а sidecar повертає створені
 * документи одразу після запису на диск — до першого парсингу.
 */
export function uploadDocuments(
  collectionId: string,
  files: File[],
  meta: UploadMeta = {},
  signal?: AbortSignal,
): Promise<DocumentDto[]> {
  const form = new FormData();
  for (const file of files) form.append("files", file, file.name);
  form.append("docType", meta.docType ?? "textbook");
  form.append("language", meta.language ?? "uk");
  if (meta.year != null) form.append("year", String(meta.year));
  if (meta.author) form.append("author", meta.author);
  if (meta.ingestMode) form.append("ingestMode", meta.ingestMode);
  return transport().request<DocumentDto[]>(`/collections/${collectionId}/documents`, {
    method: "POST",
    form,
    signal,
  });
}

export const deleteDocument = (id: string) => api.del<void>(`/documents/${id}`);
export const reingestDocument = (id: string, mode: IngestMode = "FAST") =>
  api.post<DocumentDto>(`/documents/${id}/reingest`, { mode });
export const cancelDocument = (id: string) =>
  api.post<{ cancelled: boolean; detail: string }>(`/documents/${id}/cancel`);
export const documentPages = (id: string) => api.get<PageDto[]>(`/documents/${id}/pages`);
export const documentJobs = (id: string) =>
  api.get<Record<string, unknown>[]>(`/documents/${id}/jobs`);

/** URL для pdf.js. Байти йдуть через sidecar із підтримкою Range. */
export const documentFileUrl = (id: string) => transport().url(`/documents/${id}/file`);

// -------------------------------------------------------------------- чат
/**
 * Параметр запиту тут snake_case — і це не помилка: `to_camel` перейменовує
 * поля pydantic-моделей, а `assistant_id` у `list_sessions` — звичайний
 * аргумент функції, тобто лишається як є.
 */
export const listSessions = (assistantId: string) =>
  api.get<ChatSession[]>(`/sessions?assistant_id=${encodeURIComponent(assistantId)}`);
export const createSession = (assistantId: string, title = "") =>
  api.post<ChatSession>("/sessions", { assistantId, title });
export const listMessages = (sessionId: string) =>
  api.get<ChatMessage[]>(`/sessions/${sessionId}/messages`);
/** Стерти повідомлення, лишивши саму розмову в історії. */
export const clearMessages = (sessionId: string) =>
  api.del<void>(`/sessions/${sessionId}/messages`);
/** Видалити розмову цілком. Повідомлення прибирає каскад на бекенді. */
export const deleteSession = (sessionId: string) => api.del<void>(`/sessions/${sessionId}`);

/** Стрім відповіді. AbortSignal — це і є кнопка «Зупинити». */
export const askQuestion = (payload: ChatIn, signal: AbortSignal) =>
  streamEvents("/chat", { method: "POST", body: payload, signal });

export const whyThisAnswer = (messageId: string) =>
  api.get<WhyThisAnswer>(`/messages/${messageId}/why`);

export const sendFeedback = (
  messageId: string,
  verdict: "up" | "down",
  note = "",
  chunkUid?: string,
) => api.post<{ ok: boolean }>("/feedback", { messageId, verdict, note, chunkUid: chunkUid ?? null });

// --------------------------------------------------------- пошук по базі
/**
 * Повнотекстовий пошук по колекції.
 *
 * Половина того, що викладачеві реально треба, — це grep, а не чат. Окремої
 * ручки пошуку бекенд не має, але `dev/retrieval-playground` — це рівно вона:
 * той самий гібридний конвеєр, ті самі пороги, лише без генерації. Тому екран
 * «Огляд бази знань» ходить сюди, а не в чат.
 */
export const searchCollection = (
  collectionId: string,
  query: string,
  topK = 20,
  rerank = true,
  signal?: AbortSignal,
) =>
  api.get<PlaygroundResult>(
    `/dev/retrieval-playground?collectionId=${encodeURIComponent(collectionId)}` +
      `&q=${encodeURIComponent(query)}&topK=${topK}&rerank=${rerank ? "true" : "false"}`,
    signal,
  );

// ------------------------------------------------------- глобальні події
/**
 * Єдиний мультиплексований канал. `lastEventId` дає відновлення після
 * розриву: без нього викладач втрачає рівно ті події, що сталися під час
 * обриву, — тобто найчастіше `doc.ready`.
 */
export async function* appEvents(
  signal: AbortSignal,
  lastEventId?: number | null,
): AsyncGenerator<AppEvent> {
  const suffix = lastEventId != null ? `?lastEventId=${lastEventId}` : "";
  for await (const frame of streamEvents(`/events${suffix}`, { signal })) {
    yield {
      type: frame.event,
      id: frame.id ?? 0,
      data: frame.payload,
    } as AppEvent;
  }
}
