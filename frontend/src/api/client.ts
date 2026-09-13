/**
 * Транспорт до локального sidecar.
 *
 * Один інваріант, який робить майбутню міграцію у веб-платформу видаленням
 * оболонки: УСЯ комунікація — це fetch і SSE проти HTTP API. Жодного
 * `invoke()` Tauri, жодного IPC. Оболонка володіє лише вікном, нативними
 * діалогами й життєвим циклом sidecar.
 *
 * Другий інваріант: якщо sidecar не відповідає, застосунок не падає, а
 * переходить у ДЕМОНСТРАЦІЙНИЙ режим на типізованих даних
 * (`./mock/server.ts`). Це те саме рішення, що `ASISTENT_STUB=1` на бекенді:
 * інтерфейс має підніматися без жодної завантаженої моделі й без жодного
 * запущеного процесу. Режим ЧЕСНО позначається в шапці — мовчазна підміна
 * даних була б гіршою за їх відсутність.
 */

import type { HealthStatus } from "./types";

const ENV_BASE = (import.meta.env.VITE_API_BASE as string | undefined)?.replace(/\/$/, "");

/** У dev Vite проксіює /api на 127.0.0.1:8765; у збірці ходимо напряму. */
export const API_BASE = ENV_BASE ?? (import.meta.env.DEV ? "/api" : "http://127.0.0.1:8765/api");

export class ApiError extends Error {
  readonly status: number;
  readonly code: string;

  constructor(message: string, opts: { status?: number; code?: string } = {}) {
    super(message);
    this.name = "ApiError";
    this.status = opts.status ?? 0;
    this.code = opts.code ?? "UNKNOWN";
  }
}

export interface RequestOptions {
  method?: string;
  body?: unknown;
  /** FormData передається як є, без JSON-серіалізації й без Content-Type. */
  form?: FormData;
  signal?: AbortSignal;
  headers?: Record<string, string>;
}

export interface Transport {
  readonly kind: "http" | "demo";
  request<T>(path: string, options?: RequestOptions): Promise<T>;
  /** Сирий Response — для SSE, ZIP-діагностики й байтів PDF (Range). */
  raw(path: string, options?: RequestOptions): Promise<Response>;
  /** Абсолютний URL ресурсу; потрібен pdf.js, який сам качає файл. */
  url(path: string): string;
}

// ------------------------------------------------------------------- HTTP
const httpTransport: Transport = {
  kind: "http",

  url(path) {
    return `${API_BASE}${path}`;
  },

  async raw(path, options = {}) {
    const { method = "GET", body, form, signal, headers = {} } = options;
    const init: RequestInit = { method, signal, headers: { ...headers } };
    if (form) {
      // Content-Type НЕ виставляємо: browser сам додасть boundary, а ручний
      // заголовок його з'їдає й multipart мовчки приходить порожнім.
      init.body = form;
    } else if (body !== undefined) {
      init.body = JSON.stringify(body);
      (init.headers as Record<string, string>)["Content-Type"] = "application/json";
    }
    let response: Response;
    try {
      response = await fetch(`${API_BASE}${path}`, init);
    } catch (cause) {
      if (signal?.aborted) throw cause;
      throw new ApiError("Немає зв'язку з локальним сервером застосунку.", {
        code: "SIDECAR_UNREACHABLE",
      });
    }
    if (!response.ok) throw await toApiError(response);
    return response;
  },

  async request<T>(path: string, options?: RequestOptions): Promise<T> {
    const response = await this.raw(path, options);
    if (response.status === 204) return undefined as T;
    const text = await response.text();
    return (text ? JSON.parse(text) : undefined) as T;
  },
};

/**
 * Тіло помилки FastAPI — це `{detail}`; `errorCode` додають лише власні
 * обробники (`OUTBOUND_BLOCKED`, `UNSAFE_DATA_DIR`, `NOT_FOUND`). Плюс
 * помилки валідації, де `detail` — МАСИВ, і наївне `String(detail)` дало б
 * викладачеві «[object Object]».
 */
async function toApiError(response: Response): Promise<ApiError> {
  let code = `HTTP_${response.status}`;
  let message = `Сервер відповів помилкою ${response.status}.`;
  try {
    const payload = (await response.json()) as {
      detail?: unknown;
      errorCode?: string;
      message?: string;
    };
    if (payload.errorCode) code = payload.errorCode;
    if (typeof payload.detail === "string" && payload.detail) {
      message = payload.detail;
    } else if (Array.isArray(payload.detail)) {
      const parts = payload.detail
        .map((item) => (item as { msg?: string }).msg)
        .filter(Boolean);
      if (parts.length) message = parts.join("; ");
    } else if (payload.message) {
      message = payload.message;
    }
  } catch {
    /* тіло не JSON — лишаємо типовий текст */
  }
  return new ApiError(message, { status: response.status, code });
}

// ------------------------------------------------------- вибір транспорту
let current: Transport = httpTransport;
let mode: "http" | "demo" | "unknown" = "unknown";

export function transport(): Transport {
  return current;
}

export function apiMode(): "http" | "demo" | "unknown" {
  return mode;
}

export function useTransport(next: Transport): void {
  current = next;
  mode = next.kind;
}

/**
 * Одноразова перевірка на старті: чи живий sidecar.
 *
 * Таймаут 1.5 с, а не 5: якщо sidecar не піднявся, викладач має побачити
 * інтерфейс, а не білий екран на п'ять секунд.
 */
export async function probeBackend(): Promise<HealthStatus> {
  if (import.meta.env.VITE_API_MODE === "demo") return installDemo();
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 1500);
  try {
    const health = await httpTransport.request<HealthStatus>("/health", {
      signal: controller.signal,
    });
    mode = "http";
    current = httpTransport;
    return { ...health, demo: false };
  } catch {
    return installDemo();
  } finally {
    clearTimeout(timer);
  }
}

async function installDemo(): Promise<HealthStatus> {
  const { installDemoTransport } = await import("./mock/server");
  installDemoTransport();
  return current.request<HealthStatus>("/health");
}

// -------------------------------------------------------------------- SSE
export interface SseFrame {
  id: number | null;
  event: string;
  data: string;
}

/**
 * Читання text/event-stream із fetch-відповіді.
 *
 * Чому не EventSource: він уміє лише GET і не має заголовків, а стрім
 * відповіді — це POST із тілом питання. Плюс AbortController дає ЧЕСНУ
 * кнопку «Зупинити»: розрив з'єднання доходить до sidecar, той бачить
 * `request.is_disconnected()` і скасовує генерацію в LM Studio — інакше
 * локальна модель догенерує 600 токенів у нікуди, тримаючи GPU.
 */
export async function* readSse(
  response: Response,
  signal?: AbortSignal,
): AsyncGenerator<SseFrame> {
  const body = response.body;
  if (!body) throw new ApiError("Порожня відповідь потоку.", { code: "EMPTY_STREAM" });
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  const onAbort = () => void reader.cancel().catch(() => undefined);
  signal?.addEventListener("abort", onAbort);

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      // Кадри розділені порожнім рядком. Довжину роздільника беремо з самого
      // збігу, а не з константи: \r\n\r\n трапляється на Windows-проксі, і
      // зсув на фіксовані 2 символи там залишав би \r на початку кадру.
      const separator = /\r?\n\r?\n/;
      for (;;) {
        const match = separator.exec(buffer);
        if (!match) break;
        const raw = buffer.slice(0, match.index);
        buffer = buffer.slice(match.index + match[0].length);
        const frame = parseFrame(raw);
        if (frame) yield frame;
      }
    }
    const tail = parseFrame(buffer);
    if (tail) yield tail;
  } finally {
    signal?.removeEventListener("abort", onAbort);
    try {
      await reader.cancel();
    } catch {
      /* потік уже закритий */
    }
  }
}

export function parseFrame(raw: string): SseFrame | null {
  if (!raw.trim()) return null;
  let event = "message";
  let id: number | null = null;
  const dataLines: string[] = [];
  for (const line of raw.split(/\r?\n/)) {
    // `: ping` — коментар-пульс, тримає з'єднання живим крізь проксі.
    if (line.startsWith(":")) continue;
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("id:")) {
      const parsed = Number.parseInt(line.slice(3).trim(), 10);
      id = Number.isFinite(parsed) ? parsed : null;
    } else if (line.startsWith("data:")) dataLines.push(line.slice(5).replace(/^ /, ""));
  }
  if (!dataLines.length) return null;
  return { id, event, data: dataLines.join("\n") };
}

/** POST/GET → розібрані кадри потоку разом з іменем події та id. */
export async function* streamEvents(
  path: string,
  options: RequestOptions = {},
): AsyncGenerator<{ id: number | null; event: string; payload: unknown }> {
  const response = await current.raw(path, {
    ...options,
    headers: { Accept: "text/event-stream", ...(options.headers ?? {}) },
  });
  for await (const frame of readSse(response, options.signal)) {
    if (frame.data === "[DONE]") return;
    let payload: unknown;
    try {
      payload = JSON.parse(frame.data);
    } catch {
      // Пошкоджений кадр не має вбивати стрім: наступний токен може бути
      // цілим, а обірвана відповідь виглядає для викладача як падіння.
      continue;
    }
    yield { id: frame.id, event: frame.event, payload };
  }
}
