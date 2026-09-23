/**
 * Транспорт до локального sidecar.
 *
 * Один інваріант, який робить майбутню міграцію у веб-платформу видаленням
 * оболонки: УСЯ комунікація — це fetch і SSE проти HTTP API. Жодного
 * `invoke()` Tauri, жодного IPC. Оболонка володіє лише вікном, нативними
 * діалогами й життєвим циклом sidecar.
 *
 * Другий інваріант: ДЕМОНСТРАЦІЙНИЙ режим на типізованих даних
 * (`./mock/server.ts`) існує для РОЗРОБКИ ІНТЕРФЕЙСУ — щоб екрани піднімалися
 * без жодного запущеного процесу. У ЗАПАКОВАНОМУ ЗАСТОСУНКУ він не вмикається
 * ніколи.
 *
 * ЧОМУ ЦЕ ЖОРСТКЕ ПРАВИЛО, А НЕ ОБЕРЕЖНІСТЬ.
 * Раніше `probeBackend` робив ОДИН запит із таймаутом 1.5 с і при будь-якій
 * помилці мовчки вмикав демо назавжди. Вікно `main` створюється разом із
 * процесом і вантажить SPA одразу (воно лише приховане), а Python-рантайм на
 * 1.3 ГБ підіймається кілька секунд — тож у встановленому застосунку SPA
 * стукала в порожній порт і йшла в демо ЩОРАЗУ, ще до того, як uvicorn зробить
 * bind. Викладач бачив вигадані документи, вигадані оцінки якості розбору й
 * вигадані цитати як справжні; його власні матеріали при цьому нікуди не
 * зберігалися. У продукті, чия цінність тримається на довірі до цитат, це
 * найгірша з можливих поведінок — гірша за білий екран.
 *
 * Тому тепер: чекаємо на sidecar до дедлайну (дзеркало `STARTUP_TIMEOUT` в
 * `src-tauri/src/sidecar.rs`), а не 1.5 с; у демо падаємо ЛИШЕ за явним
 * `VITE_API_MODE=demo` або в dev-збірці; у продакшн-збірці не відповівший
 * sidecar — це чесна помилка старту, а не підроблені дані.
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
 * Скільки загалом чекаємо на sidecar у ЗАПАКОВАНОМУ застосунку.
 *
 * ДЗЕРКАЛО `STARTUP_TIMEOUT` із `src-tauri/src/sidecar.rs`. Оболонка чекає на
 * `/api/health` рівно стільки ж і лише після цього показує головне вікно, тож
 * менше значення тут означало б, що SPA здається раніше за оболонку — тобто
 * рівно той баг, заради якого це написано.
 */
export const STARTUP_DEADLINE_MS = 60_000;

/**
 * Дедлайн у dev-збірці. Короткий свідомо: там демо — робочий інструмент, і
 * змушувати розробника інтерфейсу дивитись хвилину в порожній екран лише
 * заради симетрії з продакшном немає сенсу.
 */
export const DEV_DEADLINE_MS = 2_000;

/**
 * Таймаут ОДНІЄЇ спроби. Лишається коротким, щоб зависла відповідь не з'їла
 * весь бюджет: поки порт не слухається, `fetch` падає миттєво, а от відкрите,
 * але мовчазне з'єднання інакше блокувало б до самого дедлайну.
 */
const ATTEMPT_TIMEOUT_MS = 1_500;
const RETRY_INTERVAL_MS = 250;

/** Sidecar не відповів за відведений час. У продакшні це кінець старту. */
export class BackendUnavailableError extends Error {
  readonly attempts: number;
  readonly elapsedMs: number;
  readonly lastError: unknown;

  constructor(attempts: number, elapsedMs: number, lastError: unknown) {
    super(`Локальний сервер застосунку не відповів за ${Math.round(elapsedMs / 1000)} с.`);
    this.name = "BackendUnavailableError";
    this.attempts = attempts;
    this.elapsedMs = elapsedMs;
    this.lastError = lastError;
  }
}

export interface ProbeOptions {
  /** Загальний бюджет очікування. */
  deadlineMs?: number;
  attemptTimeoutMs?: number;
  retryIntervalMs?: number;
  /**
   * Чи дозволено тихо перейти на демонстраційні дані, коли бюджет вичерпано.
   * За замовчуванням — ЛИШЕ в dev-збірці. Не вмикати для постачання.
   */
  allowDemoFallback?: boolean;
  /** Точки впливу для тестів: без них тест чекав би реальні секунди. */
  now?: () => number;
  sleep?: (ms: number) => Promise<void>;
  onAttempt?: (attempt: number, elapsedMs: number) => void;
}

/**
 * Перевірка на старті: чи живий sidecar. Повторюється до дедлайну.
 *
 * Повертає `health` із `demo: false`, якщо це справжній sidecar. Якщо бюджет
 * вичерпано — або демо (dev), або `BackendUnavailableError` (постачання).
 */
export async function probeBackend(options: ProbeOptions = {}): Promise<HealthStatus> {
  // Явний намір розробника — єдиний спосіб дістати демо в зібраному вигляді.
  if (import.meta.env.VITE_API_MODE === "demo") return installDemo();

  const allowDemoFallback = options.allowDemoFallback ?? Boolean(import.meta.env.DEV);
  const deadlineMs =
    options.deadlineMs ?? (allowDemoFallback ? DEV_DEADLINE_MS : STARTUP_DEADLINE_MS);
  const attemptTimeoutMs = options.attemptTimeoutMs ?? ATTEMPT_TIMEOUT_MS;
  const retryIntervalMs = options.retryIntervalMs ?? RETRY_INTERVAL_MS;
  const now = options.now ?? (() => Date.now());
  const sleep = options.sleep ?? ((ms: number) => new Promise<void>((done) => setTimeout(done, ms)));

  const startedAt = now();
  let attempts = 0;
  let lastError: unknown;

  // Дедлайн перевіряється ПІСЛЯ спроби: одна спроба відбувається завжди,
  // навіть із нульовим бюджетом, інакше `deadlineMs: 0` у тесті не означав би
  // нічого осмисленого.
  for (;;) {
    attempts += 1;
    options.onAttempt?.(attempts, now() - startedAt);

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), attemptTimeoutMs);
    try {
      const health = await httpTransport.request<HealthStatus>("/health", {
        signal: controller.signal,
      });
      mode = "http";
      current = httpTransport;
      return { ...health, demo: false };
    } catch (error) {
      lastError = error;
    } finally {
      clearTimeout(timer);
    }

    if (now() - startedAt >= deadlineMs) break;
    await sleep(retryIntervalMs);
  }

  if (allowDemoFallback) return installDemo();
  throw new BackendUnavailableError(attempts, now() - startedAt, lastError);
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
