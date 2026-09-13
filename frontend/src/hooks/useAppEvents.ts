/**
 * Єдине підключення до каналу подій — на весь застосунок.
 *
 * ОДНЕ, А НЕ ПО ОДНОМУ НА ЕКРАН. Браузери історично тримають не більше
 * шести одночасних HTTP/1.1-з'єднань на походження. Окремий канал на екран
 * документів, на банер черги й на чат з'їв би половину бюджету саме тоді,
 * коли застосунок одночасно індексує, відповідає й качає модель — і
 * наступний запит просто повис би в черзі браузера без жодної помилки.
 *
 * СТАН ЖИВЕ ПОЗА REACT-QUERY. Прогрес приходить кілька разів на секунду;
 * писати його в кеш запитів означало б перемальовувати список документів на
 * кожен кадр. Тому подія оновлює окреме легке сховище, а `invalidateQueries`
 * викликається лише на РІДКІСНИХ переходах (`doc.ready`, `job.failed`) —
 * тобто тоді, коли справді змінилися дані, а не число на смузі.
 */

import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef } from "react";
import { create } from "zustand";

import { appEvents } from "@/api/endpoints";
import type {
  DocReadyEvent,
  JobFailedEvent,
  JobProgressEvent,
  ModelDownloadEvent,
} from "@/api/types";
import { EtaRegistry } from "@/lib/eta";

/** Живий стан одного завдання індексації. */
export interface LiveJob {
  jobId: string;
  documentId: string;
  stage: string;
  fraction: number;
  /** Оцінка сервера; якщо її немає — беремо власну (`etaLocal`). */
  etaSeconds: number | null;
  /** Наша оцінка з ковзної медіани — стабільніша за серверну після обриву. */
  etaLocal: number | null;
  updatedAt: number;
}

interface LiveState {
  connected: boolean;
  lastEventId: number | null;
  jobs: Record<string, LiveJob>;
  /** Останній успішно доданий матеріал — для кроку довіри після успіху. */
  lastReady: DocReadyEvent | null;
  lastFailure: JobFailedEvent | null;
  downloads: Record<string, ModelDownloadEvent>;

  applyProgress: (event: JobProgressEvent) => void;
  applyReady: (event: DocReadyEvent) => void;
  applyFailure: (event: JobFailedEvent) => void;
  applyDownload: (event: ModelDownloadEvent) => void;
  setConnected: (connected: boolean) => void;
  setLastEventId: (id: number | null) => void;
  dismissReady: () => void;
}

/**
 * Реєстр ETA живе поза сховищем: він мутабельний за побудовою і не має
 * викликати перемальовування сам по собі. Записується лише результат.
 */
const eta = new EtaRegistry();

export const useLive = create<LiveState>((set) => ({
  connected: false,
  lastEventId: null,
  jobs: {},
  lastReady: null,
  lastFailure: null,
  downloads: {},

  applyProgress(event) {
    if (!event.docId) return;
    const snapshot = eta.observe(event.jobId, event.current, event.total, Date.now());
    set((state) => ({
      jobs: {
        ...state.jobs,
        [event.docId as string]: {
          jobId: event.jobId,
          documentId: event.docId as string,
          stage: event.stage,
          fraction: event.fraction,
          etaSeconds: event.etaSeconds,
          etaLocal: snapshot.remainingSeconds,
          updatedAt: Date.now(),
        },
      },
    }));
  },

  applyReady(event) {
    eta.forget(`job-${event.docId}`);
    set((state) => {
      const jobs = { ...state.jobs };
      delete jobs[event.docId];
      return { jobs, lastReady: event };
    });
  },

  applyFailure(event) {
    set((state) => {
      const jobs = { ...state.jobs };
      if (event.docId) delete jobs[event.docId];
      return { jobs, lastFailure: event };
    });
  },

  applyDownload(event) {
    set((state) => ({ downloads: { ...state.downloads, [event.downloadId]: event } }));
  },

  setConnected(connected) {
    set({ connected });
  },
  setLastEventId(id) {
    set({ lastEventId: id });
  },
  dismissReady() {
    set({ lastReady: null });
  },
}));

/** Скільки завдань зараз реально виконується. */
export function activeJobIds(jobs: Record<string, LiveJob>): string[] {
  return Object.values(jobs).map((job) => job.jobId);
}

/** Сумарний залишок для банера черги. */
export function queueRemaining(jobs: Record<string, LiveJob>): number | null {
  const values = Object.values(jobs);
  if (!values.length) return null;
  // Пріоритет серверній оцінці там, де вона є: вона знає про завдання, яких
  // ми ще не бачили (щойно поставлені в чергу іншим вікном).
  const known = values
    .map((job) => job.etaSeconds ?? job.etaLocal)
    .filter((value): value is number => value != null && Number.isFinite(value));
  if (!known.length) return null;
  return known.reduce((sum, value) => sum + value, 0);
}

const RECONNECT_MS = 2000;

/**
 * Підключення. Викликається РІВНО ОДИН РАЗ, у кореневому компоненті.
 *
 * Перепідключення з `lastEventId`: без нього після кожного розриву
 * втрачаються саме ті події, що сталися під час обриву, — найчастіше
 * `doc.ready`, тобто документ назавжди лишався б «в обробці» на екрані.
 */
export function useAppEventsBridge(enabled: boolean): void {
  const queryClient = useQueryClient();
  // Лічильник поколінь, а не булевий ref. У StrictMode React монтує ефект
  // двічі (mount → unmount → mount), і `useRef(false)`, спільний для обох
  // запусків, давав гонку: перший, уже перерваний потік доходив до
  // `setConnected(false)` ПІСЛЯ того, як другий уже виставив `true`, і
  // індикатор назавжди застрягав у стані «Немає зв'язку» попри живий стрім.
  // Кожен запуск ефекту тепер має власне покоління й чіпає стан лише поки
  // лишається актуальним.
  const generation = useRef(0);

  useEffect(() => {
    if (!enabled) return;
    const myGeneration = ++generation.current;
    const isCurrent = () => generation.current === myGeneration;
    const controller = new AbortController();
    let retryTimer: number | undefined;

    const run = async () => {
      while (isCurrent()) {
        try {
          useLive.getState().setConnected(true);
          for await (const event of appEvents(controller.signal, useLive.getState().lastEventId)) {
            if (!isCurrent()) break;
            if (event.id) useLive.getState().setLastEventId(event.id);
            switch (event.type) {
              case "job.progress":
                useLive.getState().applyProgress(event.data);
                break;
              case "doc.ready":
                useLive.getState().applyReady(event.data);
                void queryClient.invalidateQueries({ queryKey: ["documents"] });
                void queryClient.invalidateQueries({ queryKey: ["assistants"] });
                break;
              case "job.failed":
                useLive.getState().applyFailure(event.data);
                void queryClient.invalidateQueries({ queryKey: ["documents"] });
                break;
              case "model.download":
                useLive.getState().applyDownload(event.data);
                break;
              case "worker.status":
                break;
              default:
                // Кадри чату приходять і сюди (другий екземпляр того самого
                // ходу), але їх споживає той компонент, що поставив питання:
                // він читає їх із тіла власного POST /chat, де вони приходять
                // без гонки з цим каналом.
                break;
            }
          }
        } catch {
          /* обрив — перепідключаємось нижче */
        }
        // Застаріле покоління мовчить: інакше воно скине прапорець, який уже
        // належить новішому, живому підключенню.
        if (!isCurrent() || controller.signal.aborted) return;
        useLive.getState().setConnected(false);
        await new Promise((resolve) => {
          retryTimer = window.setTimeout(resolve, RECONNECT_MS);
        });
      }
    };

    void run();
    return () => {
      controller.abort();
      if (retryTimer) window.clearTimeout(retryTimer);
      // Так само: розмонтування скидає прапорець лише якщо після нього не
      // встиг стартувати новіший ефект.
      if (isCurrent()) useLive.getState().setConnected(false);
    };
  }, [enabled, queryClient]);
}
