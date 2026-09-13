/**
 * Один хід розмови: питання → пошук → генерація → цитати.
 *
 * ДВОФАЗНИЙ СТАТУС — НЕ ПРИКРАСА. Перший токен 12-мільярдної моделі йде
 * 2–6 секунд, і весь цей час екран мовчить. Мовчання довше за дві секунди
 * читається як зависання, і викладач натискає «Запитати» ще раз — тобто
 * ставить на ту саму зайняту відеокарту другий запит. Тому статус
 * змінюється двічі до появи тексту: «Шукаю в матеріалах…» (це справді
 * швидко) → «Формую відповідь…» (це справді довго). Обидва рядки прив'язані
 * до РЕАЛЬНИХ подій потоку, а не до таймера: фальшивий прогрес гірший за
 * його відсутність.
 *
 * ЗУПИНКА ЧЕСНА. `AbortController` рве HTTP-з'єднання, sidecar бачить
 * `request.is_disconnected()` і скасовує генерацію в LM Studio. Без цього
 * «Зупинити» лише ховало б текст, доки модель дописувала б 600 токенів у
 * нікуди, тримаючи GPU зайнятою.
 */

import { useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useRef, useState } from "react";

import { askQuestion } from "@/api/endpoints";
import type {
  ChatCitationsEvent,
  ChatDebugEvent,
  ChatDoneEvent,
  ChatErrorEvent,
  ChatTokenEvent,
  CitationDto,
} from "@/api/types";
import type { ChatPhase } from "@/lib/stages";

export interface LiveTurn {
  question: string;
  messageId: string | null;
  text: string;
  citations: CitationDto[];
  merged: CitationDto[];
  unresolved: string[];
  phase: ChatPhase;
  abstained: boolean;
  error: { code: string; message: string; hint: string } | null;
  ttftMs: number | null;
  tokensOut: number | null;
  /** Скільки фрагментів знайшов пошук — показується ще до перших токенів. */
  found: number | null;
  documents: number | null;
}

const EMPTY_TURN: LiveTurn = {
  question: "",
  messageId: null,
  text: "",
  citations: [],
  merged: [],
  unresolved: [],
  phase: "idle",
  abstained: false,
  error: null,
  ttftMs: null,
  tokensOut: null,
  found: null,
  documents: null,
};

export interface UseChatResult {
  live: LiveTurn | null;
  busy: boolean;
  ask: (question: string) => void;
  stop: () => void;
  /** Прибрати завершений хід із живого шару після того, як історія оновилась. */
  dismiss: () => void;
}

export function useChat(sessionId: string | undefined): UseChatResult {
  const [live, setLive] = useState<LiveTurn | null>(null);
  const controller = useRef<AbortController | null>(null);
  const queryClient = useQueryClient();

  // Обрив при розмонтуванні: перехід на інший екран мусить скасовувати
  // генерацію так само, як кнопка «Зупинити». Інакше вкладку закрито, а
  // модель усе ще пише.
  useEffect(() => () => controller.current?.abort(), []);

  const stop = useCallback(() => {
    controller.current?.abort();
    controller.current = null;
    setLive((current) => (current ? { ...current, phase: "done" } : current));
  }, []);

  const dismiss = useCallback(() => setLive(null), []);

  const ask = useCallback(
    (question: string) => {
      const trimmed = question.trim();
      if (!trimmed || !sessionId) return;
      controller.current?.abort();
      const abort = new AbortController();
      controller.current = abort;

      setLive({ ...EMPTY_TURN, question: trimmed, phase: "retrieving" });

      const run = async () => {
        let sawToken = false;
        try {
          for await (const frame of askQuestion({ sessionId, message: trimmed }, abort.signal)) {
            switch (frame.event) {
              case "chat.debug": {
                const data = frame.payload as ChatDebugEvent;
                setLive((current) =>
                  current
                    ? {
                        ...current,
                        messageId: data.messageId ?? current.messageId,
                        // Перехід у другу фазу відбувається саме тут: пошук
                        // завершено, далі — очікування першого токена.
                        phase: sawToken ? current.phase : "generating",
                        abstained: data.abstained ?? current.abstained,
                        found: typeof data.found === "number" ? data.found : current.found,
                        documents:
                          typeof data.documents === "number" ? data.documents : current.documents,
                      }
                    : current,
                );
                break;
              }
              case "chat.token": {
                const data = frame.payload as ChatTokenEvent;
                sawToken = true;
                setLive((current) =>
                  current
                    ? {
                        ...current,
                        messageId: data.messageId ?? current.messageId,
                        phase: "streaming",
                        text: current.text + data.delta,
                      }
                    : current,
                );
                break;
              }
              case "chat.citations": {
                const data = frame.payload as ChatCitationsEvent;
                setLive((current) =>
                  current
                    ? {
                        ...current,
                        citations: data.citations ?? [],
                        merged: data.merged ?? [],
                        unresolved: data.unresolved ?? [],
                      }
                    : current,
                );
                break;
              }
              case "chat.error": {
                const data = frame.payload as ChatErrorEvent;
                setLive((current) =>
                  current
                    ? {
                        ...current,
                        phase: "error",
                        error: {
                          code: data.errorCode,
                          message: data.message,
                          hint: data.hint ?? "",
                        },
                      }
                    : current,
                );
                break;
              }
              case "chat.done": {
                const data = frame.payload as ChatDoneEvent;
                setLive((current) =>
                  current
                    ? {
                        ...current,
                        messageId: data.messageId ?? current.messageId,
                        phase: current.phase === "error" ? "error" : "done",
                        abstained: data.abstained ?? current.abstained,
                        ttftMs: data.ttftMs,
                        tokensOut: data.tokensOut,
                      }
                    : current,
                );
                break;
              }
              default:
                break;
            }
          }
        } catch (error) {
          if (abort.signal.aborted) {
            // Свідома зупинка. Половина відповіді лишається на екрані й в
            // історії — саме так її зберігає бекенд у `finally`.
            setLive((current) => (current ? { ...current, phase: "done" } : current));
          } else {
            setLive((current) =>
              current
                ? {
                    ...current,
                    phase: "error",
                    error: {
                      code: "CHAT_FAILED",
                      message:
                        error instanceof Error
                          ? error.message
                          : "Не вдалося отримати відповідь.",
                      hint: "",
                    },
                  }
                : current,
            );
          }
        } finally {
          if (controller.current === abort) controller.current = null;
          // Історія перечитується лише ПІСЛЯ ходу: бекенд створює порожній
          // рядок відповіді ще до генерації, і оновлення посеред стріму
          // показало б порожню відповідь поруч із тією, що пишеться.
          void queryClient.invalidateQueries({ queryKey: ["messages", sessionId] });
          void queryClient.invalidateQueries({ queryKey: ["sessions"] });
        }
      };

      void run();
    },
    [queryClient, sessionId],
  );

  const busy = live != null && (live.phase === "retrieving" || live.phase === "generating" || live.phase === "streaming");

  return { live, busy, ask, stop, dismiss };
}
