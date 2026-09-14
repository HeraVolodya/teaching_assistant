/**
 * Історія розмов асистента.
 *
 * ЧОМУ ЦЕ ВЗАГАЛІ ПОТРІБНО. Досі кнопка «Нове питання» створювала сесію, яка
 * ставала найсвіжішою й НАЗАВЖДИ затуляла попередню: рядки в `chat_sessions`
 * накопичувались, але дістатися до них з інтерфейсу було нічим. Викладач,
 * який учора розібрав тему по розділу «Артилерійська розвідка», сьогодні не
 * міг ні знайти ту розмову, ні видалити її.
 *
 * ПАНЕЛЬ, А НЕ МОДАЛКА. Правило з шапки `Overlays.tsx`: модальне вікно
 * припустиме лише для незворотних дій і створення. Перегляд історії — ні те,
 * ні інше, тож це висувна панель збоку, яка не перекриває відповідь: часто
 * саме заради порівняння з поточною відповіддю в історію й заходять.
 *
 * НАЗВА РОЗМОВИ — ПЕРШЕ ПИТАННЯ. Її ставить бекенд (`_maybe_title`), без
 * виклику моделі. Тому щойно створена розмова ще безіменна, і показувати її
 * треба чесно — «Без назви», а не вигаданим заголовком.
 */

import { MessageSquare, Trash2, X } from "lucide-react";
import { useState } from "react";
import { useTranslation } from "react-i18next";

import type { ChatSession } from "@/api/types";
import { Button, IconButton } from "@/components/ui/Button";
import { Modal } from "@/components/ui/Overlays";
import { cn } from "@/lib/cn";
import { formatRelative } from "@/lib/format";

export function SessionList({
  sessions,
  activeId,
  onPick,
  onDelete,
  onClose,
}: {
  sessions: ChatSession[];
  activeId: string | undefined;
  onPick: (sessionId: string) => void;
  onDelete: (sessionId: string) => void;
  onClose: () => void;
}) {
  const { t } = useTranslation();
  const [confirming, setConfirming] = useState<ChatSession | null>(null);

  return (
    <aside className="flex w-64 shrink-0 flex-col border-l border-line bg-surface">
      <header className="flex items-center justify-between gap-2 border-b border-line px-3 py-2.5">
        <h2 className="text-[0.8125rem] font-semibold">{t("chat.history")}</h2>
        <IconButton title={t("common.close")} onClick={onClose}>
          <X size={15} />
        </IconButton>
      </header>

      {sessions.length ? (
        <ul className="scroll-thin min-h-0 flex-1 overflow-y-auto p-1.5">
          {sessions.map((session) => (
            <li key={session.id} className="group relative">
              <button
                type="button"
                onClick={() => onPick(session.id)}
                className={cn(
                  "w-full rounded-lg px-2.5 py-2 pr-8 text-left transition",
                  session.id === activeId ? "bg-accent/12 text-accent" : "hover:bg-raised",
                )}
              >
                <span className="block truncate text-[0.8125rem] font-medium">
                  {session.title || t("chat.sessionUntitled")}
                </span>
                <span className="mt-0.5 block text-[0.6875rem] subtle">
                  {session.messages
                    ? `${formatRelative(session.updatedAt)} · ${session.messages}`
                    : t("chat.sessionEmpty")}
                </span>
              </button>
              {/*
                Кнопка видалення показується на наведення й на фокус. Саме
                `focus-within`, а не лише `hover`: інакше з клавіатури до неї
                неможливо дійти — вона лишається прозорою й під час фокуса.
              */}
              <IconButton
                title={t("chat.deleteSessionTitle")}
                onClick={() => setConfirming(session)}
                className="absolute right-1 top-1.5 opacity-0 transition group-hover:opacity-100 group-focus-within:opacity-100"
              >
                <Trash2 size={14} />
              </IconButton>
            </li>
          ))}
        </ul>
      ) : (
        <p className="flex min-h-0 flex-1 items-center justify-center gap-2 px-4 text-center text-[0.8125rem] subtle">
          <MessageSquare size={15} />
          {t("chat.historyEmpty")}
        </p>
      )}

      <Modal
        open={confirming !== null}
        onOpenChange={(open) => !open && setConfirming(null)}
        title={t("chat.deleteSessionTitle")}
        description={t("chat.deleteSessionWarning")}
        footer={
          <>
            <Button variant="ghost" onClick={() => setConfirming(null)}>
              {t("common.cancel")}
            </Button>
            <Button
              variant="danger"
              onClick={() => {
                if (confirming) onDelete(confirming.id);
                setConfirming(null);
              }}
            >
              {t("common.delete")}
            </Button>
          </>
        }
      />
    </aside>
  );
}
