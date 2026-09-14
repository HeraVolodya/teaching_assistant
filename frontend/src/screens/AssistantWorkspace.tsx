/**
 * Робоче місце одного асистента: питання · матеріали · база знань ·
 * налаштування.
 *
 * Вкладки, а не окремі сторінки: усі чотири стосуються ОДНОГО асистента, і
 * перехід між ними не має скидати ані відкриту панель джерела, ані сесію
 * чату. Сесія створюється лениво — при першому відкритті вкладки питань, а
 * не при вході в асистента: порожні сесії в історії нікому не потрібні.
 */

import { BookOpen, FileStack, MessageSquare, SlidersHorizontal } from "lucide-react";
import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { Link, Navigate, Route, Routes, useLocation, useParams } from "react-router-dom";

import { ChatView } from "@/components/chat/ChatView";
import { QueueBanner } from "@/components/QueueBanner";
import { Badge } from "@/components/ui/Primitives";
import {
  useAssistant,
  useClearMessages,
  useCreateSession,
  useDeleteSession,
  useSessions,
} from "@/hooks/queries";
import { cn } from "@/lib/cn";
import { AssistantEditor } from "@/screens/AssistantEditor";
import { DocumentsScreen } from "@/screens/DocumentsScreen";
import { LibraryScreen } from "@/screens/LibraryScreen";

export function AssistantWorkspace() {
  const { assistantId } = useParams();
  const { t } = useTranslation();
  const location = useLocation();
  const { data: assistant, isLoading, error } = useAssistant(assistantId);
  const collectionId = assistant?.collections[0]?.id;

  if (isLoading) {
    return <p className="p-8 text-center text-[0.8125rem] subtle">{t("app.loading")}</p>;
  }
  if (error || !assistant) {
    return <Navigate to="/" replace />;
  }

  const tabs = [
    { id: "chat", label: t("nav.chat"), icon: <MessageSquare size={15} /> },
    { id: "documents", label: t("nav.documents"), icon: <FileStack size={15} /> },
    { id: "library", label: t("nav.library"), icon: <BookOpen size={15} /> },
    { id: "settings", label: t("nav.editor"), icon: <SlidersHorizontal size={15} /> },
  ];
  const active = location.pathname.split("/")[3] ?? "chat";

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="flex shrink-0 flex-wrap items-center gap-3 border-b border-line bg-surface px-4 pt-2 sm:px-6">
        <div className="flex min-w-0 items-center gap-2 pb-2">
          <span
            className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg text-[0.9375rem]"
            style={{ backgroundColor: `${assistant.colour}1f` }}
            aria-hidden
          >
            {assistant.emoji}
          </span>
          <h1 className="truncate text-[0.9375rem] font-semibold">{assistant.name}</h1>
          {assistant.collections.some((collection) => collection.dirty) ? (
            <Badge tone="warn">{t("assistants.indexStale")}</Badge>
          ) : null}
        </div>

        <nav className="-mb-px ml-auto flex gap-0.5">
          {tabs.map((tab) => (
            <Link
              key={tab.id}
              to={`/a/${assistant.id}/${tab.id}`}
              className={cn(
                "flex items-center gap-1.5 border-b-2 px-3 py-2 text-[0.8125rem] transition",
                active === tab.id
                  ? "border-accent font-medium text-ink"
                  : "border-transparent text-muted hover:text-ink",
              )}
            >
              {tab.icon}
              {tab.label}
            </Link>
          ))}
        </nav>
      </div>

      <QueueBanner collectionId={collectionId} />

      <Routes>
        <Route index element={<Navigate to="chat" replace />} />
        <Route path="chat" element={<ChatTab assistantId={assistant.id} />} />
        <Route path="documents" element={<DocumentsScreen collectionId={collectionId} />} />
        <Route path="library" element={<LibraryScreen collectionId={collectionId} />} />
        <Route path="settings" element={<AssistantEditor assistant={assistant} />} />
        <Route path="*" element={<Navigate to="chat" replace />} />
      </Routes>
    </div>
  );
}

/**
 * Вкладка питань.
 *
 * Сесія береться остання наявна, а не нова щоразу: викладач продовжує
 * розмову там, де зупинився, і уточнювальне питання спирається на цитати
 * попереднього ходу (`citation_map` накопичується між ходами на бекенді).
 */
function ChatTab({ assistantId }: { assistantId: string }) {
  const { data: sessions, isLoading } = useSessions(assistantId);
  const [sessionId, setSessionId] = useState<string | undefined>();
  const [creating, setCreating] = useState(false);
  const create = useCreateSession(assistantId);
  const clear = useClearMessages(assistantId);
  const remove = useDeleteSession(assistantId);

  useEffect(() => {
    if (isLoading || creating) return;
    // Вибрана розмова могла щойно зникнути — її видалили в панелі історії.
    // Без цієї перевірки `sessionId` вказував би на неіснуючий рядок, і
    // `useMessages` отримав би 404 замість того, щоб перемкнутись на сусідню.
    const alive = sessionId && sessions?.some((session) => session.id === sessionId);
    if (alive) return;

    const latest = sessions?.[0];
    if (latest) {
      setSessionId(latest.id);
      return;
    }
    setCreating(true);
    create
      .mutateAsync()
      .then((session) => setSessionId(session.id))
      .catch(() => undefined)
      .finally(() => setCreating(false));
    // `create` навмисно поза залежностями: мутація react-query міняє
    // ідентичність на кожен рендер, і цикл «створив → ререндер → створив»
    // народжував би порожні розмови нескінченно.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [assistantId, sessions, isLoading, sessionId, creating]);

  const startNew = () => {
    setSessionId(undefined);
    setCreating(true);
    create
      .mutateAsync()
      .then((session) => setSessionId(session.id))
      .catch(() => undefined)
      .finally(() => setCreating(false));
  };

  return (
    <ChatView
      sessionId={sessionId}
      onNewSession={startNew}
      sessions={sessions ?? []}
      onPickSession={setSessionId}
      onClearSession={(id) => void clear.mutate(id)}
      onDeleteSession={(id) => {
        // Скидаємо вибір ДО мутації: інакше між видаленням і оновленням
        // списку `useMessages` встиг би сходити за повідомленнями видаленої
        // розмови. Наступний прохід ефекту вибере сусідню або створить нову.
        if (id === sessionId) setSessionId(undefined);
        remove.mutate(id);
      }}
    />
  );
}
