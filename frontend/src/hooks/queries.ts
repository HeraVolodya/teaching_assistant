/**
 * Обгортки TanStack Query.
 *
 * Ключі запитів — рівно ті, що інвалідує міст подій (`useAppEvents`):
 * `["assistants"]`, `["documents", collectionId]`. Розходження між ними
 * непомітне доти, доки хтось не додасть матеріал і не побачить, що список
 * не оновився.
 */

import {
  useMutation,
  useQuery,
  useQueryClient,
  type UseQueryResult,
} from "@tanstack/react-query";

import * as api from "@/api/endpoints";
import type {
  Assistant,
  AssistantIn,
  ChatSession,
  DocumentDto,
  HealthStatus,
  IngestMode,
  PageDto,
  PromptPreview,
  SetupStatus,
  UploadMeta,
  WhyThisAnswer,
} from "@/api/types";

export function useHealth(): UseQueryResult<HealthStatus> {
  return useQuery({
    queryKey: ["health"],
    queryFn: api.health,
    // Кожні 20 с: здоров'я потрібне для чесного бейджа в шапці, але
    // частіше опитування нічого не додає, а на слабкій машині помітне.
    refetchInterval: 20_000,
    staleTime: 10_000,
  });
}

export function useSetupStatus(enabled = true): UseQueryResult<SetupStatus> {
  return useQuery({ queryKey: ["setup"], queryFn: api.setupStatus, enabled, staleTime: 5_000 });
}

export function useAssistants(): UseQueryResult<Assistant[]> {
  return useQuery({ queryKey: ["assistants"], queryFn: api.listAssistants });
}

export function useAssistant(id: string | undefined): UseQueryResult<Assistant> {
  return useQuery({
    queryKey: ["assistants", id],
    queryFn: () => api.getAssistant(id as string),
    enabled: Boolean(id),
  });
}

export function usePromptPreview(id: string | undefined): UseQueryResult<PromptPreview> {
  return useQuery({
    queryKey: ["prompt-preview", id],
    queryFn: () => api.promptPreview(id as string),
    enabled: Boolean(id),
    staleTime: 0,
  });
}

export function useDocuments(collectionId: string | undefined): UseQueryResult<DocumentDto[]> {
  return useQuery({
    queryKey: ["documents", collectionId],
    queryFn: () => api.listDocuments(collectionId as string),
    enabled: Boolean(collectionId),
    // Події SSE — основне джерело оновлень; періодичне опитування лишається
    // страховкою на випадок, коли канал подій обірвався непомітно.
    refetchInterval: 30_000,
  });
}

export function useDocumentPages(documentId: string | undefined): UseQueryResult<PageDto[]> {
  return useQuery({
    queryKey: ["document-pages", documentId],
    queryFn: () => api.documentPages(documentId as string),
    enabled: Boolean(documentId),
    staleTime: 5 * 60_000,
  });
}

export function useSessions(assistantId: string | undefined): UseQueryResult<ChatSession[]> {
  return useQuery({
    queryKey: ["sessions", assistantId],
    queryFn: () => api.listSessions(assistantId as string),
    enabled: Boolean(assistantId),
  });
}

export function useMessages(sessionId: string | undefined) {
  return useQuery({
    queryKey: ["messages", sessionId],
    queryFn: () => api.listMessages(sessionId as string),
    enabled: Boolean(sessionId),
  });
}

export function useWhy(messageId: string | undefined): UseQueryResult<WhyThisAnswer> {
  return useQuery({
    queryKey: ["why", messageId],
    queryFn: () => api.whyThisAnswer(messageId as string),
    enabled: Boolean(messageId),
  });
}

// ------------------------------------------------------------------ мутації
export function useCreateAssistant() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (body: AssistantIn) => api.createAssistant(body),
    onSuccess: () => void client.invalidateQueries({ queryKey: ["assistants"] }),
  });
}

export function useUpdateAssistant() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: ({ id, body }: { id: string; body: AssistantIn }) => api.updateAssistant(id, body),
    onSuccess: (assistant) => {
      client.setQueryData(["assistants", assistant.id], assistant);
      void client.invalidateQueries({ queryKey: ["assistants"] });
      void client.invalidateQueries({ queryKey: ["prompt-preview", assistant.id] });
    },
  });
}

export function useDeleteAssistant() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => api.deleteAssistant(id),
    onSuccess: () => {
      // Не лише ["assistants"]: разом з асистентом бекенд видаляє його
      // документи й розмови, і кеші під ["documents", …] / ["sessions", …]
      // лишилися б показувати те, чого вже немає ні в базі, ні на диску.
      void client.invalidateQueries({ queryKey: ["assistants"] });
      void client.invalidateQueries({ queryKey: ["documents"] });
      void client.invalidateQueries({ queryKey: ["sessions"] });
      void client.invalidateQueries({ queryKey: ["messages"] });
    },
  });
}

// ------------------------------------------------------------------ розмови
export function useCreateSession(assistantId: string | undefined) {
  const client = useQueryClient();
  return useMutation({
    // Без аргументів: назву ставить бекенд із першого питання (`_maybe_title`),
    // і передавати її звідси означало б мати два джерела істини.
    mutationFn: () => api.createSession(assistantId as string),
    // Без інвалідації новий чат не з'явився б у списку історії, доки в ньому
    // не поставлять питання: раніше `createSession` викликався повз
    // react-query, і кеш ["sessions"] про нього просто не знав.
    onSuccess: () => void client.invalidateQueries({ queryKey: ["sessions", assistantId] }),
  });
}

export function useClearMessages(assistantId: string | undefined) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (sessionId: string) => api.clearMessages(sessionId),
    onSuccess: (_data, sessionId) => {
      void client.invalidateQueries({ queryKey: ["messages", sessionId] });
      // Сесії теж: бекенд скидає назву й рухає updated_at, тож порядок і
      // підпис у списку історії змінюються.
      void client.invalidateQueries({ queryKey: ["sessions", assistantId] });
    },
  });
}

export function useDeleteSession(assistantId: string | undefined) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (sessionId: string) => api.deleteSession(sessionId),
    onSuccess: (_data, sessionId) => {
      client.removeQueries({ queryKey: ["messages", sessionId] });
      void client.invalidateQueries({ queryKey: ["sessions", assistantId] });
    },
  });
}

export function useUploadDocuments(collectionId: string | undefined) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: ({ files, meta }: { files: File[]; meta: UploadMeta }) =>
      api.uploadDocuments(collectionId as string, files, meta),
    onSuccess: () => void client.invalidateQueries({ queryKey: ["documents", collectionId] }),
  });
}

export function useDocumentAction(collectionId: string | undefined) {
  const client = useQueryClient();
  const invalidate = () => {
    void client.invalidateQueries({ queryKey: ["documents", collectionId] });
    void client.invalidateQueries({ queryKey: ["assistants"] });
  };
  return {
    remove: useMutation({ mutationFn: api.deleteDocument, onSuccess: invalidate }),
    cancel: useMutation({ mutationFn: api.cancelDocument, onSuccess: invalidate }),
    reingest: useMutation({
      mutationFn: ({ id, mode }: { id: string; mode: IngestMode }) => api.reingestDocument(id, mode),
      onSuccess: invalidate,
    }),
  };
}

export function useFeedback() {
  return useMutation({
    mutationFn: ({
      messageId,
      verdict,
      note,
      chunkUid,
    }: {
      messageId: string;
      verdict: "up" | "down";
      note?: string;
      chunkUid?: string;
    }) => api.sendFeedback(messageId, verdict, note ?? "", chunkUid),
  });
}
