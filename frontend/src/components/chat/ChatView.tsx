/**
 * Екран запитань.
 *
 * ТРИ РІШЕННЯ, ЯКІ ТУТ ГОЛОВНІ.
 *
 * 1. Двофазний статус до перших токенів (`useChat`). Пауза 2–6 с без сигналу
 *    читається як зависання, і викладач тисне «Запитати» вдруге — тобто
 *    ставить другий запит на ту саму зайняту відеокарту.
 * 2. Випадок «немає підстав» отримує ЖОВТИЙ БАНЕР із варіантами дій, а не
 *    ввічливе речення в тексті. Відмова — правильна поведінка системи, і
 *    вона мусить виглядати як усвідомлене рішення, інакше її сприймають як
 *    поламаний пошук і перестають довіряти й правильним відповідям.
 * 3. Посилання клікабельні в УСІЙ історії, не лише в останньому ході. Для
 *    старих відповідей бекенд зберігає тільки мапу «[n] → фрагмент», тому
 *    клік веде на «Чому ця відповідь», де решта даних є.
 */

import {
  ClipboardCheck,
  Copy,
  Eraser,
  FileSearch,
  History,
  MessageSquarePlus,
  RefreshCw,
  Search,
  Send,
  Square,
  ThumbsDown,
  ThumbsUp,
  TriangleAlert,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";

import type { ChatMessage, ChatSession, CitationDto } from "@/api/types";
import { AnswerText } from "@/components/chat/AnswerText";
import { SessionList } from "@/components/chat/SessionList";
import { WhyPanel } from "@/components/chat/WhyPanel";
import { Button, IconButton, Spinner } from "@/components/ui/Button";
import { Modal } from "@/components/ui/Overlays";
import { Badge, Banner, EmptyState, Textarea } from "@/components/ui/Primitives";
import { useFeedback, useMessages } from "@/hooks/queries";
import { useChat, type LiveTurn } from "@/hooks/useChat";
import { cn } from "@/lib/cn";
import { describeError } from "@/lib/errors";
import { SOURCES, formatNumber, pluralize } from "@/lib/format";
import { toPlainText } from "@/lib/markdown";
import { useUi } from "@/lib/store";

export function ChatView({
  sessionId,
  onNewSession,
  sessions = [],
  onPickSession,
  onDeleteSession,
  onClearSession,
}: {
  sessionId: string | undefined;
  onNewSession: () => void;
  /** Розмови асистента для панелі історії. Порожній масив — панель порожня. */
  sessions?: ChatSession[];
  onPickSession?: (sessionId: string) => void;
  onDeleteSession?: (sessionId: string) => void;
  onClearSession?: (sessionId: string) => void;
}) {
  const { t } = useTranslation();
  const { data: history = [] } = useMessages(sessionId);
  const { live, busy, ask, stop } = useChat(sessionId);
  const [draft, setDraft] = useState("");
  const [whyFor, setWhyFor] = useState<string | null>(null);
  const [historyOpen, setHistoryOpen] = useState(false);
  const [clearing, setClearing] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const pinned = useRef(true);

  // Автопрокрутка лише тоді, коли викладач і так унизу. Якщо він піднявся
  // перечитати попередню відповідь, смикати екран на кожному токені — це
  // зробити читання неможливим.
  useEffect(() => {
    if (pinned.current) bottomRef.current?.scrollIntoView({ block: "end" });
  }, [history.length, live?.text, live?.phase]);

  const turns = useMemo(() => buildTurns(history), [history]);
  const liveShown = live && (live.phase !== "done" || !history.some((m) => m.id === live.messageId));

  const submit = () => {
    const question = draft.trim();
    if (!question || busy) return;
    setDraft("");
    pinned.current = true;
    ask(question);
  };

  return (
    <div className="flex min-h-0 flex-1">
      <div className="flex min-h-0 min-w-0 flex-1 flex-col">
      <div
        className="scroll-thin min-h-0 flex-1 overflow-y-auto px-4 py-5 sm:px-6"
        onScroll={(event) => {
          const element = event.currentTarget;
          pinned.current = element.scrollHeight - element.scrollTop - element.clientHeight < 120;
        }}
      >
        <div className="mx-auto flex max-w-3xl flex-col gap-5">
          {!turns.length && !liveShown ? (
            <EmptyState
              icon={<Search size={28} />}
              title={t("chat.empty")}
              hint={t("chat.emptyHint")}
            />
          ) : null}

          {turns.map((turn) => (
            <Turn
              key={turn.id}
              question={turn.question}
              answer={turn.answer}
              onWhy={() => setWhyFor(turn.answerId)}
            />
          ))}

          {liveShown && live ? <LiveTurnView turn={live} onWhy={() => setWhyFor(live.messageId)} /> : null}
          <div ref={bottomRef} />
        </div>
      </div>

      <div className="shrink-0 border-t border-line bg-surface px-4 py-3 sm:px-6">
        <div className="mx-auto flex max-w-3xl items-end gap-2">
          <IconButton title={t("chat.newChat")} onClick={onNewSession}>
            <MessageSquarePlus size={17} />
          </IconButton>
          <IconButton
            title={t("chat.history")}
            active={historyOpen}
            onClick={() => setHistoryOpen((open) => !open)}
          >
            <History size={17} />
          </IconButton>
          {/*
            Очищення доступне лише тоді, коли є що очищати: кнопка над
            порожньою розмовою нічого не робить, але виглядає як робоча.
          */}
          <IconButton
            title={t("chat.clearTitle")}
            onClick={() => setClearing(true)}
            disabled={!sessionId || !history.length || busy}
          >
            <Eraser size={17} />
          </IconButton>
          <Textarea
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            onKeyDown={(event) => {
              // Enter надсилає, Shift+Enter переносить рядок. Питання
              // майже завжди однорядкове, і вимагати миші для надсилання —
              // це втрачений темп на кожному запиті.
              if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
                event.preventDefault();
                submit();
              }
            }}
            rows={1}
            placeholder={t("chat.placeholder")}
            className="max-h-40 min-h-[2.5rem] flex-1"
          />
          {busy ? (
            <Button variant="secondary" onClick={stop}>
              <Square size={14} />
              {t("chat.stop")}
            </Button>
          ) : (
            <Button variant="primary" onClick={submit} disabled={!draft.trim() || !sessionId}>
              <Send size={14} />
              {t("chat.send")}
            </Button>
          )}
        </div>
      </div>

      <WhyPanel messageId={whyFor} open={Boolean(whyFor)} onOpenChange={(open) => !open && setWhyFor(null)} />

      <Modal
        open={clearing}
        onOpenChange={setClearing}
        title={t("chat.clearTitle")}
        description={t("chat.clearWarning")}
        footer={
          <>
            <Button variant="ghost" onClick={() => setClearing(false)}>
              {t("common.cancel")}
            </Button>
            <Button
              variant="danger"
              onClick={() => {
                if (sessionId) onClearSession?.(sessionId);
                setClearing(false);
              }}
            >
              {t("chat.clear")}
            </Button>
          </>
        }
      />
      </div>

      {historyOpen ? (
        <SessionList
          sessions={sessions}
          activeId={sessionId}
          onPick={(id) => {
            onPickSession?.(id);
            setHistoryOpen(false);
          }}
          onDelete={(id) => onDeleteSession?.(id)}
          onClose={() => setHistoryOpen(false)}
        />
      ) : null}
    </div>
  );
}

// ------------------------------------------------------------------ хід
interface TurnPair {
  id: string;
  question: string;
  answer: ChatMessage | null;
  answerId: string | null;
}

function buildTurns(messages: ChatMessage[]): TurnPair[] {
  const turns: TurnPair[] = [];
  for (const message of messages) {
    if (message.role === "user") {
      turns.push({ id: message.id, question: message.content, answer: null, answerId: null });
    } else if (turns.length) {
      const last = turns[turns.length - 1];
      last.answer = message;
      last.answerId = message.id;
    }
  }
  // Порожня відповідь у хвості — це рядок, який бекенд створює ДО генерації.
  // Показувати його як порожню бульбашку означало б малювати відповідь, якої
  // ще немає.
  return turns.filter((turn) => turn.answer == null || turn.answer.content.trim() !== "");
}

function Turn({
  question,
  answer,
  onWhy,
}: {
  question: string;
  answer: ChatMessage | null;
  onWhy: () => void;
}) {
  return (
    <div className="space-y-3">
      <Question text={question} />
      {answer ? <Answer message={answer} onWhy={onWhy} /> : null}
    </div>
  );
}

function Question({ text }: { text: string }) {
  return (
    <div className="flex justify-end">
      <p className="max-w-[85%] whitespace-pre-wrap rounded-xl rounded-br-sm bg-accent/10 px-3.5 py-2.5 text-[0.875rem] leading-relaxed">
        {text}
      </p>
    </div>
  );
}

function Answer({ message, onWhy }: { message: ChatMessage; onWhy: () => void }) {
  const { t } = useTranslation();
  const [copied, setCopied] = useState<"plain" | "cited" | null>(null);
  const feedback = useFeedback();
  // Оцінка не повертається в історії: `POST /feedback` пише в таблицю
  // `feedback`, а `MessageOut` її не читає. Тому позначка живе локально —
  // до перезавантаження екрана. Це чесніше, ніж вдавати збережений стан.
  const [voted, setVoted] = useState<"up" | "down" | null>(null);

  const copy = async (mode: "plain" | "cited") => {
    // `toPlainText`, а не сирий `message.content`: відколи відповідь
    // рендериться розміткою, видиме й скопійоване розійшлися б — на екрані
    // жирний заголовок, у буфері `**заголовок**`. Викладач вставляє це в
    // конспект, а не в markdown-редактор.
    const visible = toPlainText(message.content);
    const text = mode === "plain" ? visible.replace(/\s*\[\d{1,3}\]/g, "") : visible;
    try {
      await navigator.clipboard.writeText(text);
      setCopied(mode);
      window.setTimeout(() => setCopied(null), 1600);
    } catch {
      /* буфер обміну недоступний — мовчки, кнопка просто не спрацює */
    }
  };

  return (
    <article className="card p-3.5">
      {message.abstained ? <AbstainBanner /> : null}
      <AnswerText text={message.content} citations={[]} onMarkerClick={onWhy} />

      {/*
        Історія зберігає лише мапу «[n] → фрагмент» (`MessageOut.citations`),
        без назв документів і сторінок — вони живуть у `/messages/{id}/why`.
        Тягнути її для кожної відповіді в історії означало б стільки ж
        запитів, скільки повідомлень, на кожне відкриття чату. Тому тут —
        чесний лічильник із переходом туди, де дані є.
      */}
      {message.citations.length ? (
        <button
          type="button"
          onClick={onWhy}
          className="mt-3 flex items-center gap-1.5 text-[0.75rem] text-muted transition hover:text-ink"
        >
          <FileSearch size={12} />
          {t("chat.sources")}: {pluralize(message.citations.length, SOURCES)}
        </button>
      ) : null}

      <footer className="mt-3 flex flex-wrap items-center gap-1 border-t border-line pt-2.5">
        <Button size="sm" variant="ghost" onClick={() => void copy("plain")}>
          {copied === "plain" ? <ClipboardCheck size={13} /> : <Copy size={13} />}
          {t("chat.copyAnswer")}
        </Button>
        <Button size="sm" variant="ghost" onClick={() => void copy("cited")}>
          {copied === "cited" ? <ClipboardCheck size={13} /> : <Copy size={13} />}
          {t("chat.copyWithCitations")}
        </Button>
        <Button size="sm" variant="ghost" onClick={onWhy}>
          <FileSearch size={13} />
          {t("chat.why")}
        </Button>

        <span className="ml-auto flex items-center gap-1">
          {message.ttftMs ? (
            <span className="mr-2 text-[0.6875rem] tabular-nums subtle">
              {t("chat.stats", {
                ttft: `${formatNumber(message.ttftMs)} мс`,
                tokens: formatNumber(message.tokensOut ?? 0),
              })}
            </span>
          ) : null}
          <IconButton
            title={t("chat.good")}
            active={voted === "up"}
            onClick={() => {
              feedback.mutate({ messageId: message.id, verdict: "up" });
              setVoted("up");
            }}
          >
            <ThumbsUp size={14} />
          </IconButton>
          <IconButton
            title={t("chat.bad")}
            active={voted === "down"}
            onClick={() => {
              feedback.mutate({ messageId: message.id, verdict: "down" });
              setVoted("down");
            }}
          >
            <ThumbsDown size={14} />
          </IconButton>
        </span>
      </footer>
    </article>
  );
}

function LiveTurnView({ turn, onWhy }: { turn: LiveTurn; onWhy: () => void }) {
  const { t } = useTranslation();

  return (
    <div className="space-y-3">
      <Question text={turn.question} />
      <article className="card p-3.5">
        {turn.phase === "retrieving" || turn.phase === "generating" ? (
          <Phase turn={turn} />
        ) : null}

        {turn.abstained && turn.phase !== "retrieving" ? <AbstainBanner /> : null}

        {turn.error ? <ErrorBanner code={turn.error.code} message={turn.error.message} hint={turn.error.hint} /> : null}

        {turn.text ? (
          <AnswerText
            text={turn.text}
            citations={turn.citations}
            streaming={turn.phase === "streaming"}
            onMarkerClick={onWhy}
          />
        ) : null}

        {turn.merged.length ? <SourceStrip citations={turn.merged} /> : null}

        {turn.phase === "done" || turn.phase === "error" ? (
          <footer className="mt-3 flex flex-wrap items-center gap-1 border-t border-line pt-2.5">
            <Button size="sm" variant="ghost" onClick={onWhy} disabled={!turn.messageId}>
              <FileSearch size={13} />
              {t("chat.showFragments")}
            </Button>
            {turn.ttftMs ? (
              <span className="ml-auto text-[0.6875rem] tabular-nums subtle">
                {t("chat.stats", {
                  ttft: `${formatNumber(turn.ttftMs)} мс`,
                  tokens: formatNumber(turn.tokensOut ?? 0),
                })}
              </span>
            ) : null}
          </footer>
        ) : null}
      </article>
    </div>
  );
}

function Phase({ turn }: { turn: LiveTurn }) {
  const { t } = useTranslation();
  const retrieving = turn.phase === "retrieving";
  return (
    <div className="flex items-center gap-2.5 text-[0.8125rem] subtle">
      <Spinner className="text-accent" />
      <span>{retrieving ? t("chat.retrieving") : t("chat.generating")}</span>
      {!retrieving && turn.found != null ? (
        // Число знайдених фрагментів під час очікування — не декор: воно
        // повідомляє, що система вже щось зробила, і робить довгу паузу
        // перед першим токеном зрозумілою.
        <Badge tone="neutral">
          {turn.found} фрагм. · {turn.documents ?? 0} джерел
        </Badge>
      ) : null}
    </div>
  );
}

function AbstainBanner() {
  const { t } = useTranslation();
  return (
    <div className="mb-3">
      <Banner tone="warn" title={t("chat.abstainTitle")} icon={<TriangleAlert size={16} />}>
        <p>{t("chat.abstainBody")}</p>
        <p className="mt-1.5 font-medium">{t("chat.abstainAction")}:</p>
        <ul className="mt-0.5 list-inside list-disc space-y-0.5">
          <li>{t("chat.abstainHint1")}</li>
          <li>{t("chat.abstainHint2")}</li>
          <li>{t("chat.abstainHint3")}</li>
        </ul>
      </Banner>
    </div>
  );
}

function ErrorBanner({ code, message, hint }: { code: string; message: string; hint: string }) {
  const described = describeError(code);
  return (
    <div className="mb-3">
      <Banner tone="danger" title={described.title} icon={<TriangleAlert size={16} />}>
        <p>{message || described.explain}</p>
        {(hint || described.hint) ? <p className="mt-1 subtle">{hint || described.hint}</p> : null}
      </Banner>
    </div>
  );
}

/**
 * Смужка джерел під відповіддю.
 *
 * Показує ЗЛИТІ діапазони сторінок (`merged` із кадру `chat.citations`): без
 * злиття одне джерело, процитоване тричі поспіль, дає три однакові пігулки
 * і створює хибне враження, ніби документів більше, ніж є.
 */
function SourceStrip({ citations }: { citations: CitationDto[] }) {
  const { t } = useTranslation();
  const openSource = useUi((state) => state.openSource);

  return (
    <div className="mt-3 border-t border-line pt-2.5">
      <p className="mb-1.5 text-[0.6875rem] font-medium uppercase tracking-wide subtle">
        {t("chat.sources")}
      </p>
      <div className="flex flex-wrap gap-1.5">
        {citations.map((citation) => (
          <button
            key={`${citation.chunkUid}-${citation.ordinal}`}
            type="button"
            onClick={() =>
              openSource({
                documentId: citation.documentId,
                documentTitle: citation.documentTitle,
                page: citation.pageFrom ?? 1,
                bboxes: citation.bboxes,
                quote: citation.quote,
              })
            }
            className={cn(
              "flex max-w-full items-center gap-1.5 rounded-lg border border-line bg-raised",
              "px-2 py-1 text-left text-[0.75rem] transition hover:border-accent/50 hover:bg-accent/[0.06]",
            )}
          >
            <span className="shrink-0 font-semibold text-accent">{citation.ordinal}</span>
            <span className="truncate">{citation.documentTitle}</span>
            <span className="shrink-0 subtle">{citation.pageLabel}</span>
          </button>
        ))}
      </div>
    </div>
  );
}

export { SourceStrip };

/** Кнопка «Перепитати» — той самий текст питання ще раз. */
export function RegenerateButton({ onClick, disabled }: { onClick: () => void; disabled?: boolean }) {
  const { t } = useTranslation();
  return (
    <Button size="sm" variant="ghost" onClick={onClick} disabled={disabled}>
      <RefreshCw size={13} />
      {t("chat.regenerate")}
    </Button>
  );
}
