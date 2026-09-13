/**
 * Матеріали й індексація.
 *
 * ЧОТИРИ ПРАВИЛА, ЯКИХ ЦЕЙ ЕКРАН ДОТРИМУЄТЬСЯ БЕЗ ВИНЯТКІВ.
 *
 * 1. Рядок з'являється МИТТЄВО. Файл ще вантажиться на диск, а рядок уже
 *    стоїть у таблиці зі станом «У черзі». Затримка між «перетягнув» і
 *    «щось сталося» — це те, після чого перетягують ще раз, а потім ще раз.
 * 2. НІЧОГО МОДАЛЬНОГО і нічого блокуючого. Ніколи. Індексація 500-сторінкового
 *    підручника триває десятки хвилин; діалог поверх екрана на цей час
 *    перетворив би застосунок на непридатний.
 * 3. Текст етапу з ЧИСЛОМ: «Читаю сторінки — сторінка 143 з 512». Смуга без
 *    числа після третьої хвилини читається як зависання.
 * 4. Помилка називає причину й дає КНОПКУ. «Індексація не вдалась» без дії
 *    не залишає викладачеві нічого, крім дзвінка розробникові.
 */

import {
  Ban,
  FileText,
  MoreVertical,
  RefreshCw,
  Trash2,
  TriangleAlert,
  Upload,
} from "lucide-react";
import { useCallback, useMemo, useRef, useState, type DragEvent } from "react";
import { useTranslation } from "react-i18next";

import type { DocumentDto, IngestMode } from "@/api/types";
import { Button, IconButton } from "@/components/ui/Button";
import { Menu, Modal } from "@/components/ui/Overlays";
import { Badge, Banner, EmptyState, Progress, Toggle } from "@/components/ui/Primitives";
import { useDocumentAction, useDocuments, useUploadDocuments } from "@/hooks/queries";
import { useLive } from "@/hooks/useAppEvents";
import { cn } from "@/lib/cn";
import { describeError, diagnosticsText } from "@/lib/errors";
import { formatBytes, formatDuration, formatNumber } from "@/lib/format";
import { describeStage, isPendingStatus } from "@/lib/stages";
import { useUi } from "@/lib/store";

const ACCEPTED = ".pdf,.txt,.md,.markdown";

export function DocumentsScreen({ collectionId }: { collectionId: string | undefined }) {
  const { t } = useTranslation();
  const { data: documents = [], isLoading } = useDocuments(collectionId);
  const upload = useUploadDocuments(collectionId);
  const [dragging, setDragging] = useState(false);
  const [mode, setMode] = useState<IngestMode>("FAST");
  const inputRef = useRef<HTMLInputElement>(null);
  const lastReady = useLive((state) => state.lastReady);
  const dismissReady = useLive((state) => state.dismissReady);

  const addFiles = useCallback(
    (files: FileList | File[] | null) => {
      if (!files || !collectionId) return;
      const list = [...files].filter((file) => /\.(pdf|txt|md|markdown)$/i.test(file.name));
      if (list.length) upload.mutate({ files: list, meta: { ingestMode: mode } });
    },
    [collectionId, mode, upload],
  );

  const onDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragging(false);
    addFiles(event.dataTransfer.files);
  };

  const sorted = useMemo(
    () =>
      [...documents].sort((a, b) => {
        // Проблемні — вгорі, потім ті, що в роботі, потім готові. Викладач
        // приходить сюди по одній із двох причин: додати матеріал або
        // з'ясувати, чому чогось немає у відповідях.
        const rank = (document: DocumentDto) =>
          document.status === "FAILED" ? 0 : isPendingStatus(document.status) ? 1 : 2;
        return rank(a) - rank(b) || a.title.localeCompare(b.title, "uk");
      }),
    [documents],
  );

  return (
    <div
      className="scroll-thin relative min-h-0 flex-1 overflow-y-auto px-4 py-5 sm:px-6"
      onDragOver={(event) => {
        event.preventDefault();
        setDragging(true);
      }}
      onDragLeave={(event) => {
        // Тільки коли курсор справді покинув зону: `dragleave` спрацьовує на
        // кожному вкладеному елементі, і без цієї перевірки підсвітка блимає.
        if (event.currentTarget.contains(event.relatedTarget as Node)) return;
        setDragging(false);
      }}
      onDrop={onDrop}
    >
      {dragging ? (
        <div className="pointer-events-none absolute inset-3 z-20 flex flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed border-accent bg-accent/[0.08]">
          <Upload size={26} className="text-accent" />
          <p className="text-[0.9375rem] font-medium">{t("documents.dropHere")}</p>
          <p className="text-[0.8125rem] subtle">{t("documents.dropHint")}</p>
        </div>
      ) : null}

      <div className="mx-auto max-w-5xl">
        <header className="mb-4 flex flex-wrap items-end justify-between gap-3">
          <div>
            <h2 className="text-[1.0625rem] font-semibold">{t("documents.title")}</h2>
            <p className="mt-0.5 text-[0.8125rem] subtle">{t("documents.subtitle")}</p>
          </div>
          <div className="flex items-center gap-2">
            <ModeSwitch mode={mode} onChange={setMode} />
            <Button variant="primary" onClick={() => inputRef.current?.click()} disabled={!collectionId}>
              <Upload size={15} />
              {t("documents.addFiles")}
            </Button>
            <input
              ref={inputRef}
              type="file"
              multiple
              accept={ACCEPTED}
              className="hidden"
              onChange={(event) => {
                addFiles(event.target.files);
                event.target.value = "";
              }}
            />
          </div>
        </header>

        {lastReady ? <ReadyBanner onDismiss={dismissReady} ready={lastReady} /> : null}

        {isLoading ? (
          <div className="space-y-2">
            {[0, 1, 2].map((index) => (
              <div key={index} className="card h-16 animate-pulse bg-raised" />
            ))}
          </div>
        ) : sorted.length ? (
          <ul className="space-y-1.5">
            {sorted.map((document) => (
              <DocumentRow key={document.id} document={document} collectionId={collectionId} />
            ))}
          </ul>
        ) : (
          <EmptyState
            icon={<FileText size={26} />}
            title={t("documents.empty")}
            hint={t("documents.emptyHint")}
            action={
              <Button variant="primary" onClick={() => inputRef.current?.click()} disabled={!collectionId}>
                <Upload size={15} />
                {t("documents.addFiles")}
              </Button>
            }
          />
        )}
      </div>
    </div>
  );
}

function ModeSwitch({ mode, onChange }: { mode: IngestMode; onChange: (mode: IngestMode) => void }) {
  const { t } = useTranslation();
  return (
    <div className="flex rounded-lg border border-line p-0.5" title={t("documents.modeHint")}>
      {(["FAST", "DEEP"] as const).map((value) => (
        <button
          key={value}
          type="button"
          onClick={() => onChange(value)}
          className={cn(
            "rounded-md px-2.5 py-1 text-[0.75rem] transition",
            mode === value ? "bg-raised font-medium" : "text-muted hover:text-ink",
          )}
        >
          {value === "FAST" ? t("documents.modeFast") : t("documents.modeDeep")}
        </button>
      ))}
    </div>
  );
}

function DocumentRow({
  document,
  collectionId,
}: {
  document: DocumentDto;
  collectionId: string | undefined;
}) {
  const { t } = useTranslation();
  const actions = useDocumentAction(collectionId);
  const live = useLive((state) => state.jobs[document.id]);
  const openSource = useUi((state) => state.openSource);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [showError, setShowError] = useState(false);

  const fraction = live?.fraction ?? document.job?.fraction ?? null;
  const stage = describeStage({
    status: document.status,
    stage: live?.stage ?? document.job?.stage,
    fraction,
    pageCount: document.pageCount,
  });
  const eta = live?.etaSeconds ?? live?.etaLocal ?? null;
  const failed = document.status === "FAILED";

  return (
    <li
      className={cn(
        "card flex flex-wrap items-center gap-x-3 gap-y-2 p-3",
        failed && "border-danger/35 bg-danger/[0.04]",
      )}
    >
      <FileText size={16} className={cn("shrink-0", failed ? "text-danger" : "text-muted")} />

      <div className="min-w-[10rem] flex-1">
        <p className="truncate text-[0.8125rem] font-medium">{document.title}</p>
        <p className="mt-0.5 flex flex-wrap gap-x-2 text-[0.6875rem] tabular-nums subtle">
          <span>{document.docType}</span>
          {document.pageCount ? <span>{formatNumber(document.pageCount)} с.</span> : null}
          {document.sizeBytes ? <span>{formatBytes(document.sizeBytes)}</span> : null}
          {document.chunks ? <span>{formatNumber(document.chunks)} фрагм.</span> : null}
        </p>
      </div>

      <div className="min-w-[14rem] flex-[2]">
        {failed ? (
          <button
            type="button"
            onClick={() => setShowError(true)}
            className="flex items-center gap-1.5 text-left text-[0.75rem] text-danger hover:underline"
          >
            <TriangleAlert size={13} className="shrink-0" />
            {describeError(document.errorCode).title}
          </button>
        ) : isPendingStatus(document.status) ? (
          <div>
            <p className="mb-1 truncate text-[0.75rem] subtle">{stage.detail}</p>
            <Progress value={fraction ?? 0} indeterminate={fraction == null} />
          </div>
        ) : (
          <div className="flex flex-wrap items-center gap-1.5">
            <Badge tone="ok">{stage.label}</Badge>
            {document.qualityGrade ? (
              <Badge tone={document.qualityGrade === "POOR" ? "warn" : "neutral"}>
                {QUALITY_LABEL[document.qualityGrade]}
              </Badge>
            ) : null}
          </div>
        )}
      </div>

      <div className="w-16 shrink-0 text-right text-[0.75rem] tabular-nums subtle">
        {isPendingStatus(document.status) && eta != null ? formatDuration(eta) : ""}
      </div>

      <Menu
        trigger={
          <IconButton title={t("common.more")}>
            <MoreVertical size={15} />
          </IconButton>
        }
        items={[
          {
            id: "open",
            label: t("documents.openSource"),
            icon: <FileText size={14} />,
            disabled: document.status !== "READY",
            onSelect: () =>
              openSource({
                documentId: document.id,
                documentTitle: document.title,
                page: 1,
                bboxes: [],
              }),
          },
          {
            id: "retry",
            label: t("documents.reingest"),
            icon: <RefreshCw size={14} />,
            onSelect: () => actions.reingest.mutate({ id: document.id, mode: "FAST" }),
          },
          {
            id: "retry-deep",
            label: t("documents.reingestDeep"),
            icon: <RefreshCw size={14} />,
            onSelect: () => actions.reingest.mutate({ id: document.id, mode: "DEEP" }),
          },
          {
            id: "cancel",
            label: t("documents.cancel"),
            icon: <Ban size={14} />,
            disabled: !isPendingStatus(document.status),
            separatorBefore: true,
            onSelect: () => actions.cancel.mutate(document.id),
          },
          {
            id: "delete",
            label: t("common.delete"),
            icon: <Trash2 size={14} />,
            danger: true,
            onSelect: () => setConfirmDelete(true),
          },
        ]}
      />

      <Modal
        open={confirmDelete}
        onOpenChange={setConfirmDelete}
        title={t("documents.deleteTitle")}
        description={t("documents.deleteWarning")}
        footer={
          <>
            <Button onClick={() => setConfirmDelete(false)}>{t("common.cancel")}</Button>
            <Button
              variant="danger"
              loading={actions.remove.isPending}
              onClick={() =>
                actions.remove.mutate(document.id, { onSuccess: () => setConfirmDelete(false) })
              }
            >
              {t("common.delete")}
            </Button>
          </>
        }
      >
        <p className="text-[0.8125rem]">{document.title}</p>
      </Modal>

      <ErrorDialog
        document={document}
        open={showError}
        onOpenChange={setShowError}
        onRetry={(deep) => {
          actions.reingest.mutate({ id: document.id, mode: deep ? "DEEP" : "FAST" });
          setShowError(false);
        }}
        onRemove={() => {
          actions.remove.mutate(document.id);
          setShowError(false);
        }}
      />
    </li>
  );
}

const QUALITY_LABEL: Record<string, string> = {
  POOR: "Низька якість розбору",
  FAIR: "Прийнятна якість",
  GOOD: "Добра якість",
  EXCELLENT: "Відмінна якість",
};

/**
 * Діалог помилки.
 *
 * Це єдине модальне вікно на цьому екрані — і воно відкривається лише за
 * явним кліком на повідомлення про помилку, тобто тоді, коли викладач сам
 * попросив подробиць.
 */
function ErrorDialog({
  document,
  open,
  onOpenChange,
  onRetry,
  onRemove,
}: {
  document: DocumentDto;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onRetry: (deep: boolean) => void;
  onRemove: () => void;
}) {
  const { t } = useTranslation();
  const [includeTitle, setIncludeTitle] = useState(false);
  const [copied, setCopied] = useState(false);
  const described = describeError(document.errorCode);

  const copy = async () => {
    const text = diagnosticsText({
      code: document.errorCode,
      detail: document.errorDetail,
      documentTitle: document.title,
      includeTitle,
      stage: document.job?.stage ?? null,
    });
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1800);
    } catch {
      /* буфер недоступний */
    }
  };

  return (
    <Modal open={open} onOpenChange={onOpenChange} title={described.title} description={document.title}>
      <div className="space-y-3">
        <p className="text-[0.8125rem] leading-relaxed">{described.explain}</p>
        {described.hint ? (
          <Banner tone="accent" title="Що можна зробити">
            {described.hint}
          </Banner>
        ) : null}
        {document.errorDetail ? (
          <details className="rounded-lg border border-line bg-raised p-2.5">
            <summary className="cursor-pointer text-[0.75rem] font-medium">Технічні подробиці</summary>
            <p className="mt-1.5 whitespace-pre-wrap break-words text-[0.6875rem] leading-relaxed subtle">
              {document.errorDetail}
            </p>
          </details>
        ) : null}

        <div className="rounded-lg border border-line p-2.5">
          <Toggle
            checked={includeTitle}
            onChange={setIncludeTitle}
            label={t("documents.includeTitle")}
            hint="За замовчуванням вимкнено: перелік матеріалів сам по собі є інформацією."
          />
        </div>
      </div>

      <div className="mt-5 flex flex-wrap justify-end gap-2">
        <Button onClick={() => void copy()}>{copied ? t("common.copied") : t("documents.copyDiagnostics")}</Button>
        {described.action === "remove" ? (
          <Button variant="danger" onClick={onRemove}>
            {described.actionLabel}
          </Button>
        ) : described.action === "retry-deep" ? (
          <Button variant="primary" onClick={() => onRetry(true)}>
            {described.actionLabel}
          </Button>
        ) : described.action === "none" ? null : (
          <Button variant="primary" onClick={() => onRetry(false)}>
            {described.actionLabel}
          </Button>
        )}
      </div>
    </Modal>
  );
}

/**
 * Крок довіри після успіху.
 *
 * Викладач не повірить, що його балістичні таблиці прочитано правильно,
 * доки не побачить бодай одну на екрані. Тому після завершення обробки
 * з'являється не «Готово», а число розпізнаних таблиць і формул із
 * пропозицією перевірити — і посиланням у базу знань, де їх видно поруч із
 * оригінальною сторінкою.
 */
function ReadyBanner({
  ready,
  onDismiss,
}: {
  ready: { docId: string; tables: number; formulas: number; chunks: number };
  onDismiss: () => void;
}) {
  const { t } = useTranslation();
  return (
    <div className="mb-3">
      <Banner
        tone="ok"
        title={t("documents.readyTitle")}
        action={
          <div className="flex gap-2">
            <Button size="sm" onClick={onDismiss}>
              {t("documents.readyDismiss")}
            </Button>
          </div>
        }
      >
        Розпізнано {formatNumber(ready.tables)} таблиць і {formatNumber(ready.formulas)} формул,
        отримано {formatNumber(ready.chunks)} фрагментів. Перевірити кілька можна на вкладці «База знань».
      </Banner>
    </div>
  );
}
