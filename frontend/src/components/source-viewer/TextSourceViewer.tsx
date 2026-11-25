/**
 * Переглядач джерела без PDF: сторінка тексту або аркуш таблиці Excel.
 *
 * Таблицю, CSV, XML і текстовий конспект PDF-переглядач відкрити не може,
 * а показати її треба так само, як сторінку підручника: викладач має бачити
 * цитований фрагмент у його оточенні. Тому сторінка (для таблиці — аркуш)
 * малюється тим самим розбором Markdown, що й відповідь, а сам фрагмент —
 * у рамці, до якої переглядач прокручує одразу після відкриття.
 *
 * Текст сюди приходить із файлу викладача, і йде він тим самим безпечним
 * шляхом, що й відповідь моделі: через дерево розмітки, без жодного innerHTML.
 */

import { ChevronLeft, ChevronRight } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";

import { AnswerText } from "@/components/chat/AnswerText";
import { IconButton } from "@/components/ui/Button";
import { useDocumentPreview } from "@/hooks/queries";
import { cn } from "@/lib/cn";
import { splitHighlight } from "@/lib/sourceText";

export interface TextSourceViewerProps {
  documentId: string;
  /** Фізичний номер сторінки (для таблиці — порядковий номер аркуша), 1-based. */
  page: number;
  chunkUid?: string;
  className?: string;
}

export function TextSourceViewer({ documentId, page, chunkUid, className }: TextSourceViewerProps) {
  const { t } = useTranslation();
  const [current, setCurrent] = useState(page);
  // Підсвітка належить ЦИТОВАНІЙ сторінці; на сусідніх її немає.
  const { data, isLoading, error } = useDocumentPreview(
    documentId,
    current,
    current === page ? chunkUid : undefined,
  );
  const markedRef = useRef<HTMLDivElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  const preview = data?.kind === "text" ? data : null;
  const parts = useMemo(
    () => splitHighlight(preview?.text ?? "", preview?.highlight),
    [preview],
  );

  useEffect(() => {
    if (!preview) return;
    if (markedRef.current) markedRef.current.scrollIntoView({ block: "center" });
    else scrollRef.current?.scrollTo({ top: 0 });
  }, [preview]);

  if (error) {
    return (
      <div className={cn("flex h-full items-center justify-center p-6 text-center", className)}>
        <div>
          <p className="text-[0.875rem] font-medium">{t("source.failed")}</p>
          <p className="mt-1 max-w-sm text-[0.75rem] subtle">
            {error instanceof Error ? error.message : String(error)}
          </p>
        </div>
      </div>
    );
  }

  const total = preview?.pageCount ?? 0;

  return (
    <div className={cn("flex h-full min-h-0 flex-col", className)}>
      <div className="flex shrink-0 items-center gap-1 border-b border-line px-2 py-1.5">
        <IconButton
          title={t("source.previous")}
          disabled={current <= 1}
          onClick={() => setCurrent((value) => Math.max(1, value - 1))}
        >
          <ChevronLeft size={15} />
        </IconButton>
        <span className="min-w-[5.5rem] text-center text-[0.75rem] tabular-nums subtle">
          {total ? `${current} / ${total}` : "—"}
        </span>
        <IconButton
          title={t("source.next")}
          disabled={!total || current >= total}
          onClick={() => setCurrent((value) => Math.min(total, value + 1))}
        >
          <ChevronRight size={15} />
        </IconButton>
        {preview ? (
          <span className="ml-auto truncate text-[0.6875rem] subtle">{preview.pageLabel}</span>
        ) : null}
      </div>

      <div ref={scrollRef} className="scroll-thin min-h-0 flex-1 overflow-auto px-4 py-3">
        {isLoading || !preview ? (
          <p className="py-12 text-center text-[0.8125rem] subtle">{t("source.loading")}</p>
        ) : (
          <div className="space-y-2.5">
            {parts.before.trim() ? <AnswerText text={parts.before} citations={[]} /> : null}
            {parts.marked.trim() ? (
              <div
                ref={markedRef}
                className="rounded-lg bg-warn/[0.10] px-2 py-1.5 ring-1 ring-warn/45"
                aria-label={t("source.highlight")}
              >
                <AnswerText text={parts.marked} citations={[]} />
              </div>
            ) : null}
            {parts.after.trim() ? <AnswerText text={parts.after} citations={[]} /> : null}
          </div>
        )}
      </div>
    </div>
  );
}
