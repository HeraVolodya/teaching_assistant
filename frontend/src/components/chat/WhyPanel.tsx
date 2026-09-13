/**
 * «Чому ця відповідь» — фрагменти, які асистент СПРАВДІ надіслав моделі.
 *
 * Це не режим розробника. Це єдиний екран, який перетворює відповідь із
 * «застосунок так сказав» на перевірюване твердження: видно, скільки
 * кандидатів знайшов пошук, скільки з них пережили переранжування, які
 * потрапили у відповідь — і повний текст кожного, з кнопкою відкрити
 * сторінку.
 *
 * Палець вниз пише в таблицю `feedback`. Це і є набір даних для оцінювання
 * НДР — безкоштовний і зібраний у робочому процесі, а не спеціальною
 * розміткою.
 */

import { ExternalLink, ThumbsDown, ThumbsUp } from "lucide-react";
import { useState } from "react";
import { useTranslation } from "react-i18next";

import type { WhyFragment } from "@/api/types";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Primitives";
import { Modal } from "@/components/ui/Overlays";
import { useFeedback, useWhy } from "@/hooks/queries";
import { formatScore } from "@/lib/format";
import { useUi } from "@/lib/store";

export function WhyPanel({
  messageId,
  open,
  onOpenChange,
}: {
  messageId: string | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const { t } = useTranslation();
  const { data, isLoading, error } = useWhy(open && messageId ? messageId : undefined);
  const developerMode = useUi((state) => state.developerMode);
  const openSource = useUi((state) => state.openSource);
  const feedback = useFeedback();
  const [sent, setSent] = useState<string | null>(null);

  const debug = data?.debug;

  return (
    <Modal open={open} onOpenChange={onOpenChange} title={t("why.title")} description={t("why.subtitle")} wide>
      {isLoading ? <p className="py-6 text-center text-[0.8125rem] subtle">{t("app.loading")}</p> : null}
      {error ? <p className="py-6 text-center text-[0.8125rem] text-danger">{t("common.error")}</p> : null}

      {debug ? (
        <div className="mb-3 flex flex-wrap gap-x-4 gap-y-1 rounded-lg bg-raised px-3 py-2 text-[0.75rem] subtle">
          <span>
            {t("why.counts", {
              fused: debug.fused_count,
              reranked: debug.reranked_count,
              final: debug.final_count,
            })}
          </span>
          <span>
            {t("why.latency", {
              ms: Math.round(Object.values(debug.latency_ms ?? {}).reduce((a, b) => a + b, 0)),
            })}
          </span>
          {data?.ttftMs ? <span>TTFT {data.ttftMs} мс</span> : null}
          {data?.modelId ? <span className="truncate">{data.modelId}</span> : null}
        </div>
      ) : null}

      {data?.unresolved?.length ? (
        <p className="mb-3 rounded-lg border border-warn/40 bg-warn/10 px-3 py-2 text-[0.75rem]">
          {t("why.unresolved", { list: data.unresolved.map((u) => u.emitted).join(", ") })}
        </p>
      ) : null}

      <div className="scroll-thin max-h-[52vh] space-y-2 overflow-y-auto pr-1">
        {data?.fragments.map((fragment) => (
          <FragmentCard
            key={fragment.chunkUid}
            fragment={fragment}
            onOpen={() =>
              openSource({
                documentId: fragment.documentId ?? "",
                documentTitle: fragment.documentTitle ?? "",
                page: fragment.pageFrom ?? 1,
                bboxes: fragment.bboxes ?? [],
                quote: fragment.text?.slice(0, 200),
              })
            }
            onVote={(verdict) => {
              if (!messageId) return;
              feedback.mutate({ messageId, verdict, chunkUid: fragment.chunkUid });
              setSent(fragment.chunkUid);
            }}
            voted={sent === fragment.chunkUid}
          />
        ))}
        {data && !data.fragments.length ? (
          <p className="py-6 text-center text-[0.8125rem] subtle">
            Асистент не надсилав моделі жодного фрагмента — саме тому відповіді немає.
          </p>
        ) : null}
      </div>

      {developerMode && debug ? (
        <details className="mt-3 rounded-lg border border-line bg-raised p-3">
          <summary className="cursor-pointer text-[0.75rem] font-medium">
            Службові дані пошуку
          </summary>
          <pre className="scroll-thin mt-2 max-h-56 overflow-auto whitespace-pre-wrap break-all text-[0.6875rem] leading-relaxed">
            {JSON.stringify(debug, null, 2)}
          </pre>
        </details>
      ) : null}
    </Modal>
  );
}

function FragmentCard({
  fragment,
  onOpen,
  onVote,
  voted,
}: {
  fragment: WhyFragment;
  onOpen: () => void;
  onVote: (verdict: "up" | "down") => void;
  voted: boolean;
}) {
  const { t } = useTranslation();

  if (fragment.missing) {
    return (
      <div className="rounded-lg border border-dashed border-line p-3 text-[0.8125rem] subtle">
        [{fragment.ordinal}] {t("why.missing")}
      </div>
    );
  }

  return (
    <article className="rounded-lg border border-line p-3">
      <header className="mb-2 flex flex-wrap items-center gap-2">
        <Badge tone="accent" className="px-1.5 py-0.5">
          {fragment.ordinal}
        </Badge>
        <span className="min-w-0 flex-1 truncate text-[0.8125rem] font-medium">
          {fragment.documentTitle}
        </span>
        <span className="text-[0.75rem] subtle">{fragment.pageLabel}</span>
        <Badge tone="ok">{t("why.used")}</Badge>
      </header>

      {fragment.headerPath && fragment.headerPath !== "//" ? (
        <p className="mb-1.5 text-[0.6875rem] subtle">
          {fragment.headerPath.replace(/^\/+|\/+$/g, "").split("//").join(" › ")}
        </p>
      ) : null}

      <p className="whitespace-pre-wrap text-[0.8125rem] leading-relaxed">{fragment.text}</p>

      <footer className="mt-2.5 flex flex-wrap items-center gap-2">
        <Button size="sm" variant="ghost" onClick={onOpen} disabled={!fragment.documentId}>
          <ExternalLink size={13} />
          {t("why.openSource")}
        </Button>
        <span className="ml-auto flex items-center gap-1">
          {voted ? (
            <span className="text-[0.6875rem] subtle">{t("chat.feedbackThanks")}</span>
          ) : (
            <>
              <Button size="sm" variant="ghost" title={t("chat.good")} onClick={() => onVote("up")}>
                <ThumbsUp size={13} />
              </Button>
              <Button size="sm" variant="ghost" title={t("chat.bad")} onClick={() => onVote("down")}>
                <ThumbsDown size={13} />
              </Button>
            </>
          )}
        </span>
      </footer>
    </article>
  );
}

/** Компактні скори — використовуються у списку кандидатів playground'а. */
export function ScorePair({ fused, rerank }: { fused: number | null; rerank: number | null }) {
  const { t } = useTranslation();
  return (
    <span className="flex gap-3 text-[0.6875rem] tabular-nums subtle">
      <span title={t("why.searchScore")}>п {formatScore(fused)}</span>
      <span title={t("why.rerankScore")}>р {formatScore(rerank)}</span>
    </span>
  );
}
