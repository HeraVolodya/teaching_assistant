/**
 * Смуга «Обробка 3 з 12 · залишилось ~24 хв».
 *
 * Живе НАД вкладками, а не всередині екрана матеріалів: індексація триває
 * годинами, і викладач у цей час ставить питання, а не дивиться на таблицю.
 * Дізнатися, що база ще неповна, він мусить саме там, де він є, — інакше
 * відповідь без потрібного підручника виглядає як помилка системи.
 *
 * НІЧОГО МОДАЛЬНОГО. Смуга не перекриває вміст і не вимагає реакції.
 */

import { Loader2 } from "lucide-react";
import { useTranslation } from "react-i18next";

import { useDocuments } from "@/hooks/queries";
import { queueRemaining, useLive } from "@/hooks/useAppEvents";
import { formatDuration } from "@/lib/format";
import { isPendingStatus } from "@/lib/stages";

export function QueueBanner({ collectionId }: { collectionId: string | undefined }) {
  const { t } = useTranslation();
  const { data: documents = [] } = useDocuments(collectionId);
  const jobs = useLive((state) => state.jobs);

  const pending = documents.filter((document) => isPendingStatus(document.status));
  if (!pending.length) return null;

  const running = pending.filter((document) => document.status !== "QUEUED").length;
  const remaining = queueRemaining(
    Object.fromEntries(
      Object.entries(jobs).filter(([documentId]) => pending.some((d) => d.id === documentId)),
    ),
  );

  return (
    <div className="flex shrink-0 flex-wrap items-center gap-x-3 gap-y-1 border-b border-accent/25 bg-accent/[0.07] px-4 py-1.5 text-[0.75rem] sm:px-6">
      <Loader2 size={13} className="animate-spin text-accent" />
      <span className="font-medium">
        {t("documents.queueBanner", { done: running || 1, total: pending.length })}
      </span>
      <span className="subtle">
        {remaining == null
          ? t("documents.queueEstimating")
          : t("documents.queueEta", { time: formatDuration(remaining) })}
      </span>
    </div>
  );
}
