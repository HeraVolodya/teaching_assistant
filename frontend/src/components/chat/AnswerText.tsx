/**
 * Текст відповіді з інлайновими посиланнями на джерела.
 *
 * ПІГУЛКА, А НЕ ГОЛИЙ `[1]`. Номер у квадратних дужках усередині абзацу
 * читається як друкарський брак; надрядкова пігулка читається як посилання
 * і, головне, є ЦІЛЛЮ для наведення й кліку. Наведення показує документ,
 * сторінку й дослівну цитату — тобто перевірка не вимагає нікуди йти;
 * клік відкриває сторінку з підсвіткою — тобто перевірка все ж можлива до
 * кінця.
 *
 * НОМЕР БЕЗ ДЖЕРЕЛА НЕ СТАЄ ПІГУЛКОЮ. Бекенд вирізає такі маркери
 * (`parse_citations`), але якщо один прослизне — він лишиться сірим текстом,
 * а не вдасть робоче посилання. Показати нерозв'язне посилання як робоче —
 * найгірше з можливого: воно підриває довіру до всіх інших.
 */

import { Fragment, useState, type ReactNode } from "react";

import type { CitationDto } from "@/api/types";
import { HoverPopover } from "@/components/ui/Overlays";
import { cn } from "@/lib/cn";
import { useUi } from "@/lib/store";

const MARKER = /\[(\d{1,3})\]/g;

export function AnswerText({
  text,
  citations,
  streaming = false,
  onMarkerClick,
}: {
  text: string;
  citations: CitationDto[];
  streaming?: boolean;
  /**
   * Запасний обробник для повідомлень з ІСТОРІЇ: там бекенд зберігає лише
   * мапу «[n] → фрагмент», без тексту цитати й координат, тож поповер
   * показати нічим. Клік веде на екран «Чому ця відповідь», де ці дані є.
   * Так посилання лишаються робочими в усій історії, а не лише в останньому
   * ході — інакше вчорашня відповідь на вигляд нічим не підтверджена.
   */
  onMarkerClick?: (ordinal: number) => void;
}) {
  const byOrdinal = new Map(citations.map((citation) => [citation.ordinal, citation]));
  const blocks = text.split(/\n{2,}/);

  return (
    <div className="space-y-2.5 text-[0.875rem] leading-relaxed">
      {blocks.map((block, blockIndex) => (
        <p key={blockIndex} className="whitespace-pre-wrap break-words">
          {renderInline(block, byOrdinal, onMarkerClick)}
          {streaming && blockIndex === blocks.length - 1 ? (
            <span className="ml-0.5 inline-block h-[1em] w-[2px] translate-y-[0.15em] animate-pulse bg-accent align-middle" />
          ) : null}
        </p>
      ))}
    </div>
  );
}

function renderInline(
  text: string,
  byOrdinal: Map<number, CitationDto>,
  onMarkerClick?: (ordinal: number) => void,
): ReactNode[] {
  const nodes: ReactNode[] = [];
  let lastIndex = 0;
  MARKER.lastIndex = 0;

  for (let match = MARKER.exec(text); match; match = MARKER.exec(text)) {
    if (match.index > lastIndex) nodes.push(text.slice(lastIndex, match.index));
    const ordinal = Number.parseInt(match[1], 10);
    const citation = byOrdinal.get(ordinal);
    nodes.push(
      citation ? (
        <CitationPill key={`${match.index}-${ordinal}`} citation={citation} />
      ) : onMarkerClick ? (
        <button
          key={`${match.index}-${ordinal}`}
          type="button"
          onClick={() => onMarkerClick(ordinal)}
          title="Показати фрагмент, на якому стоїть це твердження"
          className={cn(
            "mx-[1px] inline-flex min-w-[1.15rem] translate-y-[-0.35em] items-center justify-center",
            "rounded-full border border-line bg-raised px-1 align-baseline",
            "text-[0.625rem] font-semibold leading-[1.15rem] text-muted transition hover:text-ink",
          )}
        >
          {ordinal}
        </button>
      ) : (
        // Маркер без джерела: лишається текстом, свідомо приглушеним.
        <span key={`${match.index}-${ordinal}`} className="text-muted/70" title="Джерело не знайдено">
          {match[0]}
        </span>
      ),
    );
    lastIndex = match.index + match[0].length;
  }
  if (lastIndex < text.length) nodes.push(text.slice(lastIndex));
  return nodes.map((node, index) => <Fragment key={index}>{node}</Fragment>);
}

export function CitationPill({ citation }: { citation: CitationDto }) {
  const [open, setOpen] = useState(false);
  const openSource = useUi((state) => state.openSource);

  const show = () => {
    openSource({
      documentId: citation.documentId,
      documentTitle: citation.documentTitle,
      page: citation.pageFrom ?? 1,
      bboxes: citation.bboxes,
      quote: citation.quote,
    });
    setOpen(false);
  };

  return (
    <HoverPopover
      open={open}
      onOpenChange={setOpen}
      trigger={
        <button
          type="button"
          onClick={show}
          onMouseEnter={() => setOpen(true)}
          onMouseLeave={() => setOpen(false)}
          onFocus={() => setOpen(true)}
          onBlur={() => setOpen(false)}
          aria-label={`Джерело ${citation.ordinal}: ${citation.documentTitle}, ${citation.pageLabel}`}
          className={cn(
            "mx-[1px] inline-flex min-w-[1.15rem] translate-y-[-0.35em] items-center justify-center",
            "rounded-full border border-accent/35 bg-accent/12 px-1 align-baseline",
            "text-[0.625rem] font-semibold leading-[1.15rem] text-accent transition",
            "hover:bg-accent hover:text-accent-ink",
          )}
        >
          {citation.ordinal}
        </button>
      }
    >
      <div className="space-y-1.5">
        <p className="text-[0.75rem] font-semibold leading-snug">{citation.documentTitle}</p>
        <p className="text-[0.6875rem] subtle">{citation.pageLabel}</p>
        <blockquote className="border-l-2 border-warn/60 bg-warn/[0.07] px-2 py-1.5 text-[0.75rem] leading-relaxed">
          «{citation.quote}»
        </blockquote>
        <p className="text-[0.6875rem] subtle">Натисніть, щоб відкрити сторінку з підсвіткою</p>
      </div>
    </HoverPopover>
  );
}
