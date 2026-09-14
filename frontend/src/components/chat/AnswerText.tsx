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
 *
 * РОЗМІТКА ЙДЕ ЧЕРЕЗ ДЕРЕВО, А НЕ ЧЕРЕЗ РЯДОК.
 * Модель пише Markdown, хоча промпт про це не просить: жирний, списки,
 * подекуди $-формули. Раніше все це доходило до екрана символами — абзац
 * малювався одним `whitespace-pre-wrap`, і викладач читав `**1-й тип:**`.
 * Тепер `lib/markdown.ts` розбирає текст на вузли, а ЛИСТКИ типу `text`
 * віддаються сюди, у `renderInline`. Порядок саме такий і тільки такий:
 * розмітка ніколи не віддає рядок HTML, тому дістатися до `[n]` повз
 * `renderInline` нічим. Побічний виграш — усередині `код` і $формул$
 * текстових листків немає взагалі, тож `arr[1]` і `\sqrt[3]{x}` більше не
 * втрачають свої дужки на користь неіснуючого джерела.
 *
 * ФОРМУЛИ — ЧЕРЕЗ `katex.render` У ВУЗОЛ, А НЕ ЧЕРЕЗ `renderToString`.
 * У `frontend/src` нуль входжень `dangerouslySetInnerHTML`, і це варто
 * зберегти: у промпт свідомо йдуть `<table>` і `<formula>` з документів
 * (див. `chunker.sanitize_display`), модель їх час від часу відлунює, тож
 * будь-який шлях «рядок → innerHTML» відкрив би вставку розмітки, що
 * прийшла з чужого PDF. `trust: false` додатково глушить `\href` і `\url`,
 * `throwOnError: false` показує криву формулу вихідним текстом замість того,
 * щоб завалити всю відповідь.
 */

import katex from "katex";
import {
  Fragment,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";

import "katex/dist/katex.min.css";

import type { CitationDto } from "@/api/types";
import { HoverPopover } from "@/components/ui/Overlays";
import { cn } from "@/lib/cn";
import { type Block, type Inline, parseMarkdown } from "@/lib/markdown";
import { useUi } from "@/lib/store";

type MarkerClick = ((ordinal: number) => void) | undefined;
type Ordinals = Map<number, CitationDto>;

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
  const byOrdinal = useMemo<Ordinals>(
    () => new Map(citations.map((citation) => [citation.ordinal, citation])),
    [citations],
  );
  // Розбір крутиться на КОЖЕН токен стріму, тому мемоізація тут не
  // косметика: без неї 600 подій означають 600 повних проходів по тексту.
  const blocks = useMemo(() => parseMarkdown(text, { streaming }), [text, streaming]);

  return (
    <div className="space-y-2.5 text-[0.875rem] leading-relaxed">
      {blocks.map((block, index) => (
        <BlockView
          key={index}
          block={block}
          byOrdinal={byOrdinal}
          onMarkerClick={onMarkerClick}
          // Каретка живе в ОСТАННЬОМУ блоці, а не після нього: інакше в кінці
          // списку вона зривається на власний рядок під маркером.
          caret={streaming && index === blocks.length - 1}
        />
      ))}
    </div>
  );
}

function BlockView({
  block,
  byOrdinal,
  onMarkerClick,
  caret,
}: {
  block: Block;
  byOrdinal: Ordinals;
  onMarkerClick: MarkerClick;
  caret: boolean;
}) {
  switch (block.kind) {
    case "para":
      return (
        <p className="break-words">
          {block.lines.map((line, index) => (
            <Fragment key={index}>
              {index > 0 ? <br /> : null}
              {renderNodes(line, byOrdinal, onMarkerClick)}
            </Fragment>
          ))}
          {caret ? <Caret /> : null}
        </p>
      );

    case "heading": {
      const Tag = (block.level <= 2 ? "h3" : "h4") as "h3" | "h4";
      return (
        <Tag
          className={cn(
            "break-words font-semibold text-ink",
            block.level <= 2 ? "mt-1 text-[0.9375rem]" : "text-[0.875rem]",
          )}
        >
          {renderNodes(block.children, byOrdinal, onMarkerClick)}
          {caret ? <Caret /> : null}
        </Tag>
      );
    }

    case "list": {
      // `list-disc`/`list-decimal` задаються явно: preflight Tailwind скидає
      // маркери списків, і без цього вивід виглядав би «як раніше, тільки
      // без зірочок» — тобто рівно так само нечитабельно.
      const Tag = (block.ordered ? "ol" : "ul") as "ol" | "ul";
      return (
        <Tag
          start={block.ordered ? block.start : undefined}
          className={cn(
            "space-y-1 pl-5",
            block.ordered ? "list-decimal" : "list-disc",
            "marker:text-muted",
          )}
        >
          {block.items.map((item, index) => (
            <li key={index} className={cn("break-words", item.depth ? "ml-4" : "")}>
              {item.lines.map((line, lineIndex) => (
                <Fragment key={lineIndex}>
                  {lineIndex > 0 ? <br /> : null}
                  {renderNodes(line, byOrdinal, onMarkerClick)}
                </Fragment>
              ))}
              {caret && index === block.items.length - 1 ? <Caret /> : null}
            </li>
          ))}
        </Tag>
      );
    }

    case "code":
      return (
        <pre className="scroll-thin overflow-x-auto rounded-lg border border-line bg-raised px-3 py-2">
          <code className="text-[0.8125rem] leading-relaxed">{block.text}</code>
        </pre>
      );

    case "math":
      return <MathView tex={block.tex} display />;

    case "table":
      // Таблиці в цьому корпусі — норма: TableFormer витягує їх із підручників,
      // і модель охоче відповідає таблицею. Горизонтальна прокрутка замість
      // переносу: у таблиці дальностей злиплі колонки гірші за скрол.
      return (
        <div className="scroll-thin overflow-x-auto rounded-lg border border-line">
          <table className="w-full border-collapse text-[0.8125rem]">
            <thead>
              <tr className="bg-raised">
                {block.header.map((cell, index) => (
                  <th
                    key={index}
                    className="border-b border-line px-2.5 py-1.5 text-left font-semibold"
                  >
                    {renderNodes(cell, byOrdinal, onMarkerClick)}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {block.rows.map((row, rowIndex) => (
                <tr key={rowIndex} className="border-b border-line/60 last:border-0">
                  {row.map((cell, cellIndex) => (
                    <td key={cellIndex} className="px-2.5 py-1.5 align-top">
                      {renderNodes(cell, byOrdinal, onMarkerClick)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      );
  }
}

function Caret() {
  return (
    <span className="ml-0.5 inline-block h-[1em] w-[2px] translate-y-[0.15em] animate-pulse bg-accent align-middle" />
  );
}

/** Вузли розмітки → React. Листки `text` — і лише вони — йдуть у renderInline. */
function renderNodes(
  nodes: Inline[],
  byOrdinal: Ordinals,
  onMarkerClick: MarkerClick,
): ReactNode[] {
  return nodes.map((node, index) => {
    switch (node.kind) {
      case "text":
        return (
          <Fragment key={index}>{renderInline(node.text, byOrdinal, onMarkerClick)}</Fragment>
        );
      case "strong":
        return (
          <strong key={index} className="font-semibold text-ink">
            {renderNodes(node.children, byOrdinal, onMarkerClick)}
          </strong>
        );
      case "em":
        return (
          <em key={index} className="italic">
            {renderNodes(node.children, byOrdinal, onMarkerClick)}
          </em>
        );
      case "code":
        return (
          <code
            key={index}
            className="rounded border border-line bg-raised px-1 py-px text-[0.8125rem]"
          >
            {node.text}
          </code>
        );
      case "math":
        return <MathView key={index} tex={node.tex} display={node.display} />;
    }
  });
}

function MathView({ tex, display = false }: { tex: string; display?: boolean }) {
  const host = useRef<HTMLSpanElement>(null);

  useLayoutEffect(() => {
    const node = host.current;
    if (!node) return;
    try {
      katex.render(tex, node, {
        displayMode: display,
        throwOnError: false,
        // `ignore`, а не дефолт: кирилиця всередині $…$ (наприклад
        // $V_{дульна}$) інакше сипле попередження в консоль на кожен рендер.
        strict: "ignore",
        trust: false,
      });
    } catch {
      // Навіть із throwOnError:false KaTeX може впасти на зіпсованому вводі.
      // Показуємо вихідний текст — це чесніше за порожнє місце.
      node.textContent = tex;
    }
  }, [tex, display]);

  return (
    <span
      ref={host}
      className={display ? "scroll-thin block overflow-x-auto py-1" : "inline-block"}
    />
  );
}

function renderInline(
  text: string,
  byOrdinal: Ordinals,
  onMarkerClick?: (ordinal: number) => void,
): ReactNode[] {
  const nodes: ReactNode[] = [];
  let lastIndex = 0;
  // Регулярка створюється на кожен виклик: модульна з ручним `lastIndex = 0`
  // безпечна лише доти, доки виклики не вкладені, а тепер вони йдуть із
  // рекурсивного обходу дерева розмітки.
  const marker = /\[(\d{1,3})\]/g;

  for (let match = marker.exec(text); match; match = marker.exec(text)) {
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
