/**
 * База знань: що саме асистент бачить у матеріалах.
 *
 * ПОВНОТЕКСТОВИЙ ПОШУК — ПОЛОВИНА ЦІННОСТІ ЦЬОГО ЕКРАНА. Значна частина
 * того, що викладачеві реально треба, — це не діалог, а grep: «де в цих
 * чотирьох підручниках згадується дирекційний кут». Питання до моделі тут
 * зайве: воно повільніше, дорожче й додає шар, у якому можна помилитися.
 *
 * ПЕРЕМИКАЧ «ЩО БАЧИТЬ ШІ» показує посторінкові оцінки розбору. Це не
 * інструмент розробника: сторінка з низькою оцінкою розпізнавання — це
 * майбутня хибна відповідь, і побачити її ЗАРАЗ дешевше, ніж вилучати
 * потім із відповіді.
 */

import { Eye, FileText, Search } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";

import { searchCollection } from "@/api/endpoints";
import type { PageDto, PlaygroundResult } from "@/api/types";
import { Button, Spinner } from "@/components/ui/Button";
import { Badge, EmptyState, Input, Toggle } from "@/components/ui/Primitives";
import { useDocumentPages, useDocuments } from "@/hooks/queries";
import { cn } from "@/lib/cn";
import { formatNumber, formatScore } from "@/lib/format";
import { useUi } from "@/lib/store";

export function LibraryScreen({ collectionId }: { collectionId: string | undefined }) {
  const { t } = useTranslation();
  const { data: documents = [] } = useDocuments(collectionId);
  const ready = useMemo(() => documents.filter((d) => d.status === "READY"), [documents]);
  const [selected, setSelected] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [aiView, setAiView] = useState(false);
  const [results, setResults] = useState<PlaygroundResult | null>(null);
  const [searching, setSearching] = useState(false);
  const openSource = useUi((state) => state.openSource);

  useEffect(() => {
    if (!selected && ready.length) setSelected(ready[0].id);
  }, [ready, selected]);

  // Пошук із затримкою: кожне натискання клавіші — це повний гібридний
  // прохід із реранкером на CPU, тобто сотні мілісекунд. Запит на кожен
  // символ поставив би в чергу десяток непотрібних проходів.
  useEffect(() => {
    if (!collectionId || query.trim().length < 3) {
      setResults(null);
      return;
    }
    const controller = new AbortController();
    const timer = window.setTimeout(() => {
      setSearching(true);
      void searchCollection(collectionId, query.trim(), 20, true, controller.signal)
        .then(setResults)
        .catch(() => setResults(null))
        .finally(() => setSearching(false));
    }, 350);
    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [collectionId, query]);

  const document_ = ready.find((d) => d.id === selected) ?? null;

  return (
    <div className="flex min-h-0 flex-1">
      <aside className="scroll-thin w-60 shrink-0 overflow-y-auto border-r border-line p-2">
        <p className="px-2 py-1.5 text-[0.6875rem] font-semibold uppercase tracking-wide subtle">
          {t("nav.documents")}
        </p>
        {ready.map((item) => (
          <button
            key={item.id}
            type="button"
            onClick={() => setSelected(item.id)}
            className={cn(
              "flex w-full items-start gap-2 rounded-lg px-2 py-1.5 text-left text-[0.8125rem] transition",
              selected === item.id ? "bg-accent/10 font-medium" : "hover:bg-raised",
            )}
          >
            <FileText size={14} className="mt-0.5 shrink-0 text-muted" />
            <span className="min-w-0 flex-1">
              <span className="block truncate">{item.title}</span>
              <span className="block text-[0.6875rem] tabular-nums subtle">
                {formatNumber(item.pageCount)} с. · {formatNumber(item.chunks)} фрагм.
              </span>
            </span>
          </button>
        ))}
        {!ready.length ? (
          <p className="px-2 py-4 text-[0.75rem] subtle">{t("documents.empty")}</p>
        ) : null}
      </aside>

      <div className="scroll-thin min-w-0 flex-1 overflow-y-auto p-4 sm:p-5">
        <div className="mx-auto max-w-3xl space-y-4">
          <div>
            <div className="relative">
              <Search size={15} className="absolute left-3 top-1/2 -translate-y-1/2 text-muted" />
              <Input
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder={t("library.searchPlaceholder")}
                className="pl-9"
              />
              {searching ? (
                <span className="absolute right-3 top-1/2 -translate-y-1/2">
                  <Spinner className="text-accent" />
                </span>
              ) : null}
            </div>
            <p className="mt-1 text-[0.75rem] subtle">{t("library.searchHint")}</p>
          </div>

          {results ? (
            <SearchResults
              results={results}
              onOpen={(documentTitle, documentId, page) =>
                openSource({ documentId, documentTitle, page, bboxes: [] })
              }
            />
          ) : document_ ? (
            <div className="space-y-3">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div className="min-w-0">
                  <h2 className="truncate text-[0.9375rem] font-semibold">{document_.title}</h2>
                  <p className="mt-0.5 text-[0.75rem] tabular-nums subtle">
                    {formatNumber(document_.pageCount)} с. ·{" "}
                    {formatNumber(document_.chunks)} фрагм. · {document_.language.toUpperCase()}
                  </p>
                </div>
                <Button
                  size="sm"
                  onClick={() =>
                    openSource({
                      documentId: document_.id,
                      documentTitle: document_.title,
                      page: 1,
                      bboxes: [],
                    })
                  }
                >
                  <Eye size={13} />
                  {t("documents.openSource")}
                </Button>
              </div>

              <div className="card p-3">
                <Toggle
                  checked={aiView}
                  onChange={setAiView}
                  label={t("library.aiView")}
                  hint={t("library.aiViewHint")}
                />
              </div>

              {aiView ? <PageQuality documentId={document_.id} /> : null}
            </div>
          ) : (
            <EmptyState title={t("library.selectDocument")} />
          )}
        </div>
      </div>
    </div>
  );
}

function SearchResults({
  results,
  onOpen,
}: {
  results: PlaygroundResult;
  onOpen: (title: string, documentId: string, page: number) => void;
}) {
  const { t } = useTranslation();
  const { data: documents = [] } = useDocuments(results.collectionId);
  const titleToId = new Map(documents.map((d) => [d.title, d.id]));

  if (!results.after.length) {
    return <EmptyState title={t("library.searchEmpty")} hint={t("chat.abstainHint1")} />;
  }

  return (
    <div className="space-y-2">
      <p className="text-[0.75rem] tabular-nums subtle">
        {t("why.counts", {
          fused: results.debug.fusedCount,
          reranked: results.debug.rerankedCount,
          final: results.debug.finalCount,
        })}
      </p>
      {results.after.map((row) => {
        const documentId = titleToId.get(row.documentTitle);
        const page = Number.parseInt(row.pages.replace(/\D+/g, ""), 10) || 1;
        return (
          <button
            key={row.chunkUid}
            type="button"
            disabled={!documentId}
            onClick={() => documentId && onOpen(row.documentTitle, documentId, page)}
            className="card w-full p-3 text-left transition hover:border-accent/50 disabled:opacity-60"
          >
            <div className="mb-1 flex flex-wrap items-center gap-2">
              <span className="min-w-0 flex-1 truncate text-[0.8125rem] font-medium">
                {row.documentTitle}
              </span>
              <span className="text-[0.75rem] subtle">{row.pages}</span>
              <Badge tone="neutral" className="tabular-nums">
                {formatScore(row.score)}
              </Badge>
            </div>
            <p className="line-clamp-3 text-[0.8125rem] leading-relaxed subtle">{row.text}</p>
          </button>
        );
      })}
    </div>
  );
}

/**
 * Посторінкові оцінки розбору.
 *
 * Показуються ТІ САМІ числа, за якими конвеєр вирішує, чи ремонтувати
 * сторінку (`PageInfo.needs_vlm_repair`). Показати щось інше означало б дати
 * викладачеві другу, неузгоджену картину якості.
 */
function PageQuality({ documentId }: { documentId: string }) {
  const { t } = useTranslation();
  const { data: pages = [], isLoading } = useDocumentPages(documentId);
  const openSource = useUi((state) => state.openSource);

  if (isLoading) return <p className="text-[0.8125rem] subtle">{t("app.loading")}</p>;
  if (!pages.length) {
    return (
      <p className="rounded-lg border border-dashed border-line p-4 text-center text-[0.8125rem] subtle">
        Посторінкових даних немає — матеріал оброблено без збереження оцінок сторінок.
      </p>
    );
  }

  return (
    <div className="card overflow-hidden">
      <table className="w-full text-[0.75rem]">
        <thead className="bg-raised text-left">
          <tr>
            <th className="px-3 py-2 font-medium">{t("common.page")}</th>
            <th className="px-3 py-2 font-medium">Тип</th>
            <th className="px-3 py-2 text-right font-medium tabular-nums">Розбір</th>
            <th className="px-3 py-2 text-right font-medium tabular-nums">Розмітка</th>
            <th className="px-3 py-2 text-right font-medium tabular-nums">Таблиці</th>
            <th className="px-3 py-2 text-right font-medium tabular-nums">Розпізнавання</th>
          </tr>
        </thead>
        <tbody>
          {pages.map((page) => (
            <PageRow
              key={page.pageNumber}
              page={page}
              onOpen={() =>
                openSource({
                  documentId,
                  documentTitle: "",
                  page: page.pageNumber,
                  bboxes: [],
                })
              }
            />
          ))}
        </tbody>
      </table>
    </div>
  );
}

function PageRow({ page, onOpen }: { page: PageDto; onOpen: () => void }) {
  const { t } = useTranslation();
  const scanned = page.pageClass === "SCANNED" || page.pageClass === "MIXED";
  return (
    <tr
      onClick={onOpen}
      className={cn(
        "cursor-pointer border-t border-line transition hover:bg-raised",
        page.needsRepair && "bg-warn/[0.07]",
      )}
    >
      <td className="px-3 py-1.5 tabular-nums">{page.pageLabel ?? page.pageNumber}</td>
      <td className="px-3 py-1.5">
        <Badge tone={page.needsRepair ? "warn" : scanned ? "neutral" : "ok"}>
          {page.needsRepair ? t("library.needsRepair") : scanned ? t("library.scanned") : t("library.digital")}
        </Badge>
      </td>
      <Score value={page.parseScore} />
      <Score value={page.layoutScore} />
      <Score value={page.tableScore} />
      <Score value={page.ocrScore} />
    </tr>
  );
}

function Score({ value }: { value: number | null }) {
  // Порогові кольори узгоджені з `PageInfo.needs_vlm_repair`: нижче 0.5 —
  // сторінка, яку конвеєр вважає провальною. Один погляд на колонку
  // показує, звідки в майбутньому візьмуться хибні цитати.
  return (
    <td
      className={cn(
        "px-3 py-1.5 text-right tabular-nums",
        value != null && value < 0.5 ? "text-danger" : value != null && value < 0.7 ? "text-warn" : "",
      )}
    >
      {value == null ? "—" : formatScore(value)}
    </td>
  );
}
