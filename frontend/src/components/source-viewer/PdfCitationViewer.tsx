/**
 * Вбудований переглядач PDF із підсвіткою цитованого фрагмента.
 *
 * ЧОМУ НЕ СИСТЕМНИЙ ПЕРЕГЛЯДАЧ. Фрагмент `#page=N` шанують Chrome і Firefox,
 * але Safari та Preview його ІГНОРУЮТЬ, а типовий обробник PDF у Windows не
 * приймає номер сторінки через `shell open`. Тобто системний шлях не може
 * відкрити документ на потрібній сторінці ЖОДНОЇ з двох цільових платформ —
 * і це ще до того, як зайде мова про підсвітку абзацу, якої він не вміє
 * взагалі.
 *
 * ЧОМУ БАЙТИ ЧЕРЕЗ SIDECAR, А НЕ `asset://`. `GET /api/documents/{id}/file`
 * підтримує Range, тож pdf.js читає спершу хвіст із таблицею xref, а потім
 * лише потрібні сторінки. Триста мегабайтів підручника не проходять ані
 * через пам'ять, ані через очікування. Плюс той самий код працює і в
 * браузері на `vite dev`, і в оболонці, і в майбутній веб-версії — тобто
 * інваріант «оболонка володіє лише вікном» лишається цілим.
 *
 * ЧОМУ ВЛАСНИЙ РЕНДЕР, А НЕ `pdf_viewer.mjs`. Готовий `PDFViewer` тягне
 * `EventBus`, `PDFLinkService`, власну тему й повний CSS вбудованого
 * переглядача, з якого нам потрібні три класи. Власний цикл — це ~150
 * рядків, зате повний контроль над шаром підсвітки, який мусить лежати НАД
 * полотном, але ПІД текстовим шаром: інакше він перехоплює виділення
 * тексту мишею, і цитату стає неможливо скопіювати.
 */

import { ChevronLeft, ChevronRight, Minus, Plus, Scan } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";

import { documentFileUrl } from "@/api/endpoints";
import type { BBoxDto } from "@/api/types";
import { IconButton } from "@/components/ui/Button";
import { bboxesToRects, mergeRects, type HighlightRect } from "@/lib/bbox";
import { cn } from "@/lib/cn";

type PdfModule = typeof import("pdfjs-dist");
type PdfDocument = Awaited<ReturnType<PdfModule["getDocument"]>["promise"]>;
type PdfPage = Awaited<ReturnType<PdfDocument["getPage"]>>;

let pdfjsPromise: Promise<PdfModule> | null = null;

/**
 * Завантаження pdf.js — ліниве.
 *
 * Бібліотека важить близько мегабайта, а переглядач джерела відкривається
 * не в кожній сесії. Тримати її в основному бандлі означало б платити цим
 * мегабайтом за кожен холодний старт застосунку, включно з тим, коли
 * викладач просто дивиться список асистентів.
 */
async function loadPdfjs(): Promise<PdfModule> {
  if (!pdfjsPromise) {
    pdfjsPromise = import("pdfjs-dist").then(async (module) => {
      const worker = await import("pdfjs-dist/build/pdf.worker.mjs?url");
      module.GlobalWorkerOptions.workerSrc = worker.default;
      return module;
    });
  }
  return pdfjsPromise;
}

/**
 * Шрифти й таблиці кодування постачаються локально (див. плагін у
 * `vite.config.ts`). Закритий контур: жодного звернення до CDN, інакше
 * документи з базовими шрифтами Type1 рендерилися б порожніми сторінками
 * рівно на тій машині, де немає мережі, — тобто на цільовій.
 */
const STANDARD_FONTS = "/pdfjs/standard_fonts/";
const CMAPS = "/pdfjs/cmaps/";

const ZOOM_STEPS = [0.5, 0.65, 0.8, 1, 1.25, 1.5, 2, 3];

export interface PdfCitationViewerProps {
  documentId: string;
  /** Фізичний номер сторінки, 1-based. */
  page: number;
  bboxes: BBoxDto[];
  className?: string;
}

interface PageEntry {
  number: number;
  width: number;
  height: number;
}

export function PdfCitationViewer({ documentId, page, bboxes, className }: PdfCitationViewerProps) {
  const { t } = useTranslation();
  const scrollRef = useRef<HTMLDivElement>(null);
  const [document_, setDocument] = useState<PdfDocument | null>(null);
  const [pages, setPages] = useState<PageEntry[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [zoom, setZoom] = useState<number | "fit">("fit");
  const [containerWidth, setContainerWidth] = useState(0);
  const [current, setCurrent] = useState(page);

  // --------------------------------------------------------- завантаження
  useEffect(() => {
    let cancelled = false;
    let task: { destroy: () => Promise<void> } | null = null;

    void (async () => {
      setError(null);
      setDocument(null);
      setPages([]);
      try {
        const pdfjs = await loadPdfjs();
        const loading = pdfjs.getDocument({
          url: documentFileUrl(documentId),
          standardFontDataUrl: STANDARD_FONTS,
          cMapUrl: CMAPS,
          cMapPacked: true,
          // Системні шрифти НЕ підставляються: у закритому контурі набір
          // шрифтів на машині викладача невідомий, і мовчазна підміна дала б
          // сторінку, яка виглядає інакше, ніж оригінал, — а саме на її
          // вигляд він і покладається, перевіряючи цитату.
          useSystemFonts: false,
        });
        task = loading;
        const doc = await loading.promise;
        if (cancelled) {
          void loading.destroy();
          return;
        }
        /*
         * Розміри сторінок потрібні НАПЕРЕД: без них контейнери мають нульову
         * висоту, прокрутка на потрібну сторінку потрапляє в нікуди, і
         * «відкрити на сторінці 143» перетворюється на «відкрити на першій».
         *
         * Але опитувати всі сторінки не можна: на 512-сторінковому підручнику
         * це 512 послідовних `getPage`, тобто секунди очікування рівно в той
         * момент, коли викладач натиснув на цитату й чекає сторінку. Тому
         * меряються ДВІ сторінки — перша й цільова, — а решта отримує розміри
         * першої. У книжковій верстці формат сторінки однаковий; поодинокі
         * винятки (вклейка з картою) виправляються самі при рендері, коли
         * сторінка повідомляє власний viewport.
         */
        const total = doc.numPages;
        const first = await doc.getPage(1);
        const base = first.getViewport({ scale: 1 });
        first.cleanup();
        if (cancelled) return;

        const sizes = new Map<number, { width: number; height: number }>();
        sizes.set(1, { width: base.width, height: base.height });
        const target = Math.min(Math.max(1, page), total);
        if (target !== 1) {
          const targetPage = await doc.getPage(target);
          const targetViewport = targetPage.getViewport({ scale: 1 });
          sizes.set(target, { width: targetViewport.width, height: targetViewport.height });
          targetPage.cleanup();
          if (cancelled) return;
        }

        const entries: PageEntry[] = Array.from({ length: total }, (_, index) => {
          const number = index + 1;
          const size = sizes.get(number) ?? { width: base.width, height: base.height };
          return { number, width: size.width, height: size.height };
        });
        setDocument(doc);
        setPages(entries);
      } catch (cause) {
        if (!cancelled) setError(cause instanceof Error ? cause.message : String(cause));
      }
    })();

    return () => {
      cancelled = true;
      void task?.destroy().catch(() => undefined);
    };
  }, [documentId, page]);

  /** Сторінка нестандартного формату повідомляє свій розмір при рендері. */
  const measure = useCallback((number: number, width: number, height: number) => {
    setPages((current) => {
      const entry = current[number - 1];
      if (!entry || (Math.abs(entry.width - width) < 1 && Math.abs(entry.height - height) < 1)) {
        return current;
      }
      const next = [...current];
      next[number - 1] = { number, width, height };
      return next;
    });
  }, []);

  // ------------------------------------------------------------ масштаб
  useEffect(() => {
    const element = scrollRef.current;
    if (!element) return;
    const observer = new ResizeObserver(([entry]) => setContainerWidth(entry.contentRect.width));
    observer.observe(element);
    setContainerWidth(element.clientWidth);
    return () => observer.disconnect();
  }, [document_]);

  const scale = useMemo(() => {
    if (zoom !== "fit") return zoom;
    const first = pages[0];
    if (!first || !containerWidth) return 1;
    // 32px — поля навколо сторінки; без запасу остання колонка тексту
    // ховається під смугою прокрутки на 150% DPI.
    return Math.max(0.25, Math.min(3, (containerWidth - 32) / first.width));
  }, [zoom, pages, containerWidth]);

  // -------------------------------------------------- перехід на сторінку
  const scrollToPage = useCallback((target: number, smooth = true) => {
    const container = scrollRef.current;
    if (!container) return;
    const element = container.querySelector<HTMLElement>(`[data-page="${target}"]`);
    if (!element) return;
    container.scrollTo({
      top: element.offsetTop - 12,
      behavior: smooth ? "smooth" : "auto",
    });
  }, []);

  // Перший показ — БЕЗ анімації: плавна прокрутка через півкниги виглядає
  // як збій, а не як перехід, і триває довше, ніж людина готова чекати.
  useEffect(() => {
    if (!pages.length) return;
    setCurrent(page);
    const timer = window.setTimeout(() => scrollToPage(page, false), 40);
    return () => window.clearTimeout(timer);
  }, [pages.length, page, scrollToPage]);

  // ------------------------------------------------------------- рендер
  if (error) {
    return (
      <div className={cn("flex h-full items-center justify-center p-6 text-center", className)}>
        <div>
          <p className="text-[0.875rem] font-medium">{t("source.failed")}</p>
          <p className="mt-1 max-w-sm text-[0.75rem] subtle">{error}</p>
        </div>
      </div>
    );
  }

  const zoomIndex = typeof zoom === "number" ? ZOOM_STEPS.indexOf(zoom) : -1;

  return (
    <div className={cn("flex h-full min-h-0 flex-col", className)}>
      <div className="flex shrink-0 items-center gap-1 border-b border-line px-2 py-1.5">
        <IconButton
          title={t("source.zoomOut")}
          disabled={!pages.length}
          onClick={() =>
            setZoom((value) => {
              const index = typeof value === "number" ? ZOOM_STEPS.indexOf(value) : ZOOM_STEPS.indexOf(1);
              return ZOOM_STEPS[Math.max(0, index - 1)];
            })
          }
        >
          <Minus size={15} />
        </IconButton>
        <IconButton
          title={t("source.zoomIn")}
          disabled={!pages.length}
          onClick={() =>
            setZoom((value) => {
              const index = typeof value === "number" ? ZOOM_STEPS.indexOf(value) : ZOOM_STEPS.indexOf(1);
              return ZOOM_STEPS[Math.min(ZOOM_STEPS.length - 1, index + 1)];
            })
          }
        >
          <Plus size={15} />
        </IconButton>
        <IconButton title={t("source.fit")} active={zoom === "fit"} onClick={() => setZoom("fit")}>
          <Scan size={15} />
        </IconButton>
        <span className="mx-1 h-4 w-px bg-line" />
        <IconButton
          title="Попередня сторінка"
          disabled={current <= 1}
          onClick={() => {
            const next = Math.max(1, current - 1);
            setCurrent(next);
            scrollToPage(next);
          }}
        >
          <ChevronLeft size={15} />
        </IconButton>
        <span className="min-w-[5.5rem] text-center text-[0.75rem] tabular-nums subtle">
          {pages.length ? `${current} / ${pages.length}` : "—"}
        </span>
        <IconButton
          title="Наступна сторінка"
          disabled={!pages.length || current >= pages.length}
          onClick={() => {
            const next = Math.min(pages.length, current + 1);
            setCurrent(next);
            scrollToPage(next);
          }}
        >
          <ChevronRight size={15} />
        </IconButton>
        <span className="ml-auto text-[0.6875rem] subtle">
          {zoomIndex >= 0 ? `${Math.round(scale * 100)} %` : t("source.fit")}
        </span>
      </div>

      <div ref={scrollRef} className="scroll-thin min-h-0 flex-1 overflow-auto bg-[hsl(var(--bg))] py-3">
        {!pages.length ? (
          <p className="py-12 text-center text-[0.8125rem] subtle">{t("source.loading")}</p>
        ) : (
          <div className="pdfViewer">
            {pages.map((entry) => (
              <PdfPageView
                key={entry.number}
                document={document_}
                entry={entry}
                scale={scale}
                bboxes={bboxes}
                highlighted={entry.number === page}
                onVisible={setCurrent}
                onMeasured={measure}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

/**
 * Одна сторінка.
 *
 * Полотно малюється ЛИШЕ коли сторінка близька до вікна. 512-сторінковий
 * підручник у повному рендері — це кілька гігабайтів растру, тобто
 * гарантоване падіння вкладки; а від «намалювати все одразу» немає жодної
 * користі, бо видно все одно одну сторінку.
 */
function PdfPageView({
  document: doc,
  entry,
  scale,
  bboxes,
  highlighted,
  onVisible,
  onMeasured,
}: {
  document: PdfDocument | null;
  entry: PageEntry;
  scale: number;
  bboxes: BBoxDto[];
  highlighted: boolean;
  onVisible: (page: number) => void;
  onMeasured: (page: number, width: number, height: number) => void;
}) {
  const hostRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const textRef = useRef<HTMLDivElement>(null);
  const [near, setNear] = useState(false);
  const [rects, setRects] = useState<HighlightRect[]>([]);

  const width = entry.width * scale;
  const height = entry.height * scale;

  // Вікно попереднього рендеру — дві висоти екрана в кожен бік: сторінка
  // встигає намалюватися до того, як користувач до неї доскролив.
  useEffect(() => {
    const element = hostRef.current;
    if (!element) return;
    const observer = new IntersectionObserver(
      (entries) => {
        for (const item of entries) {
          setNear(item.isIntersecting);
          if (item.isIntersecting && item.intersectionRatio > 0.5) onVisible(entry.number);
        }
      },
      { root: element.closest(".scroll-thin"), rootMargin: "200% 0px", threshold: [0, 0.5] },
    );
    observer.observe(element);
    return () => observer.disconnect();
  }, [entry.number, onVisible]);

  useEffect(() => {
    if (!doc || !near) return;
    let cancelled = false;
    let page: PdfPage | null = null;
    let task: { cancel: () => void } | null = null;

    void (async () => {
      page = await doc.getPage(entry.number);
      if (cancelled || !page) return;
      const viewport = page.getViewport({ scale });
      onMeasured(entry.number, viewport.width / scale, viewport.height / scale);

      const canvas = canvasRef.current;
      if (canvas) {
        // Множник щільності пікселів обов'язковий: без нього на 150% DPI
        // сторінка виглядає розмитою, і скан 1987 року стає нечитним саме
        // там, де його й треба перевіряти очима.
        const ratio = Math.min(2, window.devicePixelRatio || 1);
        canvas.width = Math.floor(viewport.width * ratio);
        canvas.height = Math.floor(viewport.height * ratio);
        canvas.style.width = `${viewport.width}px`;
        canvas.style.height = `${viewport.height}px`;
        const context = canvas.getContext("2d");
        if (context) {
          context.setTransform(ratio, 0, 0, ratio, 0, 0);
          const render = page.render({ canvasContext: context, viewport, canvas });
          task = render;
          try {
            await render.promise;
          } catch {
            /* рендер скасовано зміною масштабу — це нормальний шлях */
          }
        }
      }
      if (cancelled) return;

      const container = textRef.current;
      if (container && page) {
        container.replaceChildren();
        const pdfjs = await loadPdfjs();
        const layer = new pdfjs.TextLayer({
          textContentSource: page.streamTextContent(),
          container,
          viewport,
        });
        await layer.render().catch(() => undefined);
      }

      if (!cancelled) {
        // Підсвітка перераховується разом із рендером, а не окремим ефектом:
        // прямокутник, порахований для попереднього масштабу, з'їхав би на
        // кілька сантиметрів — і цитата вказувала б на сусідній абзац.
        setRects(mergeRects(bboxesToRects(bboxes, entry.number, viewport)));
      }
    })();

    return () => {
      cancelled = true;
      task?.cancel();
      page?.cleanup();
    };
  }, [doc, near, entry.number, scale, bboxes, onMeasured]);

  return (
    <div
      ref={hostRef}
      data-page={entry.number}
      className="page"
      style={{ width, height }}
      aria-label={`Сторінка ${entry.number}`}
    >
      {near ? (
        <>
          <div className="canvasWrapper" style={{ width, height }}>
            <canvas ref={canvasRef} />
          </div>
          <div className="asistent-highlight-layer">
            {rects.map((rect, index) => (
              <div
                key={`${rect.left}-${rect.top}-${index}`}
                className="asistent-highlight"
                style={{
                  left: rect.left,
                  top: rect.top,
                  width: rect.width,
                  height: rect.height,
                }}
              />
            ))}
          </div>
          <div ref={textRef} className="textLayer" style={{ width, height }} />
        </>
      ) : (
        <div
          className={cn(
            "flex h-full w-full items-center justify-center text-[0.75rem] text-black/25",
            highlighted && "bg-warn/5",
          )}
        >
          {entry.number}
        </div>
      )}
    </div>
  );
}
