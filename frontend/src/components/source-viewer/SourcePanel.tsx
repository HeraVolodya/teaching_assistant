/**
 * Панель джерела праворуч.
 *
 * НЕ НОВЕ ВІКНО. Викладач має бачити відповідь і сторінку підручника
 * ОДНОЧАСНО — інакше перевірка цитати перетворюється на перемикання між
 * вікнами по пам'яті, а це рівно те, чого він не робитиме. Ширина
 * змінюється перетягуванням і запам'ятовується: на 1366×768 звичною є одна
 * пропорція, на зовнішньому моніторі — інша, і нав'язувати одну з них
 * означало б зіпсувати другий сценарій.
 */

import { PanelRightClose } from "lucide-react";
import { useCallback, useEffect, useRef } from "react";
import { useTranslation } from "react-i18next";

import { IconButton } from "@/components/ui/Button";
import { PdfCitationViewer } from "@/components/source-viewer/PdfCitationViewer";
import { useUi } from "@/lib/store";

export function SourcePanel() {
  const { t } = useTranslation();
  const source = useUi((state) => state.source);
  const width = useUi((state) => state.sourceWidth);
  const setWidth = useUi((state) => state.setSourceWidth);
  const close = useUi((state) => state.closeSource);
  const dragging = useRef(false);

  const onPointerMove = useCallback(
    (event: PointerEvent) => {
      if (!dragging.current) return;
      const percent = ((window.innerWidth - event.clientX) / window.innerWidth) * 100;
      setWidth(percent);
    },
    [setWidth],
  );

  useEffect(() => {
    const stop = () => {
      dragging.current = false;
      document.body.style.userSelect = "";
      document.body.style.cursor = "";
    };
    window.addEventListener("pointermove", onPointerMove);
    window.addEventListener("pointerup", stop);
    return () => {
      window.removeEventListener("pointermove", onPointerMove);
      window.removeEventListener("pointerup", stop);
    };
  }, [onPointerMove]);

  // Esc закриває панель — так само, як будь-який тимчасовий шар. Але панель
  // НЕ модальна: клавіатурний фокус лишається там, де був, і чат далі
  // приймає введення.
  useEffect(() => {
    if (!source) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") close();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [source, close]);

  if (!source) return null;

  return (
    <aside
      className="relative flex min-h-0 shrink-0 flex-col border-l border-line bg-surface"
      style={{ width: `${width}%` }}
      aria-label={t("source.title")}
    >
      {/*
        Смуга для перетягування — на всю висоту панелі, а не лише шапки:
        схопити її під час читання сторінки має бути можливо там, де курсор і
        так є. Клавіатурна альтернатива — стрілки на самому роздільнику.
      */}
      <div
        role="separator"
        aria-orientation="vertical"
        aria-label={t("source.widen")}
        aria-valuenow={Math.round(width)}
        aria-valuemin={30}
        aria-valuemax={60}
        tabIndex={0}
        onKeyDown={(event) => {
          if (event.key === "ArrowLeft") setWidth(width + 2);
          else if (event.key === "ArrowRight") setWidth(width - 2);
        }}
        onPointerDown={() => {
          dragging.current = true;
          document.body.style.userSelect = "none";
          document.body.style.cursor = "col-resize";
        }}
        className="absolute -left-1 top-0 z-10 h-full w-2 cursor-col-resize hover:bg-accent/25"
      />

      <div className="flex shrink-0 items-start gap-2 border-b border-line px-3 py-2">
        <div className="min-w-0 flex-1">
          <p className="truncate text-[0.8125rem] font-semibold">{source.documentTitle}</p>
          <p className="mt-0.5 text-[0.75rem] subtle">
            {t("source.page", { page: source.page })} · {t("source.highlight")}
          </p>
        </div>
        <IconButton title={t("source.close")} onClick={close}>
          <PanelRightClose size={16} />
        </IconButton>
      </div>

      {source.quote ? (
        <blockquote className="shrink-0 border-b border-line bg-warn/[0.07] px-3 py-2 text-[0.75rem] leading-relaxed">
          «{source.quote}»
        </blockquote>
      ) : null}

      <PdfCitationViewer
        // key за документом і сторінкою: перехід на іншу цитату мусить
        // повністю переініціалізувати переглядач, інакше він лишається на
        // старій сторінці з новою підсвіткою.
        key={`${source.documentId}:${source.page}`}
        documentId={source.documentId}
        page={source.page}
        bboxes={source.bboxes}
        className="min-h-0 flex-1"
      />
    </aside>
  );
}
