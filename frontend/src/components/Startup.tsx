/**
 * Екрани старту: очікування sidecar і чесна помилка, якщо він не піднявся.
 *
 * ЧОМУ ЦЕ ОКРЕМИЙ ФАЙЛ І ЧОМУ БЕЗ ЗАЛЕЖНОСТЕЙ.
 * Обидва компоненти рендеряться ДО того, як змонтовано дерево застосунку: без
 * роутера, без QueryClient, без `TooltipProvider`. Це те, що бачить викладач,
 * коли зламано все інше, — тож тут навмисно немає нічого, крім React, i18n і
 * класів теми. Будь-який імпорт із `@/hooks` чи `@/components/ui` зробив би
 * екран помилки залежним від того самого коду, який міг і впасти.
 *
 * Помилка описує наслідок («матеріали не завантажено»), а не симптом
 * («fetch failed»): рішення, яке викладач може ухвалити, — це перезапустити
 * застосунок або надіслати журнал, і саме до них веде текст.
 */

import type { ReactNode } from "react";
import { useTranslation } from "react-i18next";

/** Шлях до `sidecar.log` — дзеркало `Layout::resolve` із sidecar.rs. */
function logPath(): string {
  const agent = typeof navigator === "undefined" ? "" : navigator.userAgent;
  if (/Mac|iPhone|iPad/i.test(agent)) return "~/Library/Logs/Asistent/sidecar.log";
  return "%LOCALAPPDATA%\\Asistent\\logs\\sidecar.log";
}

function Frame({ children }: { children: ReactNode }) {
  return (
    <div className="flex h-full min-h-screen items-center justify-center bg-bg p-6 text-ink">
      <div className="w-full max-w-xl rounded-xl border border-line bg-surface p-6">{children}</div>
    </div>
  );
}

/** Показується, поки триває `probeBackend`. */
export function StartupScreen() {
  const { t } = useTranslation();
  return (
    <Frame>
      <div className="flex items-center gap-3">
        <span
          aria-hidden
          className="size-4 shrink-0 animate-spin rounded-full border-2 border-line border-t-accent"
        />
        <div>
          <p className="text-[0.9375rem] font-semibold">{t("startup.waiting")}</p>
          <p className="mt-1 text-[0.8125rem] text-muted">{t("startup.waitingHint")}</p>
        </div>
      </div>
    </Frame>
  );
}

/**
 * Показується замість застосунку, коли sidecar не відповів за відведений час.
 *
 * НЕ підставляє демонстраційні дані. Раніше саме тут застосунок мовчки
 * вмикав вигаданий корпус, і викладач отримував фальшиві цитати замість
 * повідомлення про несправність.
 */
export function StartupError({ seconds }: { seconds: number }) {
  const { t } = useTranslation();
  return (
    <Frame>
      <h1 className="text-[1.0625rem] font-semibold">{t("startup.failedTitle")}</h1>
      <p className="mt-2 text-[0.875rem] text-muted">{t("startup.failedBody", { seconds })}</p>

      <p className="mt-4 text-[0.8125rem] text-muted">{t("startup.failedLog")}</p>
      <code className="mt-1 block rounded-lg border border-line bg-raised px-2 py-1.5 font-mono text-[0.75rem] break-all">
        {logPath()}
      </code>

      <button
        type="button"
        onClick={() => window.location.reload()}
        className="mt-5 rounded-lg bg-accent px-3 py-1.5 text-[0.875rem] font-medium text-accent-ink"
      >
        {t("startup.retry")}
      </button>
    </Frame>
  );
}
