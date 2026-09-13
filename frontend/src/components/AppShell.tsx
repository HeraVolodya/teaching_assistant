/**
 * Оболонка застосунку: шапка, зміст, панель джерела.
 *
 * ДЕМОНСТРАЦІЙНИЙ РЕЖИМ ПОЗНАЧАЄТЬСЯ ЗАВЖДИ. Якщо sidecar не відповів,
 * застосунок працює на прикладі — і мовчати про це не можна. У продукті,
 * де вся цінність тримається на довірі до цитат, показати вигадані дані як
 * справжні гірше, ніж не показати нічого.
 */

import { CircleAlert, Settings } from "lucide-react";
import { Link, Outlet, useLocation } from "react-router-dom";
import { useTranslation } from "react-i18next";

import { SourcePanel } from "@/components/source-viewer/SourcePanel";
import { Badge, Hint } from "@/components/ui/Primitives";
import { useHealth } from "@/hooks/queries";
import { useLive } from "@/hooks/useAppEvents";
import { cn } from "@/lib/cn";

export function AppShell({ demo }: { demo: boolean }) {
  const { t } = useTranslation();
  const { data: health } = useHealth();
  const connected = useLive((state) => state.connected);
  const location = useLocation();

  // `ok` — це лише «сервер відповідає». Модель могла бути не піднята в
  // пам'ять, і тоді кожен запит падає з «No models loaded», а індикатор
  // раніше світився зеленим. Готовність до відповіді — це chatReady.
  const lmstudioDown =
    health && !demo && (!health.lmstudio.ok || health.lmstudio.chatReady === false);

  return (
    <div className="flex h-full min-h-0 flex-col">
      <header className="flex shrink-0 items-center gap-3 border-b border-line bg-surface px-4 py-2">
        <Link to="/" className="flex items-center gap-2 rounded-lg px-1 py-0.5">
          <span className="text-[1.0625rem]" aria-hidden>
            📗
          </span>
          <span className="text-[0.9375rem] font-semibold tracking-tight">{t("app.name")}</span>
        </Link>

        <div className="ml-auto flex items-center gap-2">
          {demo ? (
            <Hint content={t("app.demoHint")}>
              <Badge tone="warn" className="cursor-help">
                <CircleAlert size={11} />
                {t("app.demoBadge")}
              </Badge>
            </Hint>
          ) : null}

          {!demo && !connected ? (
            <Hint content="Канал живих оновлень перервано. Прогрес індексації може відставати.">
              <Badge tone="neutral" className="cursor-help">
                {t("app.offline")}
              </Badge>
            </Hint>
          ) : null}

          {lmstudioDown ? (
            <Hint content={health?.lmstudio.detail || "Пошук по матеріалах працює, але сформулювати відповідь нічим."}>
              <Link to="/setup">
                <Badge tone="danger" className="cursor-pointer">
                  LM Studio
                </Badge>
              </Link>
            </Hint>
          ) : null}

          {/*
            Саме посилання, а не кнопка всередині посилання: <button> у <a> —
            невалідна вкладеність, через яку читачі екрана оголошують елемент
            двічі, а Enter і Space починають робити різні речі.
          */}
          <Link
            to="/settings"
            title={t("nav.settings")}
            aria-label={t("nav.settings")}
            aria-current={location.pathname.startsWith("/settings") ? "page" : undefined}
            className={cn(
              "inline-flex min-h-[2rem] min-w-[2rem] items-center justify-center rounded-lg text-muted transition",
              "hover:bg-raised hover:text-ink",
              location.pathname.startsWith("/settings") && "bg-raised text-ink",
            )}
          >
            <Settings size={16} />
          </Link>
        </div>
      </header>

      {/*
        РЕЖИМ ЗАГЛУШКИ ПОЗНАЧАЄТЬСЯ СМУГОЮ НА ВСЮ ШИРИНУ, А НЕ ПІГУЛКОЮ.
        Спостережено на практиці: викладач завантажив 140-сторінковий звіт,
        побачив «Матеріал додано до бази · 140 с. · Готово» — і поставив
        питання. А в базу лягли ТРИ рядки синтетичної фікстури, бо застосунок
        працював із ASISTENT_STUB=1. Дрібний бейдж «Низька якість розбору» ще
        й збивав з пантелику: розбору не було взагалі.
        У продукті, чия цінність тримається на довірі до цитат, тихе
        підсовування підробленого вмісту — найгірша з можливих поведінок.
      */}
      {health?.stub ? (
        <div
          role="status"
          className="flex shrink-0 items-center gap-2 border-b border-amber-500/40 bg-amber-500/15 px-4 py-1.5 text-[0.8125rem] text-amber-200"
        >
          <CircleAlert size={13} className="shrink-0" />
          <span>
            <b>Режим заглушки.</b> Документи <b>не розбираються</b> — замість їхнього вмісту в базу
            потрапляє синтетичний текст, а відповіді генерує заглушка. Для роботи зі справжніми
            матеріалами запустіть <code className="rounded bg-black/25 px-1">./run.sh --real</code>.
          </span>
        </div>
      ) : null}

      <div className={cn("flex min-h-0 flex-1")}>
        <main className="flex min-h-0 min-w-0 flex-1 flex-col">
          <Outlet />
        </main>
        <SourcePanel />
      </div>
    </div>
  );
}
