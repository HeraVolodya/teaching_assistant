/**
 * Налаштування.
 *
 * ЧЕСНА МЕЖА МІЖ ТИМ, ЩО МОЖНА ЗМІНИТИ, І ТИМ, ЩО НІ. Адреса LM Studio,
 * пристрій обчислень і тека даних задаються службою при запуску
 * (`ASISTENT_*`), і ручки для їх зміни бекенд не має. Тому вони показані
 * ТІЛЬКИ ДЛЯ ЧИТАННЯ з поясненням, хто їх задає. Поле введення, яке нічого
 * не змінює, — гірше за його відсутність: воно виглядає як робоче, і
 * витрачений на нього час обертається зверненням у підтримку.
 *
 * ПОПЕРЕДЖЕННЯ ПРО ВІДЕОКАРТУ ПОКАЗУЄТЬСЯ ЗАВЖДИ, а не лише при виборі GPU.
 * Відеокарта спільна з LM Studio, і використання її для обробки матеріалів
 * дає нестачу пам'яті на етапі ЗАВАНТАЖЕННЯ мовної моделі — тобто збій
 * виглядає як «ШІ перестав працювати», без жодного зв'язку з індексацією.
 *
 * П'ЯТЬ КЛІКІВ НА ВЕРСІЇ вмикають інструменти розробника. Схований, але
 * відтворюваний жест: він не засмічує інтерфейс і водночас дозволяє
 * попросити викладача по телефону «клацніть п'ять разів на номері версії».
 */

import { Download, Monitor, Moon, Sun, TriangleAlert } from "lucide-react";
import { useState } from "react";
import { useTranslation } from "react-i18next";

import { fetchDiagnosticsBundle } from "@/api/endpoints";
import { apiMode } from "@/api/client";
import { Button } from "@/components/ui/Button";
import { Badge, Banner, Field, RadioCards, Range, Section, Toggle } from "@/components/ui/Primitives";
import { useHealth } from "@/hooks/queries";
import { cn } from "@/lib/cn";
import { formatBytes, formatDuration } from "@/lib/format";
import { useUi, type Theme } from "@/lib/store";
import { SUPPORTED_LANGUAGES, type Language } from "@/i18n";

export function SettingsScreen() {
  const { t } = useTranslation();
  const { data: health } = useHealth();
  const ui = useUi();
  const [downloading, setDownloading] = useState(false);
  const [includeTitles, setIncludeTitles] = useState(false);

  const saveBundle = async () => {
    setDownloading(true);
    try {
      const response = await fetchDiagnosticsBundle();
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `asistent-diagnostics-${new Date().toISOString().slice(0, 10)}.zip`;
      link.click();
      // Звільняємо URL із затримкою: миттєвий revoke обриває збереження
      // великого архіву на повільному диску.
      window.setTimeout(() => URL.revokeObjectURL(url), 10_000);
    } catch {
      /* помилку показує сам бекенд; тут мовчимо, щоб не вигадувати текст */
    } finally {
      setDownloading(false);
    }
  };

  return (
    <div className="scroll-thin min-h-0 flex-1 overflow-y-auto px-4 py-6 sm:px-6">
      <div className="mx-auto max-w-2xl space-y-4">
        <h1 className="text-[1.25rem] font-semibold tracking-tight">{t("settings.title")}</h1>

        <Section title={t("settings.appearance")}>
          <div className="space-y-4">
            <Field label={t("settings.theme")}>
              <div className="flex gap-1.5">
                {(
                  [
                    { value: "system", label: t("settings.themeSystem"), icon: <Monitor size={14} /> },
                    { value: "light", label: t("settings.themeLight"), icon: <Sun size={14} /> },
                    { value: "dark", label: t("settings.themeDark"), icon: <Moon size={14} /> },
                  ] as { value: Theme; label: string; icon: React.ReactNode }[]
                ).map((option) => (
                  <button
                    key={option.value}
                    type="button"
                    onClick={() => ui.setTheme(option.value)}
                    className={cn(
                      "flex flex-1 items-center justify-center gap-1.5 rounded-lg border px-3 py-2 text-[0.8125rem] transition",
                      ui.theme === option.value ? "border-accent bg-accent/10" : "border-line hover:bg-raised",
                    )}
                  >
                    {option.icon}
                    {option.label}
                  </button>
                ))}
              </div>
            </Field>

            <Field
              label={t("settings.scale")}
              hint="Впливає лише на цей застосунок. Системний масштаб Windows діє поверх цього."
            >
              <Range
                value={Math.round(ui.scale * 20)}
                min={17}
                max={27}
                onChange={(value) => ui.setScale(value / 20)}
              />
              <p className="mt-1 text-[0.75rem] tabular-nums subtle">
                {Math.round(ui.scale * 100)} %
              </p>
            </Field>

            <Field label={t("settings.language")}>
              <RadioCards<Language>
                value={ui.language}
                onChange={ui.setLanguage}
                options={SUPPORTED_LANGUAGES.map((code) => ({
                  value: code,
                  label: code === "uk" ? "Українська" : "English",
                }))}
              />
            </Field>
          </div>
        </Section>

        <Section title={t("settings.hardware")} description={t("settings.serverManaged")}>
          <div className="space-y-3">
            <ReadOnlyRow label={t("settings.lmstudioUrl")} value={health?.lmstudio.baseUrl ?? "—"} mono />
            <ReadOnlyRow
              label="Стан мовної моделі"
              value={
                health?.lmstudio.ok
                  ? (health.lmstudio.models.find((m) => m.loaded)?.key ?? "сервер працює, модель не завантажено")
                  : (health?.lmstudio.detail ?? "—")
              }
            />
            <ReadOnlyRow label="Режим обробки матеріалів" value={health?.worker.mode ?? "—"} />

            <Banner tone="warn" title={t("settings.computeDevice")} icon={<TriangleAlert size={15} />}>
              {t("settings.deviceWarning")}
            </Banner>
          </div>
        </Section>

        <Section title={t("settings.storage")}>
          <div className="space-y-3">
            <ReadOnlyRow label={t("settings.dataDir")} value={health?.database.path ?? "—"} mono />
            <ReadOnlyRow
              label="Версія схеми бази"
              value={String(health?.database.schemaVersion ?? "—")}
            />
          </div>
        </Section>

        <Section title={t("settings.diagnostics")} description={t("settings.diagnosticsHint")}>
          <div className="space-y-3">
            <Toggle
              checked={includeTitles}
              onChange={setIncludeTitles}
              label={t("settings.diagnosticsTitles")}
              hint="Вимкнено за замовчуванням: перелік завантажених матеріалів сам по собі є інформацією."
            />
            <Button onClick={() => void saveBundle()} loading={downloading} disabled={apiMode() === "demo"}>
              <Download size={14} />
              {t("settings.diagnosticsDownload")}
            </Button>
          </div>
        </Section>

        <Section title={t("settings.developer")}>
          <Toggle
            checked={ui.developerMode}
            onChange={ui.setDeveloperMode}
            label={t("settings.developerOn")}
            hint={t("settings.developerHint")}
          />
        </Section>

        <Section title={t("settings.about")}>
          <p className="text-[0.8125rem] leading-relaxed">{t("settings.aboutBody")}</p>
          <div className="mt-3 flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={() => ui.bumpVersionClicks()}
              title="Натисніть п'ять разів, щоб перемкнути інструменти розробника"
              className="rounded-lg bg-raised px-2 py-1 text-[0.75rem] tabular-nums subtle"
            >
              {t("settings.version")} {health?.version ?? "—"}
            </button>
            {health?.stub ? <Badge tone="warn">режим заглушок</Badge> : null}
            {apiMode() === "demo" ? <Badge tone="warn">{t("app.demoBadge")}</Badge> : null}
            {health?.uptimeSeconds ? (
              <span className="text-[0.75rem] subtle">
                працює {formatDuration(health.uptimeSeconds)}
              </span>
            ) : null}
          </div>
          {health?.problems.length ? (
            <ul className="mt-3 space-y-2">
              {health.problems.map((problem) => (
                <li key={problem.code}>
                  <Banner tone="warn" title={problem.message}>
                    {problem.hint}
                  </Banner>
                </li>
              ))}
            </ul>
          ) : null}
        </Section>
      </div>
    </div>
  );
}

function ReadOnlyRow({ label, value, mono = false }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="flex flex-wrap items-baseline justify-between gap-2 border-b border-line pb-2 last:border-0 last:pb-0">
      <span className="text-[0.8125rem] font-medium">{label}</span>
      <span className={cn("min-w-0 break-all text-right text-[0.75rem] subtle", mono && "font-mono")}>
        {value}
      </span>
    </div>
  );
}

/** Використовується у зведенні сховища, коли бекенд віддасть розміри. */
export function StorageRow({ label, bytes }: { label: string; bytes: number | null }) {
  return <ReadOnlyRow label={label} value={formatBytes(bytes)} />;
}
