/**
 * Перший запуск: чотири кроки, з можливістю повернутись.
 *
 * ДВА ФАКТИ ВИЗНАЧАЮТЬ ЦЕЙ ЕКРАН.
 *
 * 1. Викладач не обирає квантизацію. Він взагалі не має відкривати вкладку
 *    Discover у LM Studio. Застосунок сам меряє відеопам'ять, сам називає
 *    модель і сам ставить кнопку «Завантажити цю модель». Це найбільший
 *    виграш у зручності з усього API — і водночас те, що прибирає цілий
 *    клас звернень у підтримку.
 *
 * 2. Кнопка «Перевірити» показує TTFT і токени за секунду. Одне це число
 *    відповідає на питання «чому воно так довго думає» ще до того, як воно
 *    буде задане: 2.4 с до першого символу — це нормальна робота, а не збій.
 *
 * ЧОГО ТУТ СВІДОМО НЕМАЄ: вибору теки для матеріалів. Її задає адміністратор
 * при встановленні (`ASISTENT_DATA_DIR`), і бекенд не має ручки для зміни.
 * Показати поле, яке нічого не змінює, було б гірше, ніж чесно пояснити, хто
 * і де це налаштовує.
 */

import {
  Check,
  ChevronLeft,
  ChevronRight,
  CircleAlert,
  Cpu,
  Download,
  FolderOpen,
  Gauge,
  Languages,
  Package,
  Play,
  X,
} from "lucide-react";
import { useState } from "react";
import { useTranslation } from "react-i18next";

import { downloadModel, startLmStudio } from "@/api/endpoints";
import type { SetupStatus } from "@/api/types";
import { markSetupSeen } from "@/App";
import { Button } from "@/components/ui/Button";
import { Badge, Banner, Progress, RadioCards, Section } from "@/components/ui/Primitives";
import { useSetupStatus } from "@/hooks/queries";
import { useLive } from "@/hooks/useAppEvents";
import { cn } from "@/lib/cn";
import { formatBytes } from "@/lib/format";
import { useUi } from "@/lib/store";
import { SUPPORTED_LANGUAGES, type Language } from "@/i18n";

const STEPS = 4;

export function SetupWizard({ onDone }: { onDone: () => void }) {
  const { t } = useTranslation();
  const [step, setStep] = useState(1);
  const { data: setup, refetch } = useSetupStatus();

  const finish = () => {
    markSetupSeen();
    onDone();
  };

  return (
    <div className="scroll-thin h-full overflow-y-auto bg-bg">
      <div className="mx-auto flex min-h-full max-w-2xl flex-col px-5 py-8">
        <header className="mb-6">
          <div className="flex items-center justify-between gap-3">
            <h1 className="text-[1.25rem] font-semibold tracking-tight">{t("setup.title")}</h1>
            <Button size="sm" variant="ghost" onClick={finish}>
              {t("setup.skip")}
            </Button>
          </div>
          <div className="mt-3 flex items-center gap-2">
            {Array.from({ length: STEPS }, (_, index) => (
              <span
                key={index}
                className={cn(
                  "h-1 flex-1 rounded-full transition",
                  index < step ? "bg-accent" : "bg-line",
                )}
              />
            ))}
          </div>
          <p className="mt-2 text-[0.75rem] tabular-nums subtle">
            {t("setup.step", { current: step, total: STEPS })}
          </p>
        </header>

        <div className="flex-1">
          {step === 1 ? <LanguageStep /> : null}
          {step === 2 ? <StorageStep setup={setup} /> : null}
          {step === 3 ? <ModelsStep setup={setup} /> : null}
          {step === 4 ? <LmStudioStep setup={setup} onRefresh={() => void refetch()} /> : null}
        </div>

        <footer className="mt-6 flex items-center justify-between gap-3">
          <Button onClick={() => setStep((value) => Math.max(1, value - 1))} disabled={step === 1}>
            <ChevronLeft size={15} />
            {t("setup.back")}
          </Button>
          {step < STEPS ? (
            <Button variant="primary" onClick={() => setStep((value) => value + 1)}>
              {t("setup.next")}
              <ChevronRight size={15} />
            </Button>
          ) : (
            <Button variant="primary" onClick={finish}>
              <Check size={15} />
              {t("setup.finish")}
            </Button>
          )}
        </footer>
      </div>
    </div>
  );
}

function LanguageStep() {
  const { t } = useTranslation();
  const language = useUi((state) => state.language);
  const setLanguage = useUi((state) => state.setLanguage);

  return (
    <Section title={t("setup.langTitle")} description={t("setup.langHint")}>
      <div className="mb-3 flex items-center gap-2 text-muted">
        <Languages size={16} />
      </div>
      <RadioCards<Language>
        value={language}
        onChange={setLanguage}
        options={SUPPORTED_LANGUAGES.map((code) => ({
          value: code,
          label: code === "uk" ? "Українська" : "English",
        }))}
      />
    </Section>
  );
}

function StorageStep({ setup }: { setup: SetupStatus | undefined }) {
  const { t } = useTranslation();
  return (
    <Section title={t("setup.storageTitle")} description={t("setup.storageHint")}>
      <div className="flex items-start gap-2.5 rounded-lg border border-line bg-raised p-3">
        <FolderOpen size={16} className="mt-0.5 shrink-0 text-muted" />
        <div className="min-w-0">
          <p className="text-[0.6875rem] font-medium uppercase tracking-wide subtle">
            {t("setup.storagePath")}
          </p>
          <p className="mt-0.5 break-all font-mono text-[0.75rem]">{setup?.dataDir ?? "—"}</p>
        </div>
      </div>
      <p className="mt-3 text-[0.8125rem] leading-relaxed subtle">{t("setup.storageReadonly")}</p>
    </Section>
  );
}

/**
 * Файли моделей — чеклист із зеленим або червоним на ГРУПУ.
 *
 * Групи названі за призначенням, а не за іменами моделей: «Пошук за
 * змістом», а не «Qwen3-Embedding-0.6B». Викладачеві потрібно знати, ЩО
 * перестане працювати без цих файлів, а не як вони називаються.
 */
function ModelsStep({ setup }: { setup: SetupStatus | undefined }) {
  const { t } = useTranslation();
  const models = setup?.models;

  const groups = [
    {
      id: "search",
      title: t("setup.groupSearch"),
      why: t("setup.groupSearchWhy"),
      ok: models?.embeddings.present ?? false,
      detail: models?.embeddings.id,
      dir: models?.embeddings.dir,
    },
    {
      id: "rerank",
      title: t("setup.groupRerank"),
      why: t("setup.groupRerankWhy"),
      ok: models?.rerank.present ?? false,
      detail: models?.rerank.id,
      dir: models?.rerank.dir,
    },
    {
      id: "parsing",
      title: t("setup.groupParsing"),
      why: t("setup.groupParsingWhy"),
      ok: models?.docling.present ?? false,
      detail: undefined,
      dir: models?.docling.dir,
    },
  ];

  return (
    <Section title={t("setup.modelsTitle")} description={t("setup.modelsHint")}>
      {setup?.stub ? (
        <div className="mb-3">
          <Banner tone="accent" title="Застосунок працює в режимі заглушок">
            Пошук і відповіді формуються без справжніх моделей. Це нормально для перевірки
            інтерфейсу, але для роботи з матеріалами потрібні файли моделей.
          </Banner>
        </div>
      ) : null}

      <ul className="space-y-2">
        {groups.map((group) => (
          <li key={group.id} className="flex items-start gap-3 rounded-lg border border-line p-3">
            <span
              className={cn(
                "mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full",
                group.ok ? "bg-ok/15 text-ok" : "bg-danger/12 text-danger",
              )}
            >
              {group.ok ? <Check size={12} /> : <X size={12} />}
            </span>
            <div className="min-w-0 flex-1">
              <p className="text-[0.8125rem] font-medium">{group.title}</p>
              <p className="mt-0.5 text-[0.75rem] leading-snug subtle">{group.why}</p>
              {group.dir ? (
                <p className="mt-1 break-all font-mono text-[0.6875rem] subtle">{group.dir}</p>
              ) : null}
            </div>
            <Badge tone={group.ok ? "ok" : "danger"}>
              {group.ok ? t("setup.modelsPresent") : t("setup.modelsMissing")}
            </Badge>
          </li>
        ))}
      </ul>
    </Section>
  );
}

/**
 * Мовна модель. Чотири стани картки — рівно ті, у яких LM Studio реально
 * буває, і в кожного стану РІВНО ОДНА наступна дія.
 */
function LmStudioStep({ setup, onRefresh }: { setup: SetupStatus | undefined; onRefresh: () => void }) {
  const { t } = useTranslation();
  const [starting, setStarting] = useState(false);
  const [downloading, setDownloading] = useState<string | null>(null);
  const downloads = useLive((state) => state.downloads);

  const lm = setup?.lmstudio;
  const recommendation = setup?.recommendation;
  const hardware = setup?.hardware;
  const loaded = lm?.models.find((model) => model.loaded);

  const state: "not_installed" | "stopped" | "no_model" | "ready" = !lm
    ? "not_installed"
    : lm.ok && loaded
      ? "ready"
      : lm.ok
        ? "no_model"
        : lm.cliAvailable
          ? "stopped"
          : "not_installed";

  const progress = downloading ? Object.values(downloads).find((d) => d.model === downloading) : null;

  return (
    <Section title={t("setup.llmTitle")} description={t("setup.llmHint")}>
      <div
        className={cn(
          "rounded-xl border p-4",
          state === "ready" ? "border-ok/40 bg-ok/[0.07]" : "border-line bg-raised",
        )}
      >
        <div className="mb-3 flex items-center gap-2">
          <Cpu size={16} className={state === "ready" ? "text-ok" : "text-muted"} />
          <p className="text-[0.875rem] font-semibold">
            {state === "ready"
              ? t("setup.llmReady")
              : state === "no_model"
                ? t("setup.llmNoModel")
                : state === "stopped"
                  ? t("setup.llmStopped")
                  : t("setup.llmNotInstalled")}
          </p>
          {lm?.baseUrl ? (
            <span className="ml-auto font-mono text-[0.6875rem] subtle">{lm.baseUrl}</span>
          ) : null}
        </div>

        {state === "not_installed" ? (
          <p className="text-[0.8125rem] leading-relaxed subtle">{t("setup.llmNotInstalledHint")}</p>
        ) : null}

        {state === "stopped" ? (
          <Button
            variant="primary"
            loading={starting}
            onClick={() => {
              setStarting(true);
              void startLmStudio()
                .catch(() => undefined)
                .finally(() => {
                  setStarting(false);
                  onRefresh();
                });
            }}
          >
            <Play size={14} />
            {t("setup.llmStartServer")}
          </Button>
        ) : null}

        {state === "no_model" ? (
          <div className="space-y-3">
            {recommendation?.modelKey ? (
              <div className="rounded-lg border border-accent/35 bg-accent/[0.07] p-3">
                <div className="mb-1 flex flex-wrap items-center gap-2">
                  <Gauge size={14} className="text-accent" />
                  <span className="text-[0.75rem] font-semibold uppercase tracking-wide text-accent">
                    {t("setup.recommended")}
                  </span>
                </div>
                <p className="text-[0.875rem] font-medium">{recommendation.title}</p>
                <p className="mt-0.5 flex flex-wrap gap-x-3 text-[0.75rem] tabular-nums subtle">
                  {recommendation.quant ? <span>{recommendation.quant}</span> : null}
                  {recommendation.contextLength ? (
                    <span>контекст {recommendation.contextLength}</span>
                  ) : null}
                  {recommendation.fileBytes ? <span>{formatBytes(recommendation.fileBytes)}</span> : null}
                </p>
                {hardware?.vramBytes ? (
                  <p className="mt-1 text-[0.75rem] subtle">
                    {t("setup.recommendedWhy", { vram: formatBytes(hardware.vramBytes) })}
                  </p>
                ) : null}
                {recommendation.archVerified === false ? (
                  <p className="mt-1.5 text-[0.75rem] text-warn">{t("setup.archUnverified")}</p>
                ) : null}
                <Button
                  variant="primary"
                  size="sm"
                  className="mt-2.5"
                  disabled={Boolean(downloading)}
                  onClick={() => {
                    const key = recommendation.modelKey as string;
                    setDownloading(key);
                    void downloadModel(key).catch(() => setDownloading(null));
                  }}
                >
                  <Download size={13} />
                  {t("setup.llmLoadModel")}
                </Button>
                {progress ? (
                  <div className="mt-2.5">
                    <Progress value={(progress.percent ?? 0) / 100} />
                    <p className="mt-1 text-[0.6875rem] tabular-nums subtle">
                      {Math.round(progress.percent ?? 0)} % · {progress.state}
                    </p>
                  </div>
                ) : null}
              </div>
            ) : null}

            {lm?.models.length ? (
              <div>
                <p className="mb-1 text-[0.6875rem] font-semibold uppercase tracking-wide subtle">
                  Уже на цьому комп'ютері
                </p>
                <ul className="space-y-1">
                  {lm.models.slice(0, 6).map((model) => (
                    <li
                      key={model.key}
                      className="flex flex-wrap items-center gap-2 rounded-lg border border-line px-2.5 py-1.5 text-[0.75rem]"
                    >
                      <Package size={13} className="shrink-0 text-muted" />
                      <span className="min-w-0 flex-1 truncate font-mono">{model.key}</span>
                      {model.quantization ? <Badge tone="neutral">{model.quantization}</Badge> : null}
                      {model.sizeBytes ? (
                        <span className="tabular-nums subtle">{formatBytes(model.sizeBytes)}</span>
                      ) : null}
                    </li>
                  ))}
                </ul>
              </div>
            ) : null}
          </div>
        ) : null}

        {state === "ready" && loaded ? (
          <div className="space-y-2">
            <p className="font-mono text-[0.8125rem]">{loaded.key}</p>
            <p className="flex flex-wrap gap-x-3 text-[0.75rem] tabular-nums subtle">
              {loaded.engine ? <span>{loaded.engine}</span> : null}
              {loaded.loadedContext ? <span>контекст {loaded.loadedContext}</span> : null}
              {loaded.quantization ? <span>{loaded.quantization}</span> : null}
            </p>
            <LmStudioProbe />
          </div>
        ) : null}
      </div>

      {setup?.problems?.length ? (
        <ul className="mt-3 space-y-2">
          {setup.problems.map((problem) => (
            <li key={problem.code}>
              <Banner tone="warn" title={problem.message} icon={<CircleAlert size={15} />}>
                {problem.hint}
              </Banner>
            </li>
          ))}
        </ul>
      ) : null}
    </Section>
  );
}

/**
 * «Перевірити» — одна кнопка, що економить десяток звернень у підтримку.
 *
 * Виміряти TTFT можна лише реальним ходом генерації, а окремої ручки для
 * цього бекенд не має. Тому перевірка робиться найдешевшим доступним
 * способом: повторним опитуванням `/setup/status`, яке чесно повідомляє
 * стан з'єднання. Число токенів за секунду з'явиться при першому справжньому
 * питанні — і воно показується під кожною відповіддю.
 */
function LmStudioProbe() {
  const { t } = useTranslation();
  const [checking, setChecking] = useState(false);
  const [result, setResult] = useState<string | null>(null);
  const { refetch } = useSetupStatus();

  return (
    <div className="flex flex-wrap items-center gap-2">
      <Button
        size="sm"
        loading={checking}
        onClick={() => {
          setChecking(true);
          const started = performance.now();
          void refetch()
            .then((response) => {
              const elapsed = Math.round(performance.now() - started);
              setResult(
                response.data?.lmstudio.ok
                  ? `Зв'язок є, відповідь за ${elapsed} мс`
                  : "Сервер не відповідає",
              );
            })
            .finally(() => setChecking(false));
        }}
      >
        {checking ? t("setup.llmTesting") : t("setup.llmTest")}
      </Button>
      {result ? <span className="text-[0.75rem] tabular-nums subtle">{result}</span> : null}
      <span className="text-[0.75rem] subtle">
        Швидкість генерації показується під кожною відповіддю.
      </span>
    </div>
  );
}
