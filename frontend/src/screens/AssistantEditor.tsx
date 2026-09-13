/**
 * Редактор асистента — тут живе відмінність продукту.
 *
 * ВИКЛАДАЧЕВІ НЕ ДАЮТЬ СИРИЙ ПРОМПТ ЯК ОСНОВНИЙ ІНТЕРФЕЙС. Він не мусить
 * знати, що таке промпт, і тим паче не мусить вгадувати, які формулювання
 * локальна 12-мільярдна модель виконує надійно. Він отримує структуровані
 * контроли, а вони компілюються у дві різні речі:
 *
 *   • у ТЕКСТ інструкцій (`config.instructions`) — те, що піде в промпт;
 *   • у ПОЛІТИКУ ПОШУКУ (`confidence_required`, `final_top_k`,
 *     `max_per_document`) — те, що виконується в КОДІ.
 *
 * Друге важливіше за перше. Промптом неможливо надійно змусити модель
 * відмовитись відповідати; порогом скора реранкера — можна. Тому слайдер
 * «потрібна впевненість» показує поруч числа, у які він перетворюється:
 * інакше «висока» читається як ввічливе прохання, а не як 0.55.
 */

import { ChevronDown, Code2, Plus, Save } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";

import type { Assistant, AssistantConfig } from "@/api/types";
import { Button } from "@/components/ui/Button";
import {
  Badge,
  Banner,
  ChipInput,
  Field,
  Input,
  RadioCards,
  Range,
  Section,
  Textarea,
  Toggle,
} from "@/components/ui/Primitives";
import { usePromptPreview, useUpdateAssistant } from "@/hooks/queries";
import { cn } from "@/lib/cn";
import {
  PERSONA_MAX_TOKENS,
  PROMPT_BLOCKS,
  appendBlock,
  compileInstructions,
  compilePromptPreview,
  estimateTokensUk,
  splitInstructions,
} from "@/lib/prompt";
import { useUi } from "@/lib/store";

const CONFIDENCE_ORDER: AssistantConfig["confidence_required"][] = ["low", "medium", "high"];

export function AssistantEditor({ assistant }: { assistant: Assistant }) {
  const { t } = useTranslation();
  const update = useUpdateAssistant();
  const developerMode = useUi((state) => state.developerMode);
  const { data: serverPreview } = usePromptPreview(assistant.id);

  const [name, setName] = useState(assistant.name);
  const [description, setDescription] = useState(assistant.description);
  const [freeText, setFreeText] = useState(() => splitInstructions(assistant.config.instructions).free);
  const [config, setConfig] = useState<AssistantConfig>(assistant.config);
  const [rawMode, setRawMode] = useState(false);
  const [rawText, setRawText] = useState(assistant.config.instructions);
  const [advanced, setAdvanced] = useState(false);
  const [dirty, setDirty] = useState(false);

  // Перезавантаження ззовні (інше вікно, збереження з іншої вкладки) не має
  // затирати незбережені правки: це найдорожча можлива несподіванка для
  // того, хто півгодини складав інструкції.
  useEffect(() => {
    if (dirty) return;
    setName(assistant.name);
    setDescription(assistant.description);
    setFreeText(splitInstructions(assistant.config.instructions).free);
    setRawText(assistant.config.instructions);
    setConfig(assistant.config);
  }, [assistant, dirty]);

  const patch = (next: Partial<AssistantConfig>) => {
    setConfig((current) => ({ ...current, ...next }));
    setDirty(true);
  };

  const compiledInstructions = useMemo(
    () =>
      rawMode
        ? rawText
        : compileInstructions(freeText, {
            topicsCovered: config.topics_covered,
            topicsRefused: config.topics_refused,
            onMissing: config.on_missing_information,
            answerLanguage: config.answer_language,
            alwaysCitePages: config.always_cite_pages,
          }),
    [rawMode, rawText, freeText, config],
  );

  const localPreview = useMemo(
    () => compilePromptPreview(compiledInstructions, config),
    [compiledInstructions, config],
  );

  // Прев'ю з сервера — джерело правди, але воно відстає від правок на один
  // збережений стан. Доки є незбережені зміни, показуємо локальне: інакше
  // викладач змінює контроль і не бачить жодної реакції.
  const preview = dirty || !serverPreview ? localPreview : { ...localPreview, system: serverPreview.system };
  const personaTokens = estimateTokensUk(compiledInstructions);

  const save = () => {
    update.mutate(
      {
        id: assistant.id,
        body: {
          name: name.trim() || assistant.name,
          description,
          colour: assistant.colour,
          emoji: assistant.emoji,
          config: { ...config, instructions: compiledInstructions },
        },
      },
      { onSuccess: () => setDirty(false) },
    );
  };

  const confidenceIndex = CONFIDENCE_ORDER.indexOf(config.confidence_required);

  return (
    <div className="scroll-thin min-h-0 flex-1 overflow-y-auto px-4 py-5 sm:px-6">
      <div className="mx-auto max-w-5xl space-y-4 pb-20">
        <Section title={t("assistants.nameLabel")}>
          <div className="grid gap-3 sm:grid-cols-2">
            <Field label={t("assistants.nameLabel")} htmlFor="editor-name">
              <Input
                id="editor-name"
                value={name}
                onChange={(event) => {
                  setName(event.target.value);
                  setDirty(true);
                }}
              />
            </Field>
            <Field label={t("assistants.descriptionLabel")} htmlFor="editor-description">
              <Input
                id="editor-description"
                value={description}
                onChange={(event) => {
                  setDescription(event.target.value);
                  setDirty(true);
                }}
              />
            </Field>
          </div>
        </Section>

        <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_20rem]">
          <div className="space-y-4">
            <Section
              title={t("editor.instructions")}
              description={rawMode ? t("editor.rawWarning") : t("editor.instructionsHint")}
              action={
                developerMode ? (
                  <Button
                    size="sm"
                    variant={rawMode ? "primary" : "secondary"}
                    onClick={() => {
                      if (!rawMode) setRawText(compiledInstructions);
                      setRawMode((value) => !value);
                      setDirty(true);
                    }}
                  >
                    <Code2 size={13} />
                    {t("editor.rawToggle")}
                  </Button>
                ) : null
              }
            >
              <Textarea
                rows={9}
                value={rawMode ? rawText : freeText}
                onChange={(event) => {
                  if (rawMode) setRawText(event.target.value);
                  else setFreeText(event.target.value);
                  setDirty(true);
                }}
                placeholder="Опишіть роль асистента одним-двома абзацами…"
              />
              <div className="mt-2 flex flex-wrap items-center gap-2 text-[0.75rem]">
                <Badge tone={personaTokens > PERSONA_MAX_TOKENS ? "danger" : "neutral"}>
                  {t("editor.tokensUsed", { used: personaTokens, limit: PERSONA_MAX_TOKENS })}
                </Badge>
                {personaTokens > PERSONA_MAX_TOKENS ? (
                  <span className="text-danger">
                    {t("editor.tokensOver", { limit: PERSONA_MAX_TOKENS })}
                  </span>
                ) : null}
              </div>
            </Section>

            <Section title={t("editor.scope")}>
              <div className="space-y-3">
                <Field label={t("editor.topicsCovered")}>
                  <ChipInput
                    values={config.topics_covered}
                    onChange={(next) => patch({ topics_covered: next })}
                    placeholder={t("editor.chipPlaceholder")}
                    tone="accent"
                  />
                </Field>
                <Field label={t("editor.topicsRefused")}>
                  <ChipInput
                    values={config.topics_refused}
                    onChange={(next) => patch({ topics_refused: next })}
                    placeholder={t("editor.chipPlaceholder")}
                    tone="danger"
                  />
                </Field>
              </div>
            </Section>

            <Section title={t("editor.missing")}>
              <RadioCards
                value={config.on_missing_information}
                onChange={(next) => patch({ on_missing_information: next })}
                options={[
                  {
                    value: "no_information",
                    label: t("editor.missingNoInformation"),
                    hint: "Найбезпечніший варіант: асистент прямо каже, що в матеріалах цього немає.",
                  },
                  {
                    value: "general_knowledge_marked",
                    label: t("editor.missingGeneral"),
                    hint: "Відповідь без посилань буде позначена окремо. Перевіряти її доведеться вам.",
                  },
                  { value: "refuse", label: t("editor.missingRefuse") },
                ]}
              />
            </Section>

            <Section title={t("editor.confidence")}>
              <Range
                value={confidenceIndex}
                min={0}
                max={2}
                onChange={(index) => patch({ confidence_required: CONFIDENCE_ORDER[index] })}
                labels={[t("editor.confidenceLow"), t("editor.confidenceMedium"), t("editor.confidenceHigh")]}
              />
              <p className="mt-3 text-[0.8125rem] leading-relaxed">{preview.policy.explain}</p>
              <p className="mt-2 rounded-lg bg-raised px-2.5 py-2 text-[0.75rem] tabular-nums subtle">
                {t("editor.confidenceExecuted", {
                  threshold: preview.policy.threshold.toFixed(2),
                  count: preview.policy.minSupporting,
                })}
              </p>
            </Section>

            <Section title={t("editor.language")}>
              <div className="space-y-3">
                <RadioCards
                  value={config.answer_language}
                  onChange={(next) => patch({ answer_language: next })}
                  options={[
                    { value: "match_question", label: t("editor.languageMatch") },
                    { value: "always_uk", label: t("editor.languageUk") },
                  ]}
                />
                <Toggle
                  checked={config.always_cite_pages}
                  onChange={(next) => patch({ always_cite_pages: next })}
                  label={t("editor.citePages")}
                />
              </div>
            </Section>

            <section className="card">
              <button
                type="button"
                onClick={() => setAdvanced((value) => !value)}
                className="flex w-full items-center justify-between gap-2 p-4 text-left"
              >
                <span>
                  <span className="block text-[0.9375rem] font-semibold">{t("editor.advanced")}</span>
                  <span className="mt-0.5 block text-[0.8125rem] subtle">{t("editor.advancedHint")}</span>
                </span>
                <ChevronDown size={16} className={cn("shrink-0 transition", advanced && "rotate-180")} />
              </button>
              {advanced ? (
                <div className="grid gap-3 border-t border-line p-4 sm:grid-cols-2">
                  <NumberField
                    label={t("editor.finalTopK")}
                    value={config.final_top_k}
                    min={1}
                    max={20}
                    onChange={(value) => patch({ final_top_k: value })}
                  />
                  <NumberField
                    label={t("editor.maxPerDocument")}
                    value={config.max_per_document}
                    min={1}
                    max={10}
                    onChange={(value) => patch({ max_per_document: value })}
                    hint="Менше значення змушує відповідь спиратися на більше різних матеріалів."
                  />
                  <NumberField
                    label={t("editor.rerankTopK")}
                    value={config.rerank_top_k}
                    min={5}
                    max={200}
                    onChange={(value) => patch({ rerank_top_k: value })}
                  />
                  <NumberField
                    label={t("editor.maxTokens")}
                    value={config.max_tokens}
                    min={100}
                    max={4000}
                    step={50}
                    onChange={(value) => patch({ max_tokens: value })}
                  />
                  <Field
                    label={t("editor.temperature")}
                    hint="Вище 0.4 локальні моделі починають вигадувати числа. Типове значення — 0,2."
                  >
                    <Range
                      value={Math.round(config.temperature * 10)}
                      min={0}
                      max={10}
                      onChange={(value) => patch({ temperature: value / 10 })}
                    />
                    <p className="mt-1 text-[0.75rem] tabular-nums subtle">
                      {config.temperature.toFixed(1)}
                    </p>
                  </Field>
                </div>
              ) : null}
            </section>
          </div>

          <aside className="space-y-4">
            <Section title={t("editor.blocks")}>
              <div className="space-y-3">
                {PROMPT_BLOCKS.map((group) => (
                  <div key={group.id}>
                    <p className="mb-1 text-[0.6875rem] font-semibold uppercase tracking-wide subtle">
                      {group.title}
                    </p>
                    <div className="flex flex-wrap gap-1">
                      {group.blocks.map((block) => (
                        <button
                          key={block.id}
                          type="button"
                          title={block.text}
                          disabled={rawMode}
                          onClick={() => {
                            setFreeText((current) => appendBlock(current, block));
                            setDirty(true);
                          }}
                          className={cn(
                            "inline-flex items-center gap-1 rounded-lg border border-line bg-raised",
                            "px-2 py-1 text-[0.75rem] transition hover:border-accent/50 hover:bg-accent/[0.07]",
                            "disabled:pointer-events-none disabled:opacity-40",
                          )}
                        >
                          <Plus size={11} />
                          {block.label}
                        </button>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </Section>

            <Section title={t("editor.preview")} description={t("editor.previewHint")}>
              <pre className="scroll-thin max-h-80 overflow-auto whitespace-pre-wrap break-words rounded-lg bg-raised p-2.5 text-[0.75rem] leading-relaxed">
                {preview.system}
              </pre>
              {serverPreview ? (
                <p className="mt-2 text-[0.6875rem] tabular-nums subtle">
                  Правила: {serverPreview.rulesTokens} токенів · фрагментів у відповіді:{" "}
                  {serverPreview.finalTopK} · поріг: {serverPreview.confidenceThreshold.toFixed(2)}
                </p>
              ) : null}
            </Section>
          </aside>
        </div>
      </div>

      {dirty ? (
        <div className="pointer-events-none sticky bottom-3 z-10 flex justify-center">
          <div className="pointer-events-auto flex items-center gap-3 rounded-xl border border-line bg-surface px-3 py-2 shadow-lg">
            <span className="text-[0.8125rem] subtle">Є незбережені зміни</span>
            <Button variant="primary" onClick={save} loading={update.isPending}>
              <Save size={14} />
              {t("common.save")}
            </Button>
          </div>
        </div>
      ) : null}

      {update.isError ? (
        <div className="mx-auto mt-3 max-w-5xl">
          <Banner tone="danger" title={t("common.error")}>
            {(update.error as Error).message}
          </Banner>
        </div>
      ) : null}
    </div>
  );
}

function NumberField({
  label,
  value,
  min,
  max,
  step = 1,
  hint,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  hint?: string;
  onChange: (value: number) => void;
}) {
  return (
    <Field label={label} hint={hint}>
      <Input
        type="number"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(event) => {
          const next = Number.parseFloat(event.target.value);
          if (Number.isFinite(next)) onChange(Math.min(max, Math.max(min, next)));
        }}
      />
    </Field>
  );
}
