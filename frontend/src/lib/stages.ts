/**
 * Етап обробки → рядок, який бачить викладач.
 *
 * ЖОДНОГО ТЕХНІЧНОГО ТЕРМІНА, ЯКОГО ВІН НЕ ЗНАВ РАНІШЕ. «PARSE» — це не
 * «парсинг», а «Читаю сторінки»; «CHUNK» — не «чанкування», а «Ділю на
 * фрагменти»; «INDEX» — не «побудова векторного індексу», а «Готую пошук».
 *
 * І ЗАВЖДИ З ЧИСЛОМ. «Обробка…» без числа читається як зависання вже на
 * тридцятій секунді, а індексація підручника на 512 сторінок триває хвилини.
 * Тому кожен рядок, коли є з чого, отримує «— сторінка 143 з 512».
 */

import type { DocStatus, JobStage } from "@/api/types";

export interface StageText {
  /** Коротка назва етапу — те, що стоїть у колонці статусу. */
  label: string;
  /** Повний рядок із числами — те, що стоїть у колонці «що зараз робиться». */
  detail: string;
  /** Індексація активна: рядок треба анімувати, а не малювати статично. */
  active: boolean;
}

const STAGE_LABELS: Record<string, string> = {
  PROBE: "Оцінюю документ",
  PARSE: "Читаю сторінки",
  ENRICH_FIGURES: "Описую рисунки",
  VLM_REPAIR: "Ремонтую складні сторінки",
  CHUNK: "Ділю на фрагменти",
  INDEX: "Готую пошук",
  READY: "Готово",
};

const STATUS_LABELS: Record<DocStatus, string> = {
  QUEUED: "У черзі",
  PROBING: "Оцінюю документ",
  PARSING: "Читаю сторінки",
  ENRICHING: "Описую рисунки",
  CHUNKING: "Ділю на фрагменти",
  INDEXING: "Готую пошук",
  READY: "Готово",
  FAILED: "Не вдалося",
  CANCELLED: "Скасовано",
};

const ACTIVE_STATUSES = new Set<DocStatus>([
  "PROBING",
  "PARSING",
  "ENRICHING",
  "CHUNKING",
  "INDEXING",
]);

export function isActiveStatus(status: DocStatus): boolean {
  return ACTIVE_STATUSES.has(status);
}

export function isPendingStatus(status: DocStatus): boolean {
  return status === "QUEUED" || ACTIVE_STATUSES.has(status);
}

export function stageLabel(stage: JobStage | string | null | undefined): string {
  if (!stage) return "";
  return STAGE_LABELS[stage] ?? String(stage);
}

/**
 * Зібрати рядок етапу.
 *
 * Номер сторінки виводиться з ваги прогресу, а не з окремого лічильника:
 * подія `job.progress` несе `current`/`total` В ОДИНИЦЯХ ВАГИ (сторінка зі
 * сканом «важить» більше за цифрову), і другого числа в ній немає. Тому
 * сторінка обчислюється як частка від відомої кількості сторінок документа
 * — приблизно, але монотонно, а це саме те, що потрібно для смуги прогресу.
 * Коли кількість сторінок ще невідома (перед PROBE), число не вигадується.
 */
export function describeStage(input: {
  status: DocStatus;
  stage?: JobStage | string | null;
  fraction?: number | null;
  pageCount?: number | null;
}): StageText {
  const { status, stage, fraction, pageCount } = input;
  const active = ACTIVE_STATUSES.has(status);
  const label = stage && active ? stageLabel(stage) : STATUS_LABELS[status];

  if (status === "QUEUED") {
    return { label, detail: "Очікує черги — обробка почнеться автоматично", active: false };
  }
  if (!active) return { label, detail: "", active: false };

  const pages = pageCount ?? 0;
  const done = fraction == null ? null : Math.max(0, Math.min(1, fraction));
  if (pages > 0 && done != null && (stage === "PARSE" || stage === "VLM_REPAIR" || !stage)) {
    const page = Math.min(pages, Math.max(1, Math.round(done * pages)));
    return { label, detail: `${label} — сторінка ${page} з ${pages}`, active: true };
  }
  if (done != null) {
    return { label, detail: `${label} — ${Math.round(done * 100)} %`, active: true };
  }
  return { label, detail: label, active: true };
}

/** Дві фази чату до перших токенів. Див. `ChatView` — там пояснено чому. */
export type ChatPhase = "idle" | "retrieving" | "generating" | "streaming" | "done" | "error";

export const CHAT_PHASE_TEXT: Record<Exclude<ChatPhase, "idle" | "done" | "error">, string> = {
  retrieving: "Шукаю в матеріалах…",
  generating: "Формую відповідь…",
  streaming: "",
};
