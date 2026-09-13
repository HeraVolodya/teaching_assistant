/**
 * Форматування чисел і часу українською.
 *
 * Окремий файл, бо українська відміна — це не косметика: «3 документа»
 * замість «3 документи» читається як недбалість рівно там, де застосунок
 * має викликати довіру (картка асистента, банер черги). `Intl.PluralRules`
 * тут не використовується свідомо: він дає категорію, але не форму, і все
 * одно довелося б тримати таблицю форм.
 */

export type PluralForms = [one: string, few: string, many: string];

/**
 * Українська відміна: 1 → одна форма, 2–4 → друга, решта → третя.
 * Виняток 11–14 обов'язковий: без нього виходить «11 документ».
 */
export function plural(n: number, forms: PluralForms): string {
  const abs = Math.abs(Math.trunc(n));
  const tens = abs % 100;
  if (tens >= 11 && tens <= 14) return forms[2];
  const ones = abs % 10;
  if (ones === 1) return forms[0];
  if (ones >= 2 && ones <= 4) return forms[1];
  return forms[2];
}

export function pluralize(n: number, forms: PluralForms): string {
  return `${formatNumber(n)} ${plural(n, forms)}`;
}

export const DOCUMENTS: PluralForms = ["документ", "документи", "документів"];
export const PAGES: PluralForms = ["сторінка", "сторінки", "сторінок"];
export const FRAGMENTS: PluralForms = ["фрагмент", "фрагменти", "фрагментів"];
export const TABLES: PluralForms = ["таблицю", "таблиці", "таблиць"];
export const FORMULAS: PluralForms = ["формулу", "формули", "формул"];
export const SOURCES: PluralForms = ["джерело", "джерела", "джерел"];
export const MINUTES: PluralForms = ["хвилина", "хвилини", "хвилин"];

/** Нерозривний пробіл між розрядами: «3 412», а не «3,412». */
export function formatNumber(n: number): string {
  return new Intl.NumberFormat("uk-UA").format(Math.round(n));
}

export function formatBytes(bytes: number | null | undefined): string {
  if (bytes == null || !Number.isFinite(bytes)) return "—";
  if (bytes < 1024) return `${Math.round(bytes)} Б`;
  const units = ["КБ", "МБ", "ГБ", "ТБ"];
  let value = bytes / 1024;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  // Один знак після коми до сотні, далі — жодного: «1,4 ГБ» інформативно,
  // «482,3 МБ» — це три зайві символи в тісній колонці таблиці.
  const digits = value >= 100 ? 0 : 1;
  return `${value.toFixed(digits).replace(".", ",")} ${units[unit]}`;
}

/**
 * Тривалість для ETA. Округлення НАГОРУ до одиниці, що показується:
  * «залишилось ~0 хв» під час двадцятихвилинної індексації — це втрата
 * довіри, яку потім не повернути жодною точністю.
 */
export function formatDuration(seconds: number | null | undefined): string {
  if (seconds == null || !Number.isFinite(seconds) || seconds < 0) return "—";
  if (seconds < 60) return `${Math.max(1, Math.round(seconds))} с`;
  const minutes = seconds / 60;
  if (minutes < 60) return `${Math.max(1, Math.ceil(minutes))} хв`;
  const hours = Math.floor(minutes / 60);
  const rest = Math.round(minutes % 60);
  return rest ? `${hours} год ${rest} хв` : `${hours} год`;
}

/** «щойно» / «12 хв тому» / «вчора» / дата. */
export function formatRelative(iso: string | null | undefined, now = Date.now()): string {
  if (!iso) return "—";
  const at = parseServerDate(iso);
  if (at == null) return "—";
  const diff = Math.max(0, now - at) / 1000;
  if (diff < 90) return "щойно";
  if (diff < 3600) return `${Math.round(diff / 60)} хв тому`;
  if (diff < 22 * 3600) return `${Math.round(diff / 3600)} год тому`;
  if (diff < 48 * 3600) return "учора";
  return new Intl.DateTimeFormat("uk-UA", { day: "numeric", month: "long" }).format(new Date(at));
}

/**
 * Дати з бекенда — `utcnow()`, тобто «YYYY-MM-DD HH:MM:SS» БЕЗ зони.
 * `new Date("2026-09-06 12:00:00")` браузери тлумачать як МІСЦЕВИЙ час, і
 * різниця з UTC перетворює «щойно» на «3 год тому» — тому зона додається
 * явно, а не залишається на розсуд рушія.
 */
export function parseServerDate(raw: string): number | null {
  if (!raw) return null;
  const normalized = /\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}$/.test(raw.trim())
    ? `${raw.trim().replace(" ", "T")}Z`
    : raw;
  const at = Date.parse(normalized);
  return Number.isFinite(at) ? at : null;
}

export function formatPercent(fraction: number | null | undefined): string {
  if (fraction == null || !Number.isFinite(fraction)) return "—";
  return `${Math.round(Math.max(0, Math.min(1, fraction)) * 100)} %`;
}

/** Скор реранкера в UI: три знаки — це шум, два — рівно те, що видно оком. */
export function formatScore(score: number | null | undefined): string {
  if (score == null || !Number.isFinite(score)) return "—";
  return score.toFixed(2).replace(".", ",");
}
