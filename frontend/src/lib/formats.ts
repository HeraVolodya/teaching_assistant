/**
 * Які файли приймає база знань.
 *
 * Дзеркало `backend/app/ingestion/formats.py` — тест бекенда
 * (`test_ingestion_formats_parity`) звіряє обидва списки, бо розходження тут
 * непомітне: файл, якого немає в цьому списку, екран мовчки відкидав би ще до
 * відправлення, і викладач бачив би лише, що «нічого не сталося».
 */

export const ACCEPTED_EXTENSIONS = [
  // PDF і текст — читаються як є
  ".pdf",
  ".txt",
  ".text",
  ".md",
  ".markdown",
  // Word — через PDF-копію (справжні сторінки)
  ".doc",
  ".docx",
  ".docm",
  ".dot",
  ".dotx",
  ".rtf",
  ".odt",
  // PowerPoint — через PDF-копію (слайди)
  ".ppt",
  ".pptx",
  ".pptm",
  ".pps",
  ".ppsx",
  ".odp",
  // DjVu — через PDF-копію й розпізнавання
  ".djvu",
  ".djv",
  // Таблиці — аркуш за аркушем
  ".xlsx",
  ".xlsm",
  ".xls",
  ".csv",
  ".ods",
  // XML — дані або текст
  ".xml",
] as const;

/** Значення атрибута `accept` поля вибору файлів. */
export const ACCEPT_ATTRIBUTE = ACCEPTED_EXTENSIONS.join(",");

const ACCEPTED = new Set<string>(ACCEPTED_EXTENSIONS);

export function extensionOf(name: string): string {
  const dot = name.lastIndexOf(".");
  return dot > 0 ? name.slice(dot).toLowerCase() : "";
}

export function isAcceptedFile(name: string): boolean {
  return ACCEPTED.has(extensionOf(name));
}

/** Розділити вибрані файли: що відправити, а про що сказати викладачеві. */
export function splitAccepted<T extends { name: string }>(files: Iterable<T>): {
  accepted: T[];
  rejected: string[];
} {
  const accepted: T[] = [];
  const rejected: string[] = [];
  for (const file of files) {
    if (isAcceptedFile(file.name)) accepted.push(file);
    else rejected.push(file.name);
  }
  return { accepted, rejected };
}

const SHEETS = new Set([".xlsx", ".xlsm", ".xls", ".csv", ".ods"]);
const SLIDES = new Set([".ppt", ".pptx", ".pptm", ".pps", ".ppsx", ".odp"]);

/** Одиниця «сторінки» в рядку матеріалу: у таблиці це аркуші, у презентації — слайди. */
export function pageUnit(originalName: string): string {
  const extension = extensionOf(originalName);
  if (SHEETS.has(extension)) return "арк.";
  if (SLIDES.has(extension)) return "слайд.";
  return "с.";
}
