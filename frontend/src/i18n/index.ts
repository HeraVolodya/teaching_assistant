/**
 * Ініціалізація i18next.
 *
 * Українська — мова за замовчуванням і `fallbackLng`. Це не декларація:
 * якщо ключ загубиться в англійському словнику, викладач побачить
 * український текст, а не порожнє місце або сирий ключ на кшталт
 * `chat.abstainTitle`. Зворотний порядок (fallback на англійську) у застосунку
 * для української військової академії був би гіршою з двох поведінок.
 *
 * ЩО ЛОКАЛІЗОВАНО, А ЩО НІ. Через `t()` проходять підписи інтерфейсу.
 * Довгі пояснювальні тексти, прив'язані до предметної області, — пояснення
 * помилок обробки (`lib/errors.ts`) і готові формулювання інструкцій
 * (`lib/prompt.ts`) — лишаються українськими: вони є частиною ЗМІСТУ
 * продукту, а не його оболонки, і їхній переклад має сенс лише разом із
 * перекладом самої методики.
 */

import i18n from "i18next";
import { initReactI18next } from "react-i18next";

import { en } from "./en";
import { uk } from "./uk";

export const SUPPORTED_LANGUAGES = ["uk", "en"] as const;
export type Language = (typeof SUPPORTED_LANGUAGES)[number];

const STORAGE_KEY = "asistent.language";

export function storedLanguage(): Language {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw && (SUPPORTED_LANGUAGES as readonly string[]).includes(raw)) return raw as Language;
  } catch {
    /* приватний режим браузера — читати нічого */
  }
  return "uk";
}

export function setLanguage(language: Language): void {
  void i18n.changeLanguage(language);
  document.documentElement.lang = language;
  try {
    localStorage.setItem(STORAGE_KEY, language);
  } catch {
    /* не критично: мова просто не переживе перезапуск */
  }
}

export function initI18n(): typeof i18n {
  if (i18n.isInitialized) return i18n;
  void i18n.use(initReactI18next).init({
    resources: {
      uk: { translation: uk },
      en: { translation: en },
    },
    lng: storedLanguage(),
    fallbackLng: "uk",
    // Інтерполяція {{…}} у нас лише з чисел і вже безпечних рядків, а React
    // і так екранує вивід. Подвійне екранування ламало б лапки в назвах.
    interpolation: { escapeValue: false },
    returnNull: false,
  });
  document.documentElement.lang = storedLanguage();
  return i18n;
}

export default i18n;
