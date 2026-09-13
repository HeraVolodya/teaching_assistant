/**
 * Локальний стан інтерфейсу.
 *
 * МЕЖА ЖОРСТКА: тут живе лише те, що НЕ належить серверу — тема, масштаб,
 * режим розробника, відкрита панель джерела. Усе, що приходить з API,
 * тримає TanStack Query. Змішувати їх означало б мати дві копії списку
 * документів, які розходяться рівно тоді, коли прийшла подія SSE.
 *
 * Налаштування вигляду зберігаються в localStorage, а не на сервері,
 * свідомо: це властивість робочого місця, а не бази знань, і вона мусить
 * пережити навіть той випадок, коли sidecar не піднявся взагалі.
 */

import { create } from "zustand";

import type { BBoxDto } from "@/api/types";
import { setLanguage, storedLanguage, type Language } from "@/i18n";

export type Theme = "system" | "light" | "dark";

/** Джерело, відкрите в бічній панелі. */
export interface SourceTarget {
  documentId: string;
  documentTitle: string;
  page: number;
  bboxes: BBoxDto[];
  quote?: string;
}

interface UiState {
  theme: Theme;
  /** Множник до 16px. Системний DPI множиться поверх нього браузером. */
  scale: number;
  language: Language;
  developerMode: boolean;
  /** Ширина панелі джерела у відсотках вікна. */
  sourceWidth: number;
  source: SourceTarget | null;
  /** Скільки разів натиснуто на версію — 5 вмикає інструменти розробника. */
  versionClicks: number;

  setTheme: (theme: Theme) => void;
  setScale: (scale: number) => void;
  setLanguage: (language: Language) => void;
  setDeveloperMode: (on: boolean) => void;
  setSourceWidth: (percent: number) => void;
  openSource: (target: SourceTarget) => void;
  closeSource: () => void;
  bumpVersionClicks: () => boolean;
}

const KEYS = {
  theme: "asistent.theme",
  scale: "asistent.scale",
  developer: "asistent.developer",
  sourceWidth: "asistent.sourceWidth",
} as const;

function read<T>(key: string, fallback: T, parse: (raw: string) => T): T {
  try {
    const raw = localStorage.getItem(key);
    return raw == null ? fallback : parse(raw);
  } catch {
    return fallback;
  }
}

function write(key: string, value: string): void {
  try {
    localStorage.setItem(key, value);
  } catch {
    /* приватний режим — налаштування просто не переживе перезапуск */
  }
}

/**
 * Тема застосовується атрибутом на <html>, а не класом на кореневому
 * компоненті: інакше вона не діє на портали Radix (діалоги, поповери), які
 * рендеряться в `document.body` поза деревом застосунку.
 */
export function applyTheme(theme: Theme): void {
  const root = document.documentElement;
  const dark =
    theme === "dark" ||
    (theme === "system" && window.matchMedia("(prefers-color-scheme: dark)").matches);
  root.dataset.theme = dark ? "dark" : "light";
}

export function applyScale(scale: number): void {
  document.documentElement.style.setProperty("--ui-scale", String(scale));
}

export const useUi = create<UiState>((set, get) => ({
  theme: read<Theme>(KEYS.theme, "system", (raw) => raw as Theme),
  scale: read(KEYS.scale, 1, (raw) => {
    const value = Number.parseFloat(raw);
    // Межі не косметичні: нижче 0.85 зникають підписи в таблиці документів,
    // вище 1.35 тризначна панель джерела перестає вміщатися поруч із чатом
    // на 1366×768 — типовій роздільності службового ноутбука.
    return Number.isFinite(value) ? Math.min(1.35, Math.max(0.85, value)) : 1;
  }),
  language: storedLanguage(),
  developerMode: read(KEYS.developer, false, (raw) => raw === "1"),
  sourceWidth: read(KEYS.sourceWidth, 46, (raw) => {
    const value = Number.parseFloat(raw);
    return Number.isFinite(value) ? Math.min(60, Math.max(30, value)) : 46;
  }),
  source: null,
  versionClicks: 0,

  setTheme(theme) {
    write(KEYS.theme, theme);
    applyTheme(theme);
    set({ theme });
  },
  setScale(scale) {
    const clamped = Math.min(1.35, Math.max(0.85, scale));
    write(KEYS.scale, String(clamped));
    applyScale(clamped);
    set({ scale: clamped });
  },
  setLanguage(language) {
    setLanguage(language);
    set({ language });
  },
  setDeveloperMode(on) {
    write(KEYS.developer, on ? "1" : "0");
    set({ developerMode: on, versionClicks: 0 });
  },
  setSourceWidth(percent) {
    const clamped = Math.min(60, Math.max(30, percent));
    write(KEYS.sourceWidth, String(clamped));
    set({ sourceWidth: clamped });
  },
  openSource(target) {
    set({ source: target });
  },
  closeSource() {
    set({ source: null });
  },
  bumpVersionClicks() {
    const next = get().versionClicks + 1;
    if (next >= 5) {
      get().setDeveloperMode(!get().developerMode);
      return true;
    }
    set({ versionClicks: next });
    return false;
  },
}));

/** Викликається один раз на старті, до першого рендера. */
export function bootstrapUi(): void {
  const { theme, scale } = useUi.getState();
  applyTheme(theme);
  applyScale(scale);
  // Системна тема може змінитися під час роботи — типово ввечері, за
  // розкладом ОС. Якщо не слухати, застосунок лишиться світлим на темному
  // робочому столі до наступного запуску.
  window
    .matchMedia("(prefers-color-scheme: dark)")
    .addEventListener("change", () => {
      if (useUi.getState().theme === "system") applyTheme("system");
    });
}
