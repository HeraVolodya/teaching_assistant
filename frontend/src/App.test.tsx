/**
 * @vitest-environment jsdom
 *
 * Димовий тест рендера: кожен екран мусить піднятися й показати зміст.
 *
 * ЧОМУ ЦЕ Є, ХОЧА «РЕНДЕР-ТЕСТИ ПЕРЕВІРЯЮТЬ ПЕРЕВАЖНО САМІ СЕБЕ». Помилка на
 * кшталт зверненого до `undefined` поля або невірного порядку хуків не
 * виглядає як помилка в типах і не ловиться тестами логіки — вона виглядає як
 * БІЛЕ ВІКНО в застосунку. Тут не перевіряється верстка; перевіряється рівно
 * одне: екран не падає й показує щось осмислене на демонстраційних даних.
 *
 * Тест іде проти демонстраційного транспорту — того самого, що рятує UI, коли
 * sidecar не піднявся. Тобто заразом перевіряється й він.
 */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeAll, describe, expect, it } from "vitest";

import App from "./App";
import { installDemoTransport } from "./api/mock/server";
import { initI18n } from "./i18n";
import type { HealthStatus } from "./api/types";

// jsdom не має цих API, а без них падає будь-який компонент із темою,
// вимірюванням ширини чи лінивим рендером сторінок PDF.
beforeAll(() => {
  window.matchMedia = ((query: string) => ({
    matches: false,
    media: query,
    onchange: null,
    addEventListener: () => undefined,
    removeEventListener: () => undefined,
    addListener: () => undefined,
    removeListener: () => undefined,
    dispatchEvent: () => false,
  })) as unknown as typeof window.matchMedia;

  class Observer {
    observe(): void {}
    unobserve(): void {}
    disconnect(): void {}
    takeRecords(): [] {
      return [];
    }
  }
  window.ResizeObserver = Observer as unknown as typeof ResizeObserver;
  window.IntersectionObserver = Observer as unknown as typeof IntersectionObserver;
  Element.prototype.scrollIntoView = () => undefined;
  Element.prototype.scrollTo = () => undefined;

  installDemoTransport();
  initI18n();
});

const DEMO_HEALTH: HealthStatus = {
  ok: true,
  version: "0.1.0-demo",
  stub: true,
  uptimeSeconds: 1,
  python: "—",
  platform: "demo",
  database: { ok: true, detail: "", schemaVersion: 1, path: "(пам'ять)" },
  lmstudio: { ok: false, chatReady: false, apiVersion: "none", detail: "", models: [] },
  worker: { mode: "demo" },
  jobs: { active: 0, queued: 0 },
  problems: [],
  eventSubscribers: 0,
  demo: true,
};

let root: Root | null = null;
let host: HTMLDivElement | null = null;

afterEach(() => {
  act(() => root?.unmount());
  host?.remove();
  root = null;
  host = null;
});

/** Змонтувати застосунок за вказаним маршрутом і дати запитам відпрацювати. */
async function mountAt(path: string): Promise<string> {
  host = document.createElement("div");
  document.body.append(host);
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });

  await act(async () => {
    root = createRoot(host as HTMLDivElement);
    root.render(
      <QueryClientProvider client={client}>
        <MemoryRouter initialEntries={[path]}>
          <App initialHealth={DEMO_HEALTH} />
        </MemoryRouter>
      </QueryClientProvider>,
    );
  });

  // Демонстраційний транспорт навмисно відповідає із затримкою ~90 мс, щоб
  // стани завантаження встигали з'явитися; чекаємо кілька таких кіл.
  for (let round = 0; round < 8; round += 1) {
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 60));
    });
  }
  return document.body.textContent ?? "";
}

describe("димовий рендер екранів", () => {
  it("список асистентів показує картки з демонстраційного корпусу", async () => {
    const text = await mountAt("/");
    expect(text).toContain("Асистенти");
    expect(text).toContain("Артилерійська підготовка");
    expect(text).toContain("Військова топографія");
    // Демонстраційний режим мусить чесно позначатися.
    expect(text).toContain("Демонстраційні дані");
  });

  it("картка асистента показує червону позначку про необроблений матеріал", async () => {
    const text = await mountAt("/");
    // У корпусі є документ зі статусом FAILED — картка зобов'язана це показати.
    expect(text).toContain("Є матеріал, який не вдалося обробити");
  });

  it("екран питань піднімається з полем введення", async () => {
    const text = await mountAt("/a/as-artillery/chat");
    expect(text).toContain("Поставте перше питання");
    expect(document.querySelector("textarea")).not.toBeNull();
  });

  it("екран матеріалів показує рядки, стани й помилку з дією", async () => {
    const text = await mountAt("/a/as-artillery/documents");
    expect(text).toContain("Основи балістики та стрільби");
    expect(text).toContain("Скан розпізнано ненадійно");
    // Смуга черги: у корпусі є документ в обробці.
    expect(text).toContain("Обробка");
  });

  it("редактор асистента показує контроли й скомпільований промпт", async () => {
    const text = await mountAt("/a/as-artillery/settings");
    expect(text).toContain("Межі компетенції");
    expect(text).toContain("Потрібна впевненість");
    // Прев'ю мусить містити правила, які реально піде в модель.
    expect(text).toContain("навчальний асистент викладача Національної академії");
    // Пороги показуються числами, а не словами.
    expect(text).toMatch(/0[.,]35/);
  });

  it("база знань показує дерево матеріалів і поле пошуку", async () => {
    const text = await mountAt("/a/as-artillery/library");
    expect(text).toContain("База знань");
    expect(text).toContain("Основи балістики та стрільби");
  });

  it("налаштування показують тему, масштаб і версію", async () => {
    const text = await mountAt("/settings");
    expect(text).toContain("Налаштування");
    expect(text).toContain("Тема");
    expect(text).toContain("Версія");
  });

  it("майстер першого запуску проходиться без падінь", async () => {
    const text = await mountAt("/setup");
    expect(text).toContain("Перший запуск");
    expect(text).toContain("Мова інтерфейсу");
  });

  it("невідомий маршрут веде на головну, а не в порожній екран", async () => {
    const text = await mountAt("/такого-немає");
    expect(text).toContain("Асистенти");
  });
});
