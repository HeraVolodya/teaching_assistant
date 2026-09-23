/**
 * @vitest-environment jsdom
 *
 * Зворотний бік суворості: демонстраційний режим МУСИТЬ лишитися робочим
 * інструментом розробки інтерфейсу.
 *
 * Правило «у постачанні демо не вмикається саме» (див. `client.test.ts`) легко
 * перевиконати й викинути фолбек зовсім — тоді `npm run dev` без піднятого
 * бекенда дає порожній екран, і робота над екранами, які потребують
 * неприємних станів (документ у помилці, утримання від відповіді), стає
 * неможливою без повного стенду.
 *
 * Окремий файл і jsdom — через те, що `mock/server.ts` планує симуляцію
 * індексації через `window.setTimeout` на рівні модуля: під node його імпорт
 * падає ще до першого тесту.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const realFetch = globalThis.fetch;

beforeEach(() => {
  // Порт не слухається — рівно те, що бачить розробник без запущеного sidecar.
  globalThis.fetch = vi
    .fn()
    .mockRejectedValue(new TypeError("fetch failed")) as unknown as typeof fetch;
});

afterEach(() => {
  globalThis.fetch = realFetch;
  vi.restoreAllMocks();
});

describe("probeBackend у розробці", () => {
  it("після вичерпання бюджету переходить на демонстраційні дані", async () => {
    vi.resetModules();
    const { probeBackend, apiMode } = await import("./client");

    const health = await probeBackend({
      allowDemoFallback: true,
      deadlineMs: 0,
      now: () => 0,
      sleep: async () => undefined,
    });

    expect(apiMode()).toBe("demo");
    // Режим мусить лишатися ЧЕСНО позначеним: саме на цих двох полях тримаються
    // бейдж «Демонстраційні дані» і смуга «Режим заглушки» в шапці.
    expect(health.demo).toBe(true);
    expect(health.stub).toBe(true);
  });
});
