/**
 * Старт застосунку: чекати на sidecar, а не підміняти його прикладом.
 *
 * ЩО САМЕ ТУТ ОХОРОНЯЄТЬСЯ І ЧОМУ ЦЕ НЕ ДРІБНИЦЯ.
 * `probeBackend` робив ОДНУ спробу з таймаутом 1.5 с і при будь-якій помилці
 * мовчки вмикав демонстраційний транспорт назавжди. У запакованому застосунку
 * вікно вантажить SPA одразу при старті процесу, а Python-рантайм підіймається
 * кілька секунд — тож SPA програвала цю гонку ЩОРАЗУ й показувала викладачеві
 * вигаданий корпус із вигаданими цитатами замість його власних матеріалів.
 * Помилка не виглядала як помилка: інтерфейс працював, документи «додавалися»,
 * відповіді приходили з покликаннями на сторінки — просто все це було вигадане.
 *
 * Тому тести нижче фіксують рівно дві речі, і обидві — про довіру, а не про
 * зручність: проба МУСИТЬ повторюватись, і в постачанні вона НЕ МАЄ права
 * тихо перейти на демо.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

type Client = typeof import("./client");

/** Свіжий модуль на кожен тест: транспорт і режим — модульний стан. */
async function freshClient(): Promise<Client> {
  vi.resetModules();
  return import("./client");
}

/**
 * Годинник, який рухає лише `sleep`. Без нього тест на вичерпання бюджету
 * чекав би реальні секунди, а тест на повтори — реальні мілісекунди між ними.
 */
function fakeClock() {
  let elapsed = 0;
  return {
    now: () => elapsed,
    sleep: async (ms: number) => {
      elapsed += ms;
    },
  };
}

function healthResponse(): Response {
  return new Response(JSON.stringify({ ok: true, stub: false, version: "0.1.0" }), {
    status: 200,
    headers: { "content-type": "application/json" },
  });
}

const realFetch = globalThis.fetch;
let fetchMock: ReturnType<typeof vi.fn>;

beforeEach(() => {
  fetchMock = vi.fn();
  globalThis.fetch = fetchMock as unknown as typeof fetch;
});

afterEach(() => {
  globalThis.fetch = realFetch;
  vi.restoreAllMocks();
});

describe("probeBackend", () => {
  it("повторює спроби, доки sidecar не піднявся", async () => {
    // Рівно те, що бачить SPA на холодному старті: порт ще не слухається.
    fetchMock
      .mockRejectedValueOnce(new TypeError("fetch failed"))
      .mockRejectedValueOnce(new TypeError("fetch failed"))
      .mockRejectedValueOnce(new TypeError("fetch failed"))
      .mockResolvedValueOnce(healthResponse());

    const { probeBackend, apiMode } = await freshClient();
    const clock = fakeClock();
    const health = await probeBackend({
      allowDemoFallback: false,
      deadlineMs: 60_000,
      retryIntervalMs: 250,
      ...clock,
    });

    expect(fetchMock).toHaveBeenCalledTimes(4);
    expect(health.demo).toBe(false);
    expect(apiMode()).toBe("http");
  });

  it("у постачанні НЕ підміняє sidecar демонстраційними даними", async () => {
    fetchMock.mockRejectedValue(new TypeError("fetch failed"));

    const { probeBackend, apiMode, BackendUnavailableError } = await freshClient();
    const clock = fakeClock();

    await expect(
      probeBackend({ allowDemoFallback: false, deadlineMs: 1_000, retryIntervalMs: 250, ...clock }),
    ).rejects.toBeInstanceOf(BackendUnavailableError);

    // Головне твердження тесту: транспорт лишився HTTP-івським. Якби він став
    // "demo", застосунок показав би вигаданий корпус як справжній.
    expect(apiMode()).not.toBe("demo");
  });

  it("витрачає весь бюджет, а не здається після першої відмови", async () => {
    fetchMock.mockRejectedValue(new TypeError("fetch failed"));

    const { probeBackend } = await freshClient();
    const clock = fakeClock();
    await probeBackend({
      allowDemoFallback: false,
      deadlineMs: 1_000,
      retryIntervalMs: 250,
      ...clock,
    }).catch(() => undefined);

    // 1000 мс бюджету при кроці 250 мс — п'ять спроб (нульова плюс чотири).
    expect(fetchMock).toHaveBeenCalledTimes(5);
  });

  // Зворотний бік — що демо все ще вмикається в розробці — перевіряється в
  // `client.demo.test.ts`: `mock/server.ts` звертається до `window` на рівні
  // модуля, тож той тест мусить жити в jsdom, а цей файл лишається на node.

  it("робить принаймні одну спробу навіть із нульовим бюджетом", async () => {
    fetchMock.mockResolvedValueOnce(healthResponse());

    const { probeBackend } = await freshClient();
    const health = await probeBackend({ allowDemoFallback: false, deadlineMs: 0, ...fakeClock() });

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(health.demo).toBe(false);
  });
});
