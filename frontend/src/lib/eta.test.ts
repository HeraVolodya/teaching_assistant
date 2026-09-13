import { describe, expect, it } from "vitest";

import { EtaRegistry, EtaTracker, median } from "./eta";

describe("median", () => {
  it("повертає null на порожньому наборі", () => {
    expect(median([])).toBeNull();
  });

  it("бере середнє двох центральних при парній кількості", () => {
    expect(median([1, 2, 3, 4])).toBe(2.5);
  });

  it("не залежить від порядку", () => {
    expect(median([9, 1, 5])).toBe(median([1, 5, 9]));
  });
});

describe("EtaTracker", () => {
  it("мовчить, доки вимірів менше двох", () => {
    const tracker = new EtaTracker();
    tracker.observe(0, 0);
    tracker.observe(10, 1000);
    expect(tracker.snapshot(10, 100).secondsPerUnit).toBeNull();
  });

  it("рахує секунди на одиницю ваги", () => {
    const tracker = new EtaTracker();
    // Рівна швидкість: 10 одиниць за секунду → 0.1 с на одиницю.
    for (let step = 0; step <= 5; step += 1) tracker.observe(step * 10, step * 1000);
    const snapshot = tracker.snapshot(50, 100);
    expect(snapshot.secondsPerUnit).toBeCloseTo(0.1, 5);
    expect(snapshot.remainingSeconds).toBeCloseTo(5, 5);
  });

  /**
   * Ключовий тест модуля. Одна сканована сторінка коштує на порядок дорожче
   * за цифрову; наївне середнє після неї показує вчетверо більший час, і ETA
   * стрибає туди-сюди на очах у користувача. Медіана до одного викиду
   * байдужа — саме заради цього вона тут і стоїть.
   */
  it("не піддається одиничному викиду так, як середнє", () => {
    const tracker = new EtaTracker();
    let at = 0;
    let weight = 0;
    // Дев'ять швидких кроків по 0.1 с/одиницю…
    for (let i = 0; i < 9; i += 1) {
      weight += 10;
      at += 1000;
      tracker.observe(weight, at);
    }
    // …і один вкрай повільний: 10 с на ті самі 10 одиниць (1 с/одиницю).
    weight += 10;
    at += 10_000;
    tracker.observe(weight, at);

    const snapshot = tracker.snapshot(weight, weight + 100);
    const naiveMean = (9 * 0.1 + 1.0) / 10; // 0.19 — майже вдвічі більше
    expect(snapshot.secondsPerUnit).toBeCloseTo(0.1, 3);
    expect(snapshot.secondsPerUnit as number).toBeLessThan(naiveMean);
    // Прогноз на 100 одиниць: медіана каже 10 с, середнє сказало б 19 с.
    expect(snapshot.remainingSeconds).toBeCloseTo(10, 1);
  });

  it("реагує на СТІЙКУ зміну режиму, а не лише на викид", () => {
    const tracker = new EtaTracker();
    let at = 0;
    let weight = 0;
    for (let i = 0; i < 9; i += 1) {
      weight += 10;
      at += 1000;
      tracker.observe(weight, at);
    }
    // Пішов суцільний скан: п'ять кроків по 1 с/одиницю поспіль.
    for (let i = 0; i < 5; i += 1) {
      weight += 10;
      at += 10_000;
      tracker.observe(weight, at);
    }
    const snapshot = tracker.snapshot(weight, weight + 10);
    expect(snapshot.secondsPerUnit as number).toBeGreaterThan(0.5);
  });

  it("скидає історію, коли прогрес пішов назад (повтор завдання)", () => {
    const tracker = new EtaTracker();
    tracker.observe(10, 1000);
    tracker.observe(20, 2000);
    tracker.observe(30, 3000);
    expect(tracker.snapshot(30, 100).secondsPerUnit).not.toBeNull();
    tracker.observe(0, 4000); // документ поставили в чергу заново
    expect(tracker.snapshot(0, 100).secondsPerUnit).toBeNull();
  });

  it("ігнорує події без реального приросту", () => {
    const tracker = new EtaTracker();
    tracker.observe(10, 1000);
    // Heartbeat без прогресу: ділення на нульовий приріст дало б Infinity.
    tracker.observe(10, 2000);
    tracker.observe(10, 3000);
    const snapshot = tracker.snapshot(10, 100);
    expect(snapshot.secondsPerUnit).toBeNull();
    expect(Number.isFinite(snapshot.remainingSeconds ?? 0)).toBe(true);
  });

  it("тримає лише останні N вимірів", () => {
    const tracker = new EtaTracker(3);
    let at = 0;
    let weight = 0;
    for (let i = 0; i < 10; i += 1) {
      weight += 10;
      at += 10_000; // повільно
      tracker.observe(weight, at);
    }
    for (let i = 0; i < 3; i += 1) {
      weight += 10;
      at += 1000; // швидко
      tracker.observe(weight, at);
    }
    // Вікно з трьох: стара повільна історія вже витіснена повністю.
    expect(tracker.snapshot(weight, weight).secondsPerUnit).toBeCloseTo(0.1, 5);
  });
});

describe("EtaRegistry", () => {
  it("не вигадує оцінку, доки жодне завдання її не має", () => {
    const registry = new EtaRegistry();
    registry.observe("a", 0, 100, 0);
    expect(registry.totalRemaining(["a"])).toBeNull();
  });

  it("додає залишки послідовних завдань, а не бере максимум", () => {
    const registry = new EtaRegistry();
    for (const id of ["a", "b"]) {
      registry.observe(id, 0, 100, 0);
      registry.observe(id, 10, 100, 1000);
      registry.observe(id, 20, 100, 2000);
    }
    // По 80 одиниць залишку в кожного при 0.1 с/одиницю → 8 с кожне.
    const total = registry.totalRemaining(["a", "b"]);
    expect(total).toBeCloseTo(16, 1);
  });

  it("ділить на кількість воркерів", () => {
    const registry = new EtaRegistry();
    for (const id of ["a", "b"]) {
      registry.observe(id, 0, 100, 0);
      registry.observe(id, 10, 100, 1000);
      registry.observe(id, 20, 100, 2000);
    }
    expect(registry.totalRemaining(["a", "b"], 2)).toBeCloseTo(8, 1);
  });

  it("забуває завершене завдання", () => {
    const registry = new EtaRegistry();
    registry.observe("a", 10, 100, 1000);
    registry.forget("a");
    expect(registry.snapshot("a")).toBeNull();
  });
});
