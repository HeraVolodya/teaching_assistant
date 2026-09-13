import { describe, expect, it } from "vitest";

import type { BBoxDto } from "@/api/types";
import { bboxToRect, bboxesToRects, mergeRects, pagesOf, scrollOffsetFor } from "./bbox";
import type { ViewportLike } from "./bbox";

/**
 * Дублікат перетворення pdf.js для сторінки A4 без повороту.
 *
 * Матриця viewport при scale=s і rotation=0 — це [s, 0, 0, -s, -s·x0, s·y1],
 * тобто масштаб плюс ПЕРЕВЕРТАННЯ осі Y. Саме це перевертання і є причиною,
 * чому bbox не можна переносити в CSS відніманням координат.
 */
function makeViewport(scale = 1, height = 842, width = 595): ViewportLike {
  return {
    width: width * scale,
    height: height * scale,
    convertToViewportPoint: (x, y) => [x * scale, (height - y) * scale],
  };
}

/** Сторінка, повернута на 90°: осі міняються місцями. */
function rotatedViewport(scale = 1, height = 842, width = 595): ViewportLike {
  return {
    width: height * scale,
    height: width * scale,
    convertToViewportPoint: (x, y) => [y * scale, x * scale],
  };
}

const bbox = (over: Partial<BBoxDto> = {}): BBoxDto => ({
  page: 1,
  l: 60,
  b: 700,
  r: 535,
  t: 712,
  ...over,
});

describe("bboxToRect", () => {
  it("перевертає вісь Y так, як це робить pdf.js", () => {
    const rect = bboxToRect(bbox(), makeViewport());
    expect(rect).not.toBeNull();
    // Верх у PDF (t=712) — це 842−712 = 130 від верхнього краю в CSS.
    expect(rect?.top).toBeCloseTo(130, 5);
    expect(rect?.left).toBeCloseTo(60, 5);
    expect(rect?.height).toBeCloseTo(12, 5);
    expect(rect?.width).toBeCloseTo(475, 5);
  });

  /**
   * Головна пастка модуля: не можна припускати, який кут парсер поклав у
   * `t`, а який у `b`. Після перевертання осі вони МІНЯЮТЬСЯ МІСЦЯМИ, тому
   * прямокутник збирається як min/max по чотирьох кутах — і результат
   * зобов'язаний бути однаковим при будь-якому порядку.
   */
  it("не залежить від того, який кут названо верхнім", () => {
    const straight = bboxToRect(bbox({ t: 712, b: 700 }), makeViewport());
    const swapped = bboxToRect(bbox({ t: 700, b: 712 }), makeViewport());
    expect(swapped).toEqual(straight);
  });

  it("масштабується разом із viewport", () => {
    const rect = bboxToRect(bbox(), makeViewport(2));
    expect(rect?.top).toBeCloseTo(260, 5);
    expect(rect?.height).toBeCloseTo(24, 5);
  });

  it("працює на поверненій сторінці", () => {
    const rect = bboxToRect(bbox(), rotatedViewport());
    expect(rect).not.toBeNull();
    // Осі помінялися: висота фрагмента стала його шириною.
    expect(rect?.width).toBeCloseTo(12, 5);
    expect(rect?.height).toBeCloseTo(475, 5);
  });

  it("обрізає прямокутник по межах сторінки", () => {
    const rect = bboxToRect(bbox({ l: -50, r: 900 }), makeViewport());
    expect(rect?.left).toBe(0);
    expect((rect?.left ?? 0) + (rect?.width ?? 0)).toBeLessThanOrEqual(595);
  });

  it("відкидає вироджені прямокутники", () => {
    // Нульова висота — артефакт координат, а не текст. Порожня рамка на
    // сторінці читалася б як помилка підсвітки.
    expect(bboxToRect(bbox({ t: 700, b: 700 }), makeViewport())).toBeNull();
  });

  it("відкидає нечислові координати", () => {
    expect(bboxToRect(bbox({ t: Number.NaN }), makeViewport())).toBeNull();
  });
});

describe("bboxesToRects", () => {
  it("бере лише прямокутники потрібної сторінки", () => {
    const rects = bboxesToRects(
      [bbox({ page: 1 }), bbox({ page: 2, b: 400, t: 412 }), bbox({ page: 1, b: 680, t: 692 })],
      1,
      makeViewport(),
    );
    expect(rects).toHaveLength(2);
  });

  it("pagesOf повертає впорядкований набір без повторів", () => {
    expect(pagesOf([bbox({ page: 5 }), bbox({ page: 2 }), bbox({ page: 5 })])).toEqual([2, 5]);
  });
});

describe("mergeRects", () => {
  it("зливає сусідні рядки одного абзацу", () => {
    // Три рядки по 12 px із проміжком 2 px — це один абзац, а не три
    // окремі смуги з рамками.
    const rects = [
      { left: 60, top: 100, width: 475, height: 12 },
      { left: 60, top: 114, width: 475, height: 12 },
      { left: 60, top: 128, width: 300, height: 12 },
    ];
    const merged = mergeRects(rects);
    expect(merged).toHaveLength(1);
    expect(merged[0].top).toBe(100);
    expect(merged[0].height).toBe(40);
  });

  it("не зливає рядки з різних колонок", () => {
    const merged = mergeRects([
      { left: 60, top: 100, width: 200, height: 12 },
      { left: 320, top: 100, width: 200, height: 12 },
    ]);
    expect(merged).toHaveLength(2);
  });

  it("не зливає рядки, розділені великим проміжком", () => {
    const merged = mergeRects([
      { left: 60, top: 100, width: 475, height: 12 },
      { left: 60, top: 400, width: 475, height: 12 },
    ]);
    expect(merged).toHaveLength(2);
  });

  it("порожній і одиничний набори лишаються як є", () => {
    expect(mergeRects([])).toEqual([]);
    const single = [{ left: 1, top: 2, width: 3, height: 4 }];
    expect(mergeRects(single)).toEqual(single);
  });
});

describe("scrollOffsetFor", () => {
  it("ставить фрагмент у верхню третину вікна", () => {
    expect(scrollOffsetFor({ left: 0, top: 900, width: 10, height: 10 }, 600)).toBe(700);
  });

  it("не прокручує вище початку документа", () => {
    expect(scrollOffsetFor({ left: 0, top: 50, width: 10, height: 10 }, 600)).toBe(0);
  });
});
