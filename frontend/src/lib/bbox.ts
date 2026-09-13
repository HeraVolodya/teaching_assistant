/**
 * Геометрія підсвітки цитованого фрагмента.
 *
 * Саме це робить цитату ПЕРЕВІРЮВАНОЮ, а не декоративною: викладач бачить не
 * лише «с. 143», а конкретний абзац, обведений на сторінці. Без підсвітки
 * посилання на 512-сторінковий підручник перевіряється очима хвилину, і
 * тому не перевіряється ніколи.
 *
 * ДВІ ПАСТКИ, ЧЕРЕЗ ЯКІ ТУТ НЕ «ПРОСТО ВІДНЯТИ КООРДИНАТИ».
 *
 * 1. Початок координат. У PDF він у НИЖНЬОМУ лівому куті, у CSS — у
 *    верхньому. Пряме `top = bbox.top` дає прямокутник, дзеркально
 *    відображений по вертикалі: він потрапляє в правильний рядок лише на
 *    сторінках, де фрагмент випадково посередині. Правильний шлях —
 *    `viewport.convertToViewportPoint()`, який несе в собі і переворот осі,
 *    і масштаб, і поворот сторінки, і зсув `viewBox` (сторінки зі зміщеним
 *    CropBox інакше зсунуті на десятки пунктів).
 *
 * 2. Порядок кутів. Після перетворення «верх» і «низ» міняються місцями, а
 *    при повороті на 90° міняються ще й осі. Тому перетворюються ВСІ ЧОТИРИ
 *    кути, а прямокутник збирається як min/max — це єдиний варіант, що
 *    працює для будь-якого повороту й для обох домовленостей про напрям осі
 *    Y, тобто не покладається на здогад про те, що саме поклав у bbox
 *    парсер.
 */

import type { BBoxDto } from "@/api/types";

/** Мінімальний контракт, потрібний від `PageViewport` pdf.js. */
export interface ViewportLike {
  width: number;
  height: number;
  convertToViewportPoint(x: number, y: number): number[];
}

export interface HighlightRect {
  left: number;
  top: number;
  width: number;
  height: number;
}

/** Тонші прямокутники — це артефакти координат, а не текст. */
const MIN_SIDE_PX = 2;

export function bboxToRect(bbox: BBoxDto, viewport: ViewportLike): HighlightRect | null {
  const corners = [
    viewport.convertToViewportPoint(bbox.l, bbox.t),
    viewport.convertToViewportPoint(bbox.r, bbox.t),
    viewport.convertToViewportPoint(bbox.l, bbox.b),
    viewport.convertToViewportPoint(bbox.r, bbox.b),
  ];
  const xs = corners.map((point) => point[0]);
  const ys = corners.map((point) => point[1]);
  if (xs.some((v) => !Number.isFinite(v)) || ys.some((v) => !Number.isFinite(v))) return null;

  const left = Math.max(0, Math.min(...xs));
  const right = Math.min(viewport.width, Math.max(...xs));
  const top = Math.max(0, Math.min(...ys));
  const bottom = Math.min(viewport.height, Math.max(...ys));
  const width = right - left;
  const height = bottom - top;
  if (width < MIN_SIDE_PX || height < MIN_SIDE_PX) return null;
  return { left, top, width, height };
}

export function bboxesToRects(
  bboxes: readonly BBoxDto[],
  page: number,
  viewport: ViewportLike,
): HighlightRect[] {
  const out: HighlightRect[] = [];
  for (const bbox of bboxes) {
    if (bbox.page !== page) continue;
    const rect = bboxToRect(bbox, viewport);
    if (rect) out.push(rect);
  }
  return out;
}

/** Сторінки, на яких фрагмент має підсвітку. Порядок зростання, без повторів. */
export function pagesOf(bboxes: readonly BBoxDto[]): number[] {
  return [...new Set(bboxes.map((b) => b.page))].sort((a, b) => a - b);
}

/**
 * Об'єднати прямокутники сусідніх рядків в один блок.
 *
 * Парсер віддає bbox на кожен рядок тексту, і десяток окремих смужок із
 * рамками виглядає як помилка рендерингу. Рядки, що перекриваються по
 * горизонталі й розділені менше ніж половиною висоти рядка, зливаються.
 */
export function mergeRects(rects: readonly HighlightRect[], gapRatio = 0.5): HighlightRect[] {
  if (rects.length < 2) return [...rects];
  const sorted = [...rects].sort((a, b) => a.top - b.top || a.left - b.left);
  const out: HighlightRect[] = [];
  let current = { ...sorted[0] };

  for (let i = 1; i < sorted.length; i += 1) {
    const next = sorted[i];
    const gap = next.top - (current.top + current.height);
    const overlapsX =
      next.left < current.left + current.width && current.left < next.left + next.width;
    if (overlapsX && gap <= Math.max(current.height, next.height) * gapRatio) {
      const left = Math.min(current.left, next.left);
      const top = Math.min(current.top, next.top);
      const right = Math.max(current.left + current.width, next.left + next.width);
      const bottom = Math.max(current.top + current.height, next.top + next.height);
      current = { left, top, width: right - left, height: bottom - top };
    } else {
      out.push(current);
      current = { ...next };
    }
  }
  out.push(current);
  return out;
}

/**
 * Куди прокрутити, щоб фрагмент опинився у ВЕРХНІЙ третині вікна.
 *
 * Не по центру: цитата майже завжди має продовження нижче, і центрування
 * ховає його під нижнім краєм панелі. Верхня третина лишає контекст видимим.
 */
export function scrollOffsetFor(rect: HighlightRect, viewportHeight: number): number {
  return Math.max(0, rect.top - viewportHeight / 3);
}
