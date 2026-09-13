/**
 * Демонстраційний режим мусить показувати ті самі шляхи, що й справжній —
 * включно з неприємними. Тест стежить, щоб корпус не «покращили» до стану,
 * у якому утримання й помилки обробки стають невидимими.
 */

import { describe, expect, it } from "vitest";

import { DEMO_ASSISTANTS, DEMO_CHUNKS, DEMO_DOCUMENTS, demoAnswer, demoSearch } from "./fixtures";
import { PAGE_HEIGHT, PAGE_WIDTH, buildDemoPdf, demoLineBBox } from "./pdf";

describe("демонстраційний корпус", () => {
  it("містить документ у помилці — інакше екран матеріалів показує лише щастя", () => {
    expect(DEMO_DOCUMENTS.some((document) => document.status === "FAILED")).toBe(true);
  });

  it("містить документ в обробці — інакше не видно ані смуги, ані ETA", () => {
    expect(
      DEMO_DOCUMENTS.some((document) => document.status === "INDEXING" || document.status === "QUEUED"),
    ).toBe(true);
  });

  it("кожен асистент має рівно одну колекцію", () => {
    for (const assistant of DEMO_ASSISTANTS) expect(assistant.collections).toHaveLength(1);
  });

  it("кожен фрагмент належить наявному документові", () => {
    const ids = new Set(DEMO_DOCUMENTS.map((document) => document.id));
    for (const chunk of DEMO_CHUNKS) expect(ids.has(chunk.documentId)).toBe(true);
  });
});

describe("demoSearch", () => {
  it("знаходить за словоформою, а не лише за точним збігом", () => {
    // «траєкторії» проти «траєкторія» в тексті — типовий випадок для
    // української, і саме на ньому наївний пошук по підрядку мовчить.
    const found = demoSearch("елементи траєкторії", "col-artillery");
    expect(found.length).toBeGreaterThan(0);
  });

  /** Найважливіший шлях: питання без підстав у корпусі. */
  it("повертає порожньо на питанні поза корпусом", () => {
    expect(demoSearch("порядок нарахування відпускних", "col-artillery")).toHaveLength(0);
  });

  it("не змішує колекції різних асистентів", () => {
    const found = demoSearch("дирекційний кут", "col-artillery");
    expect(found.every((chunk) => chunk.documentId !== "doc-topo")).toBe(true);
  });

  it("дотримується квоти на документ", () => {
    const found = demoSearch("траєкторія кут дальність підготовка стрільби", "col-artillery");
    const perDocument = new Map<string, number>();
    for (const chunk of found) {
      perDocument.set(chunk.documentId, (perDocument.get(chunk.documentId) ?? 0) + 1);
    }
    for (const count of perDocument.values()) expect(count).toBeLessThanOrEqual(2);
  });

  it("не шукає в документах, які ще не оброблені", () => {
    const found = demoSearch("заняття курсанти норматив", "col-artillery");
    // doc-methodic має статус INDEXING, тож його фрагменти в пошук не йдуть.
    expect(found.every((chunk) => chunk.documentId !== "doc-methodic")).toBe(true);
  });
});

describe("demoAnswer", () => {
  it("нумерує джерела від одиниці й поспіль", () => {
    const chunks = demoSearch("траєкторія", "col-artillery");
    const answer = demoAnswer("Що таке траєкторія?", chunks);
    const markers = [...answer.matchAll(/\[(\d+)\]/g)].map((match) => Number(match[1]));
    expect(markers).toEqual(chunks.map((_, index) => index + 1));
  });

  it("без фрагментів чесно відмовляється, а не вигадує", () => {
    const answer = demoAnswer("Питання поза корпусом", []);
    expect(answer).toContain("не знайшов");
    expect(answer).not.toMatch(/\[\d+\]/);
  });
});

describe("buildDemoPdf", () => {
  it("починається сигнатурою PDF і закінчується EOF", () => {
    const bytes = buildDemoPdf(3);
    const text = new TextDecoder("latin1").decode(bytes);
    expect(text.startsWith("%PDF-1.")).toBe(true);
    expect(text.trimEnd().endsWith("%%EOF")).toBe(true);
  });

  it("оголошує стільки сторінок, скільки просили", () => {
    const text = new TextDecoder("latin1").decode(buildDemoPdf(5));
    expect(text).toContain("/Count 5");
  });

  /**
   * Зміщення в таблиці xref рахуються по довжині рядка — це коректно лише
   * доки файл лишається ASCII. Один двобайтовий символ зсунув би всю
   * таблицю, і pdf.js відмовився б відкрити файл.
   */
  it("лишається чисто ASCII", () => {
    const bytes = buildDemoPdf(2);
    expect(bytes.every((byte) => byte < 128)).toBe(true);
  });

  it("зміщення xref вказують на початки об'єктів", () => {
    const bytes = buildDemoPdf(2);
    const text = new TextDecoder("latin1").decode(bytes);
    const startxref = Number(/startxref\n(\d+)/.exec(text)?.[1]);
    expect(text.slice(startxref, startxref + 4)).toBe("xref");

    const offsets = [...text.matchAll(/^(\d{10}) 00000 n $/gm)].map((match) => Number(match[1]));
    expect(offsets.length).toBeGreaterThan(3);
    offsets.forEach((offset, index) => {
      expect(text.slice(offset).startsWith(`${index + 1} 0 obj`)).toBe(true);
    });
  });

  it("координати рядків лежать усередині сторінки", () => {
    for (let line = 0; line < 20; line += 1) {
      const box = demoLineBBox(1, line);
      expect(box.l).toBeGreaterThanOrEqual(0);
      expect(box.r).toBeLessThanOrEqual(PAGE_WIDTH);
      expect(box.b).toBeGreaterThan(0);
      expect(box.t).toBeLessThan(PAGE_HEIGHT);
      expect(box.t).toBeGreaterThan(box.b);
    }
  });
});
