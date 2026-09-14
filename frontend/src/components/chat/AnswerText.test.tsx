/**
 * @vitest-environment jsdom
 *
 * Рендер відповіді: розмітка стає розміткою, а цитати лишаються цитатами.
 *
 * ЧОМУ ЦЕЙ ТЕСТ Є, ХОЧА КОНФІГ КАЖЕ «РЕНДЕР-ТЕСТИ ПЕРЕВІРЯЮТЬ САМІ СЕБЕ».
 * Тут перевіряється не верстка, а межа між двома шарами, яку легко зрушити
 * непомітно: `lib/markdown` розбирає текст на вузли, і КОЖЕН листок типу
 * `text` мусить пройти через `renderInline`, інакше `[1]` перестане бути
 * клікабельним джерелом. Помилка в цьому місці не валить екран і не ловиться
 * типами — вона просто тихо перетворює перевірні твердження на звичайний
 * текст. Саме та поломка, заради якої продукт і робиться, лишилася б
 * непоміченою до першої лекції.
 */

import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, describe, expect, it } from "vitest";

import type { CitationDto } from "@/api/types";

import { AnswerText } from "./AnswerText";

const CITATION: CitationDto = {
  ordinal: 1,
  chunkUid: "c1",
  documentId: "d1",
  documentTitle: "art_pozvidka",
  pageFrom: 32,
  pageTo: 33,
  pageLabel: "с. 32–33",
  quote: "Ця формула виражає залежність між кутовими і лінійними величинами.",
  bboxes: [],
  language: "uk",
};

let host: HTMLDivElement | null = null;
let root: Root | null = null;

function render(node: React.ReactElement): HTMLDivElement {
  host = document.createElement("div");
  document.body.append(host);
  root = createRoot(host);
  act(() => {
    root?.render(node);
  });
  return host;
}

afterEach(() => {
  act(() => root?.unmount());
  host?.remove();
  root = null;
  host = null;
});

describe("розмітка", () => {
  it("жирний стає <strong>, а зірочки зникають", () => {
    const el = render(<AnswerText text="**1-й тип:** визначають відстань" citations={[]} />);
    expect(el.querySelector("strong")?.textContent).toBe("1-й тип:");
    expect(el.textContent).not.toContain("**");
  });

  it("курсив стає <em>", () => {
    const el = render(<AnswerText text="*Примітка:* далі" citations={[]} />);
    expect(el.querySelector("em")?.textContent).toBe("Примітка:");
  });

  it("нумерований список стає <ol> з пунктами", () => {
    const el = render(<AnswerText text={"1. перший\n2. другий\n3. третій"} citations={[]} />);
    expect(el.querySelectorAll("ol > li")).toHaveLength(3);
    expect(el.textContent).not.toMatch(/^\s*1\./);
  });

  it("маркований список стає <ul>", () => {
    const el = render(<AnswerText text={"* перший\n* другий"} citations={[]} />);
    expect(el.querySelectorAll("ul > li")).toHaveLength(2);
  });

  it("таблиця стає <table>", () => {
    const el = render(
      <AnswerText text={"| Дальність | Заряд |\n|---|---|\n| 17400 | повний |"} citations={[]} />,
    );
    expect(el.querySelectorAll("table th")).toHaveLength(2);
    expect(el.querySelectorAll("table tbody td")).toHaveLength(2);
  });
});

describe("цитати", () => {
  it("[1] стає пігулкою з доступним підписом", () => {
    const el = render(<AnswerText text="Дальність 17 400 м [1]." citations={[CITATION]} />);
    const pill = el.querySelector("button[aria-label]");
    expect(pill?.getAttribute("aria-label")).toContain("art_pozvidka");
    expect(el.textContent).not.toContain("[1]");
  });

  it("[1] усередині жирного лишається пігулкою", () => {
    const el = render(<AnswerText text="**Висновок [1]:** далі" citations={[CITATION]} />);
    expect(el.querySelector("strong button[aria-label]")).not.toBeNull();
  });

  it("[1] у пункті списку лишається пігулкою", () => {
    const el = render(<AnswerText text="1. **1-й тип:** відстань [1]." citations={[CITATION]} />);
    expect(el.querySelector("li button[aria-label]")).not.toBeNull();
  });

  it("маркер без джерела не вдає робоче посилання", () => {
    const el = render(<AnswerText text="Твердження [7]." citations={[]} />);
    expect(el.querySelector("button[aria-label]")).toBeNull();
    expect(el.textContent).toContain("[7]");
  });

  it("маркер в історії стає кнопкою «чому»", () => {
    const clicks: number[] = [];
    const el = render(
      <AnswerText text="Твердження [2]." citations={[]} onMarkerClick={(n) => clicks.push(n)} />,
    );
    const button = el.querySelector("button");
    expect(button).not.toBeNull();
    act(() => button?.dispatchEvent(new MouseEvent("click", { bubbles: true })));
    expect(clicks).toEqual([2]);
  });
});

describe("формули", () => {
  it("$\\beta$ рендериться через KaTeX, а не показує долари", () => {
    const el = render(<AnswerText text="кут ($\beta$) між напрямками" citations={[]} />);
    expect(el.querySelector(".katex")).not.toBeNull();
    expect(el.textContent).not.toContain("$");
  });

  it("дужки у \\sqrt[3]{x} не стають пігулкою", () => {
    const el = render(<AnswerText text="корінь $\sqrt[3]{x}$ тут" citations={[CITATION]} />);
    expect(el.querySelector("button[aria-label]")).toBeNull();
  });
});

describe("каретка стріму", () => {
  it("зʼявляється лише під час набору", () => {
    const live = render(<AnswerText text="Відповідь" citations={[]} streaming />);
    expect(live.querySelectorAll(".animate-pulse")).toHaveLength(1);
    act(() => root?.unmount());
    host?.remove();

    const done = render(<AnswerText text="Відповідь" citations={[]} />);
    expect(done.querySelectorAll(".animate-pulse")).toHaveLength(0);
  });

  it("у кінці списку стоїть усередині останнього пункту", () => {
    const el = render(<AnswerText text={"1. перший\n2. другий"} citations={[]} streaming />);
    const items = el.querySelectorAll("li");
    expect(items[items.length - 1].querySelector(".animate-pulse")).not.toBeNull();
  });
});

describe("реальна відповідь про формулу тисячної", () => {
  const ANSWER = [
    "Формула тисячних — це формула, яка виражає залежність між кутовими та лінійними величинами [1].",
    "",
    "За допомогою цієї формули можна розвʼязувати три типи задач:",
    "1. **1-й тип:** Визначають відстань ($l$) між точками, знаючи кут ($\\beta$) та дальність ($D$) [1].",
    "2. **2-й тип:** Визначають значення кута ($\\beta$) у поділках кутоміра [1].",
    "",
    "*Примітка:* Лінійна величина збільшується на 5 % [1].",
  ].join("\n");

  it("на екрані немає жодного символу розмітки", () => {
    const el = render(<AnswerText text={ANSWER} citations={[CITATION]} />);
    const text = el.textContent ?? "";
    expect(text).not.toContain("**");
    expect(text).not.toContain("$");
    expect(text).not.toContain("[1]");
  });

  it("усі чотири посилання лишилися клікабельними", () => {
    const el = render(<AnswerText text={ANSWER} citations={[CITATION]} />);
    expect(el.querySelectorAll("button[aria-label]")).toHaveLength(4);
  });

  it("структура: абзац, абзац, список, абзац", () => {
    const el = render(<AnswerText text={ANSWER} citations={[CITATION]} />);
    expect(el.querySelectorAll("ol > li")).toHaveLength(2);
    expect(el.querySelectorAll("strong")).toHaveLength(2);
    expect(el.querySelectorAll("em")).toHaveLength(1);
    expect(el.querySelectorAll(".katex").length).toBeGreaterThanOrEqual(4);
  });
});
