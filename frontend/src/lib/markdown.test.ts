import { describe, expect, it } from "vitest";

import { type Inline, parseInline, parseMarkdown, toPlainText } from "./markdown";

/** Плоский текст листків типу `text` — саме вони підуть у renderInline. */
function textLeaves(nodes: Inline[]): string[] {
  const out: string[] = [];
  const walk = (list: Inline[]) => {
    for (const node of list) {
      if (node.kind === "text") out.push(node.text);
      else if (node.kind === "strong" || node.kind === "em") walk(node.children);
    }
  };
  walk(nodes);
  return out;
}

describe("інваріант цитат", () => {
  /**
   * НАЙВАЖЛИВІШИЙ ТЕСТ У ФАЙЛІ.
   *
   * Маркер `[n]` мусить дійти до renderInline як звичайний текст — саме там
   * він стає клікабельною пігулкою. Якщо парсер його з'їсть, розірве або
   * сховає у вузол, до якого renderInline не дістанеться, викладач втратить
   * можливість перевірити твердження. Це ядро довіри до продукту.
   */
  it("маркер доходить до текстового листка неушкодженим", () => {
    expect(textLeaves(parseInline("Дальність 17 400 м [1]."))).toContain("Дальність 17 400 м [1].");
  });

  it("маркер усередині жирного лишається в тексті", () => {
    const nodes = parseInline("**Висновок [2]:** далі");
    expect(textLeaves(nodes).join("")).toContain("[2]");
  });

  it("маркер у пункті списку лишається в тексті", () => {
    const [block] = parseMarkdown("- перший пункт [3]");
    expect(block.kind).toBe("list");
    if (block.kind !== "list") return;
    expect(textLeaves(block.items[0].lines[0]).join("")).toContain("[3]");
  });

  it("квадратні дужки у формулі НЕ стають текстом — і тому не стануть пігулкою", () => {
    // \sqrt[3]{x}: сьогодні регулярка цитат з'їдає [3] і показує його як
    // джерело, а parse_citations на бекенді ще й вирізає його з відповіді.
    const nodes = parseInline("корінь $\\sqrt[3]{x}$ тут");
    const math = nodes.find((node) => node.kind === "math");
    expect(math).toBeDefined();
    expect(textLeaves(nodes).join("")).not.toContain("[3]");
  });

  it("квадратні дужки у коді не стають текстом", () => {
    const nodes = parseInline("виклик `arr[1]` у циклі");
    expect(textLeaves(nodes).join("")).not.toContain("[1]");
  });
});

describe("наголоси", () => {
  it("жирний", () => {
    const [node] = parseInline("**1-й тип:**");
    expect(node).toEqual({ kind: "strong", children: [{ kind: "text", text: "1-й тип:" }] });
  });

  it("курсив", () => {
    const [node] = parseInline("*Примітка:*");
    expect(node).toEqual({ kind: "em", children: [{ kind: "text", text: "Примітка:" }] });
  });

  it("підкреслення НЕ курсив — інакше ламаються chunk_uid і snake_case", () => {
    expect(parseInline("поле chunk_uid і snake_case")).toEqual([
      { kind: "text", text: "поле chunk_uid і snake_case" },
    ]);
  });

  it("екранована зірочка лишається символом", () => {
    expect(parseInline("5 \\* 3")).toEqual([{ kind: "text", text: "5 * 3" }]);
  });

  it("незакритий жирний поза стрімом лишається літералом", () => {
    expect(parseInline("**недописаний")).toEqual([{ kind: "text", text: "**недописаний" }]);
  });

  it("незакритий жирний під час стріму доростає", () => {
    expect(parseInline("**недописаний", true)).toEqual([
      { kind: "strong", children: [{ kind: "text", text: "недописаний" }] },
    ]);
  });
});

describe("формули", () => {
  it("одинична змінна", () => {
    expect(parseInline("кут ($\\beta$) між")).toContainEqual({
      kind: "math",
      tex: "\\beta",
      display: false,
    });
  });

  it("долар-валюта не стає формулою", () => {
    expect(parseInline("коштує $100 і $200")).toEqual([
      { kind: "text", text: "коштує $100 і $200" },
    ]);
  });

  it("блокова формула окремим блоком", () => {
    const [block] = parseMarkdown("$$\\frac{l}{D} \\cdot 1000$$");
    expect(block).toEqual({ kind: "math", tex: "\\frac{l}{D} \\cdot 1000" });
  });
});

describe("блоки", () => {
  it("нумерований список", () => {
    const [block] = parseMarkdown("1. перший\n2. другий\n3. третій");
    expect(block.kind).toBe("list");
    if (block.kind !== "list") return;
    expect(block.ordered).toBe(true);
    expect(block.start).toBe(1);
    expect(block.items).toHaveLength(3);
  });

  it("марковий список через * і -", () => {
    const [block] = parseMarkdown("* перший\n* другий");
    expect(block.kind).toBe("list");
    if (block.kind !== "list") return;
    expect(block.ordered).toBe(false);
  });

  it("самотнє «1.» не створює списку — інакше він блимав би під час стріму", () => {
    const [block] = parseMarkdown("1.");
    expect(block.kind).toBe("para");
  });

  it("вкладеність за відступом", () => {
    const [block] = parseMarkdown("- верхній\n  - вкладений");
    if (block.kind !== "list") throw new Error("очікувався список");
    expect(block.items.map((item) => item.depth)).toEqual([0, 1]);
  });

  it("заголовок", () => {
    const [block] = parseMarkdown("## Розділ");
    expect(block).toEqual({
      kind: "heading",
      level: 2,
      children: [{ kind: "text", text: "Розділ" }],
    });
  });

  it("огорожа коду", () => {
    const [block] = parseMarkdown("```python\nx = 1\n```");
    expect(block).toEqual({ kind: "code", text: "x = 1", lang: "python" });
  });

  it("таблиця", () => {
    const [block] = parseMarkdown("| А | Б |\n|---|---|\n| 1 | 2 |");
    expect(block.kind).toBe("table");
    if (block.kind !== "table") return;
    expect(block.header).toHaveLength(2);
    expect(block.rows).toHaveLength(1);
  });

  it("одинарний перенос зберігається як окремий рядок абзацу", () => {
    const [block] = parseMarkdown("перший рядок\nдругий рядок");
    expect(block.kind).toBe("para");
    if (block.kind !== "para") return;
    expect(block.lines).toHaveLength(2);
  });

  it("порожній рядок розділяє абзаци", () => {
    expect(parseMarkdown("перший\n\nдругий")).toHaveLength(2);
  });
});

describe("реальна відповідь про формулу тисячної", () => {
  const ANSWER = [
    "Формула тисячних — це формула, яка виражає залежність між кутовими та лінійними величинами [1].",
    "",
    "За допомогою цієї формули можна розв'язувати три типи задач:",
    "1. **1-й тип:** Визначають відстань ($l$) між двома точками, знаючи кут ($\\beta$) та дальність ($D$) [1].",
    "2. **2-й тип:** Визначають значення кута ($\\beta$) у поділках кутоміра [1].",
    "",
    "*Примітка:* Лінійна величина збільшується на 5 % [1].",
  ].join("\n");

  it("розбирається на абзац, абзац, список, абзац", () => {
    const blocks = parseMarkdown(ANSWER);
    expect(blocks.map((block) => block.kind)).toEqual(["para", "para", "list", "para"]);
  });

  it("усі чотири маркери лишаються доступними для пігулок", () => {
    const collect = (nodes: Inline[]) => textLeaves(nodes).join("");
    const blocks = parseMarkdown(ANSWER);
    let markers = 0;
    for (const block of blocks) {
      const chunks =
        block.kind === "para"
          ? block.lines.map(collect)
          : block.kind === "list"
            ? block.items.flatMap((item) => item.lines.map(collect))
            : [];
      markers += chunks.join("").split("[1]").length - 1;
    }
    expect(markers).toBe(4);
  });
});

describe("toPlainText", () => {
  it("прибирає розмітку для буфера обміну", () => {
    expect(toPlainText("**Жирний** і *курсив* [1]")).toBe("Жирний і курсив [1]");
  });

  it("список стає читабельним", () => {
    expect(toPlainText("1. перший\n2. другий")).toBe("1. перший\n2. другий");
  });
});
