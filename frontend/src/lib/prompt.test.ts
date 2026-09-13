import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import {
  COMPILED_MARKER,
  PERSONA_MAX_TOKENS,
  PROMPT_BLOCKS,
  SYSTEM_RULES_UK,
  appendBlock,
  compileInstructions,
  compilePromptPreview,
  estimateTokensUk,
  retrievalPolicy,
  splitInstructions,
} from "./prompt";

const BASE = {
  topicsCovered: [],
  topicsRefused: [],
  onMissing: "no_information" as const,
  answerLanguage: "match_question" as const,
  alwaysCitePages: false,
};

describe("SYSTEM_RULES_UK", () => {
  /**
   * НАЙВАЖЛИВІШИЙ ТЕСТ У ФАЙЛІ.
   *
   * Префікс мусить бути БАЙТ-ІДЕНТИЧНИМ до `prompt_builder.py`: LM Studio
   * перевикористовує префіксний KV-кеш, і розходження в один символ
   * інвалідує його для всіх асистентів одразу — тобто повертає повний
   * prefill на кожне перемикання. Помітити це в інтерфейсі неможливо: він
   * просто стає повільнішим. Тому рядок звіряється з першоджерелом.
   */
  it("збігається з бекендом байт у байт", () => {
    const path = fileURLToPath(
      new URL("../../../backend/app/generation/prompt_builder.py", import.meta.url),
    );
    let source: string;
    try {
      source = readFileSync(path, "utf8");
    } catch {
      // У збірці без бекенда (наприклад, у контейнері фронтенду) звіряти
      // нема з чим — тест не має падати з цієї причини.
      return;
    }
    const match = /SYSTEM_RULES_UK = \(\n([\s\S]*?)\n\)\n/.exec(source);
    expect(match, "SYSTEM_RULES_UK не знайдено у prompt_builder.py").not.toBeNull();

    // Склеюємо рядковий літерал Python: послідовність рядків у дужках.
    const python = (match as RegExpExecArray)[1]
      .split("\n")
      .map((line) => line.trim())
      .join("")
      .replace(/"\s*"/g, "")
      .replace(/^"|"$/g, "")
      .replace(/\\n/g, "\n");

    expect(SYSTEM_RULES_UK).toBe(python);
  });

  it("містить рівно п'ять правил", () => {
    expect(SYSTEM_RULES_UK.split("\n").filter((line) => /^\d\./.test(line))).toHaveLength(5);
  });

  it("вкладається в бюджет 180 токенів", () => {
    expect(estimateTokensUk(SYSTEM_RULES_UK)).toBeLessThanOrEqual(180);
  });
});

describe("estimateTokensUk", () => {
  it("порожній текст — нуль токенів", () => {
    expect(estimateTokensUk("")).toBe(0);
  });

  it("бере максимум із двох оцінок, тобто не занижує", () => {
    // Довге слово без пробілів: оцінка за словами дала б 2, за символами — 15.
    const long = "а".repeat(48);
    expect(estimateTokensUk(long)).toBeGreaterThan(10);
  });

  it("зростає монотонно з довжиною", () => {
    const short = estimateTokensUk("коротко");
    const long = estimateTokensUk("коротко ".repeat(20));
    expect(long).toBeGreaterThan(short);
  });
});

describe("retrievalPolicy", () => {
  /**
   * Числа мусять збігатися з `AssistantConfig.confidence_threshold()` і
   * `min_supporting_chunks()`. Розходження тут означає, що редактор показує
   * викладачеві не той поріг, за яким система реально відмовляється
   * відповідати, — тобто бреше про найважливішу свою властивість.
   */
  it("відтворює пороги бекенда", () => {
    expect(retrievalPolicy({ confidence_required: "low" })).toMatchObject({
      threshold: 0.15,
      minSupporting: 1,
    });
    expect(retrievalPolicy({ confidence_required: "medium" })).toMatchObject({
      threshold: 0.35,
      minSupporting: 1,
    });
    expect(retrievalPolicy({ confidence_required: "high" })).toMatchObject({
      threshold: 0.55,
      minSupporting: 2,
    });
  });
});

describe("compileInstructions", () => {
  it("ставить текст викладача першим", () => {
    const result = compileInstructions("Мій текст.", BASE);
    expect(result.indexOf("Мій текст.")).toBe(0);
    expect(result.indexOf(COMPILED_MARKER)).toBeGreaterThan(0);
  });

  it("переносить теми в текст інструкцій", () => {
    const result = compileInstructions("", {
      ...BASE,
      topicsCovered: ["балістика", "топографія"],
      topicsRefused: ["кадрові питання"],
    });
    expect(result).toContain("балістика, топографія");
    expect(result).toContain("кадрові питання");
  });

  it("прибирає порожні й повторні теми без огляду на регістр", () => {
    const result = compileInstructions("", {
      ...BASE,
      topicsCovered: ["Балістика", "  ", "балістика", "Топографія"],
    });
    expect(result).toContain("Балістика, Топографія");
    expect(result).not.toContain("Балістика, балістика");
  });

  it("кожен варіант «немає відповіді» дає свій текст", () => {
    const a = compileInstructions("", { ...BASE, onMissing: "no_information" });
    const b = compileInstructions("", { ...BASE, onMissing: "general_knowledge_marked" });
    const c = compileInstructions("", { ...BASE, onMissing: "refuse" });
    expect(new Set([a, b, c]).size).toBe(3);
    expect(b).toContain("За межами завантажених матеріалів");
  });

  /**
   * Регресія, заради якої існує маркер. Без нього кожне збереження
   * дописувало б службовий блок ще раз, і через кілька правок персона
   * складалася б із копій переліку тем — гарантовано пробиваючи ліміт 250
   * токенів і обрізаючись рівно посередині.
   */
  it("не накопичує службовий блок при повторних збереженнях", () => {
    const structured = { ...BASE, topicsCovered: ["балістика"] };
    let stored = compileInstructions("Текст викладача.", structured);
    for (let round = 0; round < 5; round += 1) {
      const { free } = splitInstructions(stored);
      stored = compileInstructions(free, structured);
    }
    expect(stored.split(COMPILED_MARKER)).toHaveLength(2);
    expect(stored.match(/Твоя предметна область/g)).toHaveLength(1);
    expect(stored.startsWith("Текст викладача.")).toBe(true);
  });

  it("splitInstructions повертає ручний текст як є, коли маркера немає", () => {
    const raw = "Довільний текст, написаний вручну через режим розробника.";
    expect(splitInstructions(raw)).toEqual({ free: raw, compiled: "" });
  });
});

describe("compilePromptPreview", () => {
  it("без персони віддає лише правила", () => {
    const preview = compilePromptPreview("", { confidence_required: "medium" });
    expect(preview.system).toBe(SYSTEM_RULES_UK);
    expect(preview.overLimit).toBe(false);
  });

  it("персона йде ПІСЛЯ правил, а не перед ними", () => {
    const preview = compilePromptPreview("Роль.", { confidence_required: "medium" });
    expect(preview.system.indexOf(SYSTEM_RULES_UK)).toBe(0);
    expect(preview.system).toContain("РОЛЬ АСИСТЕНТА:");
  });

  it("позначає перевищення ліміту персони", () => {
    const huge = "дуже довгий опис ролі ".repeat(80);
    const preview = compilePromptPreview(huge, { confidence_required: "low" });
    expect(preview.personaTokens).toBeGreaterThan(PERSONA_MAX_TOKENS);
    expect(preview.overLimit).toBe(true);
  });
});

describe("PROMPT_BLOCKS", () => {
  it("усі ідентифікатори унікальні", () => {
    const ids = PROMPT_BLOCKS.flatMap((group) => group.blocks.map((block) => block.id));
    expect(new Set(ids).size).toBe(ids.length);
  });

  it("кожен блок — коротке наказове речення", () => {
    for (const group of PROMPT_BLOCKS) {
      for (const block of group.blocks) {
        expect(estimateTokensUk(block.text)).toBeLessThan(60);
        expect(block.text.endsWith(".")).toBe(true);
      }
    }
  });

  it("appendBlock не дублює вже доданий блок", () => {
    const block = PROMPT_BLOCKS[0].blocks[0];
    const once = appendBlock("", block);
    expect(appendBlock(once, block)).toBe(once);
  });
});
