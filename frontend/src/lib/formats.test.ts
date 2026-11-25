import { describe, expect, it } from "vitest";

import { ACCEPT_ATTRIBUTE, isAcceptedFile, pageUnit, splitAccepted } from "./formats";

describe("формати бази знань", () => {
  it("приймає Word, PowerPoint, DjVu, Excel і XML поряд із PDF", () => {
    for (const name of [
      "Підручник.PDF",
      "Методичка.docx",
      "Наказ.doc",
      "Лекція.pptx",
      "Настанова.djvu",
      "Нормативи.xlsx",
      "Старі норми.xls",
      "Журнал.csv",
      "Каталог.xml",
      "Конспект.md",
    ]) {
      expect(isAcceptedFile(name), name).toBe(true);
    }
  });

  it("відкидає невідоме й називає відкинуте", () => {
    const files = [{ name: "a.docx" }, { name: "setup.exe" }, { name: "без розширення" }];
    const { accepted, rejected } = splitAccepted(files);
    expect(accepted.map((file) => file.name)).toEqual(["a.docx"]);
    expect(rejected).toEqual(["setup.exe", "без розширення"]);
  });

  it("атрибут accept містить кожне розширення", () => {
    expect(ACCEPT_ATTRIBUTE.split(",")).toContain(".djvu");
    expect(ACCEPT_ATTRIBUTE.split(",")).toContain(".xls");
  });

  it("одиниця сторінки: аркуші для таблиць, слайди для презентацій", () => {
    expect(pageUnit("Норми.xlsx")).toBe("арк.");
    expect(pageUnit("Лекція.pptx")).toBe("слайд.");
    expect(pageUnit("Підручник.pdf")).toBe("с.");
  });
});
