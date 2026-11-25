import { describe, expect, it } from "vitest";

import { splitHighlight } from "./sourceText";

const PAGE = [
  "# Аркуш «Норми»",
  "",
  "Аркуш «Норми», рядки 2–3.",
  "| Норматив | Час |",
  "| --- | --- |",
  "| Розгортання | 90 с |",
  "",
  "Аркуш «Норми», рядки 4–5.",
  "| Норматив | Час |",
  "| --- | --- |",
  "| Згортання | 70 с |",
].join("\n");

describe("підсвітка фрагмента в тексті", () => {
  it("розширює діапазон до цілих рядків, не розриваючи таблицю", () => {
    const from = PAGE.indexOf("рядки 4–5") + 3;
    const to = PAGE.indexOf("70 с") + 2;
    const parts = splitHighlight(PAGE, { from, to });
    expect(parts.marked.startsWith("Аркуш «Норми», рядки 4–5.")).toBe(true);
    expect(parts.marked.endsWith("| Згортання | 70 с |")).toBe(true);
    expect(parts.before + parts.marked + parts.after).toBe(PAGE);
    expect(parts.before).toContain("Розгортання");
    expect(parts.marked).not.toContain("Розгортання");
  });

  it("без діапазону — уся сторінка без рамки", () => {
    expect(splitHighlight(PAGE, null)).toEqual({ before: PAGE, marked: "", after: "" });
    expect(splitHighlight(PAGE, { from: 5, to: 5 }).marked).toBe("");
  });

  it("діапазон за межами тексту обрізається", () => {
    const parts = splitHighlight("один\nдва", { from: 5, to: 999 });
    expect(parts).toEqual({ before: "один\n", marked: "два", after: "" });
  });
});
