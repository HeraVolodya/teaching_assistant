import { describe, expect, it } from "vitest";

import { filenameFromDisposition } from "./download";

describe("filenameFromDisposition", () => {
  it("бере кириличну назву з filename* (саме так її віддає бекенд)", () => {
    const header =
      "attachment; filename=\"document.pptx\"; filename*=UTF-8''" +
      encodeURIComponent("Обов'язки старшого офіцера батареї.pptx");
    expect(filenameFromDisposition(header)).toBe("Обов'язки старшого офіцера батареї.pptx");
  });

  it("без filename* — проста назва", () => {
    expect(filenameFromDisposition('attachment; filename="table.xlsx"')).toBe("table.xlsx");
  });

  it("зіпсоване кодування не валить збереження, а падає на просту назву", () => {
    expect(filenameFromDisposition("attachment; filename=\"a.docx\"; filename*=UTF-8''%E0%A4%A")).toBe(
      "a.docx",
    );
  });

  it("немає заголовка — немає назви (UI підставить запасну)", () => {
    expect(filenameFromDisposition(null)).toBeNull();
    expect(filenameFromDisposition("inline")).toBeNull();
  });
});
