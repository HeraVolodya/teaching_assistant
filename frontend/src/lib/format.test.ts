import { describe, expect, it } from "vitest";

import {
  DOCUMENTS,
  PAGES,
  formatBytes,
  formatDuration,
  formatPercent,
  formatScore,
  parseServerDate,
  plural,
  pluralize,
} from "./format";
import { describeError, diagnosticsText, isKnownErrorCode } from "./errors";
import { describeStage, isPendingStatus, stageLabel } from "./stages";

describe("plural", () => {
  it("відміняє за українськими правилами", () => {
    expect(plural(1, DOCUMENTS)).toBe("документ");
    expect(plural(2, DOCUMENTS)).toBe("документи");
    expect(plural(5, DOCUMENTS)).toBe("документів");
  });

  /** Без цього винятку виходить «11 документ» — класична помилка. */
  it("правильно відміняє 11–14", () => {
    for (const n of [11, 12, 13, 14]) expect(plural(n, DOCUMENTS)).toBe("документів");
    expect(plural(21, DOCUMENTS)).toBe("документ");
    expect(plural(22, PAGES)).toBe("сторінки");
    expect(plural(111, DOCUMENTS)).toBe("документів");
  });

  it("нуль — форма множини", () => {
    expect(plural(0, DOCUMENTS)).toBe("документів");
  });

  it("pluralize склеює число з формою", () => {
    expect(pluralize(3, DOCUMENTS)).toBe("3 документи");
  });
});

describe("formatBytes", () => {
  it("масштабує одиниці", () => {
    expect(formatBytes(512)).toBe("512 Б");
    expect(formatBytes(1024)).toBe("1,0 КБ");
    expect(formatBytes(48_233_984)).toContain("МБ");
    expect(formatBytes(7_800_000_000)).toContain("ГБ");
  });

  it("невідомий розмір не показується нулем", () => {
    expect(formatBytes(null)).toBe("—");
  });
});

describe("formatDuration", () => {
  /**
   * Округлення НАГОРУ обов'язкове: «залишилось ~0 хв» посеред
   * двадцятихвилинної індексації — це втрачена довіра, яку вже не повернути.
   */
  it("округлює нагору до одиниці, що показується", () => {
    expect(formatDuration(61)).toBe("2 хв");
    expect(formatDuration(1)).toBe("1 с");
    expect(formatDuration(0.2)).toBe("1 с");
  });

  it("переходить у години", () => {
    expect(formatDuration(3600)).toBe("1 год");
    expect(formatDuration(5400)).toBe("1 год 30 хв");
  });

  it("невідомий час не вигадується", () => {
    expect(formatDuration(null)).toBe("—");
    expect(formatDuration(-5)).toBe("—");
  });
});

describe("parseServerDate", () => {
  /**
   * Бекенд шле «YYYY-MM-DD HH:MM:SS» без зони (`utcnow()`), а браузер такий
   * рядок тлумачить як МІСЦЕВИЙ час. У Києві це три години різниці — тобто
   * «щойно» перетворюється на «3 год тому» на кожній картці.
   */
  it("тлумачить дату без зони як UTC", () => {
    const parsed = parseServerDate("2026-09-06 12:00:00");
    expect(parsed).toBe(Date.parse("2026-09-06T12:00:00Z"));
  });

  it("не ламається на сміттєвому рядку", () => {
    expect(parseServerDate("не дата")).toBeNull();
  });
});

describe("formatPercent / formatScore", () => {
  it("обрізає частку до діапазону", () => {
    expect(formatPercent(1.4)).toBe("100 %");
    expect(formatPercent(-1)).toBe("0 %");
  });

  it("скор — два знаки з комою", () => {
    expect(formatScore(0.8234)).toBe("0,82");
    expect(formatScore(null)).toBe("—");
  });
});

describe("describeError", () => {
  it("кожен відомий код має заголовок, пояснення й дію", () => {
    for (const code of [
      "NO_TEXT",
      "PARSE_FAILED",
      "OUT_OF_MEMORY",
      "UNSUPPORTED_FORMAT",
      "SCAN_LOW_OCR_CONFIDENCE",
      "LMSTUDIO_UNAVAILABLE",
    ]) {
      const described = describeError(code);
      expect(described.title.length).toBeGreaterThan(5);
      expect(described.explain.length).toBeGreaterThan(20);
      if (described.action !== "none") expect(described.actionLabel.length).toBeGreaterThan(3);
    }
  });

  /** Невідомий код не має давати порожній екран без наступного кроку. */
  it("невідомий код отримує осмислений запасний варіант", () => {
    const described = describeError("НЕВІДОМО_ЩО");
    expect(described.action).toBe("retry");
    expect(described.title.length).toBeGreaterThan(5);
    expect(isKnownErrorCode("НЕВІДОМО_ЩО")).toBe(false);
  });

  it("скасування користувачем не фарбується як збій", () => {
    expect(describeError("CANCELLED").benign).toBe(true);
  });

  it("жодне пояснення не містить технічного жаргону", () => {
    const forbidden = ["чанк", "ембед", "вектор", "парсер", "токенайзер", "індексац"];
    for (const code of ["NO_TEXT", "PARSE_FAILED", "UNSUPPORTED_FORMAT", "OUT_OF_MEMORY"]) {
      const text = describeError(code).explain.toLocaleLowerCase("uk");
      for (const word of forbidden) expect(text).not.toContain(word);
    }
  });
});

describe("diagnosticsText", () => {
  /**
   * Це військова академія: сам перелік завантажених матеріалів є
   * інформацією. Назва потрапляє в діагностику ЛИШЕ за явним дозволом.
   */
  it("не розкриває назву матеріалу без дозволу", () => {
    const text = diagnosticsText({
      code: "NO_TEXT",
      documentTitle: "Таблиці стрільби 2С1",
      includeTitle: false,
    });
    expect(text).not.toContain("2С1");
    expect(text).toContain("приховано");
  });

  it("включає назву за явним дозволом", () => {
    const text = diagnosticsText({
      code: "NO_TEXT",
      documentTitle: "Таблиці стрільби 2С1",
      includeTitle: true,
    });
    expect(text).toContain("Таблиці стрільби 2С1");
  });
});

describe("describeStage", () => {
  it("черга не показує смуги прогресу", () => {
    const stage = describeStage({ status: "QUEUED" });
    expect(stage.active).toBe(false);
    expect(stage.detail).toContain("Очікує");
  });

  /** Рядок етапу без числа після третьої хвилини читається як зависання. */
  it("додає номер сторінки, коли кількість сторінок відома", () => {
    const stage = describeStage({
      status: "PARSING",
      stage: "PARSE",
      fraction: 0.279,
      pageCount: 512,
    });
    expect(stage.detail).toBe("Читаю сторінки — сторінка 143 з 512");
  });

  it("не вигадує номер сторінки, коли їх кількість ще невідома", () => {
    const stage = describeStage({ status: "PARSING", stage: "PARSE", fraction: 0.3, pageCount: 0 });
    expect(stage.detail).not.toMatch(/сторінка \d/);
    expect(stage.detail).toContain("%");
  });

  it("для інших етапів показує відсоток", () => {
    const stage = describeStage({ status: "INDEXING", stage: "INDEX", fraction: 0.5, pageCount: 100 });
    expect(stage.detail).toBe("Готую пошук — 50 %");
  });

  it("назви етапів не містять технічних слів", () => {
    for (const raw of ["PROBE", "PARSE", "CHUNK", "INDEX"]) {
      const label = stageLabel(raw).toLocaleLowerCase("uk");
      expect(label).not.toContain("парс");
      expect(label).not.toContain("чанк");
      expect(label).not.toContain("індекс");
    }
  });

  it("розрізняє активні й завершені стани", () => {
    expect(isPendingStatus("QUEUED")).toBe(true);
    expect(isPendingStatus("PARSING")).toBe(true);
    expect(isPendingStatus("READY")).toBe(false);
    expect(isPendingStatus("FAILED")).toBe(false);
  });
});
