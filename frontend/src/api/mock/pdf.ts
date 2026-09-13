/**
 * Синтетичний PDF для демонстраційного режиму.
 *
 * НАВІЩО. Переглядач джерела — це половина цінності продукту: цитата, яку не
 * можна відкрити на потрібній сторінці з підсвіченим абзацом, залишається
 * обіцянкою. Але перевірити його без бекенда нічим, а класти в репозиторій
 * справжній підручник не можна (та й не варто — це десятки мегабайтів).
 * Тому демонстраційний режим генерує PDF сам, у пам'яті, з ТОЧНО ВІДОМИМИ
 * координатами абзаців — тобто підсвітку можна звірити оком, а не повірити
 * в неї.
 *
 * ЛАТИНКА В ТЕКСТІ — СВІДОМО. Кирилиця у вбудованих шрифтах PDF вимагає
 * власного шрифту з таблицею кодування; тягнути сюди кириличний TTF заради
 * демонстраційного файлу означало б додати мегабайт до збірки. Замість
 * тексту сторінка малює СМУГИ рядків — і саме на них лягає підсвітка, тобто
 * геометрія перевіряється рівно так само.
 */

export const PAGE_WIDTH = 595;
export const PAGE_HEIGHT = 842;

/** Скільки «рядків тексту» малюється на сторінці. */
const LINES_PER_PAGE = 26;
const LINE_TOP = 720;
const LINE_STEP = 24;
const LINE_LEFT = 62;
const LINE_WIDTH = 471;
const LINE_HEIGHT = 9;

/**
 * Координати рядка в системі PDF (початок — НИЖНІЙ лівий кут).
 * Демонстраційні цитати посилаються саме на них, тож підсвітка зобов'язана
 * лягти рівно на смугу; будь-який зсув видно негайно.
 */
export function demoLineBBox(page: number, line: number): {
  page: number;
  l: number;
  t: number;
  r: number;
  b: number;
} {
  const bottom = LINE_TOP - line * LINE_STEP;
  return { page, l: LINE_LEFT, b: bottom, r: LINE_LEFT + LINE_WIDTH, t: bottom + LINE_HEIGHT };
}

function pageContent(page: number, total: number): string {
  const parts: string[] = [
    "0.16 0.18 0.25 rg",
    `BT /F1 15 Tf ${LINE_LEFT} 780 Td (ASISTENT - demo document) Tj ET`,
    `BT /F1 10 Tf ${LINE_LEFT} 762 Td (page ${page} of ${total}) Tj ET`,
    "0.78 0.79 0.83 RG 0.6 w",
    `${LINE_LEFT} 752 m ${LINE_LEFT + LINE_WIDTH} 752 l S`,
  ];
  for (let line = 0; line < LINES_PER_PAGE; line += 1) {
    const box = demoLineBBox(page, line);
    if (box.b < 70) break;
    // Останній рядок абзацу коротший — так сторінка читається як текст,
    // а не як штрих-код, і зсув підсвітки на рядок стає помітним.
    const width = (line + 1) % 6 === 0 ? LINE_WIDTH * 0.55 : LINE_WIDTH;
    const shade = (line + 1) % 6 === 0 ? "0.88 0.88 0.9" : "0.83 0.83 0.86";
    parts.push(`${shade} rg`, `${box.l} ${box.b} ${width.toFixed(1)} ${LINE_HEIGHT} re f`);
  }
  parts.push(
    "0.55 0.56 0.62 rg",
    `BT /F1 9 Tf ${LINE_LEFT} 48 Td (Asistent demo corpus - generated locally) Tj ET`,
  );
  return parts.join("\n");
}

/**
 * Зібрати PDF із `pages` сторінок.
 *
 * Зміщення в таблиці xref рахуються по ДОВЖИНІ РЯДКА — це коректно лише
 * доки весь файл лишається ASCII. Саме тому в тексті немає кирилиці: один
 * двобайтовий символ зсунув би всю таблицю, і pdf.js відмовився б відкрити
 * файл із помилкою, яку майже неможливо пояснити.
 */
export function buildDemoPdf(pages: number): Uint8Array {
  const count = Math.max(1, Math.min(64, Math.trunc(pages)));
  const objects: string[] = [];

  const kids: string[] = [];
  for (let i = 0; i < count; i += 1) kids.push(`${4 + i * 2} 0 R`);

  objects.push("<< /Type /Catalog /Pages 2 0 R >>");
  objects.push(`<< /Type /Pages /Kids [${kids.join(" ")}] /Count ${count} >>`);
  objects.push("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>");

  for (let i = 0; i < count; i += 1) {
    const contentRef = 5 + i * 2;
    objects.push(
      `<< /Type /Page /Parent 2 0 R /MediaBox [0 0 ${PAGE_WIDTH} ${PAGE_HEIGHT}]` +
        ` /Resources << /Font << /F1 3 0 R >> >> /Contents ${contentRef} 0 R >>`,
    );
    const stream = pageContent(i + 1, count);
    objects.push(`<< /Length ${stream.length} >>\nstream\n${stream}\nendstream`);
  }

  let body = "%PDF-1.4\n";
  const offsets: number[] = [];
  objects.forEach((object, index) => {
    offsets.push(body.length);
    body += `${index + 1} 0 obj\n${object}\nendobj\n`;
  });

  const xrefStart = body.length;
  let xref = `xref\n0 ${objects.length + 1}\n0000000000 65535 f \n`;
  for (const offset of offsets) xref += `${String(offset).padStart(10, "0")} 00000 n \n`;
  const trailer =
    `trailer\n<< /Size ${objects.length + 1} /Root 1 0 R >>\nstartxref\n${xrefStart}\n%%EOF\n`;

  const text = body + xref + trailer;
  const bytes = new Uint8Array(text.length);
  for (let i = 0; i < text.length; i += 1) bytes[i] = text.charCodeAt(i) & 0xff;
  return bytes;
}
