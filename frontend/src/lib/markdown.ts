/**
 * Мінімальний Markdown для відповіді асистента.
 *
 * ЧОМУ СВІЙ, А НЕ react-markdown.
 * Застосунок постачається в закритий контур: на цільовій машині немає ні
 * мережі, ні доступу до реєстру npm. Додати залежність — це не `npm i`, а
 * перенесення ~60 пакетів дерева unified/micromark у вендор із власним
 * аудитом. Обсяг розмітки, яку реально пише Gemma, — жирний, курсив, списки,
 * заголовки, код, таблиці й поодинокі $-формули; це 300 рядків, які можна
 * покрити тестами повністю. Проєкт уже так робить: свій SSE-парсер, свої
 * українські форми множини замість Intl.PluralRules, свій медіанний ETA.
 *
 * ДЕРЕВО, А НЕ РЯДОК HTML — І ЦЕ ГОЛОВНЕ.
 * Парсер НІКОЛИ не віддає HTML. Він повертає вузли, і кожен листок типу
 * `text` іде далі в `renderInline` з AnswerText, який і робить пігулки-цитати.
 * Тому інваріант «[n] стає клікабельним джерелом» зберігається структурно:
 * дістатися до тексту повз renderInline просто нічим. Побічний виграш — у
 * `code` і `math` листків немає взагалі, тож `[1]` у прикладі коду більше не
 * вдає посилання, а `\sqrt[3]{x}` не втрачає свою трійку.
 *
 * ЩО СВІДОМО НЕ ПІДТРИМАНО (і як воно виглядає, коли трапиться):
 *   * `[текст](url)` — синтаксис прямо конфліктує з маркером цитати `[n]`.
 *     Лишається видимим текстом. Моделі тут посилань і не пишуть: у промпті
 *     джерела нумеровані, а не адресовані.
 *   * `_курсив_` — підкреслення ламає `chunk_uid`, `snake_case` і назви
 *     файлів, які в цьому корпусі трапляються постійно. Курсив лише через `*`.
 *   * Цитати блоками `>` і вкладеність глибше за один рівень — списки
 *     сплощуються до двох рівнів.
 * Кожен наступний пункт сюди варто додавати свідомо, інакше за півроку це
 * стане власним markdown-it без його тестів.
 *
 * НЕДОПИСАНА РОЗМІТКА ПІД ЧАС СТРІМУ.
 * Токени приходять по частинах, тож `**жир` без пари — норма, а не помилка.
 * У режимі `streaming` незакритий відкривач вважається відкритим до кінця
 * буфера: жирне «доростає» замість того, щоб блимати зірочками. Після
 * завершення (`streaming: false`) незакритий відкривач лишається літералом —
 * те, що модель написала, ми не домальовуємо.
 */

export type Inline =
  | { kind: "text"; text: string }
  | { kind: "strong"; children: Inline[] }
  | { kind: "em"; children: Inline[] }
  | { kind: "code"; text: string }
  | { kind: "math"; tex: string; display: boolean };

export interface ListItem {
  /** Рядки пункту: перший — сам пункт, решта — його продовження. */
  lines: Inline[][];
  /** 0 — верхній рівень, 1 — вкладений. Глибше не розрізняємо. */
  depth: number;
}

export type Block =
  | { kind: "para"; lines: Inline[][] }
  | { kind: "heading"; level: number; children: Inline[] }
  | { kind: "list"; ordered: boolean; start: number; items: ListItem[] }
  | { kind: "code"; text: string; lang: string }
  | { kind: "table"; header: Inline[][]; rows: Inline[][][] }
  | { kind: "math"; tex: string };

export interface ParseOptions {
  /** Відповідь ще набирається — див. «НЕДОПИСАНА РОЗМІТКА» в шапці. */
  streaming?: boolean;
}

const FENCE = /^\s*```(\w*)\s*$/;
const HEADING = /^(#{1,6})\s+(?=\S)(.*)$/;
// `(?=\S)` обов'язковий: рядок «1.», що прийшов сам, не має миттєво
// створювати порожній список, який зникне з наступним токеном.
const BULLET = /^(\s*)[-*•]\s+(?=\S)(.*)$/;
const ORDERED = /^(\s*)(\d{1,3})[.)]\s+(?=\S)(.*)$/;
const TABLE_ROW = /^\s*\|(.+)\|\s*$/;
const TABLE_SEP = /^\s*\|[\s:|-]+\|\s*$/;
const DISPLAY_MATH = /^\s*\$\$(.+?)\$\$\s*$/;

/** Рядок порожній або лише з пробілів. */
const blank = (line: string) => line.trim() === "";

/**
 * Розбір відповіді на блоки.
 *
 * Одинарний перенос рядка зберігається як окремий рядок усередині абзацу —
 * не як у CommonMark, де він склеюється в пробіл. Причина прикладна: модель
 * форматує перелічення саме переносами, і склеювання їх у суцільний абзац
 * зробило б відповідь помітно гіршою за теперішній `whitespace-pre-wrap`.
 */
export function parseMarkdown(text: string, options: ParseOptions = {}): Block[] {
  const streaming = options.streaming ?? false;
  const lines = text.replace(/\r\n?/g, "\n").split("\n");
  const blocks: Block[] = [];
  let index = 0;

  while (index < lines.length) {
    const line = lines[index];

    if (blank(line)) {
      index += 1;
      continue;
    }

    const fence = FENCE.exec(line);
    if (fence) {
      const lang = fence[1] ?? "";
      const body: string[] = [];
      index += 1;
      while (index < lines.length && !FENCE.test(lines[index])) {
        body.push(lines[index]);
        index += 1;
      }
      // Кінець буфера без закривальної огорожі — під час стріму це нормально.
      if (index < lines.length) index += 1;
      blocks.push({ kind: "code", text: body.join("\n"), lang });
      continue;
    }

    const display = DISPLAY_MATH.exec(line);
    if (display) {
      blocks.push({ kind: "math", tex: display[1].trim() });
      index += 1;
      continue;
    }

    const heading = HEADING.exec(line);
    if (heading) {
      blocks.push({
        kind: "heading",
        level: heading[1].length,
        children: parseInline(heading[2].trim(), streaming),
      });
      index += 1;
      continue;
    }

    if (TABLE_ROW.test(line) && index + 1 < lines.length && TABLE_SEP.test(lines[index + 1])) {
      const header = splitRow(lines[index], streaming);
      index += 2;
      const rows: Inline[][][] = [];
      while (index < lines.length && TABLE_ROW.test(lines[index]) && !TABLE_SEP.test(lines[index])) {
        rows.push(splitRow(lines[index], streaming));
        index += 1;
      }
      blocks.push({ kind: "table", header, rows });
      continue;
    }

    if (BULLET.test(line) || ORDERED.test(line)) {
      const first = ORDERED.exec(line);
      const ordered = first !== null;
      const start = first ? Number.parseInt(first[2], 10) : 1;
      const items: ListItem[] = [];

      while (index < lines.length) {
        const current = lines[index];
        const bullet = BULLET.exec(current);
        const numbered = ORDERED.exec(current);
        if (!bullet && !numbered) break;
        // Не змішуємо марковані й нумеровані в одному блоці: інакше після
        // «1. 2. 3.» дефіс мовчки продовжив би нумерацію.
        if (ordered !== (numbered !== null)) break;

        const match = (numbered ?? bullet) as RegExpExecArray;
        const indent = match[1].length;
        const body = numbered ? match[3] : match[2];
        const item: ListItem = {
          lines: [parseInline(body, streaming)],
          depth: indent >= 2 ? 1 : 0,
        };
        index += 1;

        // Продовження пункту: непорожній рядок із відступом, який сам не є
        // новим пунктом.
        while (index < lines.length) {
          const next = lines[index];
          if (blank(next) || BULLET.test(next) || ORDERED.test(next)) break;
          if (!/^\s+\S/.test(next)) break;
          item.lines.push(parseInline(next.trim(), streaming));
          index += 1;
        }
        items.push(item);
      }
      blocks.push({ kind: "list", ordered, start, items });
      continue;
    }

    const paragraph: Inline[][] = [];
    while (index < lines.length) {
      const current = lines[index];
      if (blank(current)) break;
      if (HEADING.test(current) || FENCE.test(current)) break;
      if (BULLET.test(current) || ORDERED.test(current)) break;
      if (TABLE_ROW.test(current) && index + 1 < lines.length && TABLE_SEP.test(lines[index + 1])) break;
      paragraph.push(parseInline(current, streaming));
      index += 1;
    }
    if (paragraph.length) blocks.push({ kind: "para", lines: paragraph });
  }

  return blocks;
}

function splitRow(line: string, streaming: boolean): Inline[][] {
  const inner = (TABLE_ROW.exec(line) as RegExpExecArray)[1];
  return inner.split("|").map((cell) => parseInline(cell.trim(), streaming));
}

/**
 * Інлайновий розбір.
 *
 * ПОРЯДОК МАЄ ЗНАЧЕННЯ: спершу відрізки, всередину яких не можна лізти
 * (`$формула$`, `` `код` ``), і лише потім наголоси. Інакше `**` усередині
 * LaTeX-виразу на кшталт `$a**b$` розірве формулу навпіл, а `[1]` у прикладі
 * коду стане пігулкою-цитатою.
 */
export function parseInline(text: string, streaming = false): Inline[] {
  const out: Inline[] = [];
  let buffer = "";

  const flush = () => {
    if (buffer) {
      out.push({ kind: "text", text: buffer });
      buffer = "";
    }
  };

  let i = 0;
  while (i < text.length) {
    const rest = text.slice(i);

    // Екранування: `\*` показує зірочку буквально.
    if (rest[0] === "\\" && rest.length > 1 && "*`$\\".includes(rest[1])) {
      buffer += rest[1];
      i += 2;
      continue;
    }

    if (rest.startsWith("$$")) {
      const end = rest.indexOf("$$", 2);
      if (end > 2) {
        flush();
        out.push({ kind: "math", tex: rest.slice(2, end).trim(), display: true });
        i += end + 2;
        continue;
      }
    }

    if (rest[0] === "$") {
      const tex = takeMath(rest);
      if (tex !== null) {
        flush();
        out.push({ kind: "math", tex: tex.trim(), display: false });
        i += tex.length + 2;
        continue;
      }
    }

    if (rest[0] === "`") {
      const end = rest.indexOf("`", 1);
      if (end > 1) {
        flush();
        out.push({ kind: "code", text: rest.slice(1, end) });
        i += end + 1;
        continue;
      }
    }

    if (rest.startsWith("**")) {
      const end = rest.indexOf("**", 2);
      if (end > 2) {
        flush();
        out.push({ kind: "strong", children: parseInline(rest.slice(2, end), streaming) });
        i += end + 2;
        continue;
      }
      if (streaming) {
        // Пара ще не прийшла — жирне доростає до кінця буфера.
        flush();
        out.push({ kind: "strong", children: parseInline(rest.slice(2), streaming) });
        break;
      }
    }

    if (rest[0] === "*" && rest[1] !== "*" && rest[1] !== " ") {
      const end = findEm(rest);
      if (end > 1) {
        flush();
        out.push({ kind: "em", children: parseInline(rest.slice(1, end), streaming) });
        i += end + 1;
        continue;
      }
      if (streaming) {
        flush();
        out.push({ kind: "em", children: parseInline(rest.slice(1), streaming) });
        break;
      }
    }

    buffer += rest[0];
    i += 1;
  }

  flush();
  return out;
}

/**
 * Вміст `$…$`, або null, якщо це не формула.
 *
 * Долар-валюта — головне джерело хибних спрацювань: «коштує $100 і $200»
 * не має ставати формулою. Тому відрізок мусить закритися в ТОМУ Ж рядку,
 * не починатися з пробілу й не складатися лише з цифр і розділових.
 */
function takeMath(rest: string): string | null {
  const end = rest.indexOf("$", 1);
  if (end <= 1) return null;
  const body = rest.slice(1, end);
  if (body.includes("\n")) return null;
  if (/^\s|\s$/.test(body)) return null;
  if (!/[a-zA-Z\\]/.test(body)) return null;
  return body;
}

/** Кінець `*курсиву*`: закривальна зірочка, перед якою не пробіл. */
function findEm(rest: string): number {
  for (let i = 1; i < rest.length; i += 1) {
    if (rest[i] === "*" && rest[i - 1] !== " " && rest[i - 1] !== "\\") return i;
  }
  return -1;
}

/**
 * Текст без розмітки — для кнопки «Копіювати».
 *
 * Після появи рендера видиме й скопійоване розійшлися б: на екрані жирний
 * заголовок, у буфері — `**заголовок**`. Викладач копіює у конспект, а не в
 * markdown-редактор, тому копіюємо те, що бачить.
 */
export function toPlainText(text: string): string {
  return parseMarkdown(text)
    .map((block) => blockToPlain(block))
    .filter((part) => part !== "")
    .join("\n\n");
}

function blockToPlain(block: Block): string {
  switch (block.kind) {
    case "para":
      return block.lines.map(inlineToPlain).join("\n");
    case "heading":
      return inlineToPlain(block.children);
    case "code":
      return block.text;
    case "math":
      return block.tex;
    case "list":
      return block.items
        .map((item, index) => {
          const marker = block.ordered ? `${block.start + index}.` : "•";
          const pad = item.depth ? "    " : "";
          return `${pad}${marker} ${item.lines.map(inlineToPlain).join(" ")}`;
        })
        .join("\n");
    case "table":
      return [block.header, ...block.rows]
        .map((row) => row.map(inlineToPlain).join("\t"))
        .join("\n");
  }
}

function inlineToPlain(nodes: Inline[]): string {
  return nodes
    .map((node) => {
      switch (node.kind) {
        case "text":
          return node.text;
        case "code":
          return node.text;
        case "math":
          return node.tex;
        case "strong":
        case "em":
          return inlineToPlain(node.children);
      }
    })
    .join("");
}
