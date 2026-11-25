/**
 * Підсвітка цитованого фрагмента в тексті чи аркуші таблиці.
 *
 * Бекенд віддає діапазон символів фрагмента всередині сторінки. Різати
 * Markdown посеред рядка не можна — рядок таблиці розпався б на дві
 * нечитні половини, — тому діапазон розширюється до меж рядків, і сторінка
 * малюється трьома шматками: до, сам фрагмент (у рамці) і після.
 */

export interface HighlightParts {
  before: string;
  marked: string;
  after: string;
}

export function splitHighlight(
  text: string,
  highlight: { from: number; to: number } | null | undefined,
): HighlightParts {
  if (!highlight || highlight.to <= highlight.from) {
    return { before: text, marked: "", after: "" };
  }
  const from = Math.max(0, Math.min(text.length, highlight.from));
  const to = Math.max(from, Math.min(text.length, highlight.to));
  const start = text.lastIndexOf("\n", from - 1) + 1;
  const newline = text.indexOf("\n", Math.max(start, to - 1));
  const end = newline === -1 ? text.length : newline;
  return {
    before: text.slice(0, start),
    marked: text.slice(start, end),
    after: text.slice(end),
  };
}
