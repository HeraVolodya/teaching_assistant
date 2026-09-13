/**
 * Розбір SSE — місце, де ламається стрімінг відповіді.
 *
 * Помилка тут не виглядає як помилка: вона виглядає як «модель іноді ковтає
 * слова» або «відповідь інколи обривається». Тому кожен окремий випадок із
 * реального кадру бекенда має власний тест.
 */

import { describe, expect, it } from "vitest";

import { parseFrame, readSse } from "./client";

function response(chunks: string[]): Response {
  const encoder = new TextEncoder();
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
      controller.close();
    },
  });
  return new Response(stream);
}

async function collect(chunks: string[]) {
  const out: { id: number | null; event: string; data: string }[] = [];
  for await (const frame of readSse(response(chunks))) out.push(frame);
  return out;
}

describe("parseFrame", () => {
  it("розбирає кадр із id та подією", () => {
    expect(parseFrame('id: 42\nevent: chat.token\ndata: {"delta":"а"}')).toEqual({
      id: 42,
      event: "chat.token",
      data: '{"delta":"а"}',
    });
  });

  it("ігнорує коментар-пульс", () => {
    expect(parseFrame(": ping")).toBeNull();
  });

  it("прибирає рівно один пробіл після data:", () => {
    // Другий пробіл — частина даних. У токені відповіді це видимий пробіл,
    // і з'їсти його означає склеїти слова у відповіді.
    expect(parseFrame("data:  два пробіли")?.data).toBe(" два пробіли");
  });

  it("кадр без data не є кадром", () => {
    expect(parseFrame("event: chat.done")).toBeNull();
  });

  it("нечисловий id стає null, а не NaN", () => {
    expect(parseFrame("id: abc\ndata: 1")?.id).toBeNull();
  });
});

describe("readSse", () => {
  it("читає послідовність кадрів", async () => {
    const frames = await collect([
      'event: chat.token\ndata: {"delta":"Тра"}\n\n',
      'event: chat.token\ndata: {"delta":"єкторія"}\n\n',
      'event: chat.done\ndata: {"tokensOut":2}\n\n',
    ]);
    expect(frames.map((frame) => frame.event)).toEqual([
      "chat.token",
      "chat.token",
      "chat.done",
    ]);
  });

  /**
   * Токен, розрізаний межею мережевого пакета, — норма, а не виняток:
   * українська в UTF-8 дає два байти на літеру, і TextDecoder зі
   * `stream: true` існує саме для цього. Без нього посеред відповіді
   * з'являються ромбики з питальним знаком.
   */
  it("склеює кадр, розрізаний між читаннями", async () => {
    const frames = await collect(['event: chat.token\nda', 'ta: {"delta":"Тра', 'єкторія"}\n\n']);
    expect(frames).toHaveLength(1);
    expect(JSON.parse(frames[0].data)).toEqual({ delta: "Траєкторія" });
  });

  /**
   * Windows-проксі перетворює \n\n на \r\n\r\n. Зсув на фіксовані два
   * символи залишав би \r на початку наступного кадру, і `event:` більше не
   * збігався б — тобто стрім працював би на macOS і мовчав на цільовій
   * платформі.
   */
  it("розуміє роздільник \\r\\n\\r\\n", async () => {
    const frames = await collect([
      'event: chat.token\r\ndata: {"delta":"а"}\r\n\r\n',
      'event: chat.done\r\ndata: {}\r\n\r\n',
    ]);
    expect(frames.map((frame) => frame.event)).toEqual(["chat.token", "chat.done"]);
  });

  it("віддає останній кадр без завершального роздільника", async () => {
    const frames = await collect(['event: chat.done\ndata: {"ok":true}']);
    expect(frames).toHaveLength(1);
    expect(frames[0].event).toBe("chat.done");
  });

  it("пульс між кадрами не породжує порожніх подій", async () => {
    const frames = await collect([
      ": ping\n\n",
      'event: job.progress\ndata: {"fraction":0.5}\n\n',
      ": ping\n\n",
    ]);
    expect(frames).toHaveLength(1);
  });

  it("зберігає id для відновлення після обриву", async () => {
    const frames = await collect([
      'id: 7\nevent: doc.ready\ndata: {"docId":"a"}\n\n',
      'id: 8\nevent: doc.ready\ndata: {"docId":"b"}\n\n',
    ]);
    expect(frames.map((frame) => frame.id)).toEqual([7, 8]);
  });
});
