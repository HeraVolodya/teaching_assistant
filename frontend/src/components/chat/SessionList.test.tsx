/**
 * @vitest-environment jsdom
 *
 * Панель історії розмов.
 *
 * Перевіряється не верстка, а три речі, які легко зламати мовчки: безіменна
 * розмова мусить мати ЧЕСНИЙ підпис (а не заголовок від сусідньої), вибір
 * розмови мусить доходити до батька, а видалення — питати підтвердження.
 * Остання вимога не косметична: розмова видаляється разом з усіма
 * повідомленнями й без можливості повернути.
 */

import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeAll, describe, expect, it } from "vitest";

import type { ChatSession } from "@/api/types";
import { initI18n } from "@/i18n";

import { SessionList } from "./SessionList";

const SESSIONS: ChatSession[] = [
  {
    id: "s1",
    assistantId: "a1",
    title: "Формула тисячної",
    createdAt: "2026-09-13 10:00:00",
    updatedAt: "2026-09-13 10:05:00",
    messages: 4,
  },
  {
    id: "s2",
    assistantId: "a1",
    title: "",
    createdAt: "2026-09-13 09:00:00",
    updatedAt: "2026-09-13 09:00:00",
    messages: 0,
  },
];

let host: HTMLDivElement | null = null;
let root: Root | null = null;

beforeAll(async () => {
  await initI18n();
});

function render(node: React.ReactElement): HTMLDivElement {
  host = document.createElement("div");
  document.body.append(host);
  root = createRoot(host);
  act(() => {
    root?.render(node);
  });
  return host;
}

afterEach(() => {
  act(() => root?.unmount());
  host?.remove();
  root = null;
  host = null;
});

const noop = () => undefined;

describe("SessionList", () => {
  it("показує назви розмов", () => {
    const el = render(
      <SessionList sessions={SESSIONS} activeId="s1" onPick={noop} onDelete={noop} onClose={noop} />,
    );
    expect(el.textContent).toContain("Формула тисячної");
  });

  it("безіменна розмова має чесний підпис, а не чужий заголовок", () => {
    const el = render(
      <SessionList sessions={SESSIONS} activeId="s1" onPick={noop} onDelete={noop} onClose={noop} />,
    );
    expect(el.textContent).toContain("Без назви");
    expect(el.textContent).toContain("Ще без питань");
  });

  it("вибір розмови доходить до батька", () => {
    const picked: string[] = [];
    const el = render(
      <SessionList
        sessions={SESSIONS}
        activeId="s1"
        onPick={(id) => picked.push(id)}
        onDelete={noop}
        onClose={noop}
      />,
    );
    const buttons = [...el.querySelectorAll("li button")].filter(
      (b) => !b.getAttribute("aria-label")?.includes("Видалити"),
    );
    act(() => buttons[1]?.dispatchEvent(new MouseEvent("click", { bubbles: true })));
    expect(picked).toEqual(["s2"]);
  });

  it("видалення НЕ відбувається без підтвердження", () => {
    const deleted: string[] = [];
    const el = render(
      <SessionList
        sessions={SESSIONS}
        activeId="s1"
        onPick={noop}
        onDelete={(id) => deleted.push(id)}
        onClose={noop}
      />,
    );
    const trash = el.querySelector('li button[aria-label="Видалити розмову"]');
    expect(trash).not.toBeNull();
    act(() => trash?.dispatchEvent(new MouseEvent("click", { bubbles: true })));
    // Клік по кошику лише відкриває діалог — сама розмова ще ціла.
    expect(deleted).toEqual([]);
  });

  it("порожня історія каже про це прямо", () => {
    const el = render(
      <SessionList sessions={[]} activeId={undefined} onPick={noop} onDelete={noop} onClose={noop} />,
    );
    expect(el.textContent).toContain("Попередніх розмов ще немає");
  });
});
