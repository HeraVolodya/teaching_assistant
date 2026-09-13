import { defineConfig } from "vitest/config";
import { fileURLToPath, URL } from "node:url";

// Тестуємо ЧИСТУ логіку (ETA, компіляція промпту, геометрія bbox, мок-сервер),
// а не рендер: рендер-тести без реального бекенда перевіряють переважно
// самі себе, а логіка вище — це те, де живуть справжні помилки.
export default defineConfig({
  resolve: {
    alias: { "@": fileURLToPath(new URL("./src", import.meta.url)) },
  },
  test: {
    environment: "node",
    // .tsx — це димові тести рендера, які самі просять jsdom докблоком
    // `@vitest-environment`. Вони не замінюють перевірку логіки, але ловлять
    // падіння екрана при першому ж рендері — те, що інакше виявляється
    // порожнім білим вікном у викладача.
    include: ["src/**/*.test.ts", "src/**/*.test.tsx"],
  },
});
