import { cpSync, existsSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { fileURLToPath, URL } from "node:url";

import react from "@vitejs/plugin-react";
import { defineConfig, type Plugin } from "vite";

const require_ = createRequire(import.meta.url);

/**
 * Допоміжні файли pdf.js — локально, а не з CDN.
 *
 * `standard_fonts` потрібні для базових шрифтів Type1 (Helvetica, Times), а
 * `cmaps` — для документів із CID-кодуванням. Без них pdf.js тихо малює
 * порожні сторінки, і побачити це можна лише на машині БЕЗ мережі — тобто
 * саме на цільовій, у закритому контурі. Тому файли не завантажуються, а
 * копіюються з node_modules: у режимі розробки — middleware, у збірці —
 * копіюванням у dist.
 */
function pdfjsAssets(): Plugin {
  const root = dirname(require_.resolve("pdfjs-dist/package.json"));
  return {
    name: "asistent-pdfjs-assets",
    configureServer(server) {
      server.middlewares.use("/pdfjs", (req, res, next) => {
        const rest = decodeURIComponent((req.url ?? "/").split("?")[0]);
        // Захист від виходу за межі каталогу: шлях приходить із запиту.
        if (rest.includes("..")) {
          res.statusCode = 400;
          res.end();
          return;
        }
        const file = join(root, rest);
        if (!file.startsWith(root) || !existsSync(file)) {
          next();
          return;
        }
        server.middlewares.handle(
          Object.assign(req, { url: `/@fs/${file}` }),
          res,
          next,
        );
      });
    },
    closeBundle() {
      for (const directory of ["standard_fonts", "cmaps"]) {
        const from = join(root, directory);
        if (existsSync(from)) cpSync(from, join("dist", "pdfjs", directory), { recursive: true });
      }
    },
  };
}

// Порт 5173 — dev-сервер Vite; sidecar слухає 8765. Проксі /api потрібен лише
// для того, щоб у режимі розробки не воювати з CORS: у зібраному застосунку
// SPA й API живуть на одному origin (оболонка вантажить статику з файлів,
// а запити йдуть на 127.0.0.1:8765 напряму — див. src/api/client.ts).
export default defineConfig({
  plugins: [react(), pdfjsAssets()],
  resolve: {
    alias: { "@": fileURLToPath(new URL("./src", import.meta.url)) },
  },
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8765",
        changeOrigin: false,
      },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: true,
    chunkSizeWarningLimit: 1600, // pdf.js сам по собі ~1 МБ
  },
  worker: { format: "es" },
});
