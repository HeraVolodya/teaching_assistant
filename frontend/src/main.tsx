/**
 * Точка входу.
 *
 * ПОРЯДОК КРОКІВ ТУТ МАЄ ЗНАЧЕННЯ:
 *   1. тема й масштаб застосовуються ДО першого рендера — інакше на старті
 *      блимає світлий екран у темній темі, і на кожному запуску;
 *   2. `probeBackend()` виконується ДО монтування дерева — щоб застосунок
 *      одразу знав, він на справжньому sidecar чи на демонстраційних даних,
 *      і не встиг показати помилку з'єднання як робочий стан;
 *   3. і лише потім React.
 *
 * `React.StrictMode` увімкнено свідомо, попри подвійний виклик ефектів у
 * розробці: саме він ловить незакриті SSE-підписки й скасовані рендери
 * pdf.js — тобто рівно ті помилки, які інакше виявляються на машині
 * викладача у вигляді «через годину застосунок гальмує».
 */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";

import { probeBackend } from "./api/client";
import App from "./App";
import { initI18n } from "./i18n";
import { bootstrapUi } from "./lib/store";
import "./index.css";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      // Локальний sidecar — не мережа: повтори лише ховали б справжню
      // помилку за секундами очікування. Одна спроба, чесна помилка.
      retry: 1,
      retryDelay: 400,
      staleTime: 3_000,
      refetchOnWindowFocus: false,
    },
    mutations: { retry: 0 },
  },
});

async function start(): Promise<void> {
  bootstrapUi();
  initI18n();
  const health = await probeBackend();

  const container = document.getElementById("root");
  if (!container) throw new Error("Кореневий елемент #root відсутній у index.html");

  createRoot(container).render(
    <StrictMode>
      <QueryClientProvider client={queryClient}>
        <BrowserRouter>
          <App initialHealth={health} />
        </BrowserRouter>
      </QueryClientProvider>
    </StrictMode>,
  );
}

void start();
