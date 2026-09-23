/**
 * Точка входу.
 *
 * ПОРЯДОК КРОКІВ ТУТ МАЄ ЗНАЧЕННЯ:
 *   1. тема й масштаб застосовуються ДО першого рендера — інакше на старті
 *      блимає світлий екран у темній темі, і на кожному запуску;
 *   2. екран очікування монтується ДО проби sidecar — інакше вікно стоїть
 *      порожнім усі секунди холодного старту Python, і це виглядає як
 *      зависання;
 *   3. `probeBackend()` виконується ДО монтування дерева застосунку — щоб той
 *      не встиг показати помилку з'єднання як робочий стан;
 *   4. і лише потім React із реальним `health`.
 *
 * ЩО РОБИТЬ ГІЛКА `catch`. Вона показує ЧЕСНУ помилку старту, а не підставляє
 * демонстраційні дані. Раніше цієї гілки не було взагалі: `probeBackend`
 * мовчки вмикав мок, і в запакованому застосунку — де SPA завжди програвала
 * гонку холодному старту sidecar — викладач отримував вигаданий корпус із
 * вигаданими цитатами замість власних матеріалів (див. коментар у
 * `api/client.ts`).
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

import { BackendUnavailableError, probeBackend } from "./api/client";
import type { HealthStatus } from "./api/types";
import App from "./App";
import { StartupError, StartupScreen } from "./components/Startup";
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

  const container = document.getElementById("root");
  if (!container) throw new Error("Кореневий елемент #root відсутній у index.html");
  const root = createRoot(container);

  root.render(
    <StrictMode>
      <StartupScreen />
    </StrictMode>,
  );

  let health: HealthStatus;
  try {
    health = await probeBackend();
  } catch (error) {
    const elapsed = error instanceof BackendUnavailableError ? error.elapsedMs : 0;
    // У консоль — повна причина: це єдиний слід, який лишиться, якщо викладач
    // відкриє панель розробника за нашою вказівкою.
    console.error("Старт зупинено: sidecar не відповів.", error);
    root.render(
      <StrictMode>
        <StartupError seconds={Math.round(elapsed / 1000)} />
      </StrictMode>,
    );
    return;
  }

  root.render(
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
