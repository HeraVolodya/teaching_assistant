/**
 * Кореневий компонент: маршрути й один спільний канал подій.
 *
 * Майстер першого запуску показується лише тоді, коли він справді потрібен
 * (`setup.ready === false`) АБО коли викладач відкрив його сам. Обов'язковий
 * майстер на кожному старті — найшвидший спосіб зробити так, щоб його
 * перестали читати.
 */

import { useEffect } from "react";
import { Navigate, Route, Routes, useLocation, useNavigate } from "react-router-dom";

import type { HealthStatus } from "@/api/types";
import { AppShell } from "@/components/AppShell";
import { TooltipProvider } from "@/components/ui/Primitives";
import { useSetupStatus } from "@/hooks/queries";
import { useAppEventsBridge } from "@/hooks/useAppEvents";
import { AssistantsScreen } from "@/screens/AssistantsScreen";
import { AssistantWorkspace } from "@/screens/AssistantWorkspace";
import { SettingsScreen } from "@/screens/SettingsScreen";
import { SetupWizard } from "@/screens/SetupWizard";

const SETUP_SEEN_KEY = "asistent.setupSeen";

export default function App({ initialHealth }: { initialHealth: HealthStatus }) {
  const demo = Boolean(initialHealth.demo);
  const location = useLocation();
  const navigate = useNavigate();
  const { data: setup } = useSetupStatus(!demo);

  useAppEventsBridge(true);

  // Одноразове перенаправлення в майстер. Прапорець у localStorage, а не на
  // сервері: у бекенді немає стану «майстер пройдено», і вигадувати його на
  // фронтенді значило б показувати майстер на кожній новій машині з тією
  // самою базою — тобто саме тоді, коли він і потрібен.
  useEffect(() => {
    if (demo || !setup || setup.ready) return;
    if (location.pathname.startsWith("/setup")) return;
    let seen = false;
    try {
      seen = localStorage.getItem(SETUP_SEEN_KEY) === "1";
    } catch {
      seen = false;
    }
    if (!seen) navigate("/setup", { replace: true });
  }, [demo, setup, location.pathname, navigate]);

  return (
    <TooltipProvider>
      <Routes>
        <Route path="/setup" element={<SetupWizard onDone={() => navigate("/")} />} />
        <Route element={<AppShell demo={demo} />}>
          <Route index element={<AssistantsScreen />} />
          <Route path="/a/:assistantId/*" element={<AssistantWorkspace />} />
          <Route path="/settings" element={<SettingsScreen />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </TooltipProvider>
  );
}

export function markSetupSeen(): void {
  try {
    localStorage.setItem(SETUP_SEEN_KEY, "1");
  } catch {
    /* приватний режим — майстер просто з'явиться ще раз */
  }
}
