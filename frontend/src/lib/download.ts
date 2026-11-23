/**
 * Зберегти відповідь сервера у файл — так само, як «Зібрати пакет» діагностики.
 *
 * НАЗВА ФАЙЛУ — ВІД СЕРВЕРА. Бекенд кладе її в Content-Disposition
 * («Види вогню артилерії.xlsx», RFC 6266: ASCII-запаска плюс `filename*` у
 * UTF-8). Вигадувати її вдруге тут означало б дві реалізації, що колись
 * розійдуться; запасна назва потрібна лише тоді, коли заголовка не видно
 * (сторонній проксі, демонстраційний режим).
 */

/** Назва з Content-Disposition; `filename*` (UTF-8) має пріоритет над `filename`. */
export function filenameFromDisposition(header: string | null | undefined): string | null {
  if (!header) return null;
  const extended = /filename\*\s*=\s*(?:UTF-8|utf-8)''([^;]+)/.exec(header);
  if (extended) {
    try {
      return decodeURIComponent(extended[1].trim().replace(/^"(.*)"$/, "$1"));
    } catch {
      /* зіпсоване кодування — спробуємо простий варіант нижче */
    }
  }
  const simple = /filename\s*=\s*"?([^";]+)"?/i.exec(header);
  return simple ? simple[1].trim() : null;
}

export async function saveResponseAsFile(response: Response, fallbackName: string): Promise<string> {
  const blob = await response.blob();
  const name = filenameFromDisposition(response.headers.get("content-disposition")) ?? fallbackName;
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  link.click();
  // Звільняємо URL із затримкою: миттєвий revoke обриває збереження на
  // повільному диску (та сама причина, що й у пакета діагностики).
  window.setTimeout(() => URL.revokeObjectURL(url), 10_000);
  return name;
}
