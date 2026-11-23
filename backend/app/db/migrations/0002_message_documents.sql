-- Файли, які асистент сформував на прохання викладача
-- («зроби презентацію …», «склади таблицю в Excel …»).
--
-- БАЙТИ ФАЙЛУ ТУТ НЕ ЗБЕРІГАЮТЬСЯ. Файл щоразу збирається з тексту самого
-- повідомлення (chat_messages.content + citation_map_json): так завантажений
-- документ завжди збігається з тим, що викладач бачив і перевіряв у чаті, не
-- потребує окремої теки на диску й зникає разом із повідомленням (каскад) —
-- жодних осиротілих файлів після «Очистити розмову».
CREATE TABLE message_documents (
    message_id  TEXT PRIMARY KEY REFERENCES chat_messages(id) ON DELETE CASCADE,
    format      TEXT NOT NULL CHECK (format IN ('pptx', 'docx', 'xlsx')),
    title       TEXT NOT NULL DEFAULT '',
    -- {"slides": 6} / {"sections": 4} / {"rows": 10} — підпис на картці файлу
    stats_json  TEXT NOT NULL DEFAULT '{}',
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
