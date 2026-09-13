/**
 * Компіляція налаштувань асистента у промпт і в політику пошуку.
 *
 * ЧОМУ ЦЕ ВЗАГАЛІ Є НА ФРОНТЕНДІ (перевірено по коду бекенда).
 * `routes_chat.py` передає генератору `persona=config.instructions` — і
 * більше НІЧОГО з конфігу в промпт не потрапляє. Поля `topics_covered`,
 * `topics_refused`, `on_missing_information`, `answer_language`,
 * `always_cite_pages` зберігаються, але самі по собі на відповідь не
 * впливають. Отже, або структуровані контроли компілюються в текст
 * інструкцій тут, або вони — декорація. Обрано перше.
 *
 * ЩО ВИКОНУЄТЬСЯ В КОДІ, А НЕ В ПРОМПТІ.
 * Слайдер «потрібна впевненість» НЕ дописує в промпт «будь обережним». Він
 * змінює `confidence_required`, з якого ретривер бере поріг скора реранкера
 * (0.15 / 0.35 / 0.55) і мінімум опорних фрагментів (1 / 1 / 2). Промптом
 * неможливо надійно змусити 12-мільярдну модель відмовитись відповідати;
 * порогом — можна. Тому редактор показує ці числа поруч зі слайдером: інакше
 * викладач читатиме «висока впевненість» як ввічливе прохання.
 *
 * КОРОТКО — НЕ ЗАРАДИ КРАСИ. Успішність виконання всіх інструкцій падає
 * приблизно експоненційно з їх кількістю, а персона має жорсткий ліміт 250
 * токенів на бекенді (`PERSONA_MAX_TOKENS`). Усе, що понад, буде обрізано —
 * тому редактор рахує токени й попереджає ДО збереження, а не після.
 */

import type { AssistantConfig } from "@/api/types";

// --------------------------------------------------------------- константи
/**
 * Байт-ідентичний префікс із `app/generation/prompt_builder.py`.
 *
 * Копія тут потрібна лише для попереднього перегляду, коли бекенд
 * недоступний (демонстраційний режим, перший запуск). Джерело правди —
 * `GET /assistants/{id}/prompt-preview`; розходження ловить тест
 * `prompt.test.ts`, який звіряє цей рядок із файлом Python.
 *
 * Не редагувати заради «краще звучить»: LM Studio перевикористовує
 * префіксний KV-кеш, і зміна одного байта інвалідує його для ВСІХ
 * асистентів одразу, тобто повертає повний prefill на кожне перемикання.
 */
export const SYSTEM_RULES_UK =
  "1. Ти — навчальний асистент викладача Національної академії сухопутних військ.\n" +
  "2. Відповідай лише за наданими нижче уривками з навчальних матеріалів; " +
  "власних знань не додавай.\n" +
  "3. Після кожного твердження став номер джерела у квадратних дужках: [1], [2]. " +
  "Інших номерів не вигадуй.\n" +
  "4. Якщо уривків недостатньо, прямо напиши про це — це правильна відповідь, а не невдача.\n" +
  "5. Відповідай українською, навіть якщо джерела англійською або російською.";

export const PERSONA_MAX_TOKENS = 250;
const UK_TOKENS_PER_WORD = 2.35;
const UK_CHARS_PER_TOKEN = 3.2;

/**
 * Роздільник між тим, що написав викладач, і тим, що склали контроли.
 *
 * Без нього структуровану частину неможливо відрізнити від ручного тексту
 * при наступному відкритті, і кожне збереження дописувало б її ще раз —
 * через п'ять правок персона складалася б із п'яти копій переліку тем і
 * гарантовано пробивала ліміт 250 токенів.
 */
export const COMPILED_MARKER = "⟦Складено з налаштувань — не редагувати вручну⟧";

/** Сильний український дефолт: те, що стоїть у полі для нового асистента. */
export const DEFAULT_INSTRUCTIONS =
  "Ти допомагаєш викладачеві та курсантам розібратися в навчальних матеріалах кафедри. " +
  "Пояснюй по суті, спираючись на означення й формулювання з наданих джерел. " +
  "Якщо в різних джерелах є розбіжність — назви обидві позиції та вкажи, де кожна.";

// ------------------------------------------------------------ оцінка обсягу
/**
 * Оцінка довжини в токенах БЕЗ токенайзера — дзеркало `estimate_tokens_uk`.
 * Береться максимум із двох незалежних оцінок (за словами і за символами):
 * занижена оцінка коштує дорожче за завищену, бо вона обрізає персону
 * непомітно для того, хто її писав.
 */
export function estimateTokensUk(text: string): number {
  if (!text) return 0;
  const words = (text.match(/\S+/g) ?? []).length;
  return Math.max(Math.trunc(words * UK_TOKENS_PER_WORD), Math.trunc(text.length / UK_CHARS_PER_TOKEN)) + 1;
}

// ----------------------------------------------------------- політика пошуку
export interface RetrievalPolicy {
  /** Поріг скора реранкера. Мапа з `AssistantConfig.confidence_threshold()`. */
  threshold: number;
  /** Мінімум опорних фрагментів. Мапа з `min_supporting_chunks()`. */
  minSupporting: number;
  /** Людське пояснення того, що змінює слайдер. */
  explain: string;
}

const POLICY: Record<AssistantConfig["confidence_required"], RetrievalPolicy> = {
  low: {
    threshold: 0.15,
    minSupporting: 1,
    explain:
      "Відповідатиме навіть за слабкої відповідності матеріалів. Менше відмов, більше ризику.",
  },
  medium: {
    threshold: 0.35,
    minSupporting: 1,
    explain: "Відповідатиме, коли знайдено щонайменше один переконливий фрагмент.",
  },
  high: {
    threshold: 0.55,
    minSupporting: 2,
    explain:
      "Відповідатиме лише за двох переконливих фрагментів. Більше відмов — і майже " +
      "жодного вигадування.",
  },
};

export function retrievalPolicy(config: Pick<AssistantConfig, "confidence_required">): RetrievalPolicy {
  return POLICY[config.confidence_required] ?? POLICY.medium;
}

// -------------------------------------------------- компіляція інструкцій
export interface StructuredInstructions {
  topicsCovered: string[];
  topicsRefused: string[];
  onMissing: AssistantConfig["on_missing_information"];
  answerLanguage: AssistantConfig["answer_language"];
  alwaysCitePages: boolean;
}

const MISSING_TEXT: Record<AssistantConfig["on_missing_information"], string> = {
  no_information:
    "Якщо в наданих джерелах відповіді немає — напиши, що в завантажених матеріалах цього немає. " +
    "Не додавай нічого від себе.",
  general_knowledge_marked:
    "Якщо в наданих джерелах відповіді немає — можеш відповісти із загальних знань, але почни " +
    "таку відповідь словами «За межами завантажених матеріалів:» і не став у ній номерів джерел.",
  refuse: "Якщо в наданих джерелах відповіді немає — відмовся відповідати й не пропонуй здогадів.",
};

/**
 * Зібрати повний текст `config.instructions`, який піде на бекенд.
 *
 * Порядок частин важливий: вільний текст викладача — першим. Він конкретний
 * і предметний, а службові рядки однакові в усіх асистентів; поставити їх
 * попереду означало б витратити початок персони (найкраще виконувану її
 * частину) на те, що й так повторюється.
 */
export function compileInstructions(freeText: string, structured: StructuredInstructions): string {
  const lines: string[] = [];
  const covered = cleanChips(structured.topicsCovered);
  const refused = cleanChips(structured.topicsRefused);

  if (covered.length) lines.push(`Твоя предметна область: ${covered.join(", ")}.`);
  if (refused.length) {
    lines.push(
      `Не відповідай на питання про: ${refused.join(", ")} — навіть якщо щось схоже трапилося в джерелах.`,
    );
  }
  lines.push(MISSING_TEXT[structured.onMissing]);
  if (structured.answerLanguage === "always_uk") {
    lines.push("Відповідай українською завжди, незалежно від мови питання.");
  }
  if (structured.alwaysCitePages) {
    lines.push("У кожній відповіді зазначай номери сторінок джерел.");
  }

  const free = freeText.trim();
  const compiled = lines.join("\n");
  if (!free) return `${COMPILED_MARKER}\n${compiled}`;
  return `${free}\n\n${COMPILED_MARKER}\n${compiled}`;
}

/**
 * Зворотна операція: дістати з `instructions` те, що писав викладач.
 *
 * Усе після маркера відкидається — воно буде складене заново з поточних
 * контролів. Текст без маркера (створений старішою версією або вручну через
 * перемикач розробника) повертається як є: втратити написане було б значно
 * гірше, ніж показати службові рядки в полі.
 */
export function splitInstructions(instructions: string): { free: string; compiled: string } {
  const index = instructions.indexOf(COMPILED_MARKER);
  if (index < 0) return { free: instructions, compiled: "" };
  return {
    free: instructions.slice(0, index).trimEnd(),
    compiled: instructions.slice(index + COMPILED_MARKER.length).trim(),
  };
}

function cleanChips(values: readonly string[]): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const raw of values) {
    const value = raw.trim().replace(/\s+/g, " ");
    if (!value) continue;
    const key = value.toLocaleLowerCase("uk");
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(value);
  }
  return out;
}

// ---------------------------------------------------- попередній перегляд
export interface CompiledPrompt {
  /** Те, що побачить модель як системне повідомлення. */
  system: string;
  personaTokens: number;
  rulesTokens: number;
  /** Персона довша за ліміт — бекенд її обріже. */
  overLimit: boolean;
  policy: RetrievalPolicy;
}

/**
 * Локальний прев'ю системного промпту — дзеркало `build_system_prompt`.
 * Використовується, доки не відповів `GET /assistants/{id}/prompt-preview`
 * (і в демонстраційному режимі, де бекенда немає взагалі).
 */
export function compilePromptPreview(
  instructions: string,
  config: Pick<AssistantConfig, "confidence_required">,
): CompiledPrompt {
  const persona = instructions.trim();
  const personaTokens = estimateTokensUk(persona);
  const system = persona ? `${SYSTEM_RULES_UK}\n\nРОЛЬ АСИСТЕНТА:\n${persona}` : SYSTEM_RULES_UK;
  return {
    system,
    personaTokens,
    rulesTokens: estimateTokensUk(SYSTEM_RULES_UK),
    overLimit: personaTokens > PERSONA_MAX_TOKENS,
    policy: retrievalPolicy(config),
  };
}

// ------------------------------------------------------- клікабельні блоки
export interface PromptBlock {
  id: string;
  label: string;
  text: string;
}

export interface PromptBlockGroup {
  id: string;
  title: string;
  blocks: PromptBlock[];
}

/**
 * Готові фрагменти інструкцій.
 *
 * Сенс не в економії набору тексту, а в тому, щоб викладач не мусив вигадувати
 * формулювання, які добре виконуються локальною моделлю. Кожен блок —
 * коротке наказове речення: саме таку форму 8–14-мільярдні моделі виконують
 * надійно, на відміну від абзацу з переліком побажань.
 */
export const PROMPT_BLOCKS: PromptBlockGroup[] = [
  {
    id: "role",
    title: "Роль",
    blocks: [
      {
        id: "role-lecturer",
        label: "Помічник викладача",
        text: "Ти готуєш матеріали для заняття: стисло, точно, з опорою на джерела.",
      },
      {
        id: "role-tutor",
        label: "Репетитор для курсанта",
        text: "Пояснюй так, ніби курсант бачить тему вперше: спершу суть, далі подробиці.",
      },
      {
        id: "role-examiner",
        label: "Екзаменатор",
        text: "Після відповіді додай одне контрольне запитання для самоперевірки.",
      },
    ],
  },
  {
    id: "style",
    title: "Стиль",
    blocks: [
      {
        id: "style-strict",
        label: "Строго й по суті",
        text: "Пиши сухо й по суті, без вступів і без переказу питання.",
      },
      {
        id: "style-plain",
        label: "Простою мовою",
        text: "Складні терміни пояснюй простими словами одразу після першого вживання.",
      },
      {
        id: "style-regulatory",
        label: "Мовою настанов",
        text: "Дотримуйся формулювань і термінології статутів та настанов.",
      },
    ],
  },
  {
    id: "depth",
    title: "Рівень деталізації",
    blocks: [
      {
        id: "depth-short",
        label: "Коротко",
        text: "Відповідь — не більше трьох речень, якщо питання не вимагає переліку.",
      },
      {
        id: "depth-full",
        label: "Розгорнуто",
        text: "Розкривай тему повно: означення, порядок дій, типові помилки.",
      },
      {
        id: "depth-numbers",
        label: "З числами й одиницями",
        text: "Завжди наводь числові значення з одиницями вимірювання так, як вони є в джерелі.",
      },
    ],
  },
  {
    id: "format",
    title: "Формат відповіді",
    blocks: [
      {
        id: "format-steps",
        label: "Порядок дій списком",
        text: "Порядок дій подавай пронумерованим списком, по одному кроку на рядок.",
      },
      {
        id: "format-table",
        label: "Порівняння таблицею",
        text: "Коли порівнюєш два і більше зразків, подай порівняння таблицею.",
      },
      {
        id: "format-summary",
        label: "Висновок наприкінці",
        text: "Наприкінці додай рядок «Головне:» з одним реченням висновку.",
      },
    ],
  },
  {
    id: "terminology",
    title: "Термінологія",
    blocks: [
      {
        id: "term-uk",
        label: "Українські терміни",
        text: "Уживай українську військову термінологію; іншомовний відповідник давай у дужках.",
      },
      {
        id: "term-keep-designations",
        label: "Зберігати позначення",
        text: "Позначення зразків і документів (Д-30, 2С1, ДСТУ, НАТО) переписуй дослівно.",
      },
      {
        id: "term-nato",
        label: "Пояснювати скорочення",
        text: "Кожне скорочення при першому вживанні розшифровуй.",
      },
    ],
  },
];

/** Додати блок у кінець тексту, не дублюючи вже наявний. */
export function appendBlock(text: string, block: PromptBlock): string {
  if (text.includes(block.text)) return text;
  const base = text.trimEnd();
  return base ? `${base}\n${block.text}` : block.text;
}
