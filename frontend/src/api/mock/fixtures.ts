/**
 * Дані демонстраційного режиму.
 *
 * Це НЕ «приклад із трьох рядків». Корпус підібраний так, щоб на ньому було
 * видно рівно ті властивості, заради яких продукт існує:
 *   • відповідь спирається на ТРИ різні документи (диверсифікація джерел);
 *   • є документ у помилці — щоб екран документів показував не лише щастя;
 *   • є документ у черзі — щоб працювала смуга прогресу й ETA;
 *   • є питання, відповіді на яке в корпусі немає — щоб було видно
 *     утримання з жовтим банером, а не вигадану відповідь.
 *
 * Тексти фрагментів навмисно правдоподібні за формою (означення, порядок
 * дій, числові дані з одиницями), але змістовно нейтральні: це відкриті
 * навчальні формулювання, а не матеріали академії.
 */

import type {
  Assistant,
  AssistantConfig,
  ChatSession,
  CitationDto,
  DocumentDto,
} from "../types";
import { demoLineBBox } from "./pdf";

export const DEFAULT_CONFIG: AssistantConfig = {
  instructions: "",
  topics_covered: [],
  topics_refused: [],
  on_missing_information: "no_information",
  confidence_required: "medium",
  answer_language: "match_question",
  always_cite_pages: true,
  dense_top_k: 60,
  sparse_top_k: 60,
  rerank_top_k: 50,
  final_top_k: 5,
  max_per_document: 2,
  min_distinct_documents: 2,
  relative_score_floor: 0.6,
  rrf_k: 60,
  weight_dense: 1.0,
  weight_sparse: 0.6,
  weight_char_ngram: 0.4,
  temperature: 0.2,
  max_tokens: 600,
  repeat_penalty: 1.08,
  prompt_tier: "compact",
};

export interface DemoChunk {
  chunkUid: string;
  documentId: string;
  page: number;
  line: number;
  lines: number;
  text: string;
  headerPath: string;
  /** Ключові слова, за якими демонстраційний пошук вирішує релевантність. */
  keywords: string[];
}

export const DEMO_ASSISTANTS: Assistant[] = [
  {
    id: "as-artillery",
    name: "Артилерійська підготовка",
    description: "Підручники й настанови для курсу вогневої підготовки",
    colour: "#4f46e5",
    emoji: "🎯",
    config: {
      ...DEFAULT_CONFIG,
      instructions:
        "Ти допомагаєш викладачеві та курсантам розібратися в навчальних матеріалах кафедри. " +
        "Пояснюй по суті, спираючись на означення й формулювання з наданих джерел.",
      topics_covered: ["балістика", "вогнева підготовка", "матеріальна частина"],
      confidence_required: "medium",
    },
    configVersion: 3,
    collections: [
      {
        id: "col-artillery",
        name: "Основна",
        embeddingModelId: "Qwen3-Embedding-0.6B",
        dim: 1024,
        indexGeneration: 4,
        dirty: false,
        documents: 4,
        chunks: 3128,
      },
    ],
  },
  {
    id: "as-topography",
    name: "Військова топографія",
    description: "Карти, орієнтування, підготовка даних",
    colour: "#0f766e",
    emoji: "🗺️",
    config: { ...DEFAULT_CONFIG, confidence_required: "high" },
    configVersion: 1,
    collections: [
      {
        id: "col-topography",
        name: "Основна",
        embeddingModelId: "Qwen3-Embedding-0.6B",
        dim: 1024,
        indexGeneration: 1,
        dirty: true,
        documents: 1,
        chunks: 412,
      },
    ],
  },
];

export const DEMO_DOCUMENTS: DocumentDto[] = [
  {
    id: "doc-ballistics",
    collectionId: "col-artillery",
    title: "Основи балістики та стрільби",
    originalName: "Основи балістики та стрільби.pdf",
    docType: "textbook",
    language: "uk",
    year: 2019,
    author: "Кафедра артилерії",
    pageCount: 512,
    ingestMode: "FAST",
    status: "READY",
    errorCode: null,
    errorDetail: null,
    qualityGrade: "EXCELLENT",
    sizeBytes: 48_233_984,
    chunks: 1840,
    job: null,
  },
  {
    id: "doc-manual",
    collectionId: "col-artillery",
    title: "Настанова з експлуатації гаубиці",
    originalName: "Настанова з експлуатації.pdf",
    docType: "manual",
    language: "uk",
    year: 2021,
    author: null,
    pageCount: 214,
    ingestMode: "DEEP",
    status: "READY",
    errorCode: null,
    errorDetail: null,
    qualityGrade: "GOOD",
    sizeBytes: 22_118_400,
    chunks: 812,
    job: null,
  },
  {
    id: "doc-methodic",
    collectionId: "col-artillery",
    title: "Методичні рекомендації щодо проведення занять",
    originalName: "Методичні рекомендації.pdf",
    docType: "methodology",
    language: "uk",
    year: 2023,
    author: "Інститут РВіА",
    pageCount: 96,
    ingestMode: "FAST",
    status: "INDEXING",
    errorCode: null,
    errorDetail: null,
    qualityGrade: null,
    sizeBytes: 8_912_896,
    chunks: 0,
    job: {
      id: "job-methodic",
      type: "PARSE",
      state: "RUNNING",
      stage: "PARSE",
      attempts: 1,
      fraction: 0.34,
      errorCode: null,
    },
  },
  {
    id: "doc-scan",
    collectionId: "col-artillery",
    title: "Таблиці стрільби (скан 1987 р.)",
    originalName: "Таблиці стрільби скан.pdf",
    docType: "reference",
    language: "uk",
    year: 1987,
    author: null,
    pageCount: 148,
    ingestMode: "FAST",
    status: "FAILED",
    errorCode: "SCAN_LOW_OCR_CONFIDENCE",
    errorDetail: "Середня впевненість розпізнавання 0.41 на 96 сторінках зі 148.",
    qualityGrade: "POOR",
    sizeBytes: 63_176_704,
    chunks: 0,
    job: {
      id: "job-scan",
      type: "PARSE",
      state: "FAILED",
      stage: "PARSE",
      attempts: 2,
      fraction: 0.71,
      errorCode: "SCAN_LOW_OCR_CONFIDENCE",
    },
  },
  {
    id: "doc-topo",
    collectionId: "col-topography",
    title: "Військова топографія: підготовка даних",
    originalName: "Топографія.pdf",
    docType: "textbook",
    language: "uk",
    year: 2018,
    author: null,
    pageCount: 168,
    ingestMode: "FAST",
    status: "READY",
    errorCode: null,
    errorDetail: null,
    qualityGrade: "GOOD",
    sizeBytes: 15_728_640,
    chunks: 412,
    job: null,
  },
];

export const DEMO_CHUNKS: DemoChunk[] = [
  {
    chunkUid: "c-ballistics-1",
    documentId: "doc-ballistics",
    page: 143,
    line: 4,
    lines: 3,
    headerPath: "Розділ 4. Зовнішня балістика//4.2. Траєкторія",
    text:
      "Траєкторією називають криву лінію, яку описує центр мас снаряда в польоті. " +
      "Форма траєкторії визначається початковою швидкістю, кутом кидання та опором " +
      "повітря; за відсутності опору вона була б параболою.",
    keywords: ["траєкторія", "балістика", "снаряд", "політ", "парабола", "швидкість"],
  },
  {
    chunkUid: "c-ballistics-2",
    documentId: "doc-ballistics",
    page: 144,
    line: 9,
    lines: 4,
    headerPath: "Розділ 4. Зовнішня балістика//4.3. Елементи траєкторії",
    text:
      "Основні елементи траєкторії: точка вильоту, горизонт зброї, кут кидання, вершина " +
      "траєкторії, висота траєкторії, повна горизонтальна дальність, кут падіння. " +
      "Висхідна гілка траєкторії довша і положистіша за низхідну.",
    keywords: ["траєкторія", "елементи", "кут", "дальність", "вершина", "падіння"],
  },
  {
    chunkUid: "c-manual-1",
    documentId: "doc-manual",
    page: 57,
    line: 6,
    lines: 5,
    headerPath: "Глава 3. Підготовка до стрільби//3.1. Порядок дій",
    text:
      "Перед стрільбою перевіряють кріплення прицільних пристроїв, стан противідкатних " +
      "пристроїв та рівень рідини. Контрольну перевірку прицілу виконують перед кожним " +
      "виходом на вогневу позицію та після маршу понад 50 км.",
    keywords: ["підготовка", "приціл", "перевірка", "стрільба", "позиція", "порядок"],
  },
  {
    chunkUid: "c-manual-2",
    documentId: "doc-manual",
    page: 58,
    line: 2,
    lines: 3,
    headerPath: "Глава 3. Підготовка до стрільби//3.2. Температурні поправки",
    text:
      "Температура заряду впливає на початкову швидкість снаряда: відхилення на кожні " +
      "10 °C від нормальної (+15 °C) змінює початкову швидкість приблизно на 0,5 %.",
    keywords: ["температура", "заряд", "швидкість", "поправка", "снаряд"],
  },
  {
    chunkUid: "c-methodic-1",
    documentId: "doc-methodic",
    page: 12,
    line: 7,
    lines: 3,
    headerPath: "Організація заняття//Практична частина",
    text:
      "На практичному занятті з визначення елементів траєкторії курсанти працюють " +
      "у складі обчислювальних команд по три особи; норматив на розрахунок — 6 хвилин.",
    keywords: ["заняття", "курсанти", "траєкторія", "норматив", "розрахунок"],
  },
  {
    chunkUid: "c-topo-1",
    documentId: "doc-topo",
    page: 34,
    line: 5,
    lines: 3,
    headerPath: "Розділ 2. Орієнтування//2.4. Дирекційний кут",
    text:
      "Дирекційним кутом називають кут між північним напрямком вертикальної лінії " +
      "координатної сітки та напрямком на предмет, відлічений за ходом годинникової стрілки.",
    keywords: ["дирекційний", "кут", "орієнтування", "сітка", "координати"],
  },
];

export function demoCitation(chunk: DemoChunk, ordinal: number): CitationDto {
  const document = DEMO_DOCUMENTS.find((d) => d.id === chunk.documentId);
  return {
    ordinal,
    chunkUid: chunk.chunkUid,
    documentId: chunk.documentId,
    documentTitle: document?.title ?? chunk.documentId,
    pageFrom: chunk.page,
    pageTo: chunk.page,
    pageLabel: `с. ${chunk.page}`,
    quote: chunk.text.slice(0, 180),
    bboxes: Array.from({ length: chunk.lines }, (_, i) => demoLineBBox(chunk.page, chunk.line + i)),
    language: "uk",
  };
}

export const DEMO_SESSIONS: ChatSession[] = [
  {
    id: "sess-demo",
    assistantId: "as-artillery",
    title: "Елементи траєкторії",
    createdAt: "2026-09-01 09:12:00",
    updatedAt: "2026-09-01 09:20:00",
    messages: 0,
  },
];

/**
 * Демонстраційний пошук: перекриття за ключовими словами.
 *
 * Це навмисно ПРИМІТИВНО і навмисно ж не приховано: демонстраційний режим
 * має показувати роботу інтерфейсу, а не імітувати якість пошуку. Коли
 * жодне слово не збіглося, повертається порожньо — і UI показує утримання,
 * тобто саме той шлях, який найважче перевірити на справжніх даних.
 */
export function demoSearch(query: string, collectionId: string, limit = 5): DemoChunk[] {
  const documentIds = new Set(
    DEMO_DOCUMENTS.filter((d) => d.collectionId === collectionId && d.status === "READY").map(
      (d) => d.id,
    ),
  );
  const terms = query
    .toLocaleLowerCase("uk")
    .split(/[^\p{L}\p{N}]+/u)
    .filter((t) => t.length > 2);
  if (!terms.length) return [];

  // Одного збігу зі складеного питання замало.
  // «Порядок нарахування відпускних» збігається з підручником по слову
  // «порядок» — і без цього правила демонстраційний режим відповідав би на
  // питання, яких у корпусі немає, тобто ховав би найважливіший шлях:
  // утримання. Для однослівного запиту вимога послаблюється до одного збігу.
  const required = terms.length >= 2 ? 2 : 1;

  const scored = DEMO_CHUNKS.filter((chunk) => documentIds.has(chunk.documentId))
    .map((chunk) => {
      const haystack = `${chunk.text} ${chunk.keywords.join(" ")}`.toLocaleLowerCase("uk");
      let matched = 0;
      for (const term of terms) {
        // Порівняння за префіксом замість повного збігу: українська
        // словозміна інакше не дала б жодного влучання на «траєкторії».
        const stem = term.slice(0, Math.max(4, term.length - 2));
        if (haystack.includes(stem)) matched += 1;
      }
      return { chunk, matched, score: matched / terms.length };
    })
    .filter((row) => row.matched >= required)
    .sort((a, b) => b.score - a.score);

  // Квота на документ — так само, як у справжньому конвеєрі: інакше вся
  // п'ятірка приходить з одного підручника й «поєднання джерел» не видно.
  const perDocument = new Map<string, number>();
  const out: DemoChunk[] = [];
  for (const row of scored) {
    const used = perDocument.get(row.chunk.documentId) ?? 0;
    if (used >= 2) continue;
    perDocument.set(row.chunk.documentId, used + 1);
    out.push(row.chunk);
    if (out.length >= limit) break;
  }
  return out;
}

/** Відповідь демонстраційного режиму — зведення з маркерами [n]. */
export function demoAnswer(question: string, chunks: DemoChunk[]): string {
  if (!chunks.length) {
    return (
      "У завантажених матеріалах я не знайшов підстав для відповіді на це питання. " +
      "Спробуйте переформулювати або додайте відповідний підручник."
    );
  }
  const head = question.trim().replace(/[?？]+$/u, "");
  const sentences = chunks.map((chunk, index) => {
    const first = chunk.text.split(". ")[0];
    return `${first}. [${index + 1}]`;
  });
  const docs = new Set(chunks.map((c) => c.documentId));
  const tail =
    docs.size > 1
      ? `\n\nВідомості зведено з ${docs.size} джерел; розбіжностей між ними не виявлено.`
      : "";
  return `Щодо «${head}» у матеріалах зазначено таке.\n\n${sentences.join(" ")}${tail}`;
}
