/**
 * Оцінка часу до завершення індексації.
 *
 * ЧОМУ КОВЗНА МЕДІАНА, А НЕ СЕРЕДНЄ.
 * Вартість сторінки в цьому конвеєрі різниться на порядок: цифрова сторінка
 * розбирається за 0.4 с, сканована з OCR — за 6–12 с, а сторінка з великою
 * таблицею через TableFormer ACCURATE — ще довше. У наївному середньому одна
 * така сторінка зсуває оцінку для ВСІХ наступних, і викладач бачить, як ETA
 * стрибає з «4 хв» на «40 хв» і назад. Медіана останніх N вимірів до
 * поодиноких викидів байдужа, а на зміну РЕЖИМУ (пішов суцільний скан)
 * реагує за N/2 вимірів — тобто саме тоді, коли зміна справжня.
 *
 * ВИМІР — У ОДИНИЦЯХ ВАГИ, А НЕ В СТОРІНКАХ.
 * Бекенд рахує прогрес зваженим за `cost_weight` сторінки (`probe.py`), і
 * подія `job.progress` несе `current`/`total` саме в цих одиницях. Тому
 * трекер меряє «секунд на одиницю ваги»: у сторінках оцінка була б
 * систематично хибною на будь-якому змішаному документі.
 *
 * Бекенд шле власний `etaSeconds` (експоненційне згладжування у watcher.py).
 * Ми його ПОКАЗУЄМО, коли він є, але маємо власний трекер: у режимі
 * `worker_mode="inline"` події йдуть із раннера без згладжування взагалі,
 * а після перепідключення SSE серверний стан оцінки втрачається.
 */

/** Скільки останніх вимірів тримати. 9 ≈ 15–30 с історії на типовій швидкості. */
export const DEFAULT_WINDOW = 9;

/** Менші прирости — це шум округлення `round(done, 3)`, а не робота. */
const MIN_WEIGHT_STEP = 1e-3;

/** Менші проміжки означають, що дві події прийшли пачкою після затримки. */
const MIN_TIME_STEP_MS = 120;

export function median(values: readonly number[]): number | null {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = sorted.length >> 1;
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

export interface EtaSnapshot {
  /** Секунд на одиницю ваги; null — вимірів ще замало. */
  secondsPerUnit: number | null;
  /** Скільки секунд лишилось; null — оцінити нічим. */
  remainingSeconds: number | null;
  /** Скільки вимірів у вікні. UI показує «оцінюємо…», доки їх < 2. */
  samples: number;
}

/**
 * Трекер одного завдання. Стан навмисно мутабельний і живе поза React:
 * перерахунок ETA на кожному рендері дав би оцінку, що залежить від того,
 * як часто перемальовується компонент.
 */
export class EtaTracker {
  private readonly window: number;
  private readonly rates: number[] = [];
  private lastWeight: number | null = null;
  private lastAt: number | null = null;

  constructor(window: number = DEFAULT_WINDOW) {
    this.window = Math.max(2, window);
  }

  /** Новий вимір прогресу. `weightDone` і `weightTotal` — з `job.progress`. */
  observe(weightDone: number, atMs: number): void {
    const previousWeight = this.lastWeight;
    const previousAt = this.lastAt;
    this.lastWeight = weightDone;
    this.lastAt = atMs;
    if (previousWeight == null || previousAt == null) return;

    const deltaWeight = weightDone - previousWeight;
    const deltaMs = atMs - previousAt;
    // Прогрес назад означає перезапуск завдання (повтор, відновлення після
    // падіння воркера). Стара швидкість до нового проходу стосунку не має.
    if (deltaWeight < 0) {
      this.rates.length = 0;
      return;
    }
    if (deltaWeight < MIN_WEIGHT_STEP || deltaMs < MIN_TIME_STEP_MS) return;

    this.rates.push(deltaMs / 1000 / deltaWeight);
    if (this.rates.length > this.window) this.rates.shift();
  }

  snapshot(weightDone: number, weightTotal: number): EtaSnapshot {
    const secondsPerUnit = this.rates.length >= 2 ? median(this.rates) : null;
    const left = Math.max(0, weightTotal - weightDone);
    return {
      secondsPerUnit,
      remainingSeconds: secondsPerUnit == null ? null : secondsPerUnit * left,
      samples: this.rates.length,
    };
  }

  reset(): void {
    this.rates.length = 0;
    this.lastWeight = null;
    this.lastAt = null;
  }
}

/**
 * Реєстр трекерів за id завдання. Один документ = одне активне завдання,
 * але завдань у черзі багато, і зведений банер («Обробка 3 з 12 ·
 * залишилось ~24 хв») складається саме з них.
 */
export class EtaRegistry {
  private readonly trackers = new Map<string, EtaTracker>();
  private readonly latest = new Map<string, { done: number; total: number }>();

  observe(jobId: string, weightDone: number, weightTotal: number, atMs: number): EtaSnapshot {
    let tracker = this.trackers.get(jobId);
    if (!tracker) {
      tracker = new EtaTracker();
      this.trackers.set(jobId, tracker);
    }
    tracker.observe(weightDone, atMs);
    this.latest.set(jobId, { done: weightDone, total: weightTotal });
    return tracker.snapshot(weightDone, weightTotal);
  }

  snapshot(jobId: string): EtaSnapshot | null {
    const tracker = this.trackers.get(jobId);
    const last = this.latest.get(jobId);
    if (!tracker || !last) return null;
    return tracker.snapshot(last.done, last.total);
  }

  forget(jobId: string): void {
    this.trackers.delete(jobId);
    this.latest.delete(jobId);
  }

  /**
   * Сумарний залишок по всіх активних завданнях.
   *
   * СУМА, а не максимум: воркерів зазвичай один-два, тобто документи
   * обробляються послідовно, і показати максимум означало б пообіцяти
   * викладачеві вчетверо менше часу, ніж він чекатиме насправді. Якщо
   * воркерів більше, оцінка ділиться на їх кількість.
   */
  totalRemaining(activeJobIds: readonly string[], workers = 1): number | null {
    let sum = 0;
    let known = 0;
    for (const id of activeJobIds) {
      const snapshot = this.snapshot(id);
      if (snapshot?.remainingSeconds == null) continue;
      sum += snapshot.remainingSeconds;
      known += 1;
    }
    if (!known) return null;
    // Невідомі завдання добиваються середнім по відомих — інакше банер
    // показував би менше часу, ніж уже видно в окремих рядках.
    const scaled = (sum / known) * activeJobIds.length;
    return scaled / Math.max(1, workers);
  }
}
