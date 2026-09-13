/**
 * Дрібні будівельні блоки інтерфейсу.
 *
 * Усі розміри — у rem і `min-height`, жодної фіксованої висоти в px: на
 * 125% і 150% DPI саме фіксовані висоти ламаються першими, і ламаються тихо
 * (текст обрізається знизу, а не переноситься).
 */

import * as RadioGroup from "@radix-ui/react-radio-group";
import * as Slider from "@radix-ui/react-slider";
import * as Switch from "@radix-ui/react-switch";
import * as Tooltip from "@radix-ui/react-tooltip";
import { X } from "lucide-react";
import {
  forwardRef,
  useId,
  useState,
  type HTMLAttributes,
  type InputHTMLAttributes,
  type KeyboardEvent,
  type ReactNode,
  type TextareaHTMLAttributes,
} from "react";

import { cn } from "@/lib/cn";

// ------------------------------------------------------------------ картки
export function Card({ className, ...rest }: HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("card", className)} {...rest} />;
}

export function Section({
  title,
  description,
  action,
  children,
  className,
}: {
  title: ReactNode;
  description?: ReactNode;
  action?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={cn("card p-4 sm:p-5", className)}>
      <header className="mb-4 flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <h2 className="text-[0.9375rem] font-semibold leading-tight">{title}</h2>
          {description ? <p className="mt-1 text-[0.8125rem] subtle">{description}</p> : null}
        </div>
        {action}
      </header>
      {children}
    </section>
  );
}

// ------------------------------------------------------------------ бейджі
export type BadgeTone = "neutral" | "ok" | "warn" | "danger" | "accent";

const BADGE_TONES: Record<BadgeTone, string> = {
  neutral: "bg-raised text-muted border-line",
  ok: "bg-ok/12 text-ok border-ok/30",
  warn: "bg-warn/15 text-warn border-warn/35",
  danger: "bg-danger/12 text-danger border-danger/30",
  accent: "bg-accent/12 text-accent border-accent/30",
};

export function Badge({
  tone = "neutral",
  className,
  ...rest
}: HTMLAttributes<HTMLSpanElement> & { tone?: BadgeTone }) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-md border px-1.5 py-0.5 text-[0.6875rem] font-medium",
        BADGE_TONES[tone],
        className,
      )}
      {...rest}
    />
  );
}

/** Червона крапка на картці асистента — «щось не проіндексувалося». */
export function Dot({ tone = "danger", title }: { tone?: BadgeTone; title: string }) {
  const colour =
    tone === "danger" ? "bg-danger" : tone === "warn" ? "bg-warn" : tone === "ok" ? "bg-ok" : "bg-muted";
  return <span title={title} aria-label={title} className={cn("inline-block h-2 w-2 rounded-full", colour)} />;
}

// ------------------------------------------------------------------- поля
export function Field({
  label,
  hint,
  htmlFor,
  children,
  className,
}: {
  label: ReactNode;
  hint?: ReactNode;
  htmlFor?: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("flex flex-col gap-1.5", className)}>
      <label htmlFor={htmlFor} className="text-[0.8125rem] font-medium">
        {label}
      </label>
      {children}
      {hint ? <p className="text-[0.75rem] subtle leading-snug">{hint}</p> : null}
    </div>
  );
}

const INPUT_BASE =
  "w-full rounded-lg border border-line bg-surface px-3 py-2 text-[0.875rem] outline-none " +
  "transition placeholder:text-muted/70 focus:border-accent disabled:opacity-60";

export const Input = forwardRef<HTMLInputElement, InputHTMLAttributes<HTMLInputElement>>(
  function Input({ className, ...rest }, ref) {
    return <input ref={ref} className={cn(INPUT_BASE, "min-h-[2.25rem]", className)} {...rest} />;
  },
);

export const Textarea = forwardRef<HTMLTextAreaElement, TextareaHTMLAttributes<HTMLTextAreaElement>>(
  function Textarea({ className, ...rest }, ref) {
    return <textarea ref={ref} className={cn(INPUT_BASE, "resize-y leading-relaxed", className)} {...rest} />;
  },
);

// ------------------------------------------------------------------ чіпи
/**
 * Введення тем. Кома і Enter завершують чіп; Backspace на порожньому полі
 * прибирає останній — так набирають теги всі, і будь-яка інша поведінка тут
 * читалася б як поламана.
 */
export function ChipInput({
  values,
  onChange,
  placeholder,
  tone = "neutral",
}: {
  values: string[];
  onChange: (next: string[]) => void;
  placeholder?: string;
  tone?: BadgeTone;
}) {
  const [draft, setDraft] = useState("");

  const commit = (raw: string) => {
    const value = raw.trim().replace(/[,;]+$/, "").trim();
    if (!value) return;
    const exists = values.some((v) => v.toLocaleLowerCase("uk") === value.toLocaleLowerCase("uk"));
    if (!exists) onChange([...values, value]);
    setDraft("");
  };

  const onKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.key === "Enter" || event.key === ",") {
      event.preventDefault();
      commit(draft);
    } else if (event.key === "Backspace" && !draft && values.length) {
      onChange(values.slice(0, -1));
    }
  };

  return (
    <div className="flex flex-wrap items-center gap-1.5 rounded-lg border border-line bg-surface p-1.5 focus-within:border-accent">
      {values.map((value) => (
        <Badge key={value} tone={tone} className="gap-1 py-1 pl-2 pr-1 text-[0.75rem]">
          {value}
          <button
            type="button"
            title={`Прибрати «${value}»`}
            onClick={() => onChange(values.filter((v) => v !== value))}
            className="rounded p-0.5 hover:bg-black/10"
          >
            <X size={11} />
          </button>
        </Badge>
      ))}
      <input
        value={draft}
        onChange={(event) => setDraft(event.target.value)}
        onKeyDown={onKeyDown}
        onBlur={() => commit(draft)}
        placeholder={values.length ? "" : placeholder}
        className="min-w-[8rem] flex-1 bg-transparent px-1.5 py-1 text-[0.8125rem] outline-none placeholder:text-muted/70"
      />
    </div>
  );
}

// -------------------------------------------------------------- перемикачі
export function Toggle({
  checked,
  onChange,
  label,
  hint,
  disabled,
}: {
  checked: boolean;
  onChange: (next: boolean) => void;
  label: ReactNode;
  hint?: ReactNode;
  disabled?: boolean;
}) {
  const id = useId();
  return (
    <div className="flex items-start justify-between gap-4">
      <div className="min-w-0">
        <label htmlFor={id} className="text-[0.8125rem] font-medium">
          {label}
        </label>
        {hint ? <p className="mt-0.5 text-[0.75rem] subtle leading-snug">{hint}</p> : null}
      </div>
      <Switch.Root
        id={id}
        checked={checked}
        disabled={disabled}
        onCheckedChange={onChange}
        className={cn(
          "relative h-[1.375rem] w-[2.375rem] shrink-0 rounded-full border border-line transition",
          "data-[state=checked]:border-accent data-[state=checked]:bg-accent data-[state=unchecked]:bg-raised",
          "disabled:opacity-50",
        )}
      >
        <Switch.Thumb className="block h-[1rem] w-[1rem] translate-x-[0.1875rem] rounded-full bg-white shadow transition-transform data-[state=checked]:translate-x-[1.0625rem]" />
      </Switch.Root>
    </div>
  );
}

export function RadioCards<T extends string>({
  value,
  onChange,
  options,
  name,
}: {
  value: T;
  onChange: (next: T) => void;
  options: { value: T; label: ReactNode; hint?: ReactNode }[];
  name?: string;
}) {
  return (
    <RadioGroup.Root
      value={value}
      onValueChange={(next) => onChange(next as T)}
      name={name}
      className="flex flex-col gap-2"
    >
      {options.map((option) => (
        <label
          key={option.value}
          className={cn(
            "flex cursor-pointer items-start gap-2.5 rounded-lg border p-2.5 transition",
            value === option.value ? "border-accent bg-accent/[0.06]" : "border-line hover:bg-raised",
          )}
        >
          <RadioGroup.Item
            value={option.value}
            className="mt-0.5 h-4 w-4 shrink-0 rounded-full border border-line bg-surface data-[state=checked]:border-accent"
          >
            <RadioGroup.Indicator className="flex h-full w-full items-center justify-center after:block after:h-2 after:w-2 after:rounded-full after:bg-accent" />
          </RadioGroup.Item>
          <span className="min-w-0">
            <span className="block text-[0.8125rem] font-medium leading-snug">{option.label}</span>
            {option.hint ? (
              <span className="mt-0.5 block text-[0.75rem] subtle leading-snug">{option.hint}</span>
            ) : null}
          </span>
        </label>
      ))}
    </RadioGroup.Root>
  );
}

export function Range({
  value,
  onChange,
  min,
  max,
  step = 1,
  labels,
}: {
  value: number;
  onChange: (next: number) => void;
  min: number;
  max: number;
  step?: number;
  labels?: string[];
}) {
  return (
    <div>
      <Slider.Root
        value={[value]}
        min={min}
        max={max}
        step={step}
        onValueChange={([next]) => onChange(next)}
        className="relative flex h-6 w-full touch-none select-none items-center"
      >
        <Slider.Track className="relative h-1.5 w-full grow rounded-full bg-line">
          <Slider.Range className="absolute h-full rounded-full bg-accent" />
        </Slider.Track>
        <Slider.Thumb className="block h-4 w-4 rounded-full border-2 border-accent bg-surface shadow transition hover:scale-110" />
      </Slider.Root>
      {labels ? (
        <div className="mt-1 flex justify-between text-[0.6875rem] subtle">
          {labels.map((label, index) => (
            <span key={label} className={cn(index === value - min && "font-semibold text-ink")}>
              {label}
            </span>
          ))}
        </div>
      ) : null}
    </div>
  );
}

// ------------------------------------------------------------------ прогрес
export function Progress({
  value,
  indeterminate = false,
  tone = "accent",
  className,
}: {
  value: number;
  indeterminate?: boolean;
  tone?: "accent" | "ok" | "warn" | "danger";
  className?: string;
}) {
  const percent = Math.round(Math.max(0, Math.min(1, value)) * 100);
  const bar =
    tone === "ok" ? "bg-ok" : tone === "warn" ? "bg-warn" : tone === "danger" ? "bg-danger" : "bg-accent";
  return (
    <div
      role="progressbar"
      aria-valuenow={indeterminate ? undefined : percent}
      aria-valuemin={0}
      aria-valuemax={100}
      className={cn("h-1.5 w-full overflow-hidden rounded-full bg-line", className)}
    >
      {indeterminate ? (
        // Невизначений прогрес показується ЛИШЕ доки число справді невідоме
        // (до першої події job.progress). Далі — реальна частка: анімація
        // замість числа читається як робота там, де її може не бути.
        <div
          className={cn("h-full w-1/3 animate-shimmer rounded-full", bar)}
          style={{
            backgroundImage:
              "linear-gradient(90deg, transparent, hsl(var(--accent) / 0.9), transparent)",
            backgroundSize: "200% 100%",
          }}
        />
      ) : (
        <div className={cn("h-full rounded-full transition-[width] duration-500", bar)} style={{ width: `${percent}%` }} />
      )}
    </div>
  );
}

// ------------------------------------------------------------------ порожньо
export function EmptyState({
  icon,
  title,
  hint,
  action,
}: {
  icon?: ReactNode;
  title: ReactNode;
  hint?: ReactNode;
  action?: ReactNode;
}) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 rounded-xl border border-dashed border-line px-6 py-12 text-center">
      {icon ? <div className="text-muted">{icon}</div> : null}
      <div>
        <p className="text-[0.9375rem] font-medium">{title}</p>
        {hint ? <p className="mx-auto mt-1 max-w-md text-[0.8125rem] subtle">{hint}</p> : null}
      </div>
      {action}
    </div>
  );
}

// ------------------------------------------------------------------ підказки
export function TooltipProvider({ children }: { children: ReactNode }) {
  return (
    <Tooltip.Provider delayDuration={350} skipDelayDuration={200}>
      {children}
    </Tooltip.Provider>
  );
}

export function Hint({ children, content }: { children: ReactNode; content: ReactNode }) {
  return (
    <Tooltip.Root>
      <Tooltip.Trigger asChild>{children}</Tooltip.Trigger>
      <Tooltip.Portal>
        <Tooltip.Content
          sideOffset={6}
          collisionPadding={8}
          className="z-50 max-w-xs animate-fade-in rounded-lg border border-line bg-surface px-2.5 py-1.5 text-[0.75rem] leading-snug shadow-lg"
        >
          {content}
          <Tooltip.Arrow className="fill-[hsl(var(--surface))]" />
        </Tooltip.Content>
      </Tooltip.Portal>
    </Tooltip.Root>
  );
}

// ------------------------------------------------------------------ смуги
export function Banner({
  tone = "warn",
  title,
  children,
  action,
  icon,
}: {
  tone?: "warn" | "danger" | "ok" | "accent";
  title: ReactNode;
  children?: ReactNode;
  action?: ReactNode;
  icon?: ReactNode;
}) {
  const styles: Record<string, string> = {
    warn: "border-warn/40 bg-warn/10",
    danger: "border-danger/40 bg-danger/10",
    ok: "border-ok/35 bg-ok/10",
    accent: "border-accent/35 bg-accent/10",
  };
  return (
    <div className={cn("flex flex-wrap items-start gap-3 rounded-xl border p-3", styles[tone])}>
      {icon ? <div className="mt-0.5 shrink-0">{icon}</div> : null}
      <div className="min-w-0 flex-1">
        <p className="text-[0.8125rem] font-semibold">{title}</p>
        {children ? <div className="mt-1 text-[0.8125rem] leading-relaxed">{children}</div> : null}
      </div>
      {action ? <div className="shrink-0">{action}</div> : null}
    </div>
  );
}
