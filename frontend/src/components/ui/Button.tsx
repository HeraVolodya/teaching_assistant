import { forwardRef, type ButtonHTMLAttributes } from "react";

import { cn } from "@/lib/cn";

export type ButtonVariant = "primary" | "secondary" | "ghost" | "danger" | "warn";
export type ButtonSize = "sm" | "md" | "lg";

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
  loading?: boolean;
}

const VARIANTS: Record<ButtonVariant, string> = {
  primary: "bg-accent text-accent-ink hover:brightness-110 active:brightness-95",
  secondary: "bg-raised text-ink border border-line hover:bg-bg",
  ghost: "text-ink hover:bg-raised",
  danger: "bg-danger text-white hover:brightness-110",
  warn: "bg-warn text-[hsl(30_60%_14%)] hover:brightness-105",
};

/**
 * Розміри задані в rem, а не в px.
 *
 * На 125% і 150% DPI Windows множить CSS-пікселі сам, але фіксована
 * `height: 32px` при збільшеному шрифті дає кнопку, з якої вилазить текст.
 * `min-height` у rem росте разом із базовим кеглем, тому масштаб інтерфейсу
 * лишається одним множником, а не переліком винятків.
 */
const SIZES: Record<ButtonSize, string> = {
  sm: "min-h-[1.75rem] px-2.5 text-[0.8125rem] gap-1.5",
  md: "min-h-[2.25rem] px-3.5 gap-2",
  lg: "min-h-[2.75rem] px-5 text-[0.9375rem] gap-2",
};

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  { className, variant = "secondary", size = "md", loading = false, disabled, children, ...rest },
  ref,
) {
  return (
    <button
      ref={ref}
      disabled={disabled || loading}
      className={cn(
        "inline-flex items-center justify-center rounded-lg font-medium transition",
        "disabled:pointer-events-none disabled:opacity-50",
        VARIANTS[variant],
        SIZES[size],
        className,
      )}
      {...rest}
    >
      {loading ? <Spinner className="shrink-0" /> : null}
      {children}
    </button>
  );
});

export function Spinner({ className }: { className?: string }) {
  return (
    <span
      role="status"
      aria-label="Виконується"
      className={cn(
        "inline-block h-3.5 w-3.5 animate-spin rounded-full border-2 border-current border-r-transparent",
        className,
      )}
    />
  );
}

/** Кнопка-іконка. `title` обов'язковий: без нього вона недоступна з клавіатури й для читача екрана. */
export const IconButton = forwardRef<
  HTMLButtonElement,
  ButtonHTMLAttributes<HTMLButtonElement> & { title: string; active?: boolean }
>(function IconButton({ className, title, active = false, ...rest }, ref) {
  return (
    <button
      ref={ref}
      title={title}
      aria-label={title}
      aria-pressed={active || undefined}
      className={cn(
        "inline-flex min-h-[2rem] min-w-[2rem] items-center justify-center rounded-lg text-muted transition",
        "hover:bg-raised hover:text-ink disabled:pointer-events-none disabled:opacity-40",
        active && "bg-raised text-ink",
        className,
      )}
      {...rest}
    />
  );
});
