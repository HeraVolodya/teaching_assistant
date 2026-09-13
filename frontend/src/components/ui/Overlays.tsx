/**
 * Модальні шари: діалоги, меню, поповери.
 *
 * ПРАВИЛО, ЩО ДІЄ НА ВЕСЬ ЗАСТОСУНОК: нічого модального в процесі обробки
 * матеріалів. Діалог тут — лише для НЕЗВОРОТНИХ дій (видалення) і для
 * створення, тобто там, де користувач сам щойно натиснув кнопку й чекає
 * питання. Прогрес, помилки й підказки живуть у рядках і банерах.
 */

import * as Dialog from "@radix-ui/react-dialog";
import * as DropdownMenu from "@radix-ui/react-dropdown-menu";
import * as Popover from "@radix-ui/react-popover";
import { X } from "lucide-react";
import type { ReactNode } from "react";

import { cn } from "@/lib/cn";

const OVERLAY =
  "fixed inset-0 z-40 bg-black/45 data-[state=open]:animate-fade-in backdrop-blur-[1px]";
const PANEL =
  "fixed left-1/2 top-1/2 z-50 w-[min(32rem,calc(100vw-2rem))] -translate-x-1/2 -translate-y-1/2 " +
  "rounded-xl border border-line bg-surface p-5 shadow-2xl data-[state=open]:animate-fade-in";

export function Modal({
  open,
  onOpenChange,
  title,
  description,
  children,
  footer,
  wide = false,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title: ReactNode;
  description?: ReactNode;
  children?: ReactNode;
  footer?: ReactNode;
  wide?: boolean;
}) {
  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className={OVERLAY} />
        <Dialog.Content className={cn(PANEL, wide && "w-[min(48rem,calc(100vw-2rem))]")}>
          <div className="mb-3 flex items-start justify-between gap-4">
            <div className="min-w-0">
              <Dialog.Title className="text-[0.9375rem] font-semibold">{title}</Dialog.Title>
              {description ? (
                <Dialog.Description className="mt-1 text-[0.8125rem] subtle leading-relaxed">
                  {description}
                </Dialog.Description>
              ) : null}
            </div>
            <Dialog.Close
              aria-label="Закрити"
              className="-mr-1 -mt-1 rounded-lg p-1.5 text-muted transition hover:bg-raised hover:text-ink"
            >
              <X size={16} />
            </Dialog.Close>
          </div>
          {children}
          {footer ? <div className="mt-5 flex justify-end gap-2">{footer}</div> : null}
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}

export interface MenuItem {
  id: string;
  label: ReactNode;
  icon?: ReactNode;
  onSelect: () => void;
  danger?: boolean;
  disabled?: boolean;
  separatorBefore?: boolean;
}

export function Menu({ trigger, items }: { trigger: ReactNode; items: MenuItem[] }) {
  return (
    <DropdownMenu.Root>
      <DropdownMenu.Trigger asChild>{trigger}</DropdownMenu.Trigger>
      <DropdownMenu.Portal>
        <DropdownMenu.Content
          align="end"
          sideOffset={4}
          collisionPadding={8}
          className="z-50 min-w-[12rem] animate-fade-in rounded-xl border border-line bg-surface p-1 shadow-xl"
        >
          {items.map((item) => (
            <div key={item.id}>
              {item.separatorBefore ? <DropdownMenu.Separator className="my-1 h-px bg-line" /> : null}
              <DropdownMenu.Item
                disabled={item.disabled}
                onSelect={item.onSelect}
                className={cn(
                  "flex cursor-pointer items-center gap-2 rounded-lg px-2.5 py-1.5 text-[0.8125rem] outline-none",
                  "data-[highlighted]:bg-raised data-[disabled]:pointer-events-none data-[disabled]:opacity-40",
                  item.danger && "text-danger data-[highlighted]:bg-danger/10",
                )}
              >
                {item.icon}
                {item.label}
              </DropdownMenu.Item>
            </div>
          ))}
        </DropdownMenu.Content>
      </DropdownMenu.Portal>
    </DropdownMenu.Root>
  );
}

/**
 * Поповер цитати. Відкривається за наведенням І за кліком: наведення — щоб
 * перевірити цитату не втрачаючи місця у відповіді, клік — щоб відкрити
 * джерело. Керований ззовні, бо наведення на пігулку [1] має закриватися
 * тоді, коли курсор пішов і з пігулки, і з самого поповера.
 */
export function HoverPopover({
  open,
  onOpenChange,
  trigger,
  children,
  side = "top",
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  trigger: ReactNode;
  children: ReactNode;
  side?: "top" | "bottom" | "left" | "right";
}) {
  return (
    <Popover.Root open={open} onOpenChange={onOpenChange}>
      <Popover.Anchor asChild>{trigger}</Popover.Anchor>
      <Popover.Portal>
        <Popover.Content
          side={side}
          sideOffset={8}
          collisionPadding={12}
          onOpenAutoFocus={(event) => event.preventDefault()}
          onMouseEnter={() => onOpenChange(true)}
          onMouseLeave={() => onOpenChange(false)}
          className="z-50 w-[min(26rem,calc(100vw-2rem))] animate-fade-in rounded-xl border border-line bg-surface p-3 shadow-2xl"
        >
          {children}
          <Popover.Arrow className="fill-[hsl(var(--surface))]" />
        </Popover.Content>
      </Popover.Portal>
    </Popover.Root>
  );
}
