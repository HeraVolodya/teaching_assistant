/**
 * Головний екран: сітка асистентів.
 *
 * ЧЕРВОНА КРАПКА НА КАРТЦІ — головна деталь цього екрана. Матеріал, який не
 * проіндексувався, не зникає з бази: він просто мовчки не бере участі у
 * відповідях. Викладач ставить питання, отримує відповідь без потрібного
 * підручника й вирішує, що застосунок поганий. Крапка на картці — єдине
 * місце, де це видно ДО того, як він поставить питання.
 *
 * ВИДАЛЕННЯ ВИМАГАЄ ВВЕСТИ НАЗВУ. Асистент — це десятки годин індексації;
 * «Ви впевнені? [Так]» тут не запобіжник, а формальність, яку натискають
 * не читаючи.
 */

import { CopyPlus, MoreVertical, Plus, Trash2, TriangleAlert } from "lucide-react";
import { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { useNavigate } from "react-router-dom";

import type { Assistant } from "@/api/types";
import { Button, IconButton } from "@/components/ui/Button";
import { Menu, Modal } from "@/components/ui/Overlays";
import { Badge, Dot, EmptyState, Field, Input, Textarea } from "@/components/ui/Primitives";
import {
  useAssistants,
  useCreateAssistant,
  useDeleteAssistant,
  useDocuments,
} from "@/hooks/queries";
import { cn } from "@/lib/cn";
import { DOCUMENTS, PAGES, formatNumber, pluralize } from "@/lib/format";

const PALETTE = ["#4f46e5", "#0f766e", "#b45309", "#9333ea", "#be123c", "#0369a1", "#4d7c0f"];
const EMOJI = ["📘", "🎯", "🗺️", "⚙️", "🧭", "📐", "🛡️", "📡", "🚚", "🔬"];

export function AssistantsScreen() {
  const { t } = useTranslation();
  const { data: assistants = [], isLoading } = useAssistants();
  const [creating, setCreating] = useState(false);
  const [deleting, setDeleting] = useState<Assistant | null>(null);

  return (
    <div className="scroll-thin min-h-0 flex-1 overflow-y-auto px-5 py-6 sm:px-8">
      <div className="mx-auto max-w-5xl">
        <header className="mb-6 flex flex-wrap items-end justify-between gap-3">
          <div>
            <h1 className="text-[1.25rem] font-semibold tracking-tight">{t("assistants.title")}</h1>
            <p className="mt-1 text-[0.8125rem] subtle">{t("assistants.subtitle")}</p>
          </div>
          <Button variant="primary" onClick={() => setCreating(true)}>
            <Plus size={15} />
            {t("assistants.create")}
          </Button>
        </header>

        {isLoading ? (
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {[0, 1, 2].map((index) => (
              <div key={index} className="card h-32 animate-pulse bg-raised" />
            ))}
          </div>
        ) : assistants.length ? (
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {assistants.map((assistant) => (
              <AssistantCard key={assistant.id} assistant={assistant} onDelete={() => setDeleting(assistant)} />
            ))}
          </div>
        ) : (
          <EmptyState
            title={t("assistants.empty")}
            hint={t("assistants.emptyHint")}
            action={
              <Button variant="primary" onClick={() => setCreating(true)}>
                <Plus size={15} />
                {t("assistants.create")}
              </Button>
            }
          />
        )}
      </div>

      <CreateDialog open={creating} onOpenChange={setCreating} />
      <DeleteDialog assistant={deleting} onClose={() => setDeleting(null)} />
    </div>
  );
}

function AssistantCard({ assistant, onDelete }: { assistant: Assistant; onDelete: () => void }) {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const duplicate = useCreateAssistant();
  const collectionId = assistant.collections[0]?.id;
  const { data: documents = [] } = useDocuments(collectionId);

  /*
   * Лічильники беруться з колекції, доки не завантажився список документів:
   * інакше картка на частку секунди показує «0 документів» — тобто рівно те,
   * що змушує подумати, ніби матеріали зникли.
   *
   * Сам список усе одно потрібен: кількість НЕОБРОБЛЕНИХ матеріалів (червона
   * крапка) у зведенні колекції не приходить, а це найважливіше число на
   * цьому екрані.
   */
  const stats = useMemo(() => {
    const fallbackCount = assistant.collections.reduce((sum, c) => sum + c.documents, 0);
    if (!documents.length) return { failed: 0, pages: 0, count: fallbackCount };
    return {
      failed: documents.filter((d) => d.status === "FAILED").length,
      pages: documents.reduce((sum, d) => sum + (d.pageCount || 0), 0),
      count: documents.length,
    };
  }, [documents, assistant.collections]);

  const stale = assistant.collections.some((collection) => collection.dirty);

  return (
    <article
      className={cn(
        "card group relative flex cursor-pointer flex-col p-4 transition",
        "hover:border-accent/50 hover:shadow-sm",
      )}
      onClick={() => navigate(`/a/${assistant.id}/chat`)}
    >
      <div className="mb-2.5 flex items-start gap-2.5">
        <span
          className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg text-[1.125rem]"
          style={{ backgroundColor: `${assistant.colour}1f` }}
          aria-hidden
        >
          {assistant.emoji}
        </span>
        <div className="min-w-0 flex-1">
          <h2 className="truncate text-[0.9375rem] font-semibold leading-tight">{assistant.name}</h2>
          {assistant.description ? (
            <p className="mt-0.5 line-clamp-2 text-[0.75rem] subtle">{assistant.description}</p>
          ) : null}
        </div>
        <div onClick={(event) => event.stopPropagation()}>
          <Menu
            trigger={
              <IconButton title={t("common.more")} className="opacity-0 group-hover:opacity-100 focus:opacity-100">
                <MoreVertical size={15} />
              </IconButton>
            }
            items={[
              {
                id: "duplicate",
                label: t("assistants.duplicate"),
                icon: <CopyPlus size={14} />,
                onSelect: () =>
                  duplicate.mutate({
                    name: `${assistant.name} (копія)`,
                    description: assistant.description,
                    colour: assistant.colour,
                    emoji: assistant.emoji,
                    // Копіюється КОНФІГ, а не матеріали: індекс — фізична
                    // партиція на диску, і його дублювання коштувало б
                    // стільки ж, скільки повна переіндексація.
                    config: assistant.config,
                  }),
              },
              {
                id: "delete",
                label: t("common.delete"),
                icon: <Trash2 size={14} />,
                danger: true,
                separatorBefore: true,
                onSelect: onDelete,
              },
            ]}
          />
        </div>
      </div>

      <p className="mt-auto text-[0.75rem] tabular-nums subtle">
        {pluralize(stats.count, DOCUMENTS)} · {formatNumber(stats.pages)}{" "}
        {stats.pages === 1 ? PAGES[0] : PAGES[2]}
      </p>

      <div className="mt-2 flex flex-wrap items-center gap-1.5">
        {stats.failed ? (
          <Badge tone="danger">
            <Dot tone="danger" title={t("assistants.hasFailures")} />
            {t("assistants.hasFailures")}
          </Badge>
        ) : null}
        {stale ? <Badge tone="warn">{t("assistants.indexStale")}</Badge> : null}
      </div>
    </article>
  );
}

function CreateDialog({ open, onOpenChange }: { open: boolean; onOpenChange: (open: boolean) => void }) {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const create = useCreateAssistant();
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [emoji, setEmoji] = useState(EMOJI[0]);
  const [colour, setColour] = useState(PALETTE[0]);

  const submit = () => {
    if (!name.trim()) return;
    create.mutate(
      { name: name.trim(), description: description.trim(), emoji, colour },
      {
        onSuccess: (assistant) => {
          onOpenChange(false);
          setName("");
          setDescription("");
          // Одразу на екран матеріалів: щойно створений асистент без
          // жодного підручника не вміє нічого, і показувати йому чат —
          // це запросити поставити питання, на яке не буде відповіді.
          navigate(`/a/${assistant.id}/documents`);
        },
      },
    );
  };

  return (
    <Modal
      open={open}
      onOpenChange={onOpenChange}
      title={t("assistants.create")}
      footer={
        <>
          <Button onClick={() => onOpenChange(false)}>{t("common.cancel")}</Button>
          <Button variant="primary" onClick={submit} disabled={!name.trim()} loading={create.isPending}>
            {t("common.create")}
          </Button>
        </>
      }
    >
      <div className="space-y-3">
        <Field label={t("assistants.nameLabel")} htmlFor="assistant-name">
          <Input
            id="assistant-name"
            value={name}
            autoFocus
            onChange={(event) => setName(event.target.value)}
            onKeyDown={(event) => event.key === "Enter" && submit()}
            placeholder="Наприклад: Вогнева підготовка"
          />
        </Field>
        <Field label={t("assistants.descriptionLabel")} htmlFor="assistant-description">
          <Textarea
            id="assistant-description"
            rows={2}
            value={description}
            onChange={(event) => setDescription(event.target.value)}
          />
        </Field>
        <Field label={t("assistants.emojiLabel")}>
          <div className="flex flex-wrap gap-1">
            {EMOJI.map((item) => (
              <button
                key={item}
                type="button"
                onClick={() => setEmoji(item)}
                className={cn(
                  "flex h-8 w-8 items-center justify-center rounded-lg border text-[1rem] transition",
                  emoji === item ? "border-accent bg-accent/10" : "border-line hover:bg-raised",
                )}
              >
                {item}
              </button>
            ))}
          </div>
        </Field>
        <Field label={t("assistants.colourLabel")}>
          <div className="flex flex-wrap gap-1.5">
            {PALETTE.map((item) => (
              <button
                key={item}
                type="button"
                aria-label={item}
                onClick={() => setColour(item)}
                style={{ backgroundColor: item }}
                className={cn(
                  "h-7 w-7 rounded-full transition",
                  colour === item ? "ring-2 ring-accent ring-offset-2 ring-offset-[hsl(var(--surface))]" : "",
                )}
              />
            ))}
          </div>
        </Field>
      </div>
    </Modal>
  );
}

function DeleteDialog({ assistant, onClose }: { assistant: Assistant | null; onClose: () => void }) {
  const { t } = useTranslation();
  const remove = useDeleteAssistant();
  const [typed, setTyped] = useState("");

  const matches = assistant != null && typed.trim() === assistant.name.trim();

  return (
    <Modal
      open={assistant != null}
      onOpenChange={(open) => {
        if (!open) {
          setTyped("");
          onClose();
        }
      }}
      title={t("assistants.deleteTitle")}
      description={t("assistants.deleteWarning")}
      footer={
        <>
          <Button onClick={onClose}>{t("common.cancel")}</Button>
          <Button
            variant="danger"
            disabled={!matches}
            loading={remove.isPending}
            onClick={() =>
              assistant &&
              remove.mutate(assistant.id, {
                onSuccess: () => {
                  setTyped("");
                  onClose();
                },
              })
            }
          >
            <Trash2 size={14} />
            {t("common.delete")}
          </Button>
        </>
      }
    >
      <div className="flex items-start gap-2.5 rounded-lg border border-danger/35 bg-danger/[0.07] p-3">
        <TriangleAlert size={16} className="mt-0.5 shrink-0 text-danger" />
        <div className="min-w-0 flex-1">
          <label htmlFor="delete-confirm" className="text-[0.8125rem]">
            {t("assistants.deleteConfirmLabel")}
          </label>
          <Input
            id="delete-confirm"
            value={typed}
            onChange={(event) => setTyped(event.target.value)}
            placeholder={assistant?.name}
            className="mt-2"
          />
        </div>
      </div>
    </Modal>
  );
}
