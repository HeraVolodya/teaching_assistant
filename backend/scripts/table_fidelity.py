"""Правдивість таблиць — Віха 2 плану.

Порівнює режими розпізнавання структури таблиць на еталонних балістичних і
нормативних таблицях, розмічених вручну:

    TableFormer V1 ACCURATE (+do_cell_matching)
    TableFormer V1 ACCURATE (-do_cell_matching)
    TableFormer V1 FAST                       ← дефолт macOS, див. нижче
    TableFormer V2

Чому V1, а не V2 за замовчуванням: план (§1) фіксує дві відкриті регресії V2
(#3158, #3553), тому в постачанні стоїть V1 ACCURATE. Цей скрипт існує, щоб
рішення трималося на власному вимірі, а не на посиланні на баг-трекер.

Чому це окрема віха, а не частина OCR-бейк-офу: у балістичній таблиці ціна
помилки інша. Неправильна літера у прозі коштує незручності; неправильне число
в комірці таблиці стрільби коштує неправильної поправки. І ламається воно
інакше — не розпізнаванням, а СТРУКТУРОЮ: об'єднаний заголовок «Дальність, м»
над трьома колонками, який рушій не розгорнув, зсуває всю праву частину
таблиці. Тому міряються два F1 (позиційний і змістовний), а їхня різниця і є
діагнозом (див. `app/eval/table_metrics.py`).

Apple Silicon: у `table_structure_model.py` Docling є жорсткий guard
`if device == MPS: device = CPU`. TableFormer працює на CPU на КОЖНОМУ Mac,
і issue #3202 наводить 10.4 с (MPS) проти 145.9 с (CPU) — втрата в 14 разів.
Тому на macOS ACCURATE може бути неприйнятно повільним, і колонка «с» у
таблиці результатів тут не менш важлива за F1.

Два режими роботи:

    # 1) повний: узяти PDF/зображення й прогнати Docling
    python -m scripts.table_fidelity --tables ./tables --gold ./tables/gold --out ./out

    # 2) без Docling: порівняти вже витягнуті таблиці
    python -m scripts.table_fidelity --gold ./tables/gold \\
        --predictions ./out/v1-accurate ./out/v2 --out ./out

Другий режим потрібен не лише для тестів: він дозволяє порівняти з чим завгодно —
з MinerU, з PaddleOCR-VL, з ручним експортом — доки вихід збережено як
`.csv`, `.md` або `.html` з тим самим іменем, що й еталон.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.eval import metrics as M
from app.eval.table_metrics import (
    Grid,
    TableScore,
    grid_from_html,
    grid_from_markdown,
    load_grid_csv,
    score_table,
)

__all__ = [
    "DOCLING_MODES",
    "TableCase",
    "collect_cases",
    "docling_available",
    "extract_with_docling",
    "load_grid",
    "load_predictions",
    "main",
    "render_markdown",
    "score_predictions",
]

TABLE_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}

# Режими, які має сенс порівнювати. Ключ → опції `ParseOptions` модуля приймання.
DOCLING_MODES: dict[str, dict[str, Any]] = {
    "v1-accurate": {"table_mode": "ACCURATE", "do_cell_matching": True, "table_former_version": 1},
    "v1-accurate-nocellmatch": {
        "table_mode": "ACCURATE", "do_cell_matching": False, "table_former_version": 1,
    },
    "v1-fast": {"table_mode": "FAST", "do_cell_matching": True, "table_former_version": 1},
    "v2-accurate": {"table_mode": "ACCURATE", "do_cell_matching": True, "table_former_version": 2},
}


@dataclass(frozen=True, slots=True)
class TableCase:
    """Еталонна таблиця й (за наявності) файл сторінки, з якої її знято."""

    name: str
    gold: Grid
    source: Path | None = None

    @property
    def cells(self) -> int:
        return sum(1 for row in self.gold for cell in row if cell.strip())


def load_grid(path: Path) -> Grid:
    """Завантажити сітку з `.csv`, `.tsv`, `.md` або `.html`."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return load_grid_csv(path)
    if suffix == ".tsv":
        return load_grid_csv(path, delimiter="\t")
    text = path.read_text(encoding="utf-8", errors="replace")
    if suffix in (".html", ".htm"):
        return grid_from_html(text)
    if suffix in (".md", ".markdown", ".txt"):
        return grid_from_markdown(text)
    raise ValueError(
        f"Невідомий формат таблиці: {path.name}. Підтримуються .csv, .tsv, .md, .html."
    )


def collect_cases(gold_dir: Path, tables_dir: Path | None = None) -> tuple[list[TableCase], list[str]]:
    """Зібрати еталони й зіставити їх зі сторінками за іменем без розширення."""
    warnings: list[str] = []
    sources: dict[str, Path] = {}
    if tables_dir is not None and tables_dir.is_dir():
        for path in sorted(tables_dir.iterdir()):
            if path.suffix.lower() in TABLE_SUFFIXES:
                sources[path.stem] = path

    cases: list[TableCase] = []
    for path in sorted(gold_dir.iterdir()):
        if path.suffix.lower() not in (".csv", ".tsv", ".md", ".html", ".htm"):
            continue
        try:
            grid = load_grid(path)
        except Exception as exc:
            warnings.append(f"Еталон {path.name} не прочитано: {exc}")
            continue
        if not grid:
            warnings.append(f"Еталон {path.name} порожній.")
            continue
        source = sources.get(path.stem)
        if tables_dir is not None and source is None:
            warnings.append(f"Для еталона {path.stem} немає сторінки в {tables_dir}.")
        cases.append(TableCase(name=path.stem, gold=grid, source=source))
    return cases, warnings


def load_predictions(directory: Path) -> dict[str, Grid]:
    """Прочитати вже витягнуті таблиці одного режиму з теки."""
    out: dict[str, Grid] = {}
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in (".csv", ".tsv", ".md", ".html", ".htm"):
            continue
        try:
            out[path.stem] = load_grid(path)
        except Exception:
            continue
    return out


def score_predictions(
    cases: Sequence[TableCase],
    predictions: dict[str, dict[str, Grid]],
    *,
    seconds: dict[str, dict[str, float]] | None = None,
) -> list[TableScore]:
    """Порахувати F1 для кожної пари (таблиця, режим).

    Відсутнє передбачення — це НЕ пропуск, а порожня сітка: рушій, який не
    знайшов таблицю на сторінці, помилився так само, як той, що знайшов її
    неправильно, і мовчазне виключення такого випадку завищило б його F1.
    """
    out: list[TableScore] = []
    for mode, grids in predictions.items():
        for case in cases:
            grid = grids.get(case.name)
            error = "" if grid is not None else "таблицю не знайдено"
            out.append(
                score_table(
                    case.name, mode, case.gold, grid or [],
                    seconds=(seconds or {}).get(mode, {}).get(case.name, 0.0),
                    error=error,
                )
            )
    return out


# ------------------------------------------------------------------- Docling
def docling_available() -> tuple[bool, str]:
    """Чи можна прогнати справжній Docling у цьому середовищі."""
    try:
        import docling  # type: ignore[import-not-found]
    except ImportError as exc:
        return False, f"docling не встановлено ({exc}). pip install .[worker]"
    return True, getattr(docling, "__version__", "версія невідома")


def extract_with_docling(
    case: TableCase,
    mode: str,
    *,
    options: dict[str, Any] | None = None,
) -> tuple[Grid, float, str]:
    """Витягти першу таблицю сторінки через модуль приймання.

    Свідомо через `app.ingestion.docling_pipeline.parse_document`, а не через
    сирий `DocumentConverter`: міряти треба ТОЙ конвеєр, що поїде до викладача,
    разом із його явними OCR-опціями. Вимір «чистого Docling» цікавий, але
    ним не можна обґрунтувати вибір для продукту.
    """
    from app.ingestion.docling_pipeline import ParseOptions, parse_document

    if case.source is None:
        return [], 0.0, "немає сторінки-джерела"

    spec = dict(DOCLING_MODES.get(mode, {}))
    spec.update(options or {})
    known = {f.name for f in ParseOptions.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    unknown = sorted(k for k in spec if k not in known)
    kwargs = {k: v for k, v in spec.items() if k in known}

    started = time.perf_counter()
    try:
        parsed = parse_document(case.source, ParseOptions(**kwargs))
    except Exception as exc:
        return [], time.perf_counter() - started, f"{type(exc).__name__}: {exc}"
    seconds = time.perf_counter() - started

    note = ""
    if unknown:
        # Наприклад, `table_former_version`: якщо модуль приймання ще не вміє
        # перемикати версію, режим НЕ вважається виміряним — інакше V1 і V2
        # дали б однакові числа, і висновок «різниці немає» був би хибним.
        note = f"опції не підтримані модулем приймання: {', '.join(unknown)}"

    tables = parsed.tables()
    if not tables:
        return [], seconds, note or "таблиць не знайдено"
    # Модуль приймання кладе СИТКУ в `meta["grid"]` — беремо її, а не текст:
    # `element.text` — це триплетна серіалізація («рядок, колонка = значення»),
    # яка навмисно втрачає координати, і відновлювати їх з неї означало б
    # міряти власний парсер замість TableFormer.
    meta = getattr(tables[0], "meta", {}) or {}
    grid = [[str(cell) for cell in row] for row in (meta.get("grid") or [])]
    if not grid:
        grid = grid_from_markdown(getattr(tables[0], "text", "") or "")
    if not grid:
        grid = grid_from_html(str(meta.get("html") or ""))
    return grid, seconds, note


# --------------------------------------------------------------------- звіт
@dataclass(slots=True)
class FidelityResult:
    scores: list[TableScore] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    notes: dict[str, str] = field(default_factory=dict)


def _mode_rows(scores: Sequence[TableScore]) -> list[dict[str, Any]]:
    """Зведення по режимах: середні F1, частка збігів форми, час."""
    by_mode: dict[str, list[TableScore]] = {}
    for score in scores:
        by_mode.setdefault(score.engine, []).append(score)
    rows: list[dict[str, Any]] = []
    for mode, items in sorted(by_mode.items()):
        good = [s for s in items if not s.error]
        rows.append({
            "режим": mode,
            "таблиць": len(items),
            "F1 позиц.": _mean(s.positional_f1 for s in good),
            "F1 зміст.": _mean(s.content_f1 for s in good),
            "втрата структури": _mean(s.structure_loss for s in good),
            "форма збіглась": _mean(1.0 if s.shape_ok else 0.0 for s in good),
            "порожніх комірок": _mean(s.empty_ratio for s in good),
            "потребує VLM-ремонту": sum(1 for s in good if s.needs_vlm_repair),
            "с/таблицю": _mean(s.seconds for s in items),
            "збоїв": sum(1 for s in items if s.error),
        })
    return rows


def _mean(values: Any) -> float:
    import math

    items = [v for v in values if isinstance(v, (int, float)) and not math.isnan(v)]
    return sum(items) / len(items) if items else math.nan


def render_markdown(result: FidelityResult) -> str:
    lines = [
        "# Правдивість таблиць (Віха 2)",
        "",
        "## Зведення по режимах",
        "",
        M.markdown_table(_mode_rows(result.scores)) or "_немає даних_",
        "",
        "**Як читати.** Високий змістовний F1 за низького позиційного означає, що "
        "значення прочитані, а структура поламана: лікується іншим режимом "
        "TableFormer, а не іншим OCR. Низькі обидва — провал розпізнавання. "
        "Частка порожніх комірок понад 30% — це той самий тригер VLM-ремонту, що "
        "працює в рантаймі.",
        "",
        "## Поклітинні результати",
        "",
        M.markdown_table([s.as_dict() for s in result.scores]) or "_немає даних_",
    ]
    if sys.platform == "darwin":
        lines += [
            "",
            "## Зауваження про Apple Silicon",
            "",
            "У `table_structure_model.py` Docling є жорсткий guard "
            "`if device == MPS: device = CPU`, тож TableFormer на цій машині рахувався "
            "на CPU. Issue #3202 наводить 10.4 с (MPS) проти 145.9 с (CPU) — втрата в "
            "14 разів. Колонку «с/таблицю» з macOS не можна порівнювати з Windows.",
        ]
    if result.notes:
        lines += ["", "## Обмеження вимірювання", ""]
        lines += [f"- **{mode}** — {note}" for mode, note in sorted(result.notes.items()) if note]
    if result.warnings:
        lines += ["", "## Попередження", ""]
        lines += [f"- {w}" for w in result.warnings]
    return "\n".join(lines)


def write_csv(result: FidelityResult, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "table_fidelity.csv"
    rows = [s.as_dict() for s in result.scores]
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        if rows:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return path


# ----------------------------------------------------------------------- CLI
def main(argv: Sequence[str] | None = None) -> int:
    from app import net_guard

    net_guard.install()

    parser = argparse.ArgumentParser(
        prog="table_fidelity",
        description="Поклітинний F1 таблиць: TableFormer V1 ACCURATE ±do_cell_matching проти V2.",
    )
    parser.add_argument("--gold", required=True, type=Path,
                        help="Тека з еталонними таблицями (.csv/.md/.html).")
    parser.add_argument("--tables", type=Path, default=None, help="Тека зі сторінками для прогону Docling.")
    parser.add_argument(
        "--predictions", type=Path, nargs="*", default=[],
        help="Теки з уже витягнутими таблицями; ім'я теки стає назвою режиму.",
    )
    parser.add_argument("--modes", default="", help="Режими Docling через кому; порожньо — усі відомі.")
    parser.add_argument("--out", type=Path, default=Path("./table_fidelity"), help="Куди класти звіт.")
    args = parser.parse_args(argv)

    if not args.gold.is_dir():
        raise SystemExit(f"Теки з еталонами не існує: {args.gold}")

    cases, warnings = collect_cases(args.gold, args.tables)
    if not cases:
        raise SystemExit(f"У {args.gold} немає жодного еталона (.csv/.md/.html).")

    predictions: dict[str, dict[str, Grid]] = {}
    seconds: dict[str, dict[str, float]] = {}
    notes: dict[str, str] = {}

    for directory in args.predictions:
        if not directory.is_dir():
            warnings.append(f"Теки з передбаченнями не існує: {directory}")
            continue
        predictions[directory.name] = load_predictions(directory)

    if args.tables is not None:
        ok, detail = docling_available()
        if not ok:
            warnings.append(f"Docling недоступний: {detail}. Режими TableFormer пропущено.")
        else:
            modes = [m.strip() for m in args.modes.split(",") if m.strip()] or list(DOCLING_MODES)
            for mode in modes:
                grids: dict[str, Grid] = {}
                times: dict[str, float] = {}
                for case in cases:
                    grid, elapsed, note = extract_with_docling(case, mode)
                    grids[case.name] = grid
                    times[case.name] = elapsed
                    if note:
                        notes[mode] = note
                predictions[mode] = grids
                seconds[mode] = times

    if not predictions:
        raise SystemExit(
            "Немає жодного режиму для порівняння. Подайте --tables (прогін Docling) "
            "або --predictions (готові таблиці)."
        )

    result = FidelityResult(
        scores=score_predictions(cases, predictions, seconds=seconds),
        warnings=warnings,
        notes=notes,
    )

    print(M.markdown_table(_mode_rows(result.scores)))
    for warning in result.warnings:
        print(f"[увага] {warning}")

    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = write_csv(result, args.out)
    md_path = args.out / "table_fidelity.md"
    md_path.write_text(render_markdown(result), encoding="utf-8")
    (args.out / "table_fidelity.json").write_text(
        json.dumps(
            {
                "platform": sys.platform,
                "modes": _mode_rows(result.scores),
                "scores": [s.as_dict() for s in result.scores],
                "notes": result.notes,
                "warnings": result.warnings,
            },
            ensure_ascii=False, indent=2, default=str,
        ),
        encoding="utf-8",
    )
    print(f"\nЗаписано:\n  {csv_path}\n  {md_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
