"""Вимір швидкості приймання: хв/стор., ConfidenceReport, пам'ять.

План (§1) обіцяє конкретні числа, і їх треба або підтвердити, або виправити на
реальному корпусі:

    Windows, режим «Швидко»: ~0.4–0.7 с/стор.  → 1000 стор. за 7–12 хв
    macOS,  режим «Швидко»: 1.3–2.5 с/стор.    → 1000 стор. за 22–40 хв

Розбіжність не випадкова, і скрипт має її ПОКАЗАТИ, а не згладити. У
`docling/models/table_structure_model.py` стоїть жорсткий guard
`if device == MPS: device = CPU` з коментарем «Disable MPS here, until we know
why it makes things slower». Наслідок: **TableFormer працює на CPU на кожному
Mac**, а issue #3202 наводить 10.4 с (MPS) проти 145.9 с (CPU) — втрата в
14 разів. Тому на Mac найгірший випадок — не сканована, а ТАБЛИЧНО-ЩІЛЬНА
сторінка, і саме тому дефолт `ParseOptions.resolved_table_mode()` на macOS —
`FAST`, а не `ACCURATE`.

`detect_tableformer_guard()` перевіряє наявність цього guard'а у ВСТАНОВЛЕНІЙ
версії Docling, а не покладається на те, що він там і досі є: якщо апстрім його
прибере, звіт мусить це сказати, а дефолт macOS — змінитися.

Приклад:

    python -m scripts.bench_ingest --docs ./corpus --modes FAST,DEEP --out ./bench
    python -m scripts.bench_ingest --docs ./corpus --max-pages 50 --repeat 2
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from app.domain import IngestMode, QualityGrade
from app.eval import metrics as M

__all__ = [
    "BenchResult",
    "DocumentBench",
    "GuardReport",
    "bench_document",
    "detect_tableformer_guard",
    "grade_distribution",
    "main",
    "peak_memory_mb",
    "render_markdown",
    "run_bench",
]


# --------------------------------------------------------------------- пам'ять
def peak_memory_mb() -> float:
    """Пікова резидентна пам'ять процесу, МБ.

    УВАГА на одиниці: `ru_maxrss` на macOS повертає БАЙТИ, на Linux —
    КІЛОБАЙТИ. Переплутати їх означає помилитись у 1024 рази й написати у звіт
    «12 ГБ» замість «12 МБ». На Windows `resource` немає взагалі, тому там
    фолбек на `tracemalloc` (він міряє лише пам'ять Python — тобто НЕ бачить
    алокацій усередині torch і onnxruntime, і це чесно позначено у звіті).
    """
    try:
        import resource

        raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return raw / (1024 * 1024) if sys.platform == "darwin" else raw / 1024
    except ImportError:
        try:
            import tracemalloc

            if tracemalloc.is_tracing():
                return tracemalloc.get_traced_memory()[1] / (1024 * 1024)
        except Exception:
            pass
        return float("nan")


# ------------------------------------------------------- guard TableFormer/MPS
@dataclass(frozen=True, slots=True)
class GuardReport:
    """Чи справді TableFormer падає на CPU в цій версії Docling."""

    docling_installed: bool
    version: str = ""
    guard_present: bool | None = None
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def detect_tableformer_guard() -> GuardReport:
    """Пошукати `MPS → CPU` у джерелі моделі структури таблиць."""
    try:
        import docling  # type: ignore[import-not-found]
    except ImportError as exc:
        return GuardReport(False, detail=f"docling не встановлено ({exc})")

    version = str(getattr(docling, "__version__", "невідома"))
    try:
        import inspect

        from docling.models import table_structure_model  # type: ignore[import-not-found]

        source = inspect.getsource(table_structure_model)
    except Exception as exc:
        return GuardReport(True, version, None, f"джерело недоступне: {exc}")

    lowered = source.lower()
    present = "mps" in lowered and "device = accelerator" not in lowered
    hint = "MPS" in source and "CPU" in source
    return GuardReport(
        docling_installed=True,
        version=version,
        guard_present=bool(present and hint),
        detail=(
            "guard `if device == MPS: device = CPU` знайдено — TableFormer рахується на CPU"
            if present and hint
            else "guard не знайдено; перевірити, чи апстрім його прибрав, і переглянути "
                 "дефолт table_mode для macOS"
        ),
    )


# --------------------------------------------------------------------- вимір
@dataclass(slots=True)
class DocumentBench:
    """Результат приймання одного документа в одному режимі."""

    document: str
    mode: str
    pages: int
    seconds: float
    tables: int
    pictures: int
    formulas: int
    grades: dict[str, int] = field(default_factory=dict)
    scanned_pages: int = 0
    repair_pages: int = 0
    peak_memory_mb: float = float("nan")
    warnings: list[str] = field(default_factory=list)
    error: str = ""

    @property
    def seconds_per_page(self) -> float:
        return self.seconds / self.pages if self.pages else float("nan")

    @property
    def minutes_per_1000_pages(self) -> float:
        return self.seconds_per_page * 1000 / 60.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "документ": self.document,
            "режим": self.mode,
            "сторінок": self.pages,
            "с": round(self.seconds, 3),
            "с/стор.": self.seconds_per_page,
            "хв/1000 стор.": self.minutes_per_1000_pages,
            "таблиць": self.tables,
            "рисунків": self.pictures,
            "формул": self.formulas,
            "сканованих": self.scanned_pages,
            "потребує VLM-ремонту": self.repair_pages,
            "пік пам'яті, МБ": self.peak_memory_mb,
            "попереджень": len(self.warnings),
            "помилка": self.error,
            **{f"grade:{k}": v for k, v in sorted(self.grades.items())},
        }


def grade_distribution(pages: Sequence[Any]) -> dict[str, int]:
    """Розподіл `ConfidenceReport` по сторінках.

    Оцінка сторінки — НАЙГІРША з наявних (parse / layout / table / ocr), а не
    середня: сторінка, де таблиця розібрана на 0.3, а решта на 0.95, — це
    сторінка з поламаною таблицею, і усереднення ховає рівно те, заради чого
    вимір робиться. Це той самий принцип, що й `Document.quality_grade`
    («mean_grade найгіршої сторінки» у схемі БД).
    """
    counts: dict[str, int] = {}
    for page in pages:
        scores = [
            s for s in (
                getattr(page, "parse_score", None),
                getattr(page, "layout_score", None),
                getattr(page, "table_score", None),
                getattr(page, "ocr_score", None),
            ) if isinstance(s, (int, float))
        ]
        key = QualityGrade.from_score(min(scores)).value if scores else "НЕВІДОМО"
        counts[key] = counts.get(key, 0) + 1
    return counts


def bench_document(
    path: Path,
    mode: IngestMode,
    *,
    max_pages: int | None = None,
    table_mode: str | None = None,
    device: str = "cpu",
) -> DocumentBench:
    """Прийняти один документ і виміряти час, якість і пам'ять."""
    from app.ingestion.docling_pipeline import ParseOptions, parse_document

    options = ParseOptions(
        ingest_mode=mode,
        max_pages=max_pages,
        table_mode=table_mode,  # type: ignore[arg-type]
        device=device,  # type: ignore[arg-type]
    )
    started = time.perf_counter()
    try:
        parsed = parse_document(path, options)
    except Exception as exc:
        return DocumentBench(
            document=path.name, mode=mode.value, pages=0,
            seconds=time.perf_counter() - started,
            tables=0, pictures=0, formulas=0,
            error=f"{type(exc).__name__}: {exc}",
        )
    seconds = time.perf_counter() - started

    from app.domain import PageClass, PageInfo

    repair = 0
    scanned = 0
    for page in parsed.pages:
        if getattr(page, "page_class", None) in (PageClass.SCANNED, PageClass.MIXED):
            scanned += 1
        info = PageInfo(
            page_number=page.page_number,
            parse_score=page.parse_score,
            layout_score=page.layout_score,
            table_score=page.table_score,
            lexicon_hit_rate=page.lexicon_hit_rate,
            cyrillic_ratio=page.cyrillic_ratio,
        )
        if info.needs_vlm_repair():
            repair += 1

    return DocumentBench(
        document=path.name,
        mode=mode.value,
        pages=parsed.page_count,
        seconds=seconds,
        tables=len(parsed.tables()),
        pictures=len(parsed.pictures()),
        formulas=len(parsed.formulas()),
        grades=grade_distribution(parsed.pages),
        scanned_pages=scanned,
        repair_pages=repair,
        peak_memory_mb=peak_memory_mb(),
        warnings=list(parsed.warnings),
    )


@dataclass(slots=True)
class BenchResult:
    benches: list[DocumentBench] = field(default_factory=list)
    guard: GuardReport | None = None
    environment: dict[str, Any] = field(default_factory=dict)

    def by_mode(self) -> dict[str, list[DocumentBench]]:
        out: dict[str, list[DocumentBench]] = {}
        for bench in self.benches:
            out.setdefault(bench.mode, []).append(bench)
        return out


def _environment() -> dict[str, Any]:
    return {
        "platform": sys.platform,
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", ""),
        "ASISTENT_STUB": os.environ.get("ASISTENT_STUB", ""),
    }


def run_bench(
    documents: Sequence[Path],
    modes: Sequence[IngestMode],
    *,
    max_pages: int | None = None,
    repeat: int = 1,
    table_mode: str | None = None,
    device: str = "cpu",
    verbose: bool = True,
) -> BenchResult:
    """Прогнати всі документи в усіх режимах.

    `repeat` існує заради першого прогону: він включає завантаження моделей
    Docling (секунди-десятки секунд) і тому систематично завищує с/стор. на
    коротких документах. Друга й наступні ітерації міряють усталений режим —
    саме той, у якому проходить 1000-сторінковий підручник.
    """
    result = BenchResult(guard=detect_tableformer_guard(), environment=_environment())
    for iteration in range(max(1, repeat)):
        for mode in modes:
            for path in documents:
                bench = bench_document(
                    path, mode, max_pages=max_pages, table_mode=table_mode, device=device
                )
                bench.document = (
                    bench.document if iteration == 0 else f"{bench.document} (прогін {iteration + 1})"
                )
                result.benches.append(bench)
                if verbose:
                    mark = bench.error or f"{bench.seconds_per_page:.3f} с/стор."
                    print(f"  [{mode.value}] {path.name}: {bench.pages} стор., {mark}")
    return result


# --------------------------------------------------------------------- звіт
def _mode_rows(result: BenchResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mode, benches in sorted(result.by_mode().items()):
        good = [b for b in benches if not b.error and b.pages]
        per_page = [b.seconds_per_page for b in good]
        rows.append({
            "режим": mode,
            "документів": len(benches),
            "сторінок": sum(b.pages for b in good),
            "с/стор. сер.": statistics.fmean(per_page) if per_page else float("nan"),
            "с/стор. мед.": statistics.median(per_page) if per_page else float("nan"),
            "с/стор. p95": M.percentiles(per_page, (95,))[0] if per_page else float("nan"),
            "хв/1000 стор.": (statistics.fmean(per_page) * 1000 / 60.0) if per_page else float("nan"),
            "пік пам'яті, МБ": max((b.peak_memory_mb for b in good), default=float("nan")),
            "збоїв": sum(1 for b in benches if b.error),
        })
    return rows


def _grade_rows(result: BenchResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mode, benches in sorted(result.by_mode().items()):
        total: dict[str, int] = {}
        for bench in benches:
            for grade, count in bench.grades.items():
                total[grade] = total.get(grade, 0) + count
        row: dict[str, Any] = {"режим": mode}
        row.update(dict(sorted(total.items())))
        row["потребує VLM-ремонту"] = sum(b.repair_pages for b in benches)
        rows.append(row)
    return rows


def render_markdown(result: BenchResult) -> str:
    guard = result.guard or GuardReport(False)
    lines = [
        "# Швидкість приймання документів",
        "",
        f"Платформа: `{result.environment.get('platform')}` / "
        f"`{result.environment.get('machine')}`, ядер: {result.environment.get('cpu_count')}.",
        "",
        "## Швидкість за режимами",
        "",
        M.markdown_table(_mode_rows(result)) or "_немає даних_",
        "",
        "План закладає ~0.4–0.7 с/стор. на Windows (1000 стор. за 7–12 хв) і "
        "1.3–2.5 с/стор. на macOS (22–40 хв). Розбіжність очікувана й пояснена нижче.",
        "",
        "## Розподіл ConfidenceReport",
        "",
        M.markdown_table(_grade_rows(result)) or "_немає даних_",
        "",
        "Оцінка сторінки — найгірша з чотирьох (parse / layout / table / ocr). "
        "Сторінки, позначені як такі, що потребують VLM-ремонту, — це рівень 2 "
        "конвеєра; план очікує 3–15% від обсягу.",
        "",
        "## TableFormer і Apple Silicon",
        "",
        f"- Docling встановлено: **{'так' if guard.docling_installed else 'ні'}**"
        + (f" (версія {guard.version})" if guard.version else ""),
        f"- Guard `MPS → CPU`: **{guard.guard_present}** — {guard.detail}",
        "",
        "Issue #3202 наводить 10.4 с (MPS) проти 145.9 с (CPU) на тій самій таблиці — "
        "втрата в 14 разів. Поки guard на місці, на macOS `TableFormerMode.FAST` є "
        "правильним дефолтом, а таблично-щільні підручники лишаються найгіршим "
        "випадком саме на Mac, а не на Windows.",
        "",
        "## Подокументно",
        "",
        M.markdown_table([b.as_dict() for b in result.benches]) or "_немає даних_",
    ]
    return "\n".join(lines)


def write_outputs(result: BenchResult, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "bench_ingest.csv"
    rows = [b.as_dict() for b in result.benches]
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})

    md_path = out_dir / "bench_ingest.md"
    md_path.write_text(render_markdown(result), encoding="utf-8")
    (out_dir / "bench_ingest.json").write_text(
        json.dumps(
            {
                "environment": result.environment,
                "guard": (result.guard or GuardReport(False)).as_dict(),
                "modes": _mode_rows(result),
                "grades": _grade_rows(result),
                "documents": rows,
            },
            ensure_ascii=False, indent=2, default=str,
        ),
        encoding="utf-8",
    )
    return csv_path, md_path


# ----------------------------------------------------------------------- CLI
def main(argv: Sequence[str] | None = None) -> int:
    from app import net_guard

    net_guard.install()

    parser = argparse.ArgumentParser(
        prog="bench_ingest",
        description="Вимір хв/стор. у режимах FAST і DEEP, розподіл ConfidenceReport, пам'ять.",
    )
    # Не `required`: `--guard-only` мусить працювати на машині, де корпусу ще
    # немає — це перше, що запускають, щоб зрозуміти, чого чекати від Mac.
    parser.add_argument("--docs", type=Path, default=None, help="Тека з документами (PDF/TXT/MD).")
    parser.add_argument("--modes", default="FAST", help="FAST, DEEP або обидва через кому.")
    parser.add_argument("--out", type=Path, default=Path("./bench_ingest"), help="Куди класти звіт.")
    parser.add_argument("--max-pages", type=int, default=None, help="Обмежити сторінки на документ.")
    parser.add_argument("--repeat", type=int, default=1, help="Скільки разів прогнати (перший — холодний).")
    parser.add_argument("--table-mode", default=None, choices=["ACCURATE", "FAST"],
                        help="Перекрити режим TableFormer (за замовчуванням — дефолт платформи).")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"],
                        help="Пристрій приймання. За замовчуванням CPU: GPU зайнято LM Studio.")
    parser.add_argument("--guard-only", action="store_true", help="Лише перевірити guard TableFormer.")
    args = parser.parse_args(argv)

    if args.guard_only:
        guard = detect_tableformer_guard()
        print(json.dumps(guard.as_dict(), ensure_ascii=False, indent=2))
        return 0

    if args.docs is None or not args.docs.is_dir():
        raise SystemExit(f"Теки з документами не існує: {args.docs}. Потрібен --docs.")

    documents = [
        p for p in sorted(args.docs.iterdir())
        if p.suffix.lower() in (".pdf", ".txt", ".md")
    ]
    if not documents:
        raise SystemExit(f"У {args.docs} немає жодного .pdf/.txt/.md.")

    try:
        modes = [IngestMode(m.strip().upper()) for m in args.modes.split(",") if m.strip()]
    except ValueError as exc:
        raise SystemExit(f"Невідомий режим: {exc}. Дозволені: FAST, DEEP.") from exc

    result = run_bench(
        documents, modes,
        max_pages=args.max_pages, repeat=args.repeat,
        table_mode=args.table_mode, device=args.device,
    )

    print()
    print(M.markdown_table(_mode_rows(result)))
    csv_path, md_path = write_outputs(result, args.out)
    print(f"\nЗаписано:\n  {csv_path}\n  {md_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
