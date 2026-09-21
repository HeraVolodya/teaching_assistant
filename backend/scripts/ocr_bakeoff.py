"""Український OCR-бейк-оф — Віха 1 плану, найцінніший експеримент проєкту.

    ПУБЛІКОВАНОЇ ОЦІНКИ ЯКОСТІ OCR УКРАЇНСЬКОЮ НЕ ІСНУЄ ДЛЯ ЖОДНОГО РУШІЯ.

Це записано в реєстр ризиків НДР першим рядком і означає буквально таке: вибір
OCR-рушія для української неможливо обґрунтувати посиланням — його можна лише
виміряти. Поки цих чисел немає, Етап II починати не можна, бо весь конвеєр
приймання стоїть на припущенні, якого ніхто не перевіряв.

Скрипт бере теку зі сторінками (зображення або PDF) і теку з ручними
транскрипціями, проганяє всі ДОСТУПНІ рушії і рахує CER, WER і — головне —
українські плутанини (`і/i`, `и/й`, `ї/i`, `є/e`, `ґ/г`, апостроф, гомогліфи).
Саме плутанини вирішують: рушій із CER 3%, який стабільно пише латинську «i»
замість кириличної «і», непридатний, бо ламає лематизацію й BM25, тобто
робить текст невидимим для пошуку при бездоганному вигляді на екрані.

Приклад:

    python -m scripts.ocr_bakeoff --pages ./bakeoff/pages --transcripts ./bakeoff/gold \\
        --out ./bakeoff/out --engines tesseract-ukr,apple-vision

Формат вхідних даних: файли зіставляються за ІМЕНЕМ БЕЗ РОЗШИРЕННЯ.
`pages/p001.png` ↔ `transcripts/p001.txt`. Для PDF додається номер сторінки:
`pages/book.pdf` дає `book_p1`, `book_p2`, … і шукає `transcripts/book_p1.txt`.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from app.eval import metrics as M
from app.eval.ocr_metrics import EngineReport, PageScore, aggregate, report_rows, score_page

__all__ = [
    "ENGINE_FACTORIES",
    "AppleVisionEngine",
    "EngineStatus",
    "LmStudioVlmEngine",
    "OcrEngine",
    "PagePair",
    "RapidOcrEngine",
    "TesseractEngine",
    "collect_pairs",
    "discover_engines",
    "main",
    "render_markdown",
    "run_bakeoff",
    "write_csv",
]

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
# 300 dpi — рідна роздільність більшості сканованих підручників. Рендерити
# вище безглуздо: план прямо попереджає, що апскейл скану ПОГІРШУЄ
# розпізнавання, а не покращує.
RENDER_DPI = 300


# --------------------------------------------------------------------- дані
@dataclass(frozen=True, slots=True)
class PagePair:
    """Сторінка та її ручна транскрипція."""

    name: str
    image: Path
    reference: str

    @property
    def ref_chars(self) -> int:
        return len(self.reference)


@dataclass(frozen=True, slots=True)
class EngineStatus:
    available: bool
    detail: str = ""


class OcrEngine(Protocol):
    """Контракт рушія. Один метод розпізнавання і чесна перевірка доступності."""

    key: str
    title: str

    def status(self) -> EngineStatus: ...

    def recognize(self, image: Path) -> str: ...


# ------------------------------------------------------------------ рушії
class TesseractEngine:
    """Tesseract 5 із `tessdata_best/ukr`.

    Виклик через `subprocess`, а не через `pytesseract`: обгортка нічого не
    додає, крім залежності, а сам Tesseract усе одно мусить бути в системі
    окремим бінарником. Заодно це знімає питання ліцензії обгортки.
    """

    key = "tesseract-ukr"
    title = "Tesseract 5 (ukr)"

    def __init__(self, lang: str = "ukr", psm: int = 3, binary: str = "tesseract") -> None:
        self.lang = lang
        self.psm = psm
        self.binary = binary

    def status(self) -> EngineStatus:
        path = shutil.which(self.binary)
        if not path:
            return EngineStatus(False, f"Не знайдено виконуваний файл «{self.binary}» у PATH.")
        try:
            langs = subprocess.run(
                [self.binary, "--list-langs"], capture_output=True, text=True, timeout=30, check=False
            ).stdout
        except Exception as exc:
            return EngineStatus(False, f"Не вдалося опитати мови: {exc}")
        if self.lang not in langs.split():
            return EngineStatus(
                False,
                f"Мовний пакет «{self.lang}» не встановлено. Потрібен tessdata_best/{self.lang}.traineddata.",
            )
        return EngineStatus(True, path)

    def recognize(self, image: Path) -> str:
        result = subprocess.run(
            [self.binary, str(image), "stdout", "-l", self.lang, "--psm", str(self.psm)],
            capture_output=True, text=True, timeout=300, check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"tesseract повернув {result.returncode}: {result.stderr.strip()[:200]}")
        return result.stdout


class RapidOcrEngine:
    """RapidOCR (PP-OCRv5). `eslav` — східнослов'янська модель (uk/ru/be).

    ОБЕРЕЖНО, тут два різні API. RapidOCR 2.x приймав шляхи до моделей
    (`RapidOCR(rec_model_path=...)`), 3.x — словник параметрів
    (`RapidOCR(params={"Rec.lang_type": "eslav"})`). Ми пробуємо обидві форми і
    ЗАПИСУЄМО, яка спрацювала: сенс бейк-офу в тому, щоб дізнатись, що реально
    працює на цільовій машині, а не в тому, щоб вгадати сигнатуру наперед.

    Так само важливо: RapidOCR бере ЛИШЕ ПЕРШУ мову зі списку (план, §1), тож
    «українська плюс англійська» одним викликом не буває.
    """

    key = "rapidocr"
    title = "RapidOCR"

    def __init__(self, lang: str = "eslav") -> None:
        self.lang = lang
        self.key = f"rapidocr-{lang}"
        self.title = f"RapidOCR PP-OCRv5 ({lang})"
        self._engine: Any = None
        self._signature = ""

    def _build(self) -> Any:
        from rapidocr import RapidOCR  # type: ignore[import-not-found]

        attempts: list[tuple[str, dict[str, Any]]] = [
            ("params={'Rec.lang_type': ...}", {"params": {"Rec.lang_type": self.lang}}),
            ("params={'Global.lang_rec': ...}", {"params": {"Global.lang_rec": self.lang}}),
            ("lang_rec=...", {"lang_rec": self.lang}),
            ("без параметрів", {}),
        ]
        errors: list[str] = []
        for name, kwargs in attempts:
            try:
                engine = RapidOCR(**kwargs)
            except Exception as exc:
                errors.append(f"{name}: {type(exc).__name__}: {exc}")
                continue
            self._signature = name
            if not kwargs:
                # Це НЕ те, що ми хотіли: без явної мови RapidOCR на Windows
                # бере китайську за замовчуванням (план, §1). Позначаємо явно.
                self._signature = "без параметрів (МОВА НЕ ЗАДАНА — результат недійсний)"
            return engine
        raise RuntimeError("Жодна відома сигнатура RapidOCR не спрацювала: " + "; ".join(errors))

    def status(self) -> EngineStatus:
        try:
            self._engine = self._build()
        except ImportError as exc:
            return EngineStatus(False, f"Пакет rapidocr не встановлено ({exc}). pip install .[worker]")
        except Exception as exc:
            return EngineStatus(False, str(exc))
        return EngineStatus(True, f"конструктор: {self._signature}")

    def recognize(self, image: Path) -> str:
        if self._engine is None:
            self._engine = self._build()
        result = self._engine(str(image))
        # 2.x повертав (список, час), 3.x — об'єкт із полем `txts`.
        texts = getattr(result, "txts", None)
        if texts is not None:
            return "\n".join(str(t) for t in texts)
        if isinstance(result, tuple):
            result = result[0]
        if not result:
            return ""
        lines = []
        for item in result:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                lines.append(str(item[1]))
        return "\n".join(lines)


class AppleVisionEngine:
    """Apple Vision через `ocrmac`. Лише macOS; `uk-UA` підтримується нативно.

    На Mac це єдиний рушій, для якого не треба возити ваги: моделі вже в
    системі. Саме тому він у постачанні є дефолтом для macOS (план, §1).
    """

    key = "apple-vision"
    title = "Apple Vision (uk-UA)"

    def __init__(self, languages: Sequence[str] = ("uk-UA", "ru-RU", "en-US")) -> None:
        self.languages = list(languages)

    def status(self) -> EngineStatus:
        if sys.platform != "darwin":
            return EngineStatus(False, "Apple Vision існує лише на macOS.")
        try:
            import ocrmac  # type: ignore[import-not-found]  # noqa: F401
        except ImportError as exc:
            return EngineStatus(False, f"Пакет ocrmac не встановлено ({exc}). pip install .[worker-macos]")
        return EngineStatus(True, ", ".join(self.languages))

    def recognize(self, image: Path) -> str:
        from ocrmac import ocrmac  # type: ignore[import-not-found]

        annotations = ocrmac.OCR(
            str(image), language_preference=self.languages, recognition_level="accurate"
        ).recognize()
        return "\n".join(str(a[0]) for a in annotations)


class LmStudioVlmEngine:
    """PaddleOCR-VL (або будь-яка VLM) через LM Studio.

    Чому це окремий рушій, а не «просто ще один OCR»: план (§1) розраховує його
    як РІВЕНЬ 2 — ремонт сторінок, що провалились, а не основний прохід.
    Арифметика: 8B VLM дає 30–55 с/стор., тобто 4.5–7.5 год на 500-сторінковий
    підручник; PaddleOCR-VL-1.6 (1.0B) — 5–10 с/стор. Бейк-оф має показати, чи
    варта його якість цієї ціни, і — окремо — ЧИ ВЗАГАЛІ вона знає українську:
    джерела розходяться (стаття називає російську, українську — лише вторинні
    джерела). Це другий рядок реєстру ризиків.

    Мережа: лише 127.0.0.1. `net_guard` кине виняток на будь-якому іншому
    хості, і це навмисно.
    """

    key = "paddleocr-vl-lmstudio"
    title = "PaddleOCR-VL через LM Studio"

    PROMPT_UK = (
        "Перед тобою сторінка українського навчального посібника. "
        "Перепиши ВЕСЬ її текст дослівно, зберігаючи порядок рядків. "
        "Не перекладай, не скорочуй, не коментуй. "
        "Не додавай нічого від себе. Виведи лише текст сторінки."
    )

    def __init__(
        self,
        model: str = "paddleocr-vl",
        base_url: str = "http://127.0.0.1:1234",
        timeout: float = 300.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def status(self) -> EngineStatus:
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover — httpx у базових залежностях
            return EngineStatus(False, f"httpx недоступний ({exc}).")
        try:
            response = httpx.get(f"{self.base_url}/v1/models", timeout=2.0)
        except Exception as exc:
            return EngineStatus(False, f"LM Studio не відповідає на {self.base_url}: {exc}")
        if response.status_code != 200:
            return EngineStatus(False, f"LM Studio відповів {response.status_code}.")
        body = response.text
        if self.model not in body:
            return EngineStatus(
                False,
                f"Модель «{self.model}» не знайдено серед завантажених. "
                "Завантажте її у вкладці Developer або вкажіть --vlm-model.",
            )
        return EngineStatus(True, self.base_url)

    def recognize(self, image: Path) -> str:
        import base64

        import httpx

        data = base64.b64encode(image.read_bytes()).decode("ascii")
        suffix = image.suffix.lower().lstrip(".") or "png"
        payload = {
            "model": self.model,
            "temperature": 0.0,
            "max_tokens": 4096,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.PROMPT_UK},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/{suffix};base64,{data}"},
                        },
                    ],
                }
            ],
        }
        response = httpx.post(
            f"{self.base_url}/v1/chat/completions", json=payload, timeout=self.timeout
        )
        response.raise_for_status()
        body = response.json()
        return str(body["choices"][0]["message"]["content"])


ENGINE_FACTORIES: dict[str, Any] = {
    "rapidocr-eslav": lambda: RapidOcrEngine("eslav"),
    "rapidocr-cyrillic": lambda: RapidOcrEngine("cyrillic"),
    "tesseract-ukr": TesseractEngine,
    "apple-vision": AppleVisionEngine,
    "paddleocr-vl-lmstudio": LmStudioVlmEngine,
}


def discover_engines(
    names: Sequence[str] | None = None,
    *,
    vlm_model: str = "paddleocr-vl",
) -> list[tuple[Any, EngineStatus]]:
    """Створити рушії й перевірити доступність кожного.

    Недоступний рушій НЕ зникає з результату: він потрапляє у звіт із причиною.
    Мовчазне зникнення рушія з таблиці — це найпростіший спосіб опублікувати
    порівняння, у якому переможець просто не мав суперників.
    """
    keys = list(names) if names else list(ENGINE_FACTORIES)
    out: list[tuple[Any, EngineStatus]] = []
    for key in keys:
        factory = ENGINE_FACTORIES.get(key)
        if factory is None:
            known = ", ".join(sorted(ENGINE_FACTORIES))
            raise SystemExit(f"Невідомий рушій «{key}». Відомі: {known}")
        engine = LmStudioVlmEngine(model=vlm_model) if key == "paddleocr-vl-lmstudio" else factory()
        try:
            status = engine.status()
        except Exception as exc:
            status = EngineStatus(False, f"{type(exc).__name__}: {exc}")
        out.append((engine, status))
    return out


# ------------------------------------------------------------ вхідні дані
def _render_pdf_pages(pdf: Path, out_dir: Path, *, dpi: int = RENDER_DPI) -> list[tuple[str, Path]]:
    """Розкласти PDF на PNG. `pypdfium2` — опційний імпорт (Apache-2.0/BSD).

    Свідомо НЕ PyMuPDF: AGPL-3 неприйнятна для розповсюджуваного бінарника
    державної НДР, і навіть у допоміжному скрипті тримати другий рендерер із
    несумісною ліцензією немає сенсу.
    """
    try:
        import pypdfium2 as pdfium  # type: ignore[import-not-found]
    except ImportError as exc:
        print(f"  [пропущено] {pdf.name}: pypdfium2 недоступний ({exc}).")
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    pages: list[tuple[str, Path]] = []
    document = pdfium.PdfDocument(str(pdf))
    try:
        for index in range(len(document)):
            name = f"{pdf.stem}_p{index + 1}"
            target = out_dir / f"{name}.png"
            if not target.exists():
                image = document[index].render(scale=dpi / 72.0).to_pil()
                image.save(target)
            pages.append((name, target))
    finally:
        document.close()
    return pages


def collect_pairs(
    pages_dir: Path,
    transcripts_dir: Path,
    *,
    cache_dir: Path | None = None,
    limit: int | None = None,
    dpi: int = RENDER_DPI,
) -> tuple[list[PagePair], list[str]]:
    """Зіставити сторінки з транскрипціями за іменем без розширення.

    Повертає (пари, попередження). Сторінка без транскрипції — це не помилка,
    а нормальний стан на початку розмітки, тож вона лише згадується у
    попередженнях. А от транскрипція без сторінки — майже завжди друкарська
    помилка в імені файлу, і її теж треба показати.
    """
    warnings: list[str] = []
    references: dict[str, Path] = {
        p.stem: p for p in sorted(transcripts_dir.glob("*.txt"))
    }

    candidates: list[tuple[str, Path]] = []
    for path in sorted(pages_dir.iterdir()):
        if path.suffix.lower() in IMAGE_SUFFIXES:
            candidates.append((path.stem, path))
        elif path.suffix.lower() == ".pdf":
            candidates.extend(_render_pdf_pages(path, cache_dir or (pages_dir / "_render"), dpi=dpi))

    pairs: list[PagePair] = []
    matched: set[str] = set()
    for name, image in candidates:
        reference = references.get(name)
        if reference is None:
            warnings.append(f"Немає транскрипції для сторінки {name}.")
            continue
        text = reference.read_text(encoding="utf-8", errors="replace")
        if not text.strip():
            warnings.append(f"Транскрипція {name} порожня — сторінку пропущено.")
            continue
        matched.add(name)
        pairs.append(PagePair(name=name, image=image, reference=text))

    for name in references:
        if name not in matched:
            warnings.append(f"Транскрипція {name} не має сторінки з таким іменем.")

    if limit is not None:
        pairs = pairs[:limit]
    return pairs, warnings


# --------------------------------------------------------------------- прогін
@dataclass(slots=True)
class BakeoffResult:
    scores: list[PageScore] = field(default_factory=list)
    reports: list[EngineReport] = field(default_factory=list)
    statuses: dict[str, EngineStatus] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    pages: int = 0


def run_bakeoff(
    pairs: Sequence[PagePair],
    engines: Sequence[tuple[Any, EngineStatus]],
    *,
    verbose: bool = True,
) -> BakeoffResult:
    """Прогнати всі доступні рушії по всіх сторінках."""
    result = BakeoffResult(pages=len(pairs))
    for engine, status in engines:
        result.statuses[engine.key] = status
        if not status.available:
            if verbose:
                print(f"[пропущено] {engine.title}: {status.detail}")
            continue
        if verbose:
            print(f"[прогін] {engine.title} ({status.detail})")
        for pair in pairs:
            started = time.perf_counter()
            error = ""
            text = ""
            try:
                text = engine.recognize(pair.image)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            seconds = time.perf_counter() - started
            score = score_page(
                pair.name, engine.key, pair.reference, text, seconds=seconds, error=error
            )
            result.scores.append(score)
            if verbose:
                mark = f"CER {score.cer:.4f}" if not error else f"ЗБІЙ {error[:60]}"
                print(f"    {pair.name}: {mark} ({seconds:.1f} с)")
    result.reports = aggregate(result.scores)
    return result


# --------------------------------------------------------------------- звіти
def write_csv(result: BakeoffResult, out_dir: Path) -> tuple[Path, Path]:
    """Два CSV: посторінковий (для аналізу) і зведений (для звіту)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    per_page = out_dir / "ocr_bakeoff_pages.csv"
    summary = out_dir / "ocr_bakeoff_summary.csv"

    rows = [s.as_dict() for s in result.scores]
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with per_page.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})

    summary_rows = report_rows(result.reports)
    with summary.open("w", encoding="utf-8-sig", newline="") as fh:
        if summary_rows:
            writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0]))
            writer.writeheader()
            writer.writerows(summary_rows)
    return per_page, summary


def render_markdown(result: BakeoffResult) -> str:
    """Markdown для звіту з НДР: зведення, плутанини, пропущені рушії."""
    lines = [
        "# Український OCR-бейк-оф",
        "",
        f"Сторінок у наборі: **{result.pages}**.",
        "",
        "Публікованої оцінки якості OCR українською не існує для жодного рушія — "
        "ці числа отримані власним вимірюванням на реальному корпусі.",
        "",
        "## Зведення",
        "",
        M.markdown_table(report_rows(result.reports)) or "_немає доступних рушіїв_",
        "",
        "## Українські плутанини",
        "",
        "CER однаково карає будь-яку заміну символу. Але заміна кириличної «і» на "
        "латинську «i» не змінює вигляду сторінки й майже не змінює CER, а пошук "
        "ламає повністю: лема не будується, FTS5 не знаходить, викладач бачить "
        "порожню відповідь на питання з власного підручника. Тому ці класи "
        "рахуються окремо.",
        "",
    ]
    confusion_rows: list[dict[str, Any]] = []
    for report in result.reports:
        row: dict[str, Any] = {"рушій": report.engine}
        row.update({k: v for k, v in sorted(report.confusions.items())})
        row["найчастіші заміни"] = ", ".join(
            f"{a or '∅'}→{b or '∅'}×{n}" for a, b, n in report.top_substitutions[:5]
        )
        confusion_rows.append(row)
    lines.append(M.markdown_table(confusion_rows) or "_немає даних_")

    skipped = [(key, s) for key, s in result.statuses.items() if not s.available]
    if skipped:
        lines += ["", "## Пропущені рушії", ""]
        lines += [f"- **{key}** — {status.detail}" for key, status in skipped]
        lines += [
            "",
            "Пропущений рушій лишається в таблиці з причиною: порівняння, у якому "
            "суперники тихо зникли, не є порівнянням.",
        ]
    return "\n".join(lines)


# ----------------------------------------------------------------------- CLI
def main(argv: Sequence[str] | None = None) -> int:
    from app import net_guard

    net_guard.install()

    parser = argparse.ArgumentParser(
        prog="ocr_bakeoff",
        description="Український OCR-бейк-оф: CER, WER і плутанини і/i, и/й, ї/i, є/e, ґ/г, апостроф.",
    )
    # Не `required`: `--list` мусить працювати до того, як зібрано корпус —
    # саме з нього починають, щоб дізнатись, які рушії взагалі є на машині.
    parser.add_argument("--pages", type=Path, default=None, help="Тека зі сторінками (зображення або PDF).")
    parser.add_argument("--transcripts", type=Path, default=None,
                        help="Тека з ручними транскрипціями (.txt).")
    parser.add_argument("--out", type=Path, default=Path("./ocr_bakeoff"), help="Куди класти CSV і Markdown.")
    parser.add_argument("--engines", default="", help="Список рушіїв через кому; порожньо — усі відомі.")
    parser.add_argument("--limit", type=int, default=None, help="Обмежити кількість сторінок.")
    parser.add_argument("--dpi", type=int, default=RENDER_DPI, help="DPI рендеру PDF (за замовчуванням 300).")
    parser.add_argument("--vlm-model", default="paddleocr-vl", help="Ключ моделі VLM у LM Studio.")
    parser.add_argument("--list", action="store_true", help="Лише показати доступність рушіїв.")
    args = parser.parse_args(argv)

    names = [n.strip() for n in args.engines.split(",") if n.strip()] or None
    engines = discover_engines(names, vlm_model=args.vlm_model)

    if args.list:
        for engine, status in engines:
            mark = "доступний" if status.available else "НЕДОСТУПНИЙ"
            print(f"{engine.key:26} {mark:12} {status.detail}")
        return 0

    if args.pages is None or not args.pages.is_dir():
        raise SystemExit(f"Теки зі сторінками не існує: {args.pages}. Потрібен --pages.")
    if args.transcripts is None or not args.transcripts.is_dir():
        raise SystemExit(
            f"Теки з транскрипціями не існує: {args.transcripts}. Потрібен --transcripts."
        )

    pairs, warnings = collect_pairs(
        args.pages, args.transcripts, cache_dir=args.out / "_render", limit=args.limit, dpi=args.dpi
    )
    for warning in warnings:
        print(f"[увага] {warning}")
    if not pairs:
        raise SystemExit(
            "Жодної пари «сторінка + транскрипція» не знайдено. Імена файлів мають "
            "збігатися без розширення: pages/p001.png ↔ transcripts/p001.txt."
        )

    result = run_bakeoff(pairs, engines)
    result.warnings = warnings

    print()
    print(M.markdown_table(report_rows(result.reports)))

    args.out.mkdir(parents=True, exist_ok=True)
    per_page, summary = write_csv(result, args.out)
    markdown = args.out / "ocr_bakeoff.md"
    markdown.write_text(render_markdown(result), encoding="utf-8")
    (args.out / "ocr_bakeoff.json").write_text(
        json.dumps(
            {
                "pages": result.pages,
                "engines": {k: {"available": s.available, "detail": s.detail}
                            for k, s in result.statuses.items()},
                "scores": [s.as_dict() for s in result.scores],
                "summary": report_rows(result.reports),
                "warnings": result.warnings,
                "platform": sys.platform,
                "cpu_count": os.cpu_count(),
            },
            ensure_ascii=False, indent=2, default=str,
        ),
        encoding="utf-8",
    )
    print(f"\nЗаписано:\n  {per_page}\n  {summary}\n  {markdown}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
