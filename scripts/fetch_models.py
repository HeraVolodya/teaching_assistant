#!/usr/bin/env python3
"""Завантаження ВСІХ моделей у пласке дерево `models/<model-id>/…` + manifest.json.

Це скрипт **машини розробника і CI**. У постачанні його немає, і він єдиний у
проєкті, якому дозволено ходити в мережу — тому він свідомо НЕ імпортує
`app.net_guard` і взагалі нічого з `app`.

## Чому пласке дерево, а не кеш HuggingFace

Кеш HF (`~/.cache/huggingface/hub/models--org--repo/snapshots/<sha>/…`) — це
зручність розробника й тягар у постачаному офлайн-продукті:

* він тримає файли як **симлінки** на `blobs/`, а на Windows без Developer Mode
  створення симлінка дає `WinError 1314` для непривілейованого користувача —
  саме такого, як викладач на доменній машині;
* імена каталогів містять commit-sha, тобто шлях змінюється при кожному
  оновленні моделі, і жоден інсталятор не може на нього послатися;
* він не має контрольних сум, які можна перевірити при встановленні з USB.

Пласке дерево + `manifest.json` із SHA-256 кожного файлу вирішує всі три задачі
одразу: шлях стабільний, симлінків немає, цілісність перевіряється
`verify_models.py` майстром першого запуску.

## Виняток: каталог `docling/`

`DOCLING_ARTIFACTS_PATH` має вказувати на БАТЬКІВСЬКИЙ каталог, що містить теки
виду `<org>--<repo>` — це жорсткий контракт Docling. Якщо теки немає, Docling
мовчки викликає `snapshot_download`, тобто йде в мережу. Тому моделі Docling
лежать під `models/docling/<org>--<repo>/…` з ВНУТРІШНЬОЮ структурою репозиторію
(TableFormer шукає саме `model_artifacts/tableformer/accurate/…`), а не пласко.

Приклад:
    python scripts/fetch_models.py --models-dir assets/models
    python scripts/fetch_models.py --profile tiny --models-dir /tmp/m   # для CI
    python scripts/fetch_models.py --print-fingerprint                  # ключ кешу CI
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

MANIFEST_SCHEMA = 1
MANIFEST_NAME = "manifest.json"
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
# ЛИШЕ ASCII: значення HTTP-заголовка кодується в latin-1, і кирилиця валить
# запит із UnicodeEncodeError ще до з'єднання. Помилка вилазить аж у
# http.client.putheader, тому виглядає як збій мережі, а не як наш рядок.
_USER_AGENT = "asistent-fetch-models/1.0 (NASV, air-gapped)"


@dataclass(frozen=True, slots=True)
class ModelFile:
    """Один файл моделі."""

    path: str                 # шлях у репозиторії HuggingFace
    as_: str | None = None    # локальний відносний шлях; None → лише ім'я файлу

    def relpath(self) -> str:
        return self.as_ if self.as_ is not None else self.path.rsplit("/", 1)[-1]


@dataclass(frozen=True, slots=True)
class ModelAsset:
    """Модель як одиниця постачання."""

    id: str
    repo: str
    revision: str
    dest: str                 # каталог відносно models_dir
    licence: str
    purpose: str
    approx_mb: int
    files: tuple[ModelFile, ...]
    profiles: tuple[str, ...] = ("full",)
    notes: str = ""
    aliases: tuple[str, ...] = field(default_factory=tuple)

    def url(self, f: ModelFile) -> str:
        return f"{HF_ENDPOINT}/{self.repo}/resolve/{self.revision}/{f.path}"

    def local_relpath(self, f: ModelFile) -> str:
        return f"{self.dest}/{f.relpath()}"


def _docling(path: str) -> ModelFile:
    """Моделі Docling зберігають ВНУТРІШНЮ структуру репозиторію — див. модуль-докстрінг."""
    return ModelFile(path, as_=path)


# --------------------------------------------------------------------- каталог
# ВАЖЛИВО: точні шляхи файлів у репозиторіях змінюються між релізами моделей.
# Перед першим постачанням прогнати `--check-urls` на машині з мережею: він
# зробить HEAD по кожному URL і покаже, що саме зникло. Тихого фолбеку немає
# навмисно — краще гучна помилка на машині розробника, ніж модель, якої не
# вистачить на машині викладача в закритому контурі.
CATALOG: tuple[ModelAsset, ...] = (
    # ------------------------------------------------------------- Docling: layout
    ModelAsset(
        id="docling-layout-heron",
        # Організація `docling-project`, НЕ застаріла `ds4sd`: Docling 2.126.0
        # шукає теку `repo_id.replace("/", "--")`, тож старе ім'я = чужий шлях.
        # Джерело істини — `LayoutObjectDetectionOptions().model_spec`, а не
        # сторінка моделі на HuggingFace: обидва імена там живі й віддають файли.
        repo="docling-project/docling-layout-heron",
        revision="main",
        dest="docling/docling-project--docling-layout-heron",
        licence="MIT",
        purpose="Аналіз розмітки сторінки (RT-DETRv2)",
        approx_mb=172,
        files=(
            _docling("config.json"),
            _docling("model.safetensors"),
            _docling("preprocessor_config.json"),
        ),
    ),
    # --------------------------------------------------- Docling: TableFormer V1
    # V1, не V2: у V2 дві відкриті регресії (docling#3158, #3553).
    # Возимо і accurate, і fast: на Apple Silicon TableFormer примусово на CPU
    # (жорсткий guard `if device == MPS: device = CPU` у table_structure_model.py),
    # і різниця accurate/fast там — це різниця між хвилинами й десятками хвилин.
    ModelAsset(
        id="docling-tableformer",
        repo="docling-project/docling-models",
        # Ревізія ПРИБИТА: `TableStructureModel.download_models` у Docling 2.126.0
        # тягне саме v2.3.0. `main` мовчки дав би інші ваги під тим самим шляхом.
        revision="v2.3.0",
        dest="docling/docling-project--docling-models",
        licence="CDLA-Permissive-2.0",
        purpose="Структура таблиць (TableFormer V1)",
        approx_mb=213,
        files=(
            _docling("model_artifacts/tableformer/accurate/tableformer_accurate.safetensors"),
            _docling("model_artifacts/tableformer/accurate/tm_config.json"),
            _docling("model_artifacts/tableformer/fast/tableformer_fast.safetensors"),
            _docling("model_artifacts/tableformer/fast/tm_config.json"),
        ),
    ),
    # -------------------------------------------------------- Docling: формули/код
    ModelAsset(
        id="docling-codeformula-v2",
        repo="docling-project/CodeFormulaV2",
        revision="main",
        dest="docling/docling-project--CodeFormulaV2",
        licence="CDLA-Permissive-2.0",
        purpose="Розпізнавання формул у LaTeX і блоків коду",
        approx_mb=520,
        files=(
            _docling("config.json"),
            _docling("generation_config.json"),
            _docling("model.safetensors"),
            _docling("preprocessor_config.json"),
            _docling("tokenizer.json"),
            _docling("tokenizer_config.json"),
            _docling("special_tokens_map.json"),
        ),
        notes="Без нього FORMULA-елементи лишаються нерозпізнаним растром.",
    ),
    # ------------------------------------------------ Docling: класифікація рисунків
    ModelAsset(
        id="docling-figure-classifier",
        # Це ОКРЕМИЙ репозиторій, а не перейменована `ds4sd/DocumentFigureClassifier`:
        # Docling 2.126.0 бере пресет `document_figure_classifier_v2`, який вказує
        # саме на -v2.5. Старий репозиторій живий і віддає файли, тому помилка
        # проявлялась не при завантаженні, а аж на розборі документа.
        repo="docling-project/DocumentFigureClassifier-v2.5",
        revision="main",
        dest="docling/docling-project--DocumentFigureClassifier-v2.5",
        licence="CDLA-Permissive-2.0",
        purpose="Класифікація рисунків — гейтинг дорогого VLM-опису",
        approx_mb=32,
        files=(
            _docling("config.json"),
            _docling("model.safetensors"),
            _docling("preprocessor_config.json"),
        ),
        notes="Зрізає 60–80% викликів VLM: 500-сторінкова методичка має 300–800 "
              "PictureItem, з яких реальних схем — близько 150.",
    ),
    # ------------------------------------------------------------------- RapidOCR
    # PP-OCRv5, східнослов'янська модель розпізнавання (uk/ru/be).
    # НІКОЛИ не `chinese` — це дефолт RapidOCR на Windows через OcrAutoOptions.
    ModelAsset(
        id="rapidocr-ppocrv5",
        repo="RapidAI/RapidOCR",
        revision="main",
        dest="rapidocr",
        licence="Apache-2.0",
        purpose="OCR українською (детекція + східнослов'янське розпізнавання)",
        approx_mb=25,
        files=(
            ModelFile("onnx/PP-OCRv5/det/ch_PP-OCRv5_mobile_det.onnx"),
            ModelFile("onnx/PP-OCRv5/rec/eslav_PP-OCRv5_mobile_rec.onnx"),
            ModelFile("onnx/PP-OCRv5/rec/eslav_dict.txt"),
        ),
        notes="RapidOCR бере ЛИШЕ першу мову зі списку lang=[...].",
    ),
    # ------------------------------------------------------------------ ембедер
    ModelAsset(
        id="qwen3-embedding-0.6b",
        repo="onnx-community/Qwen3-Embedding-0.6B-ONNX",
        revision="main",
        dest="embeddings/qwen3-embedding-0.6b",
        licence="Apache-2.0",
        purpose="Ембединги (1024-d, last-token pooling)",
        approx_mb=640,
        files=(
            ModelFile("onnx/model_int8.onnx"),
            ModelFile("config.json"),
            ModelFile("tokenizer.json"),
            ModelFile("tokenizer_config.json"),
            ModelFile("special_tokens_map.json"),
        ),
        notes="Ім'я файлу мусить збігатися з registry.EmbeddingModel.onnx_file.",
    ),
    # ----------------------------------------------------------------- реранкер
    ModelAsset(
        id="qwen3-reranker-0.6b",
        repo="onnx-community/Qwen3-Reranker-0.6B-ONNX",
        revision="main",
        dest="rerank/qwen3-reranker-0.6b",
        licence="Apache-2.0",
        purpose="Крос-енкодер реранкінгу",
        # 1.1 ГБ, а не 640 МБ: `model_quantized.onnx` помітно більший за int8-файл,
        # якого в репозиторії немає. Саме через цю цифру бюджет теки `models/`
        # у docs/DEPLOY.md (1.2-1.8 ГБ) занижений — фактично виходить 2.8 ГБ.
        approx_mb=1100,
        files=(
            # У цьому репозиторії НЕМАЄ `model_int8.onnx` — лише `model_quantized.onnx`
            # (той самий int8) і `model_q4.onnx`. Перевірено через HF API: запит
            # int8-файлу повертав 404, і завантаження всього профілю падало на
            # останньому кроці, вже після 585 МБ ембедера.
            ModelFile("onnx/model_quantized.onnx"),
            ModelFile("config.json"),
            ModelFile("tokenizer.json"),
            ModelFile("tokenizer_config.json"),
            ModelFile("special_tokens_map.json"),
        ),
        notes="ms-marco-моделі непридатні: підтверджений колапс на українській.",
    ),
    # ------------------------------------------------------- профіль CI: крихітні
    # PR-прогони не мають тягнути 1.8 ГБ. Ця модель існує лише щоб довести, що
    # шлях завантаження, маніфест і перевірка SHA-256 працюють end-to-end.
    ModelAsset(
        id="tiny-embedding",
        repo="sentence-transformers/all-MiniLM-L6-v2",
        revision="main",
        dest="embeddings/tiny-embedding",
        licence="Apache-2.0",
        purpose="Заглушка ембедера для CI",
        approx_mb=23,
        files=(
            ModelFile("onnx/model_quantized.onnx", as_="model_int8.onnx"),
            ModelFile("config.json"),
            ModelFile("tokenizer.json"),
            ModelFile("tokenizer_config.json"),
            ModelFile("special_tokens_map.json"),
        ),
        profiles=("tiny",),
    ),
)

PROFILES: tuple[str, ...] = ("full", "tiny")


def catalog_for(profile: str, only: tuple[str, ...] = ()) -> list[ModelAsset]:
    """Моделі профілю; `only` додатково звужує за id."""
    if profile not in PROFILES:
        raise SystemExit(f"Невідомий профіль {profile!r}. Доступні: {', '.join(PROFILES)}")
    chosen = [m for m in CATALOG if profile in m.profiles]
    if only:
        wanted = set(only)
        chosen = [m for m in chosen if m.id in wanted]
        missing = wanted - {m.id for m in chosen}
        if missing:
            raise SystemExit(f"Немає таких моделей у профілі {profile!r}: {', '.join(sorted(missing))}")
    return chosen


def fingerprint(assets: list[ModelAsset]) -> str:
    """Стабільний хеш СПЕЦИФІКАЦІЇ (не вмісту) — ключ кешу CI.

    Змінився каталог → змінився ключ → CI перезавантажує моделі. Не змінився →
    кеш GitHub Actions віддає ~1.8 ГБ за секунди замість годин.
    """
    payload = json.dumps(
        [
            {
                "id": a.id,
                "repo": a.repo,
                "revision": a.revision,
                "dest": a.dest,
                "files": [[f.path, f.relpath()] for f in a.files],
            }
            for a in sorted(assets, key=lambda a: a.id)
        ],
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# ------------------------------------------------------------------ завантаження
def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def human(n: int) -> str:
    value = float(n)
    for unit in ("Б", "КБ", "МБ", "ГБ"):
        if value < 1024 or unit == "ГБ":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} ГБ"


def _request(url: str, method: str = "GET") -> urllib.request.Request:
    req = urllib.request.Request(url, method=method)
    req.add_header("User-Agent", _USER_AGENT)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    return req


def download(url: str, dest: Path, *, retries: int = 3, quiet: bool = False) -> int:
    """Завантажити один файл. Повертає кількість байтів.

    Пишемо в `.part` і перейменовуємо: перервана мережа не має лишати
    правдоподібний огризок, який пройде за розміром і провалиться на SHA-256
    через півгодини встановлення.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(_request(url), timeout=60) as resp, part.open("wb") as out:
                total = int(resp.headers.get("Content-Length") or 0)
                done = 0
                next_report = 8 << 20
                while block := resp.read(1 << 20):
                    out.write(block)
                    done += len(block)
                    if not quiet and done >= next_report:
                        pct = f" ({done * 100 // total}%)" if total else ""
                        print(f"    …{human(done)}{pct}", file=sys.stderr)
                        next_report += 8 << 20
            part.replace(dest)
            return dest.stat().st_size
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            part.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(2 ** attempt)

    raise SystemExit(f"Не вдалося завантажити {url}: {last_error}")


def check_urls(assets: list[ModelAsset]) -> int:
    """HEAD по кожному URL. Повертає кількість недоступних."""
    bad = 0
    for asset in assets:
        for f in asset.files:
            url = asset.url(f)
            try:
                with urllib.request.urlopen(_request(url, "HEAD"), timeout=30) as resp:
                    size = int(resp.headers.get("Content-Length") or 0)
                    print(f"  OK   {human(size):>10}  {url}")
            except Exception as exc:  # noqa: BLE001 — тут цікавить будь-яка причина
                bad += 1
                print(f"  ЗБІЙ            {url}\n         {exc}", file=sys.stderr)
    return bad


def build_manifest(models_dir: Path, assets: list[ModelAsset], profile: str) -> dict:
    """Зібрати маніфест із ФАКТИЧНО наявних на диску файлів."""
    models: list[dict] = []
    total = 0
    for asset in assets:
        files: list[dict] = []
        for f in asset.files:
            rel = asset.local_relpath(f)
            path = models_dir / rel
            if not path.exists():
                raise SystemExit(f"Файл відсутній після завантаження: {path}")
            size = path.stat().st_size
            total += size
            files.append({"relpath": rel, "sha256": sha256_file(path), "bytes": size})
        models.append(
            {
                "id": asset.id,
                "version": asset.revision,
                "repo": asset.repo,
                "licence": asset.licence,
                "purpose": asset.purpose,
                "files": files,
            }
        )
    return {
        "schema": MANIFEST_SCHEMA,
        "profile": profile,
        "fingerprint": fingerprint(assets),
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "total_bytes": total,
        "models": models,
    }


def fetch(models_dir: Path, assets: list[ModelAsset], *, force: bool = False, quiet: bool = False) -> None:
    for asset in assets:
        print(f"[{asset.id}] {asset.repo}@{asset.revision} — {asset.purpose} (~{asset.approx_mb} МБ)")
        for f in asset.files:
            dest = models_dir / asset.local_relpath(f)
            if dest.exists() and not force:
                print(f"  вже є: {dest.relative_to(models_dir)}")
                continue
            size = download(asset.url(f), dest, quiet=quiet)
            print(f"  завантажено {human(size):>10}  {dest.relative_to(models_dir)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Завантажити моделі Асістента у пласке дерево.")
    parser.add_argument("--models-dir", type=Path, default=Path("assets/models"))
    parser.add_argument("--profile", default="full", choices=PROFILES)
    parser.add_argument("--only", nargs="*", default=[], help="Обмежити конкретними id моделей.")
    parser.add_argument("--force", action="store_true", help="Перезавантажити наявні файли.")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--check-urls", action="store_true", help="Лише HEAD по всіх URL.")
    parser.add_argument("--print-fingerprint", action="store_true", help="Лише ключ кешу CI.")
    parser.add_argument("--list", action="store_true", help="Показати склад профілю й вийти.")
    args = parser.parse_args(argv)

    assets = catalog_for(args.profile, tuple(args.only))

    if args.print_fingerprint:
        print(fingerprint(assets))
        return 0

    if args.list:
        total = sum(a.approx_mb for a in assets)
        for a in assets:
            print(f"{a.id:28} {a.approx_mb:>5} МБ  {a.licence:22} {a.repo}")
        print(f"{'РАЗОМ':28} {total:>5} МБ")
        return 0

    if args.check_urls:
        bad = check_urls(assets)
        print(f"Недоступних файлів: {bad}", file=sys.stderr)
        return 1 if bad else 0

    models_dir = args.models_dir.expanduser().resolve()
    models_dir.mkdir(parents=True, exist_ok=True)
    fetch(models_dir, assets, force=args.force, quiet=args.quiet)

    manifest = build_manifest(models_dir, assets, args.profile)
    (models_dir / MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"\nМаніфест: {models_dir / MANIFEST_NAME}\n"
        f"Моделей: {len(manifest['models'])}, разом {human(manifest['total_bytes'])}, "
        f"відбиток {manifest['fingerprint']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
