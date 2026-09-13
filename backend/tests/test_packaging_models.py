"""Каталог моделей, маніфест і перевірка цілісності.

Головне, що охороняють ці тести: **розкладка на диску**. Кеш HuggingFace у
постачаному офлайн-продукті — це симлінки (WinError 1314 на Windows без
Developer Mode), шляхи з commit-sha (на них неможливо послатися з інсталятора)
і відсутність контрольних сум (нічого перевірити при встановленні з USB).
Пласке дерево з `manifest.json` знімає всі три проблеми — і саме воно тут
фіксується тестом.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from helpers_packaging import script

fetch = script("fetch_models")
verify = script("verify_models")


# ------------------------------------------------------------------- каталог
def test_ідентифікатори_унікальні() -> None:
    ids = [m.id for m in fetch.CATALOG]
    assert len(ids) == len(set(ids))


def test_повний_профіль_несе_весь_конвеєр() -> None:
    ids = {m.id for m in fetch.catalog_for("full")}
    assert {
        "docling-layout-heron",       # розмітка сторінки
        "docling-tableformer",        # таблиці
        "docling-codeformula-v2",     # формули
        "docling-figure-classifier",  # гейтинг VLM
        "rapidocr-ppocrv5",           # OCR українською
        "qwen3-embedding-0.6b",       # ембединги
        "qwen3-reranker-0.6b",        # реранкінг
    } <= ids
    # Заглушка CI не має потрапити в постачання.
    assert "tiny-embedding" not in ids


def test_профіль_tiny_малий() -> None:
    tiny = fetch.catalog_for("tiny")
    assert [m.id for m in tiny] == ["tiny-embedding"]
    assert sum(m.approx_mb for m in tiny) < 100


def test_ліцензії_придатні_для_розповсюджуваного_бінарника() -> None:
    # Державна НДР із розповсюджуваним бінарником: жодної AGPL/GPL/NC.
    for model in fetch.CATALOG:
        assert not any(bad in model.licence.upper() for bad in ("AGPL", "GPL-3", "-NC")), model.id


def test_розкладка_пласка_окрім_docling() -> None:
    for model in fetch.CATALOG:
        for f in model.files:
            rel = model.local_relpath(f)
            # Жодного сліду кеш-розкладки HuggingFace.
            assert "models--" not in rel and "snapshots" not in rel and "blobs" not in rel
            if model.dest.startswith("docling/"):
                # DOCLING_ARTIFACTS_PATH вимагає тек `<org>--<repo>` з
                # ВНУТРІШНЬОЮ структурою репозиторію (TableFormer шукає
                # model_artifacts/tableformer/accurate/...).
                assert "--" in model.dest
                assert rel == f"{model.dest}/{f.path}"
            else:
                # Решта — рівно ДВА рівні: `<роль>/<ключ>/<файл>`, де роль це
                # `embeddings`, `rerank` або `rapidocr`.
                #
                # Раніше тут стояло `rel.count("/") == 1`, тобто повна пласкість.
                # Ця вимога СУПЕРЕЧИЛА рантайму: `provider.default_model_dir`
                # складає шлях як `models_dir/embeddings/<ключ>`, а
                # `reranker.default_model_dir` — як `models_dir/rerank/<ключ>`.
                # Через розбіжність скрипт завантажив 1.8 ГБ, відрапортував
                # успіх, а індексація впала з «не знайдено tokenizer.json».
                # Джерело істини — рантайм; див. test_model_layout_contract.py,
                # який тепер порівнює обидві сторони напряму.
                assert 1 <= rel.count("/") <= 2, rel
                assert rel.split("/", 1)[0] in {"embeddings", "rerank", "rapidocr"}, rel


def test_імʼя_файлу_ембедера_збігається_з_реєстром() -> None:
    from app.embeddings import registry

    model = fetch.catalog_for("full", ("qwen3-embedding-0.6b",))[0]
    names = {f.relpath() for f in model.files}
    assert registry.get("qwen3-embedding-0.6b").onnx_file in names


def test_url_будується_з_ревізії() -> None:
    model = fetch.catalog_for("full", ("qwen3-embedding-0.6b",))[0]
    url = model.url(model.files[0])
    assert url.startswith("https://huggingface.co/onnx-community/Qwen3-Embedding-0.6B-ONNX/resolve/main/")


# ---------------------------------------------------------------- відбиток
def test_відбиток_стабільний_і_чутливий() -> None:
    full = fetch.catalog_for("full")
    assert fetch.fingerprint(full) == fetch.fingerprint(list(reversed(full)))
    assert fetch.fingerprint(full) != fetch.fingerprint(fetch.catalog_for("tiny"))

    import dataclasses

    changed = [*full[:-1], dataclasses.replace(full[-1], revision="інша")]
    assert fetch.fingerprint(changed) != fetch.fingerprint(full)


# ---------------------------------------------------- маніфест і перевірка
@pytest.fixture
def fake_models(tmp_path: Path) -> tuple[Path, list, dict]:
    """Два «завантажені» файли на диску + маніфест до них."""
    asset = fetch.ModelAsset(
        id="fake",
        repo="org/repo",
        revision="v1",
        dest="fake",
        licence="MIT",
        purpose="тест",
        approx_mb=1,
        files=(fetch.ModelFile("onnx/model_int8.onnx"), fetch.ModelFile("tokenizer.json")),
    )
    (tmp_path / "fake").mkdir()
    (tmp_path / "fake" / "model_int8.onnx").write_bytes(b"onnx-payload")
    (tmp_path / "fake" / "tokenizer.json").write_text("{}", encoding="utf-8")

    manifest = fetch.build_manifest(tmp_path, [asset], "full")
    (tmp_path / fetch.MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )
    return tmp_path, [asset], manifest


def test_маніфест_несе_суму_розмір_і_версію(fake_models) -> None:
    _, _, manifest = fake_models
    entry = manifest["models"][0]
    assert entry["id"] == "fake"
    assert entry["version"] == "v1"
    assert {f["relpath"] for f in entry["files"]} == {
        "fake/model_int8.onnx",
        "fake/tokenizer.json",
    }
    for f in entry["files"]:
        assert len(f["sha256"]) == 64
        assert f["bytes"] > 0
    assert manifest["total_bytes"] == sum(f["bytes"] for f in entry["files"])


def test_перевірка_проходить_на_цілих_файлах(fake_models) -> None:
    models_dir, _, manifest = fake_models
    results = verify.verify(models_dir, manifest)
    assert all(r.ok for r in results)
    assert verify.main(["--models-dir", str(models_dir), "--quiet"]) == 0


def test_перевірка_ловить_підміну_вмісту(fake_models) -> None:
    models_dir, _, manifest = fake_models
    # Той самий розмір, інший вміст — саме те, що не ловить перевірка за розміром.
    (models_dir / "fake" / "model_int8.onnx").write_bytes(b"ONNX-PAYLOAD")

    quick = verify.verify(models_dir, manifest, deep=False)
    assert all(r.ok for r in quick), "швидка проба не мусить бачити підміну однакового розміру"

    deep = verify.verify(models_dir, manifest, deep=True)
    assert [r.status for r in deep if not r.ok] == [verify.HASH_MISMATCH]
    assert verify.main(["--models-dir", str(models_dir), "--quiet"]) == 1


def test_перевірка_ловить_обірване_копіювання(fake_models) -> None:
    models_dir, _, manifest = fake_models
    (models_dir / "fake" / "tokenizer.json").write_text("", encoding="utf-8")
    statuses = {r.relpath: r.status for r in verify.verify(models_dir, manifest, deep=False)}
    assert statuses["fake/tokenizer.json"] == verify.SIZE_MISMATCH


def test_перевірка_ловить_відсутній_файл(fake_models) -> None:
    models_dir, _, manifest = fake_models
    (models_dir / "fake" / "tokenizer.json").unlink()
    bad = [r for r in verify.verify(models_dir, manifest) if not r.ok]
    assert [r.status for r in bad] == [verify.MISSING]
    assert "відсут" in bad[0].describe()


def test_машинний_вивід_для_майстра_першого_запуску(fake_models, capsys) -> None:
    models_dir, _, _ = fake_models
    (models_dir / "fake" / "model_int8.onnx").unlink()
    assert verify.main(["--models-dir", str(models_dir), "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["failed"] == 1
    assert payload["problems"][0]["status"] == verify.MISSING
    assert payload["problems"][0]["message"]  # українською, для UI


def test_відсутній_маніфест_дає_зрозумілу_помилку(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exc:
        verify.load_manifest(tmp_path)
    assert "manifest.json" in str(exc.value)


def test_прогрес_викликається_на_кожен_файл(fake_models) -> None:
    models_dir, _, manifest = fake_models
    seen: list[tuple[int, int]] = []
    verify.verify(models_dir, manifest, on_progress=lambda done, total, _r: seen.append((done, total)))
    assert seen == [(1, 2), (2, 2)]


def test_verify_не_імпортує_мережевий_код() -> None:
    """`verify_models.py` працює на машині викладача — у ньому не має бути сокетів."""
    source = (script("verify_models").__file__)
    text = Path(source).read_text(encoding="utf-8")
    for forbidden in ("urllib", "socket", "httpx", "requests", "import fetch_models"):
        assert forbidden not in text, forbidden
