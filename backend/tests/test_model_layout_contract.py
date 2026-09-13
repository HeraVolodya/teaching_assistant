"""Гейт на розбіжність між скриптом завантаження й рантаймом.

Спостережено: `scripts/fetch_models.py` кладе моделі в
`models/qwen3-embedding-0.6b`, а рантайм шукає їх у
`models/embeddings/qwen3-embedding-0.6b`. Скрипт відрапортував успіх,
завантаживши 1.8 ГБ, а індексація впала з «не знайдено tokenizer.json».

Це найдорожчий клас помилки в офлайн-постачанні: у закритому контурі немає
куди піти й дозавантажити, а виявляється розбіжність аж на машині викладача.
Тому контракт розкладки перевіряється тестом, а не домовленістю.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from app.embeddings import registry as embed_registry
from app.embeddings.provider import default_model_dir

REPO_ROOT = Path(__file__).resolve().parents[2]
FETCH_SCRIPT = REPO_ROOT / "scripts" / "fetch_models.py"


def _load_fetch_module():
    """Скрипт живе поза пакетом `app` — імпортуємо за шляхом."""
    spec = importlib.util.spec_from_file_location("fetch_models", FETCH_SCRIPT)
    if spec is None or spec.loader is None:
        pytest.skip(f"не вдалося імпортувати {FETCH_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    # Реєстрація ДО exec_module обов'язкова: скрипт має
    # `from __future__ import annotations` + `@dataclass(slots=True)`, а
    # dataclasses при побудові slots-класу шукає власний модуль у sys.modules,
    # щоб розв'язати рядкові анотації. Без цього — AttributeError на NoneType.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def fetch():
    if not FETCH_SCRIPT.is_file():
        pytest.skip("scripts/fetch_models.py відсутній")
    return _load_fetch_module()


def _asset(fetch, asset_id: str):
    for a in fetch.CATALOG:
        if a.id == asset_id:
            return a
    pytest.fail(f"у каталозі немає моделі {asset_id!r}")


def test_embedding_destination_matches_runtime(fetch) -> None:
    """Куди кладе скрипт == де шукає провайдер ембедингів."""
    models_dir = Path("/models")
    model = embed_registry.get("qwen3-embedding-0.6b")
    expected = default_model_dir(model, models_dir).relative_to(models_dir)
    actual = Path(_asset(fetch, "qwen3-embedding-0.6b").dest)
    assert actual == expected, (
        f"скрипт кладе у {actual}, рантайм шукає в {expected} — "
        "у закритому контурі це виявиться аж на машині викладача"
    )


def test_reranker_destination_matches_runtime(fetch) -> None:
    from app.rerank import reranker as rr

    models_dir = Path("/models")
    actual = Path(_asset(fetch, "qwen3-reranker-0.6b").dest)
    expected = rr.default_model_dir(
        rr.get("qwen3-reranker-0.6b"), models_dir
    ).relative_to(models_dir)
    assert actual == expected, f"скрипт: {actual}, рантайм: {expected}"


def test_docling_assets_match_what_docling_itself_asks_for(fetch) -> None:
    """Каталог мусить називати ТІ САМІ репозиторії, що й сам Docling.

    Регресія з живої машини: Docling 2.126.0 перейшов з організації `ds4sd` на
    `docling-project`, а класифікатор рисунків заразом став окремим репозиторієм
    `DocumentFigureClassifier-v2.5`. Каталог лишився на старих іменах, і
    приймання падало на 395-сторінковому підручнику з
    «Model ... not found in artifacts_path» — ПІСЛЯ повного розбору сторінок.

    Тека рахується як `repo_id.replace("/", "--")`, тому неправильний repo_id —
    це завжди неправильний шлях, і жодна перевірка розкладки його не спіймає:
    каталог узгоджений сам із собою, розходиться він із Docling.

    Тут перевіряється й РЕВІЗІЯ: TableFormer прибитий до `v2.3.0`, і `main`
    мовчки дав би іншу версію ваг.
    """
    docling_models = pytest.importorskip(
        "docling.models", reason="docling є лише у групі worker"
    )
    from docling.datamodel.pipeline_options import LayoutObjectDetectionOptions
    from docling.models.stages.picture_classifier.document_picture_classifier import (
        DocumentPictureClassifierOptions,
    )
    from docling.models.stages.table_structure.table_structure_model import (
        TableStructureModel,
    )

    assert docling_models is not None

    layout = LayoutObjectDetectionOptions().model_spec
    picture = DocumentPictureClassifierOptions.from_preset("document_figure_classifier_v2")

    expected = {
        "docling-layout-heron": (layout.repo_id, layout.revision),
        "docling-figure-classifier": (picture.repo_id, picture.revision),
        "docling-tableformer": ("docling-project/docling-models", None),
        "docling-codeformula-v2": ("docling-project/CodeFormulaV2", None),
    }

    for asset_id, (repo_id, revision) in expected.items():
        asset = _asset(fetch, asset_id)
        assert asset.repo == repo_id, (
            f"{asset_id}: каталог тягне {asset.repo!r}, а Docling шукає {repo_id!r}"
        )
        assert Path(asset.dest) == Path("docling") / repo_id.replace("/", "--"), (
            f"{asset_id}: тека {asset.dest!r} не відповідає repo_id {repo_id!r}"
        )
        if revision is not None:
            assert asset.revision == revision, (
                f"{asset_id}: ревізія {asset.revision!r}, Docling чекає {revision!r}"
            )

    # Ревізію TableFormer Docling тримає в самому коді завантаження.
    import inspect
    import re

    src = inspect.getsource(TableStructureModel.download_models)
    pinned = re.findall(r'revision="([^"]+)"', src)
    if pinned:
        assert _asset(fetch, "docling-tableformer").revision == pinned[0], (
            f"TableFormer: Docling прибитий до {pinned[0]!r}, каталог — до "
            f"{_asset(fetch, 'docling-tableformer').revision!r}"
        )


def test_docling_destination_is_the_parent_of_org_repo_folders(fetch) -> None:
    """`DOCLING_ARTIFACTS_PATH` — це БАТЬКІВСЬКИЙ каталог тек `<org>--<repo>`.

    Якщо теки немає, Docling мовчки викликає snapshot_download — тобто йде в
    мережу, а мережевий гард це вб'є вже під час індексації.
    """
    from app.config import Paths, export_model_env

    env = export_model_env(Paths.resolve("/data"))
    artifacts = Path(env["DOCLING_ARTIFACTS_PATH"])
    assert artifacts.name == "docling"

    for asset_id in ("docling-layout-heron", "docling-tableformer"):
        dest = Path(_asset(fetch, asset_id).dest)
        assert dest.parts[0] == "docling", f"{asset_id}: dest={dest}"
        assert "--" in dest.parts[1], (
            f"{asset_id}: тека мусить мати вигляд <org>--<repo>, отримано {dest.parts[1]!r}"
        )


def _rerank_registry():
    """Реранкер імпортується ліниво — так само, як у сусідніх тестах."""
    from app.rerank import reranker as rr

    return rr


@pytest.mark.parametrize(
    ("asset_id", "runtime_onnx_file"),
    [
        ("qwen3-embedding-0.6b", lambda: embed_registry.get("qwen3-embedding-0.6b").onnx_file),
        ("qwen3-reranker-0.6b", lambda: _rerank_registry().get("qwen3-reranker-0.6b").onnx_file),
    ],
)
def test_runtime_onnx_filename_is_actually_downloaded(fetch, asset_id, runtime_onnx_file) -> None:
    """Збігатись мусить не лише КАТАЛОГ, а й ТОЧНЕ ІМ'Я файлу моделі.

    Регресія: реєстр реранкера просив `onnx/model_int8.onnx`, а скрипт клав
    `model_quantized.onnx` — файл з такою назвою у репозиторії відсутній узагалі.
    Обидва сусідні тести проходили: каталог збігався, і «якийсь .onnx» був
    на місці. Падало аж на `ort.InferenceSession` при першому пошуку.

    Пастка тут подвійна, тому перевіряються обидві частини:
      * ім'я файлу (`model_int8` проти `model_quantized`);
      * префікс — `fetch_models.py` кладе файли ПЛАСКО, за basename, тож
        шлях виду `onnx/…` у реєстрі рантайму не існує на диску ніколи.
    """
    expected = runtime_onnx_file()
    downloaded = {f.relpath() for f in _asset(fetch, asset_id).files}
    assert expected in downloaded, (
        f"{asset_id}: рантайм шукає {expected!r}, а скрипт кладе {sorted(downloaded)}. "
        "У закритому контурі це виявиться аж на машині викладача."
    )
    assert "/" not in expected, (
        f"{asset_id}: {expected!r} містить префікс каталогу, а fetch_models.py "
        "зберігає файли пласко — рантайм ніколи не знайде такий шлях"
    )


def test_every_full_profile_asset_has_a_tokenizer_when_needed(fetch) -> None:
    """Ембедер і реранкер без tokenizer.json не запустяться взагалі."""
    for asset_id in ("qwen3-embedding-0.6b", "qwen3-reranker-0.6b"):
        asset = _asset(fetch, asset_id)
        names = {f.relpath() for f in asset.files}
        assert "tokenizer.json" in names, f"{asset_id}: немає tokenizer.json"
        assert any(n.endswith(".onnx") for n in names), f"{asset_id}: немає .onnx"


def test_catalog_has_no_duplicate_destinations(fetch) -> None:
    """Дві моделі в одну теку затирали б файли одна одної."""
    seen: dict[str, str] = {}
    for a in fetch.CATALOG:
        if a.dest in seen and seen[a.dest] != a.id:
            pytest.fail(f"{a.id} і {seen[a.dest]} мають однаковий dest={a.dest!r}")
        seen[a.dest] = a.id
