"""Самотест ембедера — найцінніший запобіжник у системі (план, §4).

Категорія помилок, від якої він захищає, унікальна: помилка в проводці
ембедингів НЕ падає, НЕ логується і НЕ ловиться жодним тестом типів. Вектори
мають правильну розмірність, одиничну норму й цілком розумні косинуси — просто
recall тихо стає випадковим. Викладач бачить це як «асистент почав відповідати
дурниці», через тиждень після зміни, яку ніхто не пов'язує з причиною.

Тому чотири твердження виконуються при КОЖНОМУ старті процесу й після кожної
зміни моделі:

  1. Ембединг відомої української пари питання/уривок, отриманий через код
     застосунку (а не через окремий скрипт!), проти еталонного вектора →
     `cos >= 0.999`. Ловить: іншу ревізію ваг, інший файл ONNX, змінений
     шаблон префікса, іншу стелю обрізання, інший execution provider, зміну
     нормалізації тексту.
  2. `abs(||v|| - 1.0) < 1e-3`. Ловить: втрачену L2-нормалізацію, після якої
     косинусна метрика USearch мовчки міряє не те.
  3. Ембединг пари З префіксом і БЕЗ: з префіксом косинус мусить бути ВИЩИМ.
     Ловить перевернуту проводку (запит відформатований як документ і навпаки)
     — саме ту помилку, яку неможливо побачити очима в дампі векторів.
  4. `tokenizer(chunk).length <= max_seq - 8`; будь-яке обрізання логується.
     Ловить: чанк, більший за вікно моделі, і mismatched стелю в конфізі.

Еталон зберігається в JSON поруч із кодом. У stub-режимі (і при першому запуску
з реальною моделлю, якщо еталон не постачено) він генерується автоматично —
тоді перевірка 1 позначається `skipped`, а порівняння починає працювати з
наступного запуску. Постачання еталона з машини розробника — пункт релізного
чекліста: лише тоді перевірка 1 ловить розбіжність МІЖ машинами, а не лише
дрейф на одній.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from app.embeddings import registry
from app.embeddings.provider import (
    BaseEmbeddingProvider,
    EmbeddingProvider,
    create_provider,
    stub_enabled,
)

__all__ = [
    "NORM_TOLERANCE",
    "NO_BOOTSTRAP_ENV",
    "REFERENCE_COS_MIN",
    "REFERENCE_ENV",
    "REFERENCE_PASSAGE",
    "REFERENCE_QUESTION",
    "CheckResult",
    "SelfTestReport",
    "default_reference_path",
    "run_selftest",
]

log = logging.getLogger(__name__)

REFERENCE_ENV = "ASISTENT_EMBED_REFERENCE"
NO_BOOTSTRAP_ENV = "ASISTENT_EMBED_NO_BOOTSTRAP"

# Українська пара з предметної області асистента. Навмисно містить літерно-
# цифрове позначення (`Д-30`) і числа: саме на них найчастіше видно, що
# токенайзер чи нормалізація змінилися.
REFERENCE_QUESTION = "Яка максимальна дальність стрільби гаубиці Д-30?"
REFERENCE_PASSAGE = (
    "122-мм гаубиця Д-30 забезпечує максимальну дальність стрільби осколково-фугасним "
    "снарядом до 15 300 м, а активно-реактивним снарядом — до 21 900 м. "
    "Скорострільність — 6-8 пострілів за хвилину."
)
# Уривок довжиною з ЖОРСТКИЙ максимум чанка (контракт, §4: 3600 символів).
# Потрібен саме такий: перевірка 4 має міряти найгірший реальний випадок,
# а не типовий.
REFERENCE_LONG_CHUNK = (REFERENCE_PASSAGE + " ") * 20

REFERENCE_COS_MIN = 0.999
NORM_TOLERANCE = 1e-3


@dataclass(slots=True)
class CheckResult:
    """Результат однієї перевірки. Навмисно НЕ голий bool.

    Голий bool у цьому місці — це тікет «пошук працює погано» без жодної
    зачіпки. Тут кожна перевірка несе виміряне значення, поріг і українське
    пояснення, придатне для показу в UI діагностики.
    """

    name: str
    title: str
    ok: bool
    detail: str
    value: float | None = None
    threshold: float | None = None
    skipped: bool = False

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class SelfTestReport:
    model_id: str
    model_key: str
    dim: int
    stub: bool
    checks: list[CheckResult] = field(default_factory=list)
    reference_path: str = ""
    bootstrapped: bool = False
    truncated_inputs: int = 0
    max_tokens_seen: int = 0
    max_input_tokens: int = 0
    providers: list[str] = field(default_factory=list)
    tokenizer_appends_eos: bool | None = None
    duration_ms: float = 0.0
    error: str | None = None

    @property
    def ok(self) -> bool:
        """Пропущена перевірка не валить звіт — провалена валить."""
        return self.error is None and all(c.ok for c in self.checks)

    @property
    def failures(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.ok]

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    def format_uk(self) -> str:
        """Один людський рядок для логів і для екрана діагностики."""
        head = "OK" if self.ok else "ПОМИЛКА"
        lines = [f"Самотест ембедера [{head}] модель={self.model_id} dim={self.dim}"
                 f"{' (заглушка)' if self.stub else ''}"]
        for c in self.checks:
            mark = "—" if c.skipped else ("+" if c.ok else "!")
            lines.append(f"  {mark} {c.title}: {c.detail}")
        if self.error:
            lines.append(f"  ! Самотест не виконано: {self.error}")
        return "\n".join(lines)


# ------------------------------------------------------------------ еталони
def default_reference_path() -> Path:
    """Де лежить JSON з еталонними векторами.

    Пріоритет: явна змінна середовища → файл, що постачається поруч із кодом →
    каталог даних. У каталог поруч із кодом ми ПИШЕМО лише якщо він уже існує:
    у постачанні пакет може лежати на диску лише для читання, і падати на
    цьому самотест не має права.
    """
    override = os.environ.get(REFERENCE_ENV, "").strip()
    if override:
        return Path(override).expanduser()
    shipped = Path(__file__).with_name("selftest_reference.json")
    if shipped.is_file():
        return shipped
    try:
        from app.config import Paths

        return Paths.resolve().models_dir / "embeddings" / "selftest_reference.json"
    except Exception:                       # pragma: no cover — небезпечний каталог даних
        return shipped


def _load_references(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:    # pragma: no cover — пошкоджений файл
        log.warning("Еталонні вектори в %s не читаються (%s); перевірку 1 буде пропущено.", path, exc)
        return {}
    return data if isinstance(data, dict) else {}


def _store_reference(path: Path, key: str, entry: dict) -> bool:
    data = _load_references(path)
    data[key] = entry
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        return True
    except OSError as exc:                  # pragma: no cover — диск лише для читання
        log.warning("Не вдалося зберегти еталонні вектори в %s: %s", path, exc)
        return False


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ------------------------------------------------------------------ самотест
def run_selftest(
    provider: EmbeddingProvider | None = None,
    *,
    model_name: str | None = None,
    reference_path: Path | str | None = None,
    allow_bootstrap: bool | None = None,
) -> SelfTestReport:
    """Виконати всі чотири твердження. Ніколи не кидає — завжди повертає звіт.

    Виняток тут був би найгіршим варіантом: самотест викликається на старті
    процесу, і впасти він мусить у вигляді зрозумілого повідомлення в UI, а не
    трейсбеком у логу, який ніхто не читає.
    """
    started = time.perf_counter()
    own_provider = provider is None
    if provider is None:
        provider = create_provider(model_name)

    model = getattr(provider, "model", registry.get(model_name))
    report = SelfTestReport(
        model_id=model.id,
        model_key=provider.model_key,
        dim=provider.dim,
        stub=stub_enabled() if own_provider else provider.__class__.__name__.startswith("Stub"),
        max_input_tokens=int(getattr(provider, "max_input_tokens", 0) or 0),
        providers=list(getattr(provider, "providers", [])),
        tokenizer_appends_eos=getattr(provider, "tokenizer_appends_eos", None),
    )
    path = Path(reference_path) if reference_path is not None else default_reference_path()
    report.reference_path = str(path)
    if allow_bootstrap is None:
        flag = os.environ.get(NO_BOOTSTRAP_ENV, "").strip().lower()
        allow_bootstrap = flag not in {"1", "true", "yes", "on"}

    try:
        if isinstance(provider, BaseEmbeddingProvider):
            provider.stats.reset()

        q_vec = provider.embed_queries([REFERENCE_QUESTION])[0]
        d_vec = provider.embed_documents([REFERENCE_PASSAGE])[0]

        report.checks.append(_check_reference(provider, model, q_vec, d_vec, path, allow_bootstrap, report))
        report.checks.append(_check_norm(q_vec, d_vec))
        report.checks.append(_check_prefix_wiring(provider, model, d_vec))
        report.checks.append(_check_token_budget(provider, model))

        if isinstance(provider, BaseEmbeddingProvider):
            report.truncated_inputs = provider.stats.truncated
            report.max_tokens_seen = provider.stats.max_tokens_seen
    except Exception as exc:
        report.error = f"{type(exc).__name__}: {exc}"
        log.exception("Самотест ембедера не виконано")
    finally:
        if own_provider and hasattr(provider, "close"):
            provider.close()
        report.duration_ms = (time.perf_counter() - started) * 1000.0

    if not report.ok:
        log.error("%s", report.format_uk())
    return report


# ---------------------------------------------------------------- перевірки
def _check_reference(
    provider: EmbeddingProvider,
    model: registry.EmbeddingModel,
    q_vec: np.ndarray,
    d_vec: np.ndarray,
    path: Path,
    allow_bootstrap: bool,
    report: SelfTestReport,
) -> CheckResult:
    key = provider.model_key
    stored = _load_references(path).get(key)

    if stored is None:
        if not allow_bootstrap:
            return CheckResult(
                name="reference_vector",
                title="Збіг з еталонним вектором",
                ok=True,
                skipped=True,
                detail=f"Еталона для цієї моделі немає у {path.name}, автостворення вимкнено.",
                threshold=REFERENCE_COS_MIN,
            )
        entry = {
            "model_id": model.id,
            "dim": int(provider.dim),
            "question": REFERENCE_QUESTION,
            "passage": REFERENCE_PASSAGE,
            "query_vector": [float(x) for x in np.asarray(q_vec).ravel()],
            "document_vector": [float(x) for x in np.asarray(d_vec).ravel()],
        }
        saved = _store_reference(path, key, entry)
        report.bootstrapped = saved
        return CheckResult(
            name="reference_vector",
            title="Збіг з еталонним вектором",
            ok=True,
            skipped=True,
            detail=(
                f"Еталон створено вперше й записано у {path}; порівняння почне працювати з "
                "наступного запуску."
                if saved
                else "Еталона немає, і записати його не вдалося (каталог лише для читання)."
            ),
            threshold=REFERENCE_COS_MIN,
        )

    ref_q = np.asarray(stored.get("query_vector", []), dtype=np.float64)
    ref_d = np.asarray(stored.get("document_vector", []), dtype=np.float64)
    if ref_q.size != provider.dim or ref_d.size != provider.dim:
        return CheckResult(
            name="reference_vector",
            title="Збіг з еталонним вектором",
            ok=False,
            detail=(
                f"Еталон має розмірність {ref_q.size}, а модель — {provider.dim}. "
                "Файл еталонів застарів або належить іншій моделі."
            ),
            threshold=REFERENCE_COS_MIN,
        )

    cos_q, cos_d = _cos(q_vec, ref_q), _cos(d_vec, ref_d)
    worst = min(cos_q, cos_d)
    ok = worst >= REFERENCE_COS_MIN
    return CheckResult(
        name="reference_vector",
        title="Збіг з еталонним вектором",
        ok=ok,
        value=worst,
        threshold=REFERENCE_COS_MIN,
        detail=(
            f"cos(запит)={cos_q:.6f}, cos(документ)={cos_d:.6f} при порозі {REFERENCE_COS_MIN}."
            + (
                ""
                if ok
                else " Ембединги розійшлися з еталоном: змінилися ваги, файл ONNX, шаблон "
                "префікса, нормалізація тексту або execution provider. Переіндексація "
                "обов'язкова, інакше старі вектори несумісні з новими запитами."
            )
        ),
    )


def _check_norm(q_vec: np.ndarray, d_vec: np.ndarray) -> CheckResult:
    norms = [float(np.linalg.norm(np.asarray(v, dtype=np.float64))) for v in (q_vec, d_vec)]
    worst = max(abs(n - 1.0) for n in norms)
    ok = worst < NORM_TOLERANCE
    return CheckResult(
        name="unit_norm",
        title="Одинична L2-норма",
        ok=ok,
        value=worst,
        threshold=NORM_TOLERANCE,
        detail=(
            f"Максимальне відхилення норми від 1.0 — {worst:.2e}."
            + ("" if ok else " Без нормалізації косинусна метрика USearch міряє не те, "
                             "що ви думаєте: довгі тексти отримують перевагу за модулем.")
        ),
    )


def _check_prefix_wiring(
    provider: EmbeddingProvider,
    model: registry.EmbeddingModel,
    d_vec: np.ndarray,
) -> CheckResult:
    """Перевірка 3: з префіксом косинус мусить бути ВИЩИМ, ніж без нього."""
    if not model.query_template:
        return CheckResult(
            name="prefix_wiring",
            title="Проводка префікса запиту",
            ok=True,
            skipped=True,
            detail=f"Модель {model.id} не має шаблону запиту — перевіряти нічого.",
        )
    if not isinstance(provider, BaseEmbeddingProvider):   # pragma: no cover — чужий провайдер
        return CheckResult(
            name="prefix_wiring",
            title="Проводка префікса запиту",
            ok=True,
            skipped=True,
            detail="Провайдер не дозволяє вимкнути шаблон — перевірку пропущено.",
        )

    with_prefix = provider.encode([REFERENCE_QUESTION], side="query")[0]
    without_prefix = provider.encode([REFERENCE_QUESTION], side="query", apply_template=False)[0]
    cos_with, cos_without = _cos(with_prefix, d_vec), _cos(without_prefix, d_vec)
    ok = cos_with > cos_without
    return CheckResult(
        name="prefix_wiring",
        title="Проводка префікса запиту",
        ok=ok,
        value=cos_with - cos_without,
        threshold=0.0,
        detail=(
            f"cos з префіксом {cos_with:.4f} проти {cos_without:.4f} без нього."
            + ("" if ok else " Префікс погіршує схожість — найімовірніше, проводку перевернуто "
                             "(запит форматується як документ або навпаки) чи шаблон реєстру "
                             "застосовано не побайтово.")
        ),
    )


def _check_token_budget(provider: EmbeddingProvider, model: registry.EmbeddingModel) -> CheckResult:
    """Перевірка 4: жоден вхід не перевищує `max_seq - 8`, обрізання видиме."""
    ceiling = model.max_seq - 8
    limit = int(getattr(provider, "max_input_tokens", ceiling))
    counter = getattr(provider, "count_tokens", None)
    long_tokens = int(counter(REFERENCE_LONG_CHUNK)) if callable(counter) else 0

    truncated_before = provider.stats.truncated if isinstance(provider, BaseEmbeddingProvider) else 0
    provider.embed_documents([REFERENCE_LONG_CHUNK])
    truncated_after = provider.stats.truncated if isinstance(provider, BaseEmbeddingProvider) else 0
    truncated = truncated_after - truncated_before

    ok = limit <= ceiling
    detail = (
        f"Стеля входу {limit} токенів при max_seq={model.max_seq} (дозволено {ceiling}); "
        f"найдовший пробний чанк ({len(REFERENCE_LONG_CHUNK)} символів) — {long_tokens} токенів."
    )
    if truncated:
        detail += (
            f" Обрізано {truncated} вхід(ів): частина тексту не потрапила у вектор — "
            "перевірте жорсткий максимум чанка (3600 символів) для цієї моделі."
        )
    if not ok:
        detail += (
            " Стеля перевищує max_seq - 8: модель мовчки обріже вхід сама, і чанк "
            "втратить хвіст без жодного запису в журналі."
        )
    return CheckResult(
        name="token_budget",
        title="Бюджет токенів",
        ok=ok,
        value=float(limit),
        threshold=float(ceiling),
        detail=detail,
    )
