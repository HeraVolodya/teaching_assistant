"""Асистенти, їхні колекції й прев'ю системного промпту.

Завдання 4 НДР — «викладач створює асистента без розробника» — тримається на
одному рішенні: конфіг асистента є ВЕРСІЙОВАНИМИ ДАНИМИ (`config_json`), а не
кодом. Тому цей роутер нічого не знає про поведінку асистента; він приймає
словник, фільтрує його по відомих полях і зберігає.

ОДИН АСИСТЕНТ = ОДНА КОЛЕКЦІЯ ЗА ЗАМОВЧУВАННЯМ.
Колекція — це фізична партиція: один файл ANN-індексу. Ізоляція асистентів
таким чином фізична, а не через метаданий фільтр, і не коштує ані втрати
recall, ані ризику витоку матеріалів одного викладача у відповідь іншого.
Колекція створюється разом з асистентом, бо асистент без неї не має куди
приймати документи, а змушувати UI робити другий виклик — це гарантований
напівстворений асистент після падіння між викликами.
"""

from __future__ import annotations

import logging
import shutil
import sqlite3
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from app.api.schemas import (
    AssistantIn,
    AssistantOut,
    PromptPreviewOut,
    assistant_out,
    collection_out,
    config_from_dict,
)
from app.api.state import Services
from app.db.repositories import AssistantRepo, ChunkRepo, CollectionRepo, DocumentRepo
from app.domain import Assistant, Collection, new_id
from app.jobs.pipeline import resolve_document_path

__all__ = ["router", "services_of", "default_collection_id"]

log = logging.getLogger("asistent.api")
router = APIRouter(tags=["assistants"])


def services_of(request: Request) -> Services:
    services: Services | None = getattr(request.app.state, "services", None)
    if services is None:  # pragma: no cover — можливе лише поза lifespan
        raise HTTPException(status_code=503, detail="Застосунок ще не ініціалізовано.")
    return services


def _collections_for(con: sqlite3.Connection, assistant_id: str) -> list[Any]:
    out = []
    for collection in CollectionRepo(con).for_assistant(assistant_id):
        documents = con.execute(
            "SELECT count(*) FROM documents WHERE collection_id=?", (collection.id,)
        ).fetchone()[0]
        chunks = con.execute(
            "SELECT count(*) FROM chunks WHERE collection_id=? AND level='L2'",
            (collection.id,),
        ).fetchone()[0]
        out.append(collection_out(collection, documents=documents, chunks=chunks))
    return out


def default_collection_id(services: Services, assistant_id: str) -> str | None:
    with services.db.connection() as con:
        collections = CollectionRepo(con).for_assistant(assistant_id)
    return collections[0].id if collections else None


def _new_collection(services: Services, assistant_id: str, name: str = "Основна") -> Collection:
    """Колекція під поточну embedding-модель.

    Ключ і розмірність беруться з РЕЄСТРУ, а не з завантаженого провайдера:
    асистента мусить бути можливо створити на машині, де ваги ще не
    розпаковані (майстер першого запуску саме так і працює).
    """
    from app.embeddings import registry

    model = registry.get(services.settings.embedding_model)
    return Collection(
        id=new_id(),
        assistant_id=assistant_id,
        name=name,
        embedding_model_id=model.id,
        embedding_model_key=registry.model_key(model),
        dim=model.dim,
    )


# ------------------------------------------------------------------ CRUD
@router.get("/assistants", response_model=list[AssistantOut])
def list_assistants(request: Request) -> list[AssistantOut]:
    services = services_of(request)
    with services.db.connection() as con:
        return [
            assistant_out(a, _collections_for(con, a.id)) for a in AssistantRepo(con).list()
        ]


@router.post("/assistants", response_model=AssistantOut, status_code=201)
def create_assistant(request: Request, payload: AssistantIn) -> AssistantOut:
    services = services_of(request)
    assistant = Assistant(
        id=new_id(),
        name=payload.name.strip(),
        description=payload.description,
        colour=payload.colour,
        emoji=payload.emoji,
        config=config_from_dict(payload.config),
    )
    with services.db.transaction() as con:
        AssistantRepo(con).create(assistant)
        CollectionRepo(con).create(_new_collection(services, assistant.id))
        collections = _collections_for(con, assistant.id)
    return assistant_out(assistant, collections)


@router.get("/assistants/{assistant_id}", response_model=AssistantOut)
def get_assistant(request: Request, assistant_id: str) -> AssistantOut:
    services = services_of(request)
    with services.db.connection() as con:
        assistant = AssistantRepo(con).get(assistant_id)
        if assistant is None:
            raise HTTPException(status_code=404, detail="Асистента не знайдено.")
        return assistant_out(assistant, _collections_for(con, assistant_id))


@router.put("/assistants/{assistant_id}", response_model=AssistantOut)
def update_assistant(request: Request, assistant_id: str, payload: AssistantIn) -> AssistantOut:
    services = services_of(request)
    with services.db.transaction() as con:
        repo = AssistantRepo(con)
        current = repo.get(assistant_id)
        if current is None:
            raise HTTPException(status_code=404, detail="Асистента не знайдено.")
        current.name = payload.name.strip()
        current.description = payload.description
        current.colour = payload.colour
        current.emoji = payload.emoji
        # Часткове оновлення: PUT без `config` не має скидати налаштування
        # пошуку до дефолтів — це найдорожча можлива несподіванка для
        # викладача, який півдня калібрував асистента.
        if payload.config is not None:
            current.config = config_from_dict(payload.config)
        repo.update(current)
        updated = repo.get(assistant_id)
        assert updated is not None
        collections = _collections_for(con, assistant_id)
    return assistant_out(updated, collections)


@router.delete("/assistants/{assistant_id}", status_code=204)
def delete_assistant(request: Request, assistant_id: str) -> None:
    """Видалити асистента РАЗОМ з його чатами, документами й файлами.

    КАСКАД БД ПРИБИРАЄ НЕ ВСЕ, І ЦЕ БУЛО ТИХОЮ ВИТОКОМ МІСЦЯ.
    Схема має повний ланцюг `ON DELETE CASCADE` (асистент → колекції →
    документи → сторінки/чанки, і окремо асистент → сесії → повідомлення →
    відгуки), тож БД після видалення чиста. Але оригінали підручників лежать
    НЕ в базі, а файлами під UUID у `docs/`, і каскад до них не дотягується.
    Раніше тут видалявся лише рядок — і кожен видалений асистент лишав по
    собі свої PDF назавжди. На сканованому підручнику це 50-100 МБ, яких
    викладач ніколи не побачить і не зможе знайти: ім'я файлу — UUID, а
    зв'язок з оригінальною назвою жив у щойно видаленому рядку.

    ПОРЯДОК КРОКІВ ОБОВ'ЯЗКОВИЙ.
    1. Спершу скасувати завдання: воркер тримає PDF відкритим, і на Windows
       `unlink` зайнятого файлу кидає `PermissionError`, а не видаляє.
    2. Зібрати шляхи ДО видалення рядків — після каскаду дізнатись їх нема звідки.
    3. Видалити рядки.
    4. Аж потім чіпати диск: файл, зайнятий читачем, не має валити запит,
       який у базі вже успішно завершився.
    """
    services = services_of(request)
    with services.db.connection() as con:
        if AssistantRepo(con).get(assistant_id) is None:
            raise HTTPException(status_code=404, detail="Асистента не знайдено.")
        collections = CollectionRepo(con).for_assistant(assistant_id)
        documents = [
            document
            for collection in collections
            for document in DocumentRepo(con).for_collection(collection.id)
        ]

    for document in documents:
        services.queue.cancel_document(document.id)
    files = [resolve_document_path(services.paths, document) for document in documents]

    with services.db.transaction() as con:
        # FTS5 ПЕРЕД каскадом, і тільки вручну.
        # `chunk_fts` і `code_fts` — віртуальні таблиці; зовнішніх ключів до
        # віртуальних таблиць SQLite не підтримує, тому `ON DELETE CASCADE`
        # їх не бачить. Каскад забирає `chunks`, а їхні пошукові рядки
        # лишаються назавжди — і саме вони, а не самі чанки, найбільше
        # роздувають файл бази на великих підручниках. Порядок критичний:
        # `delete_for_document` шукає rowid через `chunks`, тож після каскаду
        # видаляти вже не було б за чим.
        chunks = ChunkRepo(con)
        for document in documents:
            chunks.delete_for_document(document.id)
        AssistantRepo(con).delete(assistant_id)

    for path in files:
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:  # pragma: no cover — файл зайнятий читачем на Windows
            log.warning("Не вдалося видалити файл %s: %s", path, exc)

    # Індекс — цілий каталог на колекцію (`index_path_for`), тож прибираємо
    # його разом із версійованими підкаталогами моделей. `ignore_errors`, бо
    # mmap-файл може бути ще відкритий: залишок каталогу нікому не заважає,
    # а провалений HTTP-запит після успішного видалення в базі — заважає.
    for collection in collections:
        shutil.rmtree(services.paths.index_dir / collection.id, ignore_errors=True)

    _reclaim_space(services)


def _reclaim_space(services: Services) -> None:
    """Повернути ОС сторінки, звільнені каскадом.

    SQLite не зменшує файл сам: видалені сторінки лишаються в ньому як вільні
    й мовчки переживають перезапуск. Для викладача це виглядає як «видалив
    підручник, а база й далі важить 400 МБ».

    `incremental_vacuum`, а не `VACUUM`: повний потребує монопольного доступу
    й тимчасової копії всього файлу, а тут поруч працює процес-воркер із
    власним з'єднанням. Інкрементальний доступний лише тому, що
    `auto_vacuum=INCREMENTAL` виставлено на кожному з'єднанні (`SQLITE_PRAGMAS`).
    """
    try:
        with services.db.connection() as con:
            con.execute("PRAGMA incremental_vacuum")
    except sqlite3.Error as exc:  # pragma: no cover — зайнята база не привід падати
        log.warning("Не вдалося повернути вільні сторінки: %s", exc)


# --------------------------------------------------------- прев'ю промпту
@router.get("/assistants/{assistant_id}/prompt-preview", response_model=PromptPreviewOut)
def prompt_preview(request: Request, assistant_id: str) -> PromptPreviewOut:
    """Що саме побачить модель — і що НЕ побачить.

    Показуємо і промпт, і числові пороги. Промптом неможливо надійно змусити
    12B-модель відмовитись відповідати; порогом реранкера — можна. Тому
    «висока впевненість» в UI мусить читатися як 0.55, а не як ввічливе
    прохання, і це прев'ю — єдине місце, де обидві половини видно разом.
    """
    services = services_of(request)
    with services.db.connection() as con:
        assistant = AssistantRepo(con).get(assistant_id)
    if assistant is None:
        raise HTTPException(status_code=404, detail="Асистента не знайдено.")

    from app.generation.prompt_builder import (
        SYSTEM_RULES_UK,
        build_prompt,
        estimate_tokens_uk,
    )

    persona = assistant.config.instructions or ""
    messages = build_prompt("Приклад питання викладача", [], assistant.config, persona=persona)
    system = next((m["content"] for m in messages if m.get("role") == "system"), SYSTEM_RULES_UK)
    sample = "\n\n".join(f"[{m['role']}]\n{m['content']}" for m in messages)
    return PromptPreviewOut(
        system=system,
        persona=persona,
        rules_tokens=estimate_tokens_uk(SYSTEM_RULES_UK),
        persona_tokens=estimate_tokens_uk(persona),
        confidence_threshold=assistant.config.confidence_threshold(),
        min_supporting_chunks=assistant.config.min_supporting_chunks(),
        final_top_k=assistant.config.final_top_k,
        max_per_document=assistant.config.max_per_document,
        on_missing_information=assistant.config.on_missing_information,
        sample=sample,
    )
