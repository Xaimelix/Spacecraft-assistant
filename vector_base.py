import json
import os
from functools import lru_cache
from threading import RLock
from time import time
from typing import Dict, Iterable, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Директория по умолчанию для хранения FAISS и сопутствующих файлов
PERSIST_DIR = os.path.join("rag_store", "faiss_index")
ID_MAP_FILENAME = "ids.json"  # карта соответствия doc_id -> список внутренних chunk_ids

# Внутренние кэши, чтобы избежать повторной загрузки инициализации тяжёлых объектов
_VECTORSTORE_CACHE: Dict[str, FAISS] = {}
_VECTORSTORE_SIGNATURES: Dict[str, float] = {}
_VECTORSTORE_LOCK = RLock()


@lru_cache(maxsize=4)
def _build_embeddings(kind: str) -> object:
    """Отдельная фабрика, кэшированная по типу/настройкам."""
    if kind == "openai":
        return OpenAIEmbeddings(model="text-embedding-3-small")

    # Формат: hf::<model_name>
    _, model_name = kind.split("::", 1)
    return HuggingFaceEmbeddings(model_name=model_name)


def _get_embeddings_model():
    provider = os.getenv("EMBEDDINGS_PROVIDER", "hf").lower()
    hf_model = os.getenv("HF_EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    if provider == "openai" and os.getenv("OPENAI_API_KEY"):
        return _build_embeddings("openai")
    return _build_embeddings(f"hf::{hf_model}")


def get_text_splitter(
    chunk_size: int = 1000, chunk_overlap: int = 200
) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )


def _persist_paths(persist_dir: Optional[str] = None) -> Tuple[str, str]:
    base_dir = persist_dir or PERSIST_DIR
    os.makedirs(base_dir, exist_ok=True)
    id_map_path = os.path.join(base_dir, ID_MAP_FILENAME)
    return base_dir, id_map_path


def _load_id_map(persist_dir: Optional[str] = None) -> Dict[str, List[str]]:
    _, id_map_path = _persist_paths(persist_dir)
    if os.path.exists(id_map_path):
        with open(id_map_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_id_map(id_map: Dict[str, List[str]], persist_dir: Optional[str] = None) -> None:
    _, id_map_path = _persist_paths(persist_dir)
    with open(id_map_path, "w", encoding="utf-8") as f:
        json.dump(id_map, f, ensure_ascii=False, indent=2)


def _index_signature(base_dir: str) -> float:
    """Возвращает подпись (максимальный mtime) для файлов индекса."""
    try:
        mtimes = []
        for fname in os.listdir(base_dir):
            path = os.path.join(base_dir, fname)
            if os.path.isfile(path):
                mtimes.append(os.path.getmtime(path))
        return max(mtimes) if mtimes else 0.0
    except FileNotFoundError:
        return 0.0


def load_vectorstore(persist_dir: Optional[str] = None) -> Optional[FAISS]:
    base_dir, _ = _persist_paths(persist_dir)
    cache_key = os.path.abspath(base_dir)
    signature = _index_signature(base_dir)

    with _VECTORSTORE_LOCK:
        cached = _VECTORSTORE_CACHE.get(cache_key)
        cached_sig = _VECTORSTORE_SIGNATURES.get(cache_key, 0.0)
        if cached and cached_sig >= signature:
            return cached

    try:
        store = FAISS.load_local(
            base_dir, _get_embeddings_model(), allow_dangerous_deserialization=True
        )
    except Exception:
        with _VECTORSTORE_LOCK:
            _VECTORSTORE_CACHE.pop(cache_key, None)
            _VECTORSTORE_SIGNATURES.pop(cache_key, None)
        return None

    with _VECTORSTORE_LOCK:
        _VECTORSTORE_CACHE[cache_key] = store
        _VECTORSTORE_SIGNATURES[cache_key] = signature or time()

    return store


def save_vectorstore(store: FAISS, persist_dir: Optional[str] = None) -> None:
    base_dir, _ = _persist_paths(persist_dir)
    store.save_local(base_dir)
    cache_key = os.path.abspath(base_dir)
    with _VECTORSTORE_LOCK:
        _VECTORSTORE_CACHE[cache_key] = store
        _VECTORSTORE_SIGNATURES[cache_key] = _index_signature(base_dir) or time()


def create_document(content: str, metadata: Optional[Dict] = None) -> Document:
    return Document(page_content=content, metadata=metadata or {})


def add_documents(
    documents: Iterable[Document],
    *,
    doc_id: Optional[str] = None,
    text_splitter: Optional[RecursiveCharacterTextSplitter] = None,
    persist_dir: Optional[str] = None,
) -> FAISS:
    """
    Добавляет документы в векторную базу. При необходимости создаёт новую базу.

    - doc_id: логический идентификатор исходного документа. Все созданные чанки
      получат metadata {"doc_id": <doc_id>} и их внутренние ids будут сохранены
      в карте id_map для последующих обновлений.
    - text_splitter: опциональная стратегия разбиения.
    - persist_dir: путь для сохранения/загрузки базы.
    """
    splitter = text_splitter or get_text_splitter()
    chunks: List[Document] = splitter.split_documents(list(documents))

    if doc_id is not None:
        for chunk in chunks:
            chunk.metadata = dict(chunk.metadata or {})
            chunk.metadata["doc_id"] = doc_id

    store = load_vectorstore(persist_dir)
    embeddings = _get_embeddings_model()

    if store is None:
        store = FAISS.from_documents(documents=chunks, embedding=embeddings)
        save_vectorstore(store, persist_dir)
        added_ids = list(store.index_to_docstore_id.values())
    else:
        added_ids = store.add_documents(chunks)
        save_vectorstore(store, persist_dir)
        if added_ids is None:
            # Fallback: собрать все ids (поведение зависит от версии)
            added_ids = list(store.index_to_docstore_id.values())

    if doc_id is not None:
        id_map = _load_id_map(persist_dir)
        existing = id_map.get(doc_id, [])
        id_map[doc_id] = list(dict.fromkeys(existing + list(added_ids)))
        _save_id_map(id_map, persist_dir)

    return store


def update_knowledge_base(
    new_documents: Iterable[Document],
    *,
    doc_id: Optional[str] = None,
    text_splitter: Optional[RecursiveCharacterTextSplitter] = None,
    persist_dir: Optional[str] = None,
) -> None:
    """Добавление новых документов в существующую базу."""
    add_documents(
        new_documents,
        doc_id=doc_id,
        text_splitter=text_splitter,
        persist_dir=persist_dir,
    )


def refresh_document(
    doc_id: str,
    new_content: str,
    *,
    text_splitter: Optional[RecursiveCharacterTextSplitter] = None,
    persist_dir: Optional[str] = None,
) -> None:
    """
    Обновляет документ по логическому идентификатору: удаляет старые чанки и добавляет новую версию.
    """
    store = load_vectorstore(persist_dir)
    if store is None:
        # База не создана — просто создадим её новой версией документа
        add_documents(
            [create_document(new_content)],
            doc_id=doc_id,
            text_splitter=text_splitter,
            persist_dir=persist_dir,
        )
        return

    id_map = _load_id_map(persist_dir)
    chunk_ids = id_map.get(doc_id, [])

    if chunk_ids:
        try:
            store.delete(chunk_ids)
            save_vectorstore(store, persist_dir)
        except Exception:
            _rebuild_store_without_ids(store, set(chunk_ids), persist_dir)

        # Очистим карту
        id_map.pop(doc_id, None)
        _save_id_map(id_map, persist_dir)

    # Добавим новую версию
    add_documents(
        [create_document(new_content)],
        doc_id=doc_id,
        text_splitter=text_splitter,
        persist_dir=persist_dir,
    )


def _rebuild_store_without_ids(
    store: FAISS, exclude_ids: set, persist_dir: Optional[str]
) -> None:
    """Пересобирает FAISS store, исключая документы с internal ids из exclude_ids."""
    keep_pairs = [
        (idx, did) for idx, did in store.index_to_docstore_id.items() if did not in exclude_ids
    ]

    if not keep_pairs:
        # Полная очистка директории индекса
        base_dir, _ = _persist_paths(persist_dir)
        for fname in os.listdir(base_dir):
            try:
                os.remove(os.path.join(base_dir, fname))
            except OSError:
                pass
        return

    kept_docs = [store.docstore.search(did) for _, did in keep_pairs]
    new_store = FAISS.from_documents(kept_docs, _get_embeddings_model())
    save_vectorstore(new_store, persist_dir)

