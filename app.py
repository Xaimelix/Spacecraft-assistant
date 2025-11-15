import os
import re
import uuid
from collections import deque
from datetime import datetime

from flask import Flask, jsonify, redirect, render_template, request, send_from_directory, url_for, session

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Опциональные импорты для оффлайн LLM
try:
    from langchain_ollama import ChatOllama
except ImportError:
    ChatOllama = None

try:
    from langchain_community.llms import HuggingFacePipeline
except ImportError:
    HuggingFacePipeline = None

from vector_base import (
    update_knowledge_base,
    create_document,
    load_vectorstore,
)

from dotenv import load_dotenv


UPLOAD_DIR = os.path.join(os.getcwd(), "uploads")
ALLOWED_EXTENSIONS = {"pdf", "txt"}
load_dotenv("api_key.env")

MAX_MEMORY_TURNS = int(os.getenv("MEMORY_MAX_TURNS", "6"))
MAX_MEMORY_USER_CONTEXT = int(os.getenv("MEMORY_USER_CONTEXT", "3"))
_conversation_memory: dict[str, deque] = {}


# Кэш для LLM модели (чтобы не пересоздавать каждый раз)
_llm_cache = None
_model_info_cache = None  # Информация о модели в кэше


def get_llm(model_mode: str = "auto"):
    """
    Получает LLM модель для генерации ответов.
    
    Args:
        model_mode: Режим выбора модели:
            - "auto": автоматический выбор (приоритет: оффлайн > онлайн)
            - "offline": только оффлайн модели (Ollama/HuggingFace)
            - "online": только онлайн модели (OpenAI/Timeweb)
    
    Returns:
        LLM модель или None, если ничего не доступно
    """
    global _llm_cache, _model_info_cache
    
    # Для режима "auto" используем кэш
    # Для других режимов всегда проверяем заново
    if model_mode == "auto":
        if _llm_cache is not None and _llm_cache is not False:
            # Если в кэше есть модель, возвращаем её с информацией о модели
            return _llm_cache, _model_info_cache or "Кэшированная модель"
        if _llm_cache is False:
            return None, None
    
    model_used = None  # Для отслеживания какой модели используется
    
    # 1. Попытка использовать Ollama (оффлайн, локальный)
    if model_mode in ("auto", "offline") and ChatOllama is not None:
        ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2")
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        try:
            llm = ChatOllama(
                model=ollama_model,
                base_url=ollama_base_url,
                temperature=0,
            )
            # Легкая проверка доступности через простой запрос
            try:
                test_response = llm.invoke([HumanMessage(content="ok")])
                if test_response and test_response.content:
                    model_used = f"Ollama ({ollama_model})"
                    print(f"✓ Используется Ollama модель: {ollama_model}")
                    if model_mode == "auto":
                        _llm_cache = llm
                        _model_info_cache = model_used
                    return llm, model_used
            except Exception:
                pass  # Ollama не отвечает, пробуем дальше
        except Exception as e:
            print(f"Ollama недоступен: {e}")
    
    # 2. Попытка использовать локальную HuggingFace модель (оффлайн)
    hf_model = os.getenv("HF_MODEL")
    if model_mode in ("auto", "offline") and hf_model and HuggingFacePipeline is not None:
        try:
            from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
            import torch
            
            print(f"Загрузка локальной модели HuggingFace: {hf_model}...")
            tokenizer = AutoTokenizer.from_pretrained(hf_model)
            model = AutoModelForCausalLM.from_pretrained(
                hf_model,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
            
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=512,
                temperature=0,
                do_sample=False,
            )
            
            llm = HuggingFacePipeline(pipeline=pipe)
            model_used = f"HuggingFace ({hf_model})"
            print(f"✓ Используется локальная HuggingFace модель: {hf_model}")
            if model_mode == "auto":
                _llm_cache = llm
                _model_info_cache = model_used
            return llm, model_used
        except Exception as e:
            print(f"Локальная HuggingFace модель недоступна: {e}")
    
    # 3. Fallback на OpenAI/Timeweb (онлайн)
    api_key = os.getenv("timeweb_api")
    if model_mode in ("auto", "online") and api_key:
        try:
            llm = ChatOpenAI(
                base_url=os.getenv("timeweb_openai_url"),
                api_key=api_key,
                model=os.getenv("OPENAI_MODEL", "gpt-5-nano"),
                temperature=0
            )
            model_used = "OpenAI/Timeweb API"
            print("✓ Используется OpenAI/Timeweb API (онлайн)")
            if model_mode == "auto":
                _llm_cache = llm
                _model_info_cache = model_used
            return llm, model_used
        except Exception as e:
            print(f"OpenAI/Timeweb недоступен: {e}")
    
    # 4. Если ничего не доступно, возвращаем None
    if model_mode == "offline":
        print("⚠ Предупреждение: оффлайн модели недоступны.")
    elif model_mode == "online":
        print("⚠ Предупреждение: онлайн API недоступен.")
    else:
        print("⚠ Предупреждение: ни одна LLM модель не доступна. Будет использован простой ответ на основе контекста.")
    
    if model_mode == "auto":
        _llm_cache = False  # Кэшируем False, чтобы не пытаться снова
        _model_info_cache = None
    return None, None


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def safe_filename(filename: str) -> str:
    """
    Создает безопасное имя файла, сохраняя кириллические символы.
    Удаляет только опасные символы для файловой системы.
    """
    # Получаем расширение
    if "." in filename:
        name, ext = filename.rsplit(".", 1)
        ext = "." + ext.lower()
    else:
        name = filename
        ext = ""
    
    # Удаляем опасные символы для файловой системы Windows/Linux
    # Сохраняем кириллицу и другие Unicode символы
    dangerous_chars = r'[<>:"/\\|?*\x00-\x1f]'
    name = re.sub(dangerous_chars, '_', name)
    
    # Удаляем ведущие/замыкающие точки и пробелы
    name = name.strip('. ')
    
    # Если имя стало пустым, используем UUID
    if not name:
        name = str(uuid.uuid4())
    
    return name + ext


def _ensure_session_id() -> str:
    """Возвращает идентификатор сессии пользователя и создает его при необходимости."""
    sid = session.get("session_id")
    if not sid:
        sid = session["session_id"] = uuid.uuid4().hex
    return sid


def _get_history(session_id: str) -> deque | None:
    return _conversation_memory.get(session_id)


def _append_history(session_id: str, role: str, content: str) -> None:
    if not content:
        return
    history = _conversation_memory.get(session_id)
    if history is None:
        history = deque(maxlen=MAX_MEMORY_TURNS * 2)
        _conversation_memory[session_id] = history
    history.append({"role": role, "content": content})


def _history_to_messages(history: deque | None) -> list:
    if not history:
        return []
    messages = []
    for item in history:
        role = item.get("role")
        content = item.get("content") or ""
        if not content:
            continue
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    return messages


def _augment_query_with_memory(question: str, history: deque | None) -> str:
    if not history:
        return question
    recent_user_messages = [
        item.get("content") or ""
        for item in history
        if item.get("role") == "user" and item.get("content")
    ]
    if not recent_user_messages or MAX_MEMORY_USER_CONTEXT <= 0:
        return question
    tail = recent_user_messages[-MAX_MEMORY_USER_CONTEXT:]
    memory_context = "\n".join(tail)
    return f"{question}\n\nПредыдущие вопросы пользователя:\n{memory_context}"


def _reset_history(session_id: str) -> None:
    _conversation_memory.pop(session_id, None)


def create_app() -> Flask:
    app = Flask(__name__, static_folder="static", template_folder="templates")
    app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "dev-secret")
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    @app.get("/healthz")
    def healthz():
        return jsonify({"status": "ok", "time": datetime.utcnow().isoformat()})

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/upload")
    def upload_page():
        return render_template("upload.html")

    @app.post("/api/upload")
    def api_upload():
        if "file" not in request.files:
            return jsonify({"ok": False, "error": "Нет файла в запросе"}), 400
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"ok": False, "error": "Имя файла пустое"}), 400
        if not allowed_file(file.filename):
            return jsonify({"ok": False, "error": "Разрешены только .pdf и .txt"}), 400
        
        # Используем безопасную функцию, сохраняющую кириллические символы
        original_filename = file.filename
        filename = safe_filename(original_filename)
        save_path = os.path.join(UPLOAD_DIR, filename)
        file.save(save_path)

        # Ingest into vector store
        print(f"Оригинальное имя: {original_filename}, сохранено как: {filename}")
        ext = filename.rsplit(".", 1)[1].lower()
        # Используем оригинальное имя файла для doc_id, если не указано явно
        logical_doc_id = request.form.get("doc_id") or original_filename

        try:
            documents: list[Document] = []
            if ext == "pdf":
                # Lazy import to keep startup fast
                from langchain_community.document_loaders import PyPDFLoader

                loader = PyPDFLoader(save_path)
                documents = loader.load()
            elif ext == "txt":
                text = open(save_path, "r", encoding="utf-8", errors="ignore").read()
                documents = [create_document(text)]

            update_knowledge_base(documents, doc_id=logical_doc_id)
            
            # Удаляем файл после успешной загрузки в базу знаний
            # (документ уже обработан и сохранен в векторной базе)
            try:
                if os.path.exists(save_path):
                    os.remove(save_path)
            except Exception as cleanup_error:
                # Логируем ошибку удаления, но не прерываем успешный ответ
                print(f"Предупреждение: не удалось удалить файл {save_path}: {cleanup_error}")
            
            return jsonify({"ok": True, "doc_id": logical_doc_id, "filename": filename})
        except Exception as e:
            # При ошибке оставляем файл для возможной отладки
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.get("/chat")
    def chat_page():
        return render_template("chat.html")
    
    @app.post("/api/chat/reset")
    def reset_chat():
        session_id = _ensure_session_id()
        had_history = bool(_get_history(session_id))
        _reset_history(session_id)
        message = (
            "Память диалога очищена."
            if had_history
            else "Память уже пуста. Можно продолжать чат."
        )
        return jsonify({"ok": True, "message": message, "memory_turns": 0})

    @app.post("/api/chat")
    def api_chat():
        # Поддержка как JSON, так и FormData (для загрузки файлов)
        uploaded_files = []
        question = ""
        model_mode = "auto"
        session_id = _ensure_session_id()
        history = _get_history(session_id)
        
        # Проверяем тип запроса: FormData или JSON
        # FormData запросы имеют request.form, даже если файлов нет
        if request.form:
            # Это FormData запрос
            question = (request.form.get("message") or "").strip()
            model_mode = request.form.get("model_mode", "auto")
            
            # Обрабатываем загруженные файлы (если есть)
            if request.files:
                files = request.files.getlist("files")
                for file in files:
                    if file.filename == "":
                        continue
                    if not allowed_file(file.filename):
                        continue
                    
                    # Сохраняем файл временно
                    original_filename = file.filename
                    filename = safe_filename(original_filename)
                    save_path = os.path.join(UPLOAD_DIR, filename)
                    file.save(save_path)
                    
                    try:
                        # Обрабатываем файл и добавляем в базу знаний
                        ext = filename.rsplit(".", 1)[1].lower()
                        documents: list[Document] = []
                        
                        if ext == "pdf":
                            from langchain_community.document_loaders import PyPDFLoader
                            loader = PyPDFLoader(save_path)
                            documents = loader.load()
                        elif ext == "txt":
                            text = open(save_path, "r", encoding="utf-8", errors="ignore").read()
                            documents = [create_document(text)]
                        
                        # Используем оригинальное имя файла как doc_id
                        logical_doc_id = original_filename
                        update_knowledge_base(documents, doc_id=logical_doc_id)
                        uploaded_files.append(original_filename)
                        
                        # Удаляем временный файл после успешной обработки
                        try:
                            if os.path.exists(save_path):
                                os.remove(save_path)
                        except Exception as cleanup_error:
                            print(f"Предупреждение: не удалось удалить файл {save_path}: {cleanup_error}")
                            
                    except Exception as e:
                        print(f"Ошибка при обработке файла {original_filename}: {e}")
                        # Продолжаем обработку других файлов даже при ошибке
        
        else:
            # Обычный JSON запрос
            data = request.get_json(silent=True) or {}
            question = (data.get("message") or "").strip()
            model_mode = data.get("model_mode", "auto")
        
        # Если нет вопроса и нет загруженных файлов
        if not question and not uploaded_files:
            return jsonify({"ok": False, "error": "Пустой запрос"}), 400
        
        # Если загружены только файлы без вопроса, возвращаем успешный ответ
        if uploaded_files and not question:
            return jsonify({
                "ok": True,
                "answer": f"Файлы успешно загружены и добавлены в базу знаний. Теперь вы можете задать вопросы на основе их содержимого.",
                "uploaded_files": uploaded_files,
                "snippets": []
            })
        
        # Валидация режима модели
        if model_mode not in ("auto", "offline", "online"):
            model_mode = "auto"

        store = load_vectorstore()
        if store is None:
            return jsonify({
                "ok": False,
                "error": "База знаний пуста. Загрузите документы на странице загрузки.",
            }), 400

        # Retrieve context with relevance filtering
        # Используем similarity_search_with_score для получения оценок релевантности
        # FAISS использует L2 расстояние, где меньше = лучше (0 = идентично)
        # Получаем больше кандидатов, затем фильтруем по порогу
        max_candidates = 10
        query_for_retrieval = _augment_query_with_memory(question, history)
        docs_with_scores = store.similarity_search_with_score(query_for_retrieval, k=max_candidates)
        
        # Порог релевантности: максимальное допустимое расстояние
        # Для L2 расстояния в FAISS: обычно < 1.0 для хороших совпадений
        # Можно настроить через переменную окружения
        relevance_threshold = float(os.getenv("RELEVANCE_THRESHOLD", "1.5"))
        
        # Фильтруем по релевантности и сортируем
        filtered_docs = [
            (doc, score) for doc, score in docs_with_scores 
            if score <= relevance_threshold
        ]
        
        # Сортируем по score (меньше = лучше) и берем топ-4
        filtered_docs.sort(key=lambda x: x[1])
        filtered_docs = filtered_docs[:4]
        
        docs = [doc for doc, _ in filtered_docs]
        
        # Логирование для отладки
        if filtered_docs:
            print(f"Найдено {len(filtered_docs)} релевантных документов:")
            for i, (doc, score) in enumerate(filtered_docs, 1):
                doc_id = (doc.metadata or {}).get("doc_id", "unknown")
                preview = doc.page_content[:100].replace("\n", " ")
                print(f"  {i}. [doc_id={doc_id}, score={score:.3f}] {preview}...")
        else:
            print(f"Предупреждение: не найдено документов с score <= {relevance_threshold}")
            # Если ничего не найдено, берем топ-2 без фильтрации
            docs = [doc for doc, _ in docs_with_scores[:2]]
            print("Используются топ-2 документа без фильтрации по порогу")
        
        context = "\n\n".join([d.page_content for d in docs]) if docs else ""
        
        # Если после фильтрации контекст пуст, сообщаем об этом
        if not context:
            return jsonify({
                "ok": False,
                "error": "В базе знаний не найдено релевантной информации для вашего запроса. Попробуйте переформулировать вопрос или загрузите соответствующие документы."
            }), 400
        
        # Получаем LLM с учетом выбранного режима
        llm, model_used = get_llm(model_mode)
        
        system_prompt = (
            "Вы — помощник на борту космического корабля. "
            "Отвечайте кратко и по делу, используя ТОЛЬКО информацию из предоставленного контекста.\n\n"
            "ВАЖНО:\n"
            "- Если контекст не содержит информации, относящейся к вопросу, "
            "честно скажите: 'В предоставленном контексте нет информации по этому вопросу.'\n"
            "- Если контекст содержит информацию, но она не полностью отвечает на вопрос, "
            "укажите это и ответьте на основе того, что есть.\n"
            "- НЕ придумывайте информацию, которой нет в контексте.\n\n"
            f"Контекст из базы знаний:\n{context}"
        )
        
        memory_messages = _history_to_messages(history)
        
        if llm:
            try:
                messages = [SystemMessage(content=system_prompt)]
                if memory_messages:
                    messages.extend(memory_messages)
                messages.append(HumanMessage(content=question))
                answer = llm.invoke(messages).content
            except Exception as e:
                print(f"Ошибка при генерации ответа LLM: {e}")
                answer = (
                    f"Ошибка при генерации ответа ({str(e)}). "
                    "Ниже приведены наиболее релевантные фрагменты из базы знаний:\n\n" + context[:1500]
                )
                model_used = None  # Сбрасываем, так как произошла ошибка
        else:
            # Fallback: простой ответ на основе контекста
            if model_mode == "offline":
                answer = (
                    "Оффлайн модели недоступны. "
                    "Вот релевантный контекст из базы знаний по вашему запросу:\n\n" + context[:1500]
                )
            elif model_mode == "online":
                answer = (
                    "Онлайн API недоступен. "
                    "Вот релевантный контекст из базы знаний по вашему запросу:\n\n" + context[:1500]
                )
            else:
                answer = (
                    "LLM модель не настроена или недоступна. "
                    "Вот релевантный контекст из базы знаний по вашему запросу:\n\n" + context[:1500]
                )
            model_used = None

        snippets = [
            {
                "doc_id": (d.metadata or {}).get("doc_id"),
                "preview": d.page_content[:250],
            }
            for d in docs
        ]

        if question:
            _append_history(session_id, "user", question)
            _append_history(session_id, "assistant", answer)
        
        current_history = _get_history(session_id)
        memory_turns = (len(current_history) // 2) if current_history else 0
        
        response_data = {
            "ok": True,
            "answer": answer,
            "snippets": snippets,
            "memory_turns": memory_turns,
        }
        if model_used:
            response_data["model_used"] = model_used
        if uploaded_files:
            response_data["uploaded_files"] = uploaded_files
        
        return jsonify(response_data)

    # Static convenience for uploaded files (only available if upload failed)
    # Note: files are deleted after successful ingestion into knowledge base
    @app.get("/uploads/<path:filename>")
    def uploaded(filename: str):
        return send_from_directory(UPLOAD_DIR, filename)

    return app


if __name__ == "__main__":
    application = create_app()
    port = int(os.getenv("PORT", "8000"))
    application.run(host="0.0.0.0", port=port, debug=True)


