import os
import re
import uuid
from datetime import datetime
from typing import Optional
import logging
from logging.handlers import RotatingFileHandler

from flask import Flask, jsonify, redirect, render_template, request, send_from_directory, url_for

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

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
load_dotenv("config.env")


# Кэш для LLM модели (чтобы не пересоздавать каждый раз)
_llm_cache = None
_model_info_cache = None  # Информация о модели в кэше


def setup_logging() -> None:
    """
    Инициализация логирования приложения и аудита.
    - Основной лог: logs/app.log
    - Аудит-лог: logs/audit.log
    Управляется переменными окружения: LOG_DIR, LOG_LEVEL.
    """
    log_dir = os.getenv("LOG_DIR", os.path.join(os.getcwd(), "logs"))
    os.makedirs(log_dir, exist_ok=True)

    # Уровень логирования
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    # Форматы
    app_format = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    audit_format = logging.Formatter(
        "%(asctime)s | %(levelname)s | AUDIT | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Основной файл логов
    app_log_path = os.path.join(log_dir, "app.log")
    app_handler = RotatingFileHandler(app_log_path, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    app_handler.setFormatter(app_format)
    app_handler.setLevel(level)

    # Аудит файл логов
    audit_log_path = os.path.join(log_dir, "audit.log")
    audit_handler = RotatingFileHandler(audit_log_path, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    audit_handler.setFormatter(audit_format)
    audit_handler.setLevel(logging.INFO)

    # Конфигурация корневого логгера (для app и библиотек)
    root_logger = logging.getLogger()
    # Избегаем дублирования хендлеров при повторной инициализации
    if not any(isinstance(h, RotatingFileHandler) for h in root_logger.handlers):
        root_logger.setLevel(level)
        root_logger.addHandler(app_handler)

    # Отдельный логгер для аудита
    audit_logger = logging.getLogger("audit")
    if not any(isinstance(h, RotatingFileHandler) for h in audit_logger.handlers):
        audit_logger.setLevel(logging.INFO)
        audit_logger.addHandler(audit_handler)


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
    
    logger = logging.getLogger(__name__)

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
                    logger.info(f"Используется Ollama модель: {ollama_model}")
                    if model_mode == "auto":
                        _llm_cache = llm
                        _model_info_cache = model_used
                    return llm, model_used
            except Exception:
                pass  # Ollama не отвечает, пробуем дальше
        except Exception as e:
            logger.warning(f"Ollama недоступен: {e}")
    
    # 2. Попытка использовать локальную HuggingFace модель (оффлайн)
    hf_model = os.getenv("HF_MODEL")
    if model_mode in ("auto", "offline") and hf_model and HuggingFacePipeline is not None:
        try:
            from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
            import torch
            
            logger.info(f"Загрузка локальной модели HuggingFace: {hf_model}...")
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
            logger.info(f"Используется локальная HuggingFace модель: {hf_model}")
            if model_mode == "auto":
                _llm_cache = llm
                _model_info_cache = model_used
            return llm, model_used
        except Exception as e:
            logger.warning(f"Локальная HuggingFace модель недоступна: {e}")
    
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
            logger.info("Используется OpenAI/Timeweb API (онлайн)")
            if model_mode == "auto":
                _llm_cache = llm
                _model_info_cache = model_used
            return llm, model_used
        except Exception as e:
            logger.warning(f"OpenAI/Timeweb недоступен: {e}")
    
    # 4. Если ничего не доступно, возвращаем None
    if model_mode == "offline":
        logger.warning("Предупреждение: оффлайн модели недоступны.")
    elif model_mode == "online":
        logger.warning("Предупреждение: онлайн API недоступен.")
    else:
        logger.warning("Предупреждение: ни одна LLM модель не доступна. Будет использован простой ответ на основе контекста.")
    
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


def create_app() -> Flask:
    app = Flask(__name__, static_folder="static", template_folder="templates")
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    setup_logging()
    logger = logging.getLogger(__name__)
    audit_logger = logging.getLogger("audit")

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
        request_id = str(uuid.uuid4())
        client_ip = request.headers.get("X-Forwarded-For", request.remote_addr)

        if "file" not in request.files:
            logging.getLogger(__name__).warning(f"[{request_id}] Загрузка: отсутствует файл | ip={client_ip}")
            return jsonify({"ok": False, "error": "Нет файла в запросе"}), 400
        file = request.files["file"]
        if file.filename == "":
            logger.warning(f"[{request_id}] Загрузка: пустое имя файла | ip={client_ip}")
            return jsonify({"ok": False, "error": "Имя файла пустое"}), 400
        if not allowed_file(file.filename):
            logger.warning(f"[{request_id}] Загрузка: недопустимое расширение '{file.filename}' | ip={client_ip}")
            return jsonify({"ok": False, "error": "Разрешены только .pdf и .txt"}), 400
        
        # Используем безопасную функцию, сохраняющую кириллические символы
        original_filename = file.filename
        filename = safe_filename(original_filename)
        save_path = os.path.join(UPLOAD_DIR, filename)
        file.save(save_path)

        # Ingest into vector store
        logger.info(f"[{request_id}] Загрузка файла: original='{original_filename}', saved_as='{filename}', ip={client_ip}")
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
                logger.warning(f"[{request_id}] Не удалось удалить файл {save_path}: {cleanup_error}")
            
            audit_logger.info(f"[{request_id}] UPLOAD ok | doc_id='{logical_doc_id}' | file='{original_filename}' | ip={client_ip}")
            return jsonify({"ok": True, "doc_id": logical_doc_id, "filename": filename})
        except Exception as e:
            # При ошибке оставляем файл для возможной отладки
            logger.exception(f"[{request_id}] Ошибка обработки загрузки файла '{original_filename}': {e}")
            audit_logger.info(f"[{request_id}] UPLOAD fail | file='{original_filename}' | error='{str(e)}' | ip={client_ip}")
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.get("/chat")
    def chat_page():
        return render_template("chat.html")

    @app.post("/api/chat")
    def api_chat():
        request_id = str(uuid.uuid4())
        client_ip = request.headers.get("X-Forwarded-For", request.remote_addr)
        # Поддержка как JSON, так и FormData (для загрузки файлов)
        uploaded_files = []
        question = ""
        model_mode = "auto"
        
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
                            logging.getLogger(__name__).warning(f"[{request_id}] Не удалось удалить временный файл {save_path}: {cleanup_error}")
                            
                    except Exception as e:
                        logging.getLogger(__name__).error(f"[{request_id}] Ошибка при обработке файла {original_filename}: {e}")
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
            audit_logger.info(f"[{request_id}] CHAT files_only | files={len(uploaded_files)} | ip={client_ip}")
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
            logging.getLogger(__name__).warning(f"[{request_id}] Запрос без базы знаний | ip={client_ip}")
            return jsonify({
                "ok": False,
                "error": "База знаний пуста. Загрузите документы на странице загрузки.",
            }), 400

        # Retrieve context with relevance filtering
        # Используем similarity_search_with_score для получения оценок релевантности
        # FAISS использует L2 расстояние, где меньше = лучше (0 = идентично)
        # Получаем больше кандидатов, затем фильтруем по порогу
        max_candidates = 10
        docs_with_scores = store.similarity_search_with_score(question, k=max_candidates)
        
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
            logger.info(f"[{request_id}] Найдено {len(filtered_docs)} релевантных документов (threshold={relevance_threshold})")
        else:
            logger.warning(f"[{request_id}] Не найдено документов с score <= {relevance_threshold}")
            # Если ничего не найдено, берем топ-2 без фильтрации
            docs = [doc for doc, _ in docs_with_scores[:2]]
            logger.info(f"[{request_id}] Используются топ-2 документа без фильтрации по порогу")
        
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
        
        if llm:
            try:
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=question),
                ]
                answer = llm.invoke(messages).content
            except Exception as e:
                logger.exception(f"[{request_id}] Ошибка при генерации ответа LLM: {e}")
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

        response_data = {"ok": True, "answer": answer, "snippets": snippets}
        if model_used:
            response_data["model_used"] = model_used
        if uploaded_files:
            response_data["uploaded_files"] = uploaded_files
        
        # Аудит запроса
        audit_logger.info(
            f"[{request_id}] CHAT ok | q_len={len(question)} | files={len(uploaded_files)} | model={model_used or 'fallback'} | ip={client_ip}"
        )
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
    # Debug отключаем, чтобы избежать двойной инициализации логгеров в reloader
    application.run(host="0.0.0.0", port=port, debug=False)


