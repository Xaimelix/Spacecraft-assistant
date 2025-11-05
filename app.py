import os
import re
import uuid
from datetime import datetime
from typing import Optional

from flask import Flask, jsonify, redirect, render_template, request, send_from_directory, url_for

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from vector_base import (
    update_knowledge_base,
    create_document,
    load_vectorstore,
)

from dotenv import load_dotenv


UPLOAD_DIR = os.path.join(os.getcwd(), "uploads")
ALLOWED_EXTENSIONS = {"pdf", "txt"}
load_dotenv("api_key.env")


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

    @app.post("/api/chat")
    def api_chat():
        data = request.get_json(silent=True) or {}
        question = (data.get("message") or "").strip()
        if not question:
            return jsonify({"ok": False, "error": "Пустой запрос"}), 400

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
        
        # Use OpenAI if configured; otherwise, return heuristic answer
        api_key = os.getenv("timeweb_api")
        if api_key:
            try:
                llm = ChatOpenAI(
                    base_url=os.getenv("timeweb_openai_url"),
                    api_key=api_key,
                    model=os.getenv("OPENAI_MODEL", "gpt-5-nano"),
                    temperature=0
                )
                messages = [
                    SystemMessage(
                        content=(
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
                    ),
                    HumanMessage(content=question),
                ]
                answer = llm.invoke(messages).content
            except Exception as e:
                answer = (
                    "LLM недоступен (" + str(e) + "). "
                    "Ниже приведены наиболее релевантные фрагменты из базы знаний.\n\n" + context[:1500]
                )
        else:
            answer = (
                "LLM не настроен (нет переменной timeweb_api). "
                "Вот релевантный контекст из базы знаний:\n\n" + context[:1500]
            )

        snippets = [
            {
                "doc_id": (d.metadata or {}).get("doc_id"),
                "preview": d.page_content[:250],
            }
            for d in docs
        ]

        return jsonify({"ok": True, "answer": answer, "snippets": snippets})

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


