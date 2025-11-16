## Spacecraft Assistant — локальный RAG-ассистент с веб‑интерфейсом

Интерактивный помощник для корабельных операторов с поиском по базе знаний (RAG) и чат‑интерфейсом. Поддерживаются оффлайн‑модели (Ollama, локальные HuggingFace) и онлайн‑модели (OpenAI/Timeweb). Документы (`.pdf`, `.txt`) загружаются через веб‑UI и индексируются в локальном FAISS.

### Возможности

- Веб‑интерфейс: загрузка документов и чат (`Flask`, страницы `index`, `upload`, `chat`)
- Локальное хранение знаний: `FAISS` + `HuggingFace/OpenAI embeddings`
- Интеллектуальный выбор LLM: `auto | offline | online`
- Безопасные имена файлов с сохранением кириллицы, временные файлы удаляются после индексации
- Логирование и аудит в `logs/app.log` и `logs/audit.log`

### Быстрый старт (Windows PowerShell)

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Скопируйте и отредактируйте переменные в config.env (см. раздел Конфигурация)
# Запуск приложения
python app.py
```

Откройте в браузере:
- `http://localhost:8000/` — главная
- `http://localhost:8000/upload` — загрузка документов
- `http://localhost:8000/chat` — чат с ассистентом

### Конфигурация (config.env)

По умолчанию приложение читает переменные из `config.env` (загружается через `python-dotenv`):

```env
# Онлайн LLM (Timeweb/OpenAI-совместимый endpoint)
timeweb_api=...
timeweb_access_id=...
timeweb_openai_url=https://agent.timeweb.cloud/api/v1/cloud-ai/agents/<id>/v1
OPENAI_MODEL=gpt-5-nano

# Оффлайн LLM
OLLAMA_MODEL=gemma3:1b          # либо ваша модель Ollama (напр.llama3.2)
# OLLAMA_BASE_URL=http://localhost:11434
HF_MODEL=                         # локальная HF модель (опционально)

# Embeddings
EMBEDDINGS_PROVIDER=hf           # hf | openai
HF_EMBEDDINGS_MODEL=intfloat/e5-base-v2

# Прочее
PORT=8000
LOG_DIR=logs
LOG_LEVEL=INFO
RELEVANCE_THRESHOLD=1.5          # порог релевантности для FAISS (меньше — лучше)
```

> Внимание: не коммитьте реальные ключи в репозиторий. Храните `config.env` локально.

### Режимы моделей

- `auto` — сначала пытается Ollama, затем локальную HuggingFace, дальше онлайн API
- `offline` — использует только Ollama/HuggingFace
- `online` — использует только Timeweb/OpenAI‑совместимый API

Режим задаётся параметром `model_mode` в `/api/chat` (или через UI).

### Как это работает

1) Загрузка документов (`.pdf`, `.txt`) на странице `/upload` или вместе с сообщением в `/chat`  
2) Разбиение текста на чанки и индексирование в `FAISS` с эмбеддингами (`HF` по умолчанию)  
3) По вопросу пользователя извлекаются релевантные чанки (similarity + порог `RELEVANCE_THRESHOLD`)  
4) LLM генерирует ответ строго по контексту; при отсутствии релевантных данных возвращается понятная ошибка/сообщение

Векторное хранилище и служебные файлы: `rag_store/faiss_index/` (включая `ids.json` для сопоставления `doc_id -> chunk_ids`).

### API (кратко)

- `GET /healthz` — проверка состояния
- `GET /` — главная
- `GET /upload` — форма загрузки
- `POST /api/upload` — загрузка файла (`pdf`/`txt`), индексация, удаление временного файла
- `GET /chat` — чат‑страница
- `POST /api/chat` — сообщение + (опциональные) файлы, режим модели `auto|offline|online`

Ответ `/api/chat` включает текст ответа, список использованных фрагментов и, при наличии, `model_used`.

### Работа оффлайн

1) Установите Ollama: `https://ollama.ai/`, запустите по умолчанию на `http://localhost:11434`  
2) Загрузите модель, например:  
```powershell
ollama pull gemma3:1b
```
3) Убедитесь, что `OLLAMA_MODEL` задан в `config.env` (или используйте `HF_MODEL` для локальной HuggingFace)

### Полезные скрипты и файлы

- `app.py` — веб‑приложение (Flask, UI + API)
- `vector_base.py` — работа с FAISS/эмбеддингами, обновление/перестройка индекса
- `ingest.py` — пример оффлайн‑индексации из файлов (опционально)
- `requirements.txt` — зависимости
- `templates/`, `static/` — фронтенд
- `logs/` — логи приложения и аудита

### Требования

- Python 3.10+
- Windows/macOS/Linux; для оффлайн LLM потребуется достаточный объём RAM/VRAM 

### Частые вопросы

- Ничего не найдено в базе знаний  
  Проверьте, что вы загрузили документы и порог `RELEVANCE_THRESHOLD` не слишком строгий.

- Ошибка онлайн‑модели (см. logs) 
  Проверьте `timeweb_api`, `timeweb_openai_url`, `OPENAI_MODEL`. Временно можно работать в режиме `offline`.
  Проверьте, что ваш env файл называется config.env

- Производительность на CPU (см. logs) 
  Для локальных HF‑моделей используйте меньшие модели или GPU, либо Ollama‑модель полегче.

---

Если нужен полностью оффлайн режим и воспроизводимая установка — смотрите `OFFLINE_SETUP.md`. Также есть `AGENT_README.md` по функциональности агентов.
