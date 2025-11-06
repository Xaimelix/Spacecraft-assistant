# ИИ Агент для Космического Корабля

## Описание

Реализация ИИ агента с использованием LangGraph, который может:
- Получать данные с датчиков корабля (температура, давление, топливо и т.д.)
- Искать информацию в документации через RAG систему
- Комбинировать данные из разных источников для ответов

## Файлы

### 1. `agent_functionality.md`
Набросок возможного функционала ИИ агента с описанием всех категорий задач, которые может выполнять агент.

### 2. `sensor_tools.py`
Тестовые функции (инструменты) для получения данных с датчиков:
- `get_temperature()` - температура систем
- `get_pressure()` - давление в отсеках
- `get_oxygen_level()` - уровень кислорода
- `get_fuel_level()` - уровень топлива
- `get_power_consumption()` - энергопотребление
- `get_radiation_level()` - уровень радиации
- `get_position()` - позиция корабля
- `get_velocity()` - скорость
- `get_orientation()` - ориентация
- `get_system_status()` - статус систем
- `get_all_sensor_data()` - все данные сразу

### 3. `agent_graph.py`
Граф агента на LangGraph, который:
- Анализирует запрос пользователя
- Определяет, нужны ли инструменты или RAG поиск
- Выполняет соответствующие действия
- Генерирует финальный ответ

## Использование

### Базовое использование

```python
from agent_graph import run_agent

# Запрос температуры
response = run_agent("Какая температура двигателя?")
print(response)

# Полная проверка систем
response = run_agent("Проверь все системы корабля")
print(response)

# Поиск в документации
response = run_agent("Как выполнить аварийное отключение?")
print(response)
```

### Прямое использование графа

```python
from agent_graph import create_agent_graph
from langchain_core.messages import HumanMessage

graph = create_agent_graph()
initial_state = {
    "messages": [HumanMessage(content="Проверь температуру и уровень топлива")],
    "rag_context": "",
}

result = graph.invoke(initial_state)
print(result["messages"][-1].content)
```

## Установка зависимостей

```bash
pip install -r requirements.txt
```

Убедитесь, что установлен `langgraph>=0.2.0`.

## Архитектура графа

```
[Router] 
    ├─> [RAG Search] ─> [Generate Response]
    ├─> [Call LLM with Tools] ─> [Tool Executor] ─> [Generate Response]
    └─> [Generate Response]
```

1. **Router** - определяет тип запроса (RAG, инструменты, или прямой ответ)
2. **RAG Search** - поиск в векторной базе знаний
3. **Call LLM with Tools** - LLM решает, какие инструменты использовать
4. **Tool Executor** - выполняет выбранные инструменты
5. **Generate Response** - генерирует финальный ответ на основе всех данных

## Интеграция с существующей системой

Агент использует:
- `get_llm()` из `app.py` для получения LLM модели
- `load_vectorstore()` из `vector_base.py` для RAG поиска
- Инструменты из `sensor_tools.py` для получения данных с датчиков

## Примеры запросов

- "Какая температура двигателя?"
- "Проверь все системы корабля"
- "Как выполнить аварийное отключение?"
- "Покажи температуру и найди в документации нормальные значения"
- "Какое давление в кабине и уровень кислорода?"

