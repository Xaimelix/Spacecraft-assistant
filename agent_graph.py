"""
Граф агента для космического корабля.
Использует LangGraph для создания агента, который может обращаться к инструментам и RAG системе.
"""

import os
from typing import Literal, TypedDict, Annotated
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages

from sensor_tools import ALL_TOOLS
from vector_base import load_vectorstore
from app import get_llm

# Загружаем переменные окружения
from dotenv import load_dotenv
load_dotenv("api_key.env")


class AgentState(TypedDict):
    """Состояние агента в графе."""
    messages: Annotated[list, add_messages]
    rag_context: str  # Контекст из RAG системы


def router(state: AgentState) -> Literal["rag", "tools", "direct"]:
    """
    Определяет, нужно ли использовать RAG, инструменты или можно ответить напрямую.
    """
    last_message = state["messages"][-1]
    if not isinstance(last_message, HumanMessage):
        return "direct"
    
    question = last_message.content.lower()
    
    # Ключевые слова для RAG (поиск в документации)
    rag_keywords = [
        "процедура", "инструкция", "документация", "как", "что такое",
        "описание", "руководство", "метод", "способ", "алгоритм"
    ]
    
    # Ключевые слова для инструментов (сенсоры)
    tool_keywords = [
        "температура", "давление", "кислород", "топливо", "энергия",
        "радиация", "позиция", "скорость", "ориентация", "статус",
        "проверь", "покажи", "получи", "мониторинг", "состояние"
    ]
    
    has_rag = any(keyword in question for keyword in rag_keywords)
    has_tools = any(keyword in question for keyword in tool_keywords)
    
    # Если есть оба типа ключевых слов, используем оба
    if has_rag and has_tools:
        return "tools"  # Сначала инструменты, потом можно добавить RAG
    elif has_tools:
        return "tools"
    elif has_rag:
        return "rag"
    else:
        return "direct"


def rag_search(state: AgentState) -> AgentState:
    """
    Выполняет поиск в RAG системе (векторной базе знаний).
    """
    last_message = state["messages"][-1]
    if not isinstance(last_message, HumanMessage):
        return state
    
    question = last_message.content
    store = load_vectorstore()
    
    if store is None:
        state["rag_context"] = "База знаний пуста. Загрузите документы для использования RAG."
        return state
    
    # Поиск релевантных документов
    docs = store.similarity_search(question, k=4)
    context = "\n\n".join([d.page_content for d in docs])
    
    state["rag_context"] = context if context else "Релевантная информация не найдена в базе знаний."
    return state


def call_llm_with_tools(state: AgentState) -> AgentState:
    """
    Вызывает LLM с инструментами для определения, какие инструменты нужно использовать.
    """
    llm, _ = get_llm()
    if llm is None:
        state["messages"].append(AIMessage(content="LLM модель недоступна."))
        return state
    
    # Связываем инструменты с LLM
    llm_with_tools = llm.bind_tools(ALL_TOOLS)
    
    # Получаем последнее сообщение пользователя
    last_message = state["messages"][-1]
    
    # Формируем системный промпт
    system_prompt = (
        "Вы — ИИ помощник на борту космического корабля. "
        "У вас есть доступ к инструментам для получения данных с датчиков. "
        "Используйте инструменты для ответа на вопросы о состоянии систем корабля. "
        "Если вопрос требует данных с датчиков, обязательно используйте соответствующие инструменты."
    )
    
    # Добавляем RAG контекст, если он есть
    if state.get("rag_context"):
        system_prompt += f"\n\nДополнительная информация из документации:\n{state['rag_context']}"
    
    messages = [
        SystemMessage(content=system_prompt),
        last_message,
    ]
    
    # Вызываем LLM
    response = llm_with_tools.invoke(messages)
    state["messages"].append(response)
    
    return state


def should_continue(state: AgentState) -> Literal["continue", "end"]:
    """
    Определяет, нужно ли продолжать выполнение (есть ли вызовы инструментов).
    """
    last_message = state["messages"][-1]
    # Если последнее сообщение содержит вызовы инструментов, продолжаем
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "continue"
    return "end"


def after_rag_route(state: AgentState) -> Literal["call_llm_with_tools", "generate_final_response"]:
    """
    Определяет следующий шаг после RAG поиска.
    """
    # Проверяем, нужны ли инструменты после RAG поиска
    last_message = state["messages"][-1]
    if isinstance(last_message, HumanMessage):
        question = last_message.content.lower()
        tool_keywords = [
            "температура", "давление", "кислород", "топливо", "энергия",
            "радиация", "позиция", "скорость", "ориентация", "статус",
            "проверь", "покажи", "получи", "мониторинг", "состояние"
        ]
        if any(kw in question for kw in tool_keywords):
            return "call_llm_with_tools"
    return "generate_final_response"


def generate_final_response(state: AgentState) -> AgentState:
    """
    Генерирует финальный ответ на основе всех собранных данных.
    """
    llm, _ = get_llm()
    if llm is None:
        # Fallback: собираем результаты из сообщений
        results = []
        for msg in state["messages"]:
            if isinstance(msg, ToolMessage):
                results.append(msg.content)
        
        if results:
            response = "Результаты:\n" + "\n".join(results)
        else:
            response = "LLM модель недоступна."
        
        state["messages"].append(AIMessage(content=response))
        return state
    
    # Формируем системный промпт
    system_prompt = (
        "Вы — ИИ помощник на борту космического корабля. "
        "Отвечайте на вопросы экипажа, используя предоставленную информацию из инструментов и документации. "
        "Отвечайте кратко и по делу. Предупреждайте о критических значениях параметров."
    )
    
    # Добавляем RAG контекст, если он есть
    if state.get("rag_context"):
        system_prompt += f"\n\nДополнительная информация из документации:\n{state['rag_context']}"
    
    # Собираем все сообщения для контекста
    messages = [SystemMessage(content=system_prompt)]
    messages.extend(state["messages"])
    
    try:
        response = llm.invoke(messages)
        state["messages"].append(response)
    except Exception as e:
        state["messages"].append(AIMessage(content=f"Ошибка при генерации ответа: {str(e)}"))
    
    return state




def create_agent_graph():
    """
    Создает граф агента с использованием LangGraph.
    """
    # Создаем ToolNode для выполнения инструментов
    tool_node = ToolNode(ALL_TOOLS)
    
    # Создаем граф
    workflow = StateGraph(AgentState)
    
    # Добавляем узлы
    workflow.add_node("router", router)  # Роутер определяет путь
    workflow.add_node("rag_search", rag_search)
    workflow.add_node("call_llm_with_tools", call_llm_with_tools)
    workflow.add_node("tool_executor", tool_node)
    workflow.add_node("generate_final_response", generate_final_response)
    
    # Устанавливаем начальную точку
    workflow.set_entry_point("router")
    
    # Добавляем условные переходы от роутера
    workflow.add_conditional_edges(
        "router",
        router,
        {
            "rag": "rag_search",
            "tools": "call_llm_with_tools",
            "direct": "generate_final_response",
        }
    )
    
    # После RAG поиска переходим к вызову LLM с инструментами или финальному ответу
    workflow.add_conditional_edges(
        "rag_search",
        after_rag_route,
        {
            "call_llm_with_tools": "call_llm_with_tools",
            "generate_final_response": "generate_final_response",
        }
    )
    
    # После вызова LLM с инструментами проверяем, нужно ли выполнять инструменты
    workflow.add_conditional_edges(
        "call_llm_with_tools",
        should_continue,
        {
            "continue": "tool_executor",
            "end": "generate_final_response",
        }
    )
    
    # После выполнения инструментов возвращаемся к LLM для финального ответа
    workflow.add_edge("tool_executor", "generate_final_response")
    
    # Генерация финального ответа - финальный шаг
    workflow.add_edge("generate_final_response", END)
    
    return workflow.compile()


def run_agent(question: str, model_mode: str = "auto") -> str:
    """
    Запускает агента с заданным вопросом.
    
    Args:
        question: Вопрос пользователя
        model_mode: Режим модели ('auto', 'offline', 'online')
    
    Returns:
        Ответ агента
    """
    # Временно устанавливаем режим модели (если нужно)
    # Для простоты используем текущую настройку из app.py
    
    # Создаем граф
    graph = create_agent_graph()
    
    # Инициализируем состояние
    initial_state = {
        "messages": [HumanMessage(content=question)],
        "rag_context": "",
    }
    
    # Запускаем граф
    try:
        result = graph.invoke(initial_state)
        
        # Извлекаем последний ответ
        last_message = result["messages"][-1]
        if isinstance(last_message, AIMessage):
            return last_message.content
        else:
            return str(last_message)
    except Exception as e:
        return f"Ошибка при выполнении агента: {str(e)}"


if __name__ == "__main__":
    # Примеры использования
    print("=== Тест 1: Запрос температуры ===")
    response1 = run_agent("Какая температура двигателя?")
    print(response1)
    print("\n")
    
    print("=== Тест 2: Полная проверка систем ===")
    response2 = run_agent("Проверь все системы корабля")
    print(response2)
    print("\n")
    
    print("=== Тест 3: Поиск в документации ===")
    response3 = run_agent("Как выполнить аварийное отключение?")
    print(response3)
    print("\n")
    
    print("=== Тест 4: Комбинированный запрос ===")
    response4 = run_agent("Проверь температуру и найди в документации информацию о нормальных значениях")
    print(response4)

