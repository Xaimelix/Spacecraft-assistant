"""
Инструменты (Tools) для ИИ агента космического корабля.
Содержит функции для получения данных с датчиков и управления системами.
"""

import random
import time
from datetime import datetime
from typing import Dict, List, Optional
from langchain_core.tools import tool


# Симуляция состояния систем корабля
_system_state = {
    "temperature": {
        "engine": 850.0,
        "cabin": 22.5,
        "life_support": 20.0,
        "electronics": 45.0,
    },
    "pressure": {
        "cabin": 101.3,  # кПа
        "fuel_tank": 500.0,
        "life_support": 101.3,
    },
    "oxygen_level": 21.0,  # процент
    "fuel_level": 75.5,  # процент
    "power_consumption": {
        "total": 8500.0,  # Вт
        "engine": 5000.0,
        "life_support": 2000.0,
        "electronics": 1500.0,
    },
    "radiation_level": 0.15,  # мЗв/ч
    "position": {
        "x": 384400000.0,  # метры (примерно расстояние до Луны)
        "y": 0.0,
        "z": 0.0,
    },
    "velocity": 1000.0,  # м/с
    "orientation": {
        "roll": 0.5,  # градусы
        "pitch": -0.2,
        "yaw": 0.0,
    },
    "systems_status": {
        "engine": "active",
        "life_support": "active",
        "navigation": "active",
        "communication": "active",
    },
}


def _update_system_state():
    """Обновляет состояние систем с небольшими случайными изменениями."""
    # Небольшие случайные изменения для реалистичности
    _system_state["temperature"]["engine"] += random.uniform(-5, 5)
    _system_state["temperature"]["cabin"] += random.uniform(-0.5, 0.5)
    _system_state["fuel_level"] -= random.uniform(0, 0.1)
    _system_state["power_consumption"]["total"] += random.uniform(-50, 50)


@tool
def get_temperature(system: str = "all") -> str:
    """
    Получает температуру указанной системы или всех систем.
    
    Args:
        system: Название системы ('engine', 'cabin', 'life_support', 'electronics', 'all')
    
    Returns:
        Строка с информацией о температуре
    """
    _update_system_state()
    
    if system == "all":
        temps = []
        for sys_name, temp in _system_state["temperature"].items():
            temps.append(f"{sys_name}: {temp:.1f}°C")
        return f"Температура систем:\n" + "\n".join(temps)
    elif system in _system_state["temperature"]:
        temp = _system_state["temperature"][system]
        return f"Температура системы '{system}': {temp:.1f}°C"
    else:
        return f"Система '{system}' не найдена. Доступные системы: {', '.join(_system_state['temperature'].keys())}"


@tool
def get_pressure(compartment: str = "cabin") -> str:
    """
    Получает давление в указанном отсеке.
    
    Args:
        compartment: Название отсека ('cabin', 'fuel_tank', 'life_support')
    
    Returns:
        Строка с информацией о давлении
    """
    _update_system_state()
    
    if compartment in _system_state["pressure"]:
        pressure = _system_state["pressure"][compartment]
        return f"Давление в отсеке '{compartment}': {pressure:.2f} кПа"
    else:
        return f"Отсек '{compartment}' не найден. Доступные отсеки: {', '.join(_system_state['pressure'].keys())}"


@tool
def get_oxygen_level() -> str:
    """
    Получает уровень кислорода в атмосфере корабля.
    
    Returns:
        Строка с информацией об уровне кислорода
    """
    _update_system_state()
    level = _system_state["oxygen_level"]
    status = "нормальный" if 19.5 <= level <= 23.5 else "критический"
    return f"Уровень кислорода: {level:.1f}% ({status})"


@tool
def get_fuel_level() -> str:
    """
    Получает уровень топлива.
    
    Returns:
        Строка с информацией об уровне топлива
    """
    _update_system_state()
    level = _system_state["fuel_level"]
    status = "нормальный" if level > 20 else "низкий" if level > 10 else "критический"
    return f"Уровень топлива: {level:.1f}% ({status})"


@tool
def get_power_consumption(system: str = "total") -> str:
    """
    Получает энергопотребление указанной системы или общее.
    
    Args:
        system: Название системы ('total', 'engine', 'life_support', 'electronics')
    
    Returns:
        Строка с информацией об энергопотреблении
    """
    _update_system_state()
    
    if system == "total":
        power = _system_state["power_consumption"]["total"]
        return f"Общее энергопотребление: {power:.1f} Вт"
    elif system in _system_state["power_consumption"]:
        power = _system_state["power_consumption"][system]
        return f"Энергопотребление системы '{system}': {power:.1f} Вт"
    else:
        return f"Система '{system}' не найдена. Доступные системы: {', '.join(_system_state['power_consumption'].keys())}"


@tool
def get_radiation_level() -> str:
    """
    Получает текущий уровень радиации.
    
    Returns:
        Строка с информацией об уровне радиации
    """
    _update_system_state()
    level = _system_state["radiation_level"]
    status = "нормальный" if level < 0.5 else "повышенный" if level < 1.0 else "опасный"
    return f"Уровень радиации: {level:.2f} мЗв/ч ({status})"


@tool
def get_position() -> str:
    """
    Получает текущую позицию корабля в пространстве.
    
    Returns:
        Строка с координатами позиции
    """
    _update_system_state()
    pos = _system_state["position"]
    return f"Позиция корабля:\nX: {pos['x']:.0f} м\nY: {pos['y']:.0f} м\nZ: {pos['z']:.0f} м"


@tool
def get_velocity() -> str:
    """
    Получает текущую скорость корабля.
    
    Returns:
        Строка с информацией о скорости
    """
    _update_system_state()
    velocity = _system_state["velocity"]
    return f"Текущая скорость: {velocity:.1f} м/с ({velocity * 3.6:.1f} км/ч)"


@tool
def get_orientation() -> str:
    """
    Получает ориентацию корабля (крен, тангаж, рыскание).
    
    Returns:
        Строка с информацией об ориентации
    """
    _update_system_state()
    orient = _system_state["orientation"]
    return f"Ориентация корабля:\nКрен (roll): {orient['roll']:.2f}°\nТангаж (pitch): {orient['pitch']:.2f}°\nРыскание (yaw): {orient['yaw']:.2f}°"


@tool
def get_system_status(system: str = "all") -> str:
    """
    Получает статус указанной системы или всех систем.
    
    Args:
        system: Название системы ('engine', 'life_support', 'navigation', 'communication', 'all')
    
    Returns:
        Строка с информацией о статусе системы
    """
    _update_system_state()
    
    if system == "all":
        statuses = []
        for sys_name, status in _system_state["systems_status"].items():
            statuses.append(f"{sys_name}: {status}")
        return "Статус систем:\n" + "\n".join(statuses)
    elif system in _system_state["systems_status"]:
        status = _system_state["systems_status"][system]
        return f"Статус системы '{system}': {status}"
    else:
        return f"Система '{system}' не найдена. Доступные системы: {', '.join(_system_state['systems_status'].keys())}"


@tool
def get_all_sensor_data() -> str:
    """
    Получает все данные с датчиков одновременно.
    
    Returns:
        Строка с полной информацией о состоянии корабля
    """
    _update_system_state()
    
    result = []
    result.append("=== СОСТОЯНИЕ КОСМИЧЕСКОГО КОРАБЛЯ ===\n")
    result.append(f"Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    result.append("\n--- Температура ---")
    for sys, temp in _system_state["temperature"].items():
        result.append(f"  {sys}: {temp:.1f}°C")
    
    result.append("\n--- Давление ---")
    for comp, press in _system_state["pressure"].items():
        result.append(f"  {comp}: {press:.2f} кПа")
    
    result.append(f"\n--- Атмосфера ---")
    result.append(f"  Кислород: {_system_state['oxygen_level']:.1f}%")
    
    result.append(f"\n--- Топливо ---")
    result.append(f"  Уровень: {_system_state['fuel_level']:.1f}%")
    
    result.append("\n--- Энергопотребление ---")
    for sys, power in _system_state["power_consumption"].items():
        result.append(f"  {sys}: {power:.1f} Вт")
    
    result.append(f"\n--- Радиация ---")
    result.append(f"  Уровень: {_system_state['radiation_level']:.2f} мЗв/ч")
    
    result.append("\n--- Навигация ---")
    pos = _system_state["position"]
    result.append(f"  Позиция: X={pos['x']:.0f}, Y={pos['y']:.0f}, Z={pos['z']:.0f} м")
    result.append(f"  Скорость: {_system_state['velocity']:.1f} м/с")
    orient = _system_state["orientation"]
    result.append(f"  Ориентация: roll={orient['roll']:.2f}°, pitch={orient['pitch']:.2f}°, yaw={orient['yaw']:.2f}°")
    
    result.append("\n--- Статус систем ---")
    for sys, status in _system_state["systems_status"].items():
        result.append(f"  {sys}: {status}")
    
    return "\n".join(result)


# Список всех доступных инструментов
ALL_TOOLS = [
    get_temperature,
    get_pressure,
    get_oxygen_level,
    get_fuel_level,
    get_power_consumption,
    get_radiation_level,
    get_position,
    get_velocity,
    get_orientation,
    get_system_status,
    get_all_sensor_data,
]

