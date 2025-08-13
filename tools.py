import json

from langchain.tools import tool

@tool
def get_weather(city: str, unit: str = "c") -> str:
    """Get the weather for a city. unit is 'c' for Celsius or 'f' for Fahrenheit."""
    data = {"zurich": {"c": 15, "f": 59}, "paris": {"c": 20, "f": 68}, "berlin": {"c": 10, "f": 50}}
    city_key = city.lower()
    temp = data.get(city_key, {"c": 0, "f": 0})[unit]
    return json.dumps({"city": city, "temperature": temp, "unit": unit})

@tool
def calculator(expression: str) -> float:
    """Rechnet einfache mathematische Ausdrücke aus."""
    return eval(expression)

@tool
def reverse_string(input_string: str) -> str:
    """Kehrt die Zeichenkette um."""
    return input_string[::-1]
