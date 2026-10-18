"""Agent météo autonome : instructions, outil Open-Meteo et graphe LangGraph."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import before_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, RemoveMessage
from langchain_core.messages.utils import trim_messages
from langchain_core.tools import BaseTool, tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field, field_validator


class WeatherQuery(BaseModel):
    """Entrée contrôlée de l'outil météo."""

    city: str = Field(
        min_length=1,
        description="Nom de la ville, par exemple Paris ou Montréal.",
    )

    @field_validator("city")
    @classmethod
    def city_not_blank(cls, value: str) -> str:
        """Refuse les noms vides ou composés uniquement d'espaces."""
        city = value.strip()
        if not city:
            raise ValueError("Le nom de la ville ne peut pas être vide.")
        return city


_WEATHER_CODES = {
    0: "ciel dégagé",
    1: "principalement dégagé",
    2: "partiellement nuageux",
    3: "couvert",
    45: "brouillard",
    48: "brouillard givrant",
    51: "bruine légère",
    53: "bruine modérée",
    55: "bruine dense",
    56: "bruine verglaçante légère",
    57: "bruine verglaçante dense",
    61: "pluie faible",
    63: "pluie modérée",
    65: "pluie forte",
    66: "pluie verglaçante faible",
    67: "pluie verglaçante forte",
    71: "neige faible",
    73: "neige modérée",
    75: "neige forte",
    77: "grains de neige",
    80: "averses faibles",
    81: "averses modérées",
    82: "averses violentes",
    85: "averses de neige faibles",
    86: "averses de neige fortes",
    95: "orage",
    96: "orage avec grêle faible",
    99: "orage avec grêle forte",
}


def make_weather_tool(timeout_seconds: int) -> BaseTool:
    """Construit l'outil météo avec le délai réseau fourni par la configuration."""

    def get_json(url: str, params: dict[str, str]) -> dict[str, Any]:
        """Appelle un endpoint Open-Meteo et valide le type de la réponse."""
        request = Request(
            f"{url}?{urlencode(params)}",
            headers={"Accept": "application/json", "User-Agent": "microchat/0.1"},
        )
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                result = json.load(response)
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Le service météo est indisponible : {exc}") from exc
        if not isinstance(result, dict):
            raise RuntimeError("Le service météo a renvoyé une réponse invalide.")
        return result

    @tool(args_schema=WeatherQuery)
    def get_weather(city: str) -> str:
        """Donne la météo actuelle d'une ville grâce au service Open-Meteo."""
        location_result = get_json(
            "https://geocoding-api.open-meteo.com/v1/search",
            {"name": city, "count": "1", "language": "fr", "format": "json"},
        )
        locations = location_result.get("results")
        if not isinstance(locations, list) or not locations:
            raise RuntimeError(f"Aucune ville trouvée pour « {city} ».")
        location = locations[0]
        forecast = get_json(
            "https://api.open-meteo.com/v1/forecast",
            {
                "latitude": str(location["latitude"]),
                "longitude": str(location["longitude"]),
                "current": (
                    "temperature_2m,relative_humidity_2m,apparent_temperature,"
                    "precipitation,weather_code,wind_speed_10m"
                ),
                "timezone": "auto",
            },
        )
        current = forecast.get("current")
        if not isinstance(current, dict):
            raise RuntimeError("Le service météo n'a pas renvoyé les conditions actuelles.")

        code = current.get("weather_code")
        condition = (
            _WEATHER_CODES.get(code, "conditions inconnues")
            if isinstance(code, int)
            else ("conditions inconnues")
        )
        place = ", ".join(
            str(part)
            for part in (
                location.get("name", city),
                location.get("admin1"),
                location.get("country"),
            )
            if part
        )
        return (
            f"Météo actuelle à {place} ({current.get('time', 'heure locale inconnue')}) : "
            f"{condition}, {current['temperature_2m']} °C "
            f"(ressenti {current['apparent_temperature']} °C), "
            f"humidité {current['relative_humidity_2m']} %, "
            f"précipitations {current['precipitation']} mm, "
            f"vent {current['wind_speed_10m']} km/h."
        )

    return get_weather


def build_agent(
    llm: BaseChatModel,
    *,
    system_prompt: str,
    instructions: str,
    token_counter: Callable[[Sequence[BaseMessage]], int],
    history_token_budget: int,
    history_low_watermark: float,
    checkpointer: InMemorySaver,
    weather_timeout_seconds: int,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Assemble l'agent météo à partir du modèle et des paramètres de l'INI."""
    weather_tool = make_weather_tool(weather_timeout_seconds)

    @before_model
    def trim_history(
        state: AgentState, runtime: Runtime[None]
    ) -> dict[str, list[BaseMessage]] | None:
        """Purge par paliers l'historique lorsque son budget de tokens est dépassé."""
        messages = state["messages"]
        if token_counter(messages) <= history_token_budget:
            return None
        trimmed = trim_messages(
            messages,
            max_tokens=max(1, int(history_token_budget * history_low_watermark)),
            token_counter=token_counter,
            strategy="last",
            start_on="human",
            allow_partial=False,
        )
        return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *trimmed]}

    return create_agent(
        model=llm,
        tools=[weather_tool],
        system_prompt=f"{system_prompt}\n\n{instructions}",
        middleware=[trim_history],
        checkpointer=checkpointer,
        name="microchat_weather_agent",
    )


def text_from_content(content: str | list[Any]) -> Iterable[str]:
    """Extrait le texte des contenus LangChain texte ou multimodaux."""
    if isinstance(content, str):
        if content:
            yield content
        return
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text")
            if isinstance(text, str) and text:
                yield text
