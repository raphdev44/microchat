"""Agent météo autonome : instructions, outil Open-Meteo et graphe LangGraph."""

from __future__ import annotations

import json
import re
import time
import unicodedata
from collections.abc import Callable, Iterable, Sequence
from datetime import date, timedelta
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from uuid import uuid4

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import after_model, before_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    ToolCall,
)
from langchain_core.messages.utils import trim_messages
from langchain_core.tools import BaseTool, ToolException, tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field, field_validator

ICON = "🌦️"
TOOL_NAMES = ("get_weather",)
MAX_FORECAST_DAYS = 16
ASK_CITY = "Pour quelle ville voulez-vous la météo ?"


class WeatherQuery(BaseModel):
    """Entrée contrôlée de l'outil météo."""

    city: str = Field(
        min_length=1,
        description="Nom de la ville, par exemple Paris ou Montréal.",
    )
    when: str | None = Field(
        default=None,
        description=(
            "Premier jour voulu, tel que l'a dit l'utilisateur : aujourd'hui, demain, "
            "après-demain, samedi, le 21, 21/10, 21 octobre, 2026-10-21 ou dans 3 jours."
        ),
    )
    days: int = Field(
        default=1,
        ge=1,
        le=MAX_FORECAST_DAYS,
        description="Nombre de jours de prévision à partir de ce jour (1 à 16).",
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
    0: ("☀️", "ciel dégagé"),
    1: ("🌤️", "principalement dégagé"),
    2: ("⛅", "partiellement nuageux"),
    3: ("☁️", "couvert"),
    45: ("🌫️", "brouillard"),
    48: ("🌫️", "brouillard givrant"),
    51: ("🌦️", "bruine légère"),
    53: ("🌦️", "bruine modérée"),
    55: ("🌦️", "bruine dense"),
    56: ("🧊", "bruine verglaçante légère"),
    57: ("🧊", "bruine verglaçante dense"),
    61: ("🌧️", "pluie faible"),
    63: ("🌧️", "pluie modérée"),
    65: ("🌧️", "pluie forte"),
    66: ("🧊", "pluie verglaçante faible"),
    67: ("🧊", "pluie verglaçante forte"),
    71: ("🌨️", "neige faible"),
    73: ("❄️", "neige modérée"),
    75: ("❄️", "neige forte"),
    77: ("🌨️", "grains de neige"),
    80: ("🌦️", "averses faibles"),
    81: ("🌧️", "averses modérées"),
    82: ("⛈️", "averses violentes"),
    85: ("🌨️", "averses de neige faibles"),
    86: ("❄️", "averses de neige fortes"),
    95: ("⛈️", "orage"),
    96: ("⛈️", "orage avec grêle faible"),
    99: ("⛈️", "orage avec grêle forte"),
}
_WEEKDAYS = ("lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche")
_MONTHS = (
    "janvier",
    "février",
    "mars",
    "avril",
    "mai",
    "juin",
    "juillet",
    "août",
    "septembre",
    "octobre",
    "novembre",
    "décembre",
)
_DAY_PATTERN = re.compile(r"(?<!\d)(\d{1,2})(?:[/.-](\d{1,2}))?(?:[/.-](\d{4}))?(?!\d)")


def _condition(code: Any) -> tuple[str, str]:
    """Associe un code WMO à son symbole et à son libellé."""
    unknown = ("🌡️", "conditions inconnues")
    return _WEATHER_CODES.get(code, unknown) if isinstance(code, int) else unknown


def _num(value: Any) -> str:
    """Affiche une mesure Open-Meteo, qui peut être absente sur les jours lointains."""
    return "n/d" if value is None else str(value)


def normalize_name(text: str) -> str:
    """Minuscules sans accents ni ponctuation, pour comparer des noms de ville."""
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    plain = "".join(char for char in decomposed if not unicodedata.combining(char))
    return " ".join(re.findall(r"[a-z0-9]+", plain))


def format_day(day: date) -> str:
    """Formate une date en français sans dépendre de la locale du système."""
    number = "1er" if day.day == 1 else str(day.day)
    return f"{_WEEKDAYS[day.weekday()]} {number} {_MONTHS[day.month - 1]}"


def _next_matching_date(today: date, day: int, month: int | None, year: int | None) -> date:
    """Renvoie la prochaine date compatible avec un jour, un mois et une année partiels."""
    if year is not None and month is not None:
        return date(year, month, day)
    if month is None:
        next_month = (today.year + today.month // 12, today.month % 12 + 1)
        months = [(today.year, today.month), next_month]
    else:
        months = [(today.year, month), (today.year + 1, month)]
    for candidate_year, candidate_month in months:
        try:
            candidate = date(candidate_year, candidate_month, day)
        except ValueError:
            continue
        if candidate >= today:
            return candidate
    raise ToolException(f"Date invalide : jour {day}.")


def resolve_day(when: str | None, today: date) -> date:
    """Convertit une expression de date française en date, relative au jour local."""
    text = " ".join((when or "").lower().replace("’", "'").split())
    if text in ("", "aujourd'hui", "aujourdhui", "maintenant", "actuellement", "ce jour"):
        return today
    if text == "demain":
        return today + timedelta(days=1)
    if text.replace("-", " ") in ("après demain", "apres demain"):
        return today + timedelta(days=2)
    if match := re.fullmatch(r"dans (\d+) jours?", text):
        return today + timedelta(days=int(match[1]))
    try:
        return date.fromisoformat(text)
    except ValueError:
        pass
    words = re.findall(r"[a-zà-ÿ]+", text)
    if match := _DAY_PATTERN.search(text):
        month = int(match[2]) if match[2] else None
        if month is None:
            month = next((i + 1 for i, name in enumerate(_MONTHS) if name in words), None)
        year_match = re.search(r"(?<!\d)(\d{4})(?!\d)", text)
        year = int(match[3] or (year_match[1] if year_match else 0)) or None
        try:
            return _next_matching_date(today, int(match[1]), month, year)
        except ValueError as exc:
            raise ToolException(f"Date invalide : « {when} ».") from exc
    for index, name in enumerate(_WEEKDAYS):
        if name in words:
            return today + timedelta(days=(index - today.weekday()) % 7)
    raise ToolException(
        f"Date non reconnue : « {when} ». Utilise par exemple demain, samedi, le 21, "
        "21/10 ou 2026-10-21."
    )


# Format Hermes/Qwen ; un bloc non fermé est accepté pendant le streaming.
_TOOL_CALL_PATTERN = re.compile(r"<tool_call>(.*?)(?:</tool_call>|$)", re.DOTALL)
# Qwen3 émet un bloc de réflexion, vide avec /no_think, que Rich rend en lignes vides.
_THINK_PATTERN = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL)


def visible_text(text: str) -> str:
    """Retire les balises de réflexion et d'appel d'outil du texte affiché."""
    return _TOOL_CALL_PATTERN.sub("", _THINK_PATTERN.sub("", text)).strip()


def parse_text_tool_calls(text: str) -> list[ToolCall]:
    """Extrait les appels d'outil écrits en texte par les modèles locaux."""
    calls: list[ToolCall] = []
    for block in _TOOL_CALL_PATTERN.findall(text):
        try:
            # raw_decode tolère les caractères parasites après le JSON.
            payload, _end = json.JSONDecoder().raw_decode(block.strip())
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict) or not isinstance(payload.get("name"), str):
            continue
        args = payload.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                continue
        if not isinstance(args, dict):
            continue
        calls.append(
            ToolCall(name=payload["name"], args=args, id=f"call_{uuid4().hex}", type="tool_call")
        )
    return calls


def make_weather_tool(timeout_seconds: int) -> BaseTool:
    """Construit l'outil météo avec le délai réseau fourni par la configuration."""

    def get_json(url: str, params: dict[str, str]) -> dict[str, Any]:
        """Appelle un endpoint Open-Meteo et valide le type de la réponse."""
        request = Request(
            f"{url}?{urlencode(params)}",
            headers={"Accept": "application/json", "User-Agent": "microchat/0.1"},
        )
        for attempt in range(2):
            try:
                with urlopen(request, timeout=timeout_seconds) as response:
                    result = json.load(response)
                break
            except HTTPError as exc:
                if attempt == 0 and (exc.code == 429 or 500 <= exc.code < 600):
                    time.sleep(0.25)
                    continue
                raise ToolException(f"Le service météo est indisponible : {exc}") from exc
            except (URLError, TimeoutError) as exc:
                if attempt == 0:
                    time.sleep(0.25)
                    continue
                raise ToolException(f"Le service météo est indisponible : {exc}") from exc
            except json.JSONDecodeError as exc:
                raise ToolException(
                    f"Le service météo a renvoyé une réponse invalide : {exc}"
                ) from exc
        if not isinstance(result, dict):
            raise ToolException("Le service météo a renvoyé une réponse invalide.")
        return result

    @tool(args_schema=WeatherQuery, return_direct=True)
    def get_weather(city: str, when: str | None = None, days: int = 1) -> str:
        """Donne la météo actuelle et les prévisions jour par jour d'une ville (16 jours max).

        Passe la date telle que l'utilisateur l'a dite dans `when` (demain, samedi, le 21…)
        et le nombre de jours demandés dans `days`.
        """
        location_result = get_json(
            "https://geocoding-api.open-meteo.com/v1/search",
            {"name": city, "count": "1", "language": "fr", "format": "json"},
        )
        locations = location_result.get("results")
        if not isinstance(locations, list) or not locations:
            raise ToolException(f"Aucune ville trouvée pour « {city} ».")
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
                "daily": (
                    "weather_code,temperature_2m_max,temperature_2m_min,"
                    "precipitation_sum,precipitation_probability_max,wind_speed_10m_max"
                ),
                "forecast_days": str(MAX_FORECAST_DAYS),
                "timezone": "auto",
            },
        )
        daily = forecast.get("daily")
        if not isinstance(daily, dict) or not daily.get("time"):
            raise ToolException("Le service météo n'a pas renvoyé de prévisions.")
        dates = [date.fromisoformat(value) for value in daily["time"]]
        # Le premier jour renvoyé est la date locale de la ville (timezone=auto).
        today = dates[0]
        start = resolve_day(when, today)
        offset = (start - today).days
        if offset < 0:
            raise ToolException(
                f"Le {format_day(start)} est passé : seules les prévisions sont disponibles."
            )
        if offset >= len(dates):
            raise ToolException(
                f"Le {format_day(start)} est trop loin : les prévisions couvrent "
                f"jusqu'au {format_day(dates[-1])}."
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
        lines = [f"**📍 {place}**", ""]
        current = forecast.get("current")
        if offset == 0 and isinstance(current, dict):
            symbol, label = _condition(current.get("weather_code"))
            lines.append(
                f"- **Maintenant** : {symbol} {label}, "
                f"🌡️ {_num(current.get('temperature_2m'))} °C "
                f"(ressenti {_num(current.get('apparent_temperature'))} °C), "
                f"💧 humidité {_num(current.get('relative_humidity_2m'))} %, "
                f"☔ {_num(current.get('precipitation'))} mm, "
                f"💨 {_num(current.get('wind_speed_10m'))} km/h"
            )
        end = min(offset + days, len(dates))
        for index in range(offset, end):

            def value(name: str, i: int = index) -> Any:
                series = daily.get(name)
                return series[i] if isinstance(series, list) and i < len(series) else None

            symbol, label = _condition(value("weather_code"))
            lines.append(
                f"- **{format_day(dates[index]).capitalize()}** : {symbol} {label}, "
                f"🌡️ {_num(value('temperature_2m_min'))} → "
                f"{_num(value('temperature_2m_max'))} °C, "
                f"☔ {_num(value('precipitation_sum'))} mm "
                f"({_num(value('precipitation_probability_max'))} %), "
                f"💨 {_num(value('wind_speed_10m_max'))} km/h"
            )
        if offset + days > len(dates):
            lines += ["", f"_Prévisions limitées au {format_day(dates[-1])}._"]
        return "\n".join(lines)

    get_weather.handle_tool_error = True
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

    @after_model
    def extract_text_tool_calls(
        state: AgentState, runtime: Runtime[None]
    ) -> dict[str, list[BaseMessage]] | None:
        """Remplace une réponse ``<tool_call>`` textuelle par un appel d'outil structuré."""
        message = state["messages"][-1]
        if not isinstance(message, AIMessage) or message.tool_calls:
            return None
        text = "".join(text_from_content(message.content))
        calls = parse_text_tool_calls(text)
        if not calls:
            return None
        # ChatLlamaCpp convertit "" en None, ce que le template Jinja de Qwen3 refuse.
        content = visible_text(text) or " "
        # Même id : add_messages remplace le message au lieu de l'ajouter.
        return {"messages": [AIMessage(content=content, tool_calls=calls, id=message.id)]}

    @after_model
    def require_named_city(
        state: AgentState, runtime: Runtime[None]
    ) -> dict[str, list[BaseMessage]] | None:
        """Demande la ville et préserve la date lorsqu'elle est fournie au tour suivant."""
        message = state["messages"][-1]
        if not isinstance(message, AIMessage):
            return None

        messages = state["messages"]
        previous_assistant = next(
            (item for item in reversed(messages[:-1]) if isinstance(item, AIMessage)), None
        )
        latest_user = next(
            (item for item in reversed(messages[:-1]) if isinstance(item, HumanMessage)), None
        )
        pending = (
            previous_assistant.additional_kwargs.get("pending_weather")
            if previous_assistant is not None and previous_assistant.content == ASK_CITY
            else None
        )

        if pending and latest_user is not None:
            city = "".join(text_from_content(latest_user.content)).strip()
            city = re.sub(r"^(?:à|a|au|aux|pour)\s+", "", city, flags=re.IGNORECASE)
            if city:
                call = ToolCall(
                    name="get_weather",
                    args={"city": city, **pending},
                    id=f"call_{uuid4().hex}",
                    type="tool_call",
                )
                return {"messages": [AIMessage(content=" ", tool_calls=[call], id=message.id)]}

        if not message.tool_calls:
            return None
        said = ""
        if latest_user is not None:
            said = f" {normalize_name(''.join(text_from_content(latest_user.content)))} "
        for call in message.tool_calls:
            city = normalize_name(str(call["args"].get("city", "")))
            if call["name"] in TOOL_NAMES and (not city or f" {city} " not in said):
                pending = {
                    "when": call["args"].get("when"),
                    "days": call["args"].get("days", 1),
                }
                return {
                    "messages": [
                        AIMessage(
                            content=ASK_CITY,
                            additional_kwargs={"pending_weather": pending},
                            id=message.id,
                        )
                    ]
                }
        return None

    return create_agent(
        model=llm,
        tools=[weather_tool],
        system_prompt=f"{system_prompt}\n\n{instructions}",
        # Les hooks after_model s'exécutent du dernier au premier.
        middleware=[trim_history, require_named_city, extract_text_tool_calls],
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
