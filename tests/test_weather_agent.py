from __future__ import annotations

import io
import json
from collections.abc import Sequence
from datetime import date, timedelta
from urllib.error import HTTPError

import pytest
from langchain_community.chat_models.llamacpp import _convert_message_to_dict
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import ToolException
from langgraph.checkpoint.memory import InMemorySaver

from microchat.agents.weather import (
    ASK_CITY,
    build_agent,
    format_day,
    make_weather_tool,
    parse_text_tool_calls,
    resolve_day,
    visible_text,
)


class ToolCallingFake(FakeMessagesListChatModel):
    """Faux modèle qui renvoie des appels d'outil déterministes."""

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self


def count_words(messages: Sequence[BaseMessage]) -> int:
    """Compteur de tokens factice pour les tests."""
    return sum(len(str(message.content).split()) for message in messages)


# Lundi 19 octobre 2026, date locale de la ville dans les faux services.
TODAY = date(2026, 10, 19)


def open_meteo(location, requests=None, current_code=2):
    """Faux urlopen : géocodage puis conditions actuelles et 16 jours de prévisions."""

    def fake_urlopen(request, timeout):
        if requests is not None:
            requests.append((request.full_url, timeout))
        if "geocoding" in request.full_url:
            data = {"results": [location]}
        else:
            days = range(16)
            data = {
                "current": {
                    "temperature_2m": 17,
                    "relative_humidity_2m": 61,
                    "apparent_temperature": 16,
                    "precipitation": 0,
                    "weather_code": current_code,
                    "wind_speed_10m": 8,
                },
                "daily": {
                    "time": [(TODAY + timedelta(days=i)).isoformat() for i in days],
                    "weather_code": [(0, 61, 95, 73)[i % 4] for i in days],
                    "temperature_2m_min": [5 + i for i in days],
                    "temperature_2m_max": [15 + i for i in days],
                    "precipitation_sum": [i / 2 for i in days],
                    "precipitation_probability_max": [10 * (i % 10) for i in days],
                    "wind_speed_10m_max": [20 + i for i in days],
                },
            }
        return io.BytesIO(json.dumps(data).encode("utf-8"))

    return fake_urlopen


NANTES = {
    "name": "Nantes",
    "admin1": "Pays de la Loire",
    "country": "France",
    "latitude": 47.22,
    "longitude": -1.55,
}


def test_weather_tool_looks_up_city_and_formats_current_data(monkeypatch):
    requests = []
    monkeypatch.setattr("microchat.agents.weather.urlopen", open_meteo(NANTES, requests))
    weather_tool = make_weather_tool(timeout_seconds=7)

    result = weather_tool.invoke({"city": "Nantes"})

    assert "**📍 Nantes, Pays de la Loire, France**" in result
    assert "- **Maintenant** : ⛅ partiellement nuageux, 🌡️ 17 °C" in result
    assert "- **Lundi 19 octobre** : ☀️ ciel dégagé, 🌡️ 5 → 15 °C" in result
    assert "Mardi" not in result
    assert len(requests) == 2
    assert all(timeout == 7 for _, timeout in requests)
    assert "forecast_days=16" in requests[1][0]


def test_weather_tool_gives_future_day_and_multi_day_forecast(monkeypatch):
    monkeypatch.setattr("microchat.agents.weather.urlopen", open_meteo(NANTES))
    weather_tool = make_weather_tool(timeout_seconds=7)

    tomorrow = weather_tool.invoke({"city": "Nantes", "when": "demain"})
    three_days = weather_tool.invoke({"city": "Nantes", "when": "le 21", "days": 3})

    assert "Maintenant" not in tomorrow
    assert "**Mardi 20 octobre** : 🌧️ pluie faible, 🌡️ 6 → 16 °C" in tomorrow
    assert "☔ 0.5 mm (10 %), 💨 21 km/h" in tomorrow
    assert [line.split("**")[1] for line in three_days.splitlines()[2:]] == [
        "Mercredi 21 octobre",
        "Jeudi 22 octobre",
        "Vendredi 23 octobre",
    ]
    assert "⛈️ orage" in three_days
    assert "❄️ neige modérée" in three_days


def test_weather_tool_reports_dates_outside_forecast_to_model(monkeypatch):
    monkeypatch.setattr("microchat.agents.weather.urlopen", open_meteo(NANTES))
    weather_tool = make_weather_tool(timeout_seconds=7)

    too_far = weather_tool.invoke({"city": "Nantes", "when": "2026-12-25"})
    unknown = weather_tool.invoke({"city": "Nantes", "when": "un jour"})
    clipped = weather_tool.invoke({"city": "Nantes", "when": "dans 14 jours", "days": 5})

    assert "trop loin" in too_far
    assert "Date non reconnue" in unknown
    assert clipped.count("°C") == 2
    assert "Prévisions limitées au mardi 3 novembre" in clipped


def test_weather_tool_retries_one_transient_service_error(monkeypatch):
    attempts = 0
    successful = open_meteo(NANTES)

    def flaky_urlopen(request, timeout):
        nonlocal attempts
        if "forecast" in request.full_url and attempts == 0:
            attempts += 1
            raise HTTPError(request.full_url, 503, "Unavailable", {}, None)
        if "forecast" in request.full_url:
            attempts += 1
        return successful(request, timeout)

    monkeypatch.setattr("microchat.agents.weather.urlopen", flaky_urlopen)
    result = make_weather_tool(timeout_seconds=1).invoke({"city": "Nantes"})

    assert "Nantes" in result
    assert attempts == 2


@pytest.mark.parametrize(
    ("when", "expected"),
    [
        (None, TODAY),
        ("Aujourd’hui", TODAY),
        ("demain", date(2026, 10, 20)),
        ("après-demain", date(2026, 10, 21)),
        ("dans 5 jours", date(2026, 10, 24)),
        ("samedi", date(2026, 10, 24)),
        ("lundi", TODAY),
        ("le 21", date(2026, 10, 21)),
        ("le 2", date(2026, 11, 2)),
        ("21/10", date(2026, 10, 21)),
        ("3 novembre", date(2026, 11, 3)),
        ("samedi 24 octobre", date(2026, 10, 24)),
        ("2026-10-30", date(2026, 10, 30)),
        ("le 18", date(2026, 11, 18)),
        ("le 31", date(2026, 10, 31)),
    ],
)
def test_resolve_day_understands_french_expressions(when, expected):
    assert resolve_day(when, TODAY) == expected


def test_resolve_day_rolls_over_to_next_year():
    assert resolve_day("le 2", date(2026, 12, 20)) == date(2027, 1, 2)


def test_resolve_day_rejects_unknown_expression():
    with pytest.raises(ToolException, match="Date non reconnue"):
        resolve_day("bientôt", TODAY)


def test_format_day_uses_french_names():
    assert format_day(date(2026, 11, 1)) == "dimanche 1er novembre"


def test_weather_agent_runs_its_tool_and_retains_turn(monkeypatch):
    monkeypatch.setattr(
        "microchat.agents.weather.urlopen",
        open_meteo({"name": "Paris", "latitude": 48.85, "longitude": 2.35}, current_code=0),
    )
    checkpointer = InMemorySaver()
    agent = build_agent(
        ToolCallingFake(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "get_weather",
                            "args": {"city": "Paris"},
                            "id": "weather-1",
                        }
                    ],
                ),
                AIMessage(content="À Paris, le ciel est dégagé et il fait 15 °C."),
            ]
        ),
        system_prompt="Réponds en français.",
        instructions="Utilise la météo quand la question porte sur le temps.",
        token_counter=count_words,
        history_token_budget=100,
        history_low_watermark=0.6,
        checkpointer=checkpointer,
        weather_timeout_seconds=5,
    )
    config = {"configurable": {"thread_id": "test"}}

    list(
        agent.stream(
            {"messages": [{"role": "user", "content": "Quel temps à Paris ?"}]},
            config=config,
            stream_mode="messages",
        )
    )

    messages = agent.get_state(config).values["messages"]
    # Le résultat de l'outil est la réponse finale : pas de reformulation par le modèle.
    assert isinstance(messages[-1], ToolMessage)
    assert "☀️ ciel dégagé" in messages[-1].content
    assert len(messages) == 3


def test_weather_agent_executes_text_tool_call_from_local_model(monkeypatch):
    urls = []

    monkeypatch.setattr("microchat.agents.weather.urlopen", open_meteo(NANTES, urls, 3))
    checkpointer = InMemorySaver()
    agent = build_agent(
        ToolCallingFake(
            responses=[
                # Sortie réelle de Qwen3 : bloc think vide et accolade fermante en trop.
                AIMessage(
                    content='<think>\n\n</think>\n\n<tool_call> {"name": "get_weather", '
                    '"arguments": {"city": "Nantes"}}} </tool_call>'
                ),
                AIMessage(content="À Nantes, le ciel est couvert et il fait 14 °C."),
            ]
        ),
        system_prompt="Réponds en français.",
        instructions="Utilise la météo quand la question porte sur le temps.",
        token_counter=count_words,
        history_token_budget=100,
        history_low_watermark=0.6,
        checkpointer=checkpointer,
        weather_timeout_seconds=5,
    )
    config = {"configurable": {"thread_id": "text-tool-call"}}

    agent.invoke({"messages": [{"role": "user", "content": "Quel temps à Nantes ?"}]}, config)

    messages = agent.get_state(config).values["messages"]
    assert any("nantes" in url.lower() for url, _timeout in urls)
    assert messages[1].tool_calls[0]["args"] == {"city": "Nantes"}
    assert "<tool_call>" not in str(messages[1].content)
    # Le template Jinja de Qwen3 échoue si llama.cpp reçoit content=None.
    assert _convert_message_to_dict(messages[1])["content"] is not None
    assert "couvert" in str(messages[2].content)
    assert messages[-1] is messages[2]


def test_weather_agent_asks_for_city_instead_of_inventing_one(monkeypatch):
    urls = []
    monkeypatch.setattr("microchat.agents.weather.urlopen", open_meteo(NANTES, urls))
    checkpointer = InMemorySaver()
    agent = build_agent(
        ToolCallingFake(
            responses=[
                AIMessage(
                    content='<tool_call>{"name": "get_weather", '
                    '"arguments": {"city": "Paris", "when": "demain"}}</tool_call>'
                ),
                AIMessage(
                    content="",
                    tool_calls=[{"name": "get_weather", "args": {"city": "nantes"}, "id": "w2"}],
                ),
            ]
        ),
        system_prompt="Réponds en français.",
        instructions="Utilise la météo.",
        token_counter=count_words,
        history_token_budget=200,
        history_low_watermark=0.6,
        checkpointer=checkpointer,
        weather_timeout_seconds=5,
    )
    config = {"configurable": {"thread_id": "no-city"}}

    agent.invoke({"messages": [{"role": "user", "content": "La météo pour demain ?"}]}, config)

    messages = agent.get_state(config).values["messages"]
    assert messages[-1].content == ASK_CITY
    assert not messages[-1].tool_calls
    assert urls == []

    agent.invoke({"messages": [{"role": "user", "content": "À Nantès"}]}, config)

    messages = agent.get_state(config).values["messages"]
    assert isinstance(messages[-1], ToolMessage)
    assert messages[1].additional_kwargs["pending_weather"] == {"when": "demain", "days": 1}
    assert messages[-1].content.find("Mardi 20 octobre") >= 0
    assert urls


def test_visible_text_hides_complete_and_partial_tool_calls():
    assert visible_text('Avant <tool_call>{"name": "x"}</tool_call> après') == "Avant  après"
    assert visible_text('<tool_call> {"name": "get_wea') == ""
    assert visible_text("<think>\n\n</think>\n\nSalut !") == "Salut !"
    assert visible_text("<think>\nje réfléch") == ""
    assert parse_text_tool_calls("Bonjour") == []
