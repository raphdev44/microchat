from __future__ import annotations

import io
import json
from collections.abc import Sequence

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langgraph.checkpoint.memory import InMemorySaver

from microchat.agents.weather import build_agent, make_weather_tool


class ToolCallingFake(FakeMessagesListChatModel):
    """Faux modèle qui renvoie des appels d'outil déterministes."""

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self


def count_words(messages: Sequence[BaseMessage]) -> int:
    """Compteur de tokens factice pour les tests."""
    return sum(len(str(message.content).split()) for message in messages)


def test_weather_tool_looks_up_city_and_formats_current_data(monkeypatch):
    requests = []

    def fake_urlopen(request, timeout):
        requests.append((request.full_url, timeout))
        if "geocoding" in request.full_url:
            data = {
                "results": [
                    {
                        "name": "Nantes",
                        "admin1": "Pays de la Loire",
                        "country": "France",
                        "latitude": 47.22,
                        "longitude": -1.55,
                    }
                ]
            }
        else:
            data = {
                "current": {
                    "time": "2026-10-18T18:00",
                    "temperature_2m": 17,
                    "relative_humidity_2m": 61,
                    "apparent_temperature": 16,
                    "precipitation": 0,
                    "weather_code": 2,
                    "wind_speed_10m": 8,
                }
            }
        return io.BytesIO(json.dumps(data).encode("utf-8"))

    monkeypatch.setattr("microchat.agents.weather.urlopen", fake_urlopen)
    weather_tool = make_weather_tool(timeout_seconds=7)

    result = weather_tool.invoke({"city": "Nantes"})

    assert "Nantes, Pays de la Loire, France" in result
    assert "partiellement nuageux" in result
    assert "17 °C" in result
    assert len(requests) == 2
    assert all(timeout == 7 for _, timeout in requests)


def test_weather_agent_runs_its_tool_and_retains_turn(monkeypatch):
    def fake_urlopen(request, timeout):
        if "geocoding" in request.full_url:
            data = {"results": [{"name": "Paris", "latitude": 48.85, "longitude": 2.35}]}
        else:
            data = {
                "current": {
                    "time": "2026-10-18T18:00",
                    "temperature_2m": 15,
                    "relative_humidity_2m": 50,
                    "apparent_temperature": 15,
                    "precipitation": 0,
                    "weather_code": 0,
                    "wind_speed_10m": 5,
                }
            }
        return io.BytesIO(json.dumps(data).encode("utf-8"))

    monkeypatch.setattr("microchat.agents.weather.urlopen", fake_urlopen)
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

    events = list(
        agent.stream(
            {"messages": [{"role": "user", "content": "Quel temps à Paris ?"}]},
            config=config,
            stream_mode="messages",
        )
    )

    assert any("15 °C" in str(chunk.content) for chunk, _metadata in events)
    state = agent.get_state(config).values
    assert len(state["messages"]) >= 4
