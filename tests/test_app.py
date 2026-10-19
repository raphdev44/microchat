from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langgraph.checkpoint.memory import InMemorySaver
from rich.console import Console

from microchat.agents import weather
from microchat.agents.weather import build_agent
from microchat.app import ChatApp, _parse_command, _select_gguf, load_config


class SimpleFakeModel(FakeMessagesListChatModel):
    """Faux modèle terminal qui n'appelle pas l'outil pendant ce test."""

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self


def count_words(messages: list[BaseMessage]) -> int:
    """Compteur déterministe utilisé par le middleware de l'agent."""
    return sum(len(str(message.content).split()) for message in messages)


def test_loads_complete_ini_configuration():
    settings = load_config(Path("config.ini"))

    assert settings.model_repo_id == "Qwen/Qwen3-1.7B-GGUF"
    assert settings.n_ctx >= 512
    assert settings.history_token_budget > 0
    assert settings.session_id == "terminal"
    assert settings.weather_timeout_seconds == 30


def test_config_rejects_missing_parameters(tmp_path):
    path = tmp_path / "config.ini"
    path.write_text("[microchat]\nmodel_repo_id = org/model\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Paramètres manquants"):
        load_config(path)


def test_config_rejects_unknown_parameters(tmp_path):
    path = Path("config.ini")
    text = path.read_text(encoding="utf-8").replace(
        "model_repo_id =", "unexpected_option = true\nmodel_repo_id ="
    )
    unknown = tmp_path / "config.ini"
    unknown.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="Paramètres inconnus"):
        load_config(unknown)


def test_selects_compact_gguf_available_in_repository():
    assert (
        _select_gguf(
            "org/model",
            ["model-F16.gguf", "model-Q4_K_M.gguf", "model-Q8_0.gguf"],
        )
        == "model-Q4_K_M.gguf"
    )


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("hello", None),
        ("/help", "help"),
        ("/reset", "clear"),
        ("/Q", "quit"),
        ("/not-a-command", "unknown"),
    ],
)
def test_parse_command(line, expected):
    assert _parse_command(line) == expected


def test_chat_app_runs_chat_and_commands(monkeypatch):
    checkpointer = InMemorySaver()
    agent = build_agent(
        SimpleFakeModel(responses=[AIMessage(content="Bonjour !")]),
        system_prompt="Réponds brièvement.",
        instructions="N'appelle pas d'outil pour ce test.",
        token_counter=count_words,
        history_token_budget=100,
        history_low_watermark=0.6,
        checkpointer=checkpointer,
        weather_timeout_seconds=3,
    )
    settings = load_config(Path("config.ini"))
    app = ChatApp(agent, checkpointer, settings, Path("fake.gguf"))
    output = io.StringIO()
    app.console = Console(file=output, force_terminal=False, width=100)
    prompts = iter(["bonjour", "/info", "/clear", "/quit"])
    monkeypatch.setattr(app.console, "input", lambda _prompt: next(prompts))

    app.run()

    rendered = output.getvalue()
    assert "Bonjour" in rendered
    assert weather.ICON not in rendered
    assert "Messages en mémoire" in rendered
    assert "Historique effacé" in rendered
    assert app._message_count() == 0


def test_weather_tool_names_match_declared_tools():
    assert weather.make_weather_tool(timeout_seconds=1).name in weather.TOOL_NAMES


def test_chat_app_shows_weather_icon_when_subagent_is_used(monkeypatch):
    spinner_phases = []

    class RecordingLive:
        def __init__(self, renderable, **kwargs):
            self.console = kwargs["console"]
            self.update(renderable)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.console.print(self.renderable)
            return None

        def update(self, renderable):
            self.renderable = renderable
            output = io.StringIO()
            Console(file=output, force_terminal=False, width=100).print(renderable)
            spinner_phases.append(output.getvalue())

    monkeypatch.setattr("microchat.app.Live", RecordingLive)
    monkeypatch.setattr(
        "microchat.agents.weather.urlopen",
        lambda request, timeout: io.BytesIO(
            json.dumps(
                {"results": [{"name": "Nantes", "latitude": 47.2, "longitude": -1.5}]}
                if "geocoding" in request.full_url
                else {
                    "current": {
                        "temperature_2m": 14,
                        "relative_humidity_2m": 70,
                        "apparent_temperature": 13,
                        "precipitation": 0,
                        "weather_code": 3,
                        "wind_speed_10m": 12,
                    },
                    "daily": {
                        "time": ["2026-10-19"],
                        "weather_code": [61],
                        "temperature_2m_min": [9],
                        "temperature_2m_max": [15],
                        "precipitation_sum": [2.4],
                        "precipitation_probability_max": [70],
                        "wind_speed_10m_max": [20],
                    },
                }
            ).encode("utf-8")
        ),
    )
    checkpointer = InMemorySaver()
    agent = build_agent(
        SimpleFakeModel(
            responses=[
                AIMessage(
                    content=(
                        '<tool_call> {"name": "get_weather", '
                        '"arguments": {"city": "Nantes"}} </tool_call>'
                    )
                ),
                AIMessage(content="Reformulation qui ne doit pas être utilisée."),
            ]
        ),
        system_prompt="Réponds brièvement.",
        instructions="Utilise la météo.",
        token_counter=count_words,
        history_token_budget=100,
        history_low_watermark=0.6,
        checkpointer=checkpointer,
        weather_timeout_seconds=3,
    )
    app = ChatApp(agent, checkpointer, load_config(Path("config.ini")), Path("fake.gguf"))
    output = io.StringIO()
    app.console = Console(file=output, force_terminal=False, width=100)

    app._answer("Quel temps à Nantes ?")

    rendered = output.getvalue()
    assert weather.ICON in rendered
    assert "☁️ couvert" in rendered
    assert "🌧️ pluie faible" in rendered
    assert "Reformulation" not in rendered
    assert app._message_count() == 2
    assert any("Consultation d'Open-Meteo" in phase for phase in spinner_phases)
    assert any("Mise en forme des prévisions" in phase for phase in spinner_phases)
