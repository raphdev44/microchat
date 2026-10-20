from __future__ import annotations

import configparser
import io
import json
from pathlib import Path
from urllib.error import HTTPError

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from rich.cells import cell_len
from rich.console import Console

from microchat.agents import weather
from microchat.agents.runtime import build_agent
from microchat.agents.weather import build_agent as build_weather_agent
from microchat.app import ChatApp, _parse_command, _select_gguf, load_config
from microchat.plugins import AgentPlugin, PluginCommand, load_plugins


class SimpleFakeModel(FakeMessagesListChatModel):
    """Faux modèle terminal qui n'appelle pas l'outil pendant ce test."""

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self


def count_words(messages: list[BaseMessage]) -> int:
    """Compteur déterministe utilisé par le middleware de l'agent."""
    return sum(len(str(message.content).split()) for message in messages)


def test_loads_complete_ini_configuration():
    settings = load_config(Path("config.ini"))

    assert settings.model_repo_id
    assert settings.n_ctx >= 512
    assert settings.history_token_budget > 0
    assert settings.session_id == "terminal"
    assert settings.plugins == ("weather", "web_search", "rag")
    assert settings.plugin_configurations["weather"]["timeout_seconds"] == "30"
    assert settings.plugin_configurations["web_search"]["timeout_seconds"] == "5"
    assert settings.plugin_configurations["web_search"]["max_results"] == "3"
    assert settings.plugin_configurations["rag"]["embedding_model"]
    assert int(settings.plugin_configurations["rag"]["chunk_size"]) > 150


def test_plugin_registry_accepts_no_active_plugins():
    assert load_plugins((), {}) == ()


def test_plugin_registry_loads_selected_entry_point(monkeypatch):
    class WeatherEntryPoint:
        name = "weather"

        @staticmethod
        def load():
            return weather.load_plugin

    monkeypatch.setattr("microchat.plugins.entry_points", lambda **_kwargs: [WeatherEntryPoint()])

    (plugin,) = load_plugins(("weather",), {"weather": {"timeout_seconds": "7"}})

    assert plugin.name == "weather"
    assert plugin.tools[0].name == "get_weather"


def test_plugin_registry_rejects_uninstalled_plugin():
    with pytest.raises(ValueError, match="Plugins non installés"):
        load_plugins(("missing-plugin",), {})


def test_config_rejects_missing_parameters(tmp_path):
    path = tmp_path / "config.ini"
    path.write_text("[microchat]\nmodel_repo_id = org/model\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Paramètres manquants"):
        load_config(path)


def test_config_rejects_unknown_parameters(tmp_path):
    path = Path("config.ini")
    parser = configparser.ConfigParser(interpolation=None)
    with path.open(encoding="utf-8") as config_file:
        parser.read_file(config_file)
    parser["microchat"]["unexpected_option"] = "true"
    unknown = tmp_path / "config.ini"
    with unknown.open("w", encoding="utf-8") as config_file:
        parser.write(config_file)

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


def test_plugin_commands_are_recognized_only_when_enabled():
    assert _parse_command("/rag_start") == "unknown"
    assert _parse_command("/rag_start", ("rag_start",)) == "rag_start"


def test_help_shows_only_commands_from_active_plugins():
    checkpointer = InMemorySaver()
    agent = build_agent(
        SimpleFakeModel(responses=[]),
        system_prompt="Réponds brièvement.",
        plugins=(),
        token_counter=count_words,
        history_token_budget=100,
        history_low_watermark=0.6,
        checkpointer=checkpointer,
    )
    settings = load_config(Path("config.ini"))
    command = PluginCommand("/rag_start", "Active le mode RAG", lambda _argument: "ok", "📖")

    def help_output(commands):
        app = ChatApp(
            agent,
            checkpointer,
            settings,
            Path("fake.gguf"),
            count_words,
            plugin_commands=commands,
        )
        output = io.StringIO()
        app.console = Console(file=output, force_terminal=False, width=100)
        app._show_help()
        return output.getvalue()

    assert "/rag_start" not in help_output({})
    assert "📖 /rag_start" in help_output({"rag_start": command})


def test_chat_app_runs_chat_and_commands(monkeypatch):
    checkpointer = InMemorySaver()
    agent = build_agent(
        SimpleFakeModel(responses=[AIMessage(content="Bonjour !")]),
        system_prompt="Réponds brièvement.",
        plugins=(),
        token_counter=count_words,
        history_token_budget=100,
        history_low_watermark=0.6,
        checkpointer=checkpointer,
    )
    settings = load_config(Path("config.ini"))
    app = ChatApp(agent, checkpointer, settings, Path("fake.gguf"), count_words)
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


def test_retrieved_document_frames_render_before_the_answer():
    checkpointer = InMemorySaver()
    agent = build_agent(
        SimpleFakeModel(responses=[AIMessage(content="Réponse finale")]),
        system_prompt="Réponds brièvement.",
        plugins=(),
        token_counter=count_words,
        history_token_budget=100,
        history_low_watermark=0.6,
        checkpointer=checkpointer,
    )
    app = ChatApp(
        agent,
        checkpointer,
        load_config(Path("config.ini")),
        Path("fake.gguf"),
        count_words,
        context_providers=(lambda: (("document.docx", "Fragment récupéré"),),),
    )
    output = io.StringIO()
    app.console = Console(file=output, force_terminal=False, width=100)

    app._answer("Question")

    rendered = output.getvalue()
    assert "Fragment documentaire" in rendered
    assert "Fragment récupéré" in rendered
    assert rendered.index("Fragment récupéré") < rendered.index("Réponse finale")


def test_weather_tool_names_match_declared_tools():
    assert weather.make_weather_tool(timeout_seconds=1).name in weather.TOOL_NAMES


@pytest.mark.parametrize(
    "days,width", [(1, 100), (2, 100), (1, 40), (2, 40), (7, 100), (7, 80), (7, 40), (7, 24)]
)
def test_chat_app_shows_weather_icon_when_subagent_is_used(monkeypatch, days, width):
    spinner_phases = []
    overflow_modes = []

    class RecordingLive:
        def __init__(self, renderable, **kwargs):
            self.console = kwargs["console"]
            overflow_modes.append(kwargs["vertical_overflow"])
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
                        "time": [f"2026-10-{19 + index}" for index in range(7)],
                        "weather_code": [61, 0, 2, 3, 61, 0, 2],
                        "temperature_2m_min": [9.2, 10.3, 11.1, 9.2, 8.4, 7.1, 10.2],
                        "temperature_2m_max": [15.4, 16.2, 17.1, 18.3, 14.2, 15.1, 16.4],
                        "precipitation_sum": [2.4, 0, 0, 0, 1, 0, 0],
                        "precipitation_probability_max": [70, 0, 0, 0, 60, 0, 0],
                        "wind_speed_10m_max": [20, 10, 12, 14, 15, 10, 12],
                    },
                }
            ).encode("utf-8")
        ),
    )
    checkpointer = InMemorySaver()
    agent = build_weather_agent(
        SimpleFakeModel(
            responses=[
                AIMessage(
                    content=(
                        '<tool_call> {"name": "get_weather", '
                        f'"arguments": {{"city": "Nantes", "days": {days}}}}} </tool_call>'
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
    app = ChatApp(
        agent,
        checkpointer,
        load_config(Path("config.ini")),
        Path("fake.gguf"),
        count_words,
        {"get_weather": weather.ICON},
    )
    output = io.StringIO()
    app.console = Console(file=output, force_terminal=False, width=width)

    app._answer(f"Quel temps à Nantes sur {days} jours ?")

    rendered = output.getvalue()
    assert weather.ICON in rendered
    if days == 1:
        assert "Couvert" in rendered
        assert "14°" in rendered
        assert "12 km/h" in rendered
        frame = next(line for line in rendered.splitlines() if "Météo · Nantes" in line)
        assert cell_len(frame) <= min(width, 64)
    else:
        assert "lun. 19" in rendered
        assert "mar. 20" in rendered
        assert "15°" in rendered and "9°" in rendered
        assert "15.4" not in rendered
        assert "Maintenant" not in rendered
        assert "km/h" not in rendered
        assert "🌧️" in rendered
        if days == 7:
            labels = ["lun. 19", "mar. 20", "mer. 21", "jeu. 22", "ven. 23", "sam. 24", "dim. 25"]
            assert all(label in rendered for label in labels)
            if width >= 80:
                headings = next(line for line in rendered.splitlines() if "lun. 19" in line)
                assert all(label in headings for label in labels)
    assert "humidité" not in rendered
    assert "ressenti" not in rendered
    assert " mm" not in rendered
    title = f"{'Météo' if days == 1 else 'Prévisions'} · Nantes"
    lines = rendered.splitlines()
    frame_index = next(index for index, line in enumerate(lines) if title in line)
    assert lines[frame_index + 1].strip("│ ") == ""
    if days > 1 and width >= 80:
        assert cell_len(lines[frame_index]) < width
        headings = next(line for line in lines if "lun. 19" in line)
        gap = headings.split("lun. 19", 1)[1].split("mar. 20", 1)[0]
        assert len(gap) >= 3 and not gap.strip()
    assert any(corner in rendered for corner in ("╭", "┌"))
    assert max(cell_len(line) for line in rendered.splitlines()) <= width
    assert "Reformulation" not in rendered
    assert app._message_count() == 2
    assert any("Appel d'un outil" in phase for phase in spinner_phases)
    assert any("Traitement du résultat" in phase for phase in spinner_phases)
    assert overflow_modes == ["ellipsis"]


def test_chat_app_shows_weather_service_error(monkeypatch):
    def unavailable(request, timeout):
        raise HTTPError(request.full_url, 503, "Unavailable", {}, None)

    monkeypatch.setattr("microchat.agents.weather.urlopen", unavailable)
    plugin = weather.load_plugin({"timeout_seconds": "1"})
    checkpointer = InMemorySaver()
    agent = build_agent(
        SimpleFakeModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "get_weather",
                            "args": {"city": "Couëron"},
                            "id": "weather-503",
                        }
                    ],
                )
            ]
        ),
        system_prompt="Réponds en français.",
        plugins=(plugin,),
        token_counter=count_words,
        history_token_budget=100,
        history_low_watermark=0.6,
        checkpointer=checkpointer,
    )
    app = ChatApp(
        agent,
        checkpointer,
        load_config(Path("config.ini")),
        Path("fake.gguf"),
        count_words,
        plugin.tool_icons,
    )
    output = io.StringIO()
    app.console = Console(file=output, force_terminal=False, width=100)

    app._answer("Quel temps fait-il à Couëron ?")

    rendered = output.getvalue()
    assert weather.ICON in rendered
    assert "Le service météo est indisponible" in rendered
    assert "(réponse vide)" not in rendered


def test_chat_app_hides_structured_tool_call_json():
    @tool
    def echo(value: str) -> str:
        """Retourne le texte fourni."""
        return value

    plugin = AgentPlugin(
        name="echo",
        instructions="Utilise echo pour répéter le texte.",
        tools=(echo,),
        tool_icons={"echo": "🔧"},
    )
    checkpointer = InMemorySaver()
    agent = build_agent(
        SimpleFakeModel(
            responses=[
                AIMessage(
                    content='{"name": "echo", "arguments": {"value": "bonjour"}}',
                    tool_calls=[{"name": "echo", "args": {"value": "bonjour"}, "id": "echo-1"}],
                ),
                AIMessage(content="Bonjour !"),
            ]
        ),
        system_prompt="Réponds en français.",
        plugins=(plugin,),
        token_counter=count_words,
        history_token_budget=100,
        history_low_watermark=0.6,
        checkpointer=checkpointer,
    )
    app = ChatApp(
        agent,
        checkpointer,
        load_config(Path("config.ini")),
        Path("fake.gguf"),
        count_words,
        plugin.tool_icons,
    )
    output = io.StringIO()
    app.console = Console(file=output, force_terminal=False, width=100)

    app._answer("Répète bonjour")

    rendered = output.getvalue()
    assert '"name": "echo"' not in rendered
    assert "Bonjour !" in rendered
