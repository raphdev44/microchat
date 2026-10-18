from __future__ import annotations

import io
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langgraph.checkpoint.memory import InMemorySaver
from rich.console import Console

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
    assert "Messages en mémoire" in rendered
    assert "Historique effacé" in rendered
    assert app._message_count() == 0
