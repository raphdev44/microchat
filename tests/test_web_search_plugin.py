from __future__ import annotations

import io
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from langchain.agents.middleware.types import ModelRequest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import PrivateAttr
from rich.console import Console

from microchat.agents import weather, web_search
from microchat.agents.runtime import build_agent
from microchat.app import ChatApp, load_config


def test_web_search_plugin_declares_tool_and_icon(monkeypatch):
    monkeypatch.setattr(web_search, "_get_ddgs", lambda: object)

    plugin = web_search.load_plugin({"region": "us-en", "timeout_seconds": "8", "max_results": "3"})

    assert plugin.name == "web_search"
    assert plugin.tools[0].name == "search_web"
    assert plugin.tool_icons == {"search_web": web_search.ICON}
    assert plugin.tool_icons == {"search_web": "🌐"}
    assert len(plugin.middleware) == 3


@pytest.mark.parametrize(
    "query",
    [
        "c'est quoi Kandorya ?",
        "qu'est-ce que la tectonique des plaques ?",
        "Qui est Ada Lovelace ?",
        "Que signifie entropie ?",
        "What is Kandorya?",
    ],
)
def test_explicit_factual_questions_require_lookup(query):
    assert web_search._requires_lookup(query)


@pytest.mark.parametrize("query", ["Salut !", "Comment vas-tu ?", "Combien font 2 + 2 ?"])
def test_conversation_and_simple_questions_skip_lookup(query):
    assert not web_search._requires_lookup(query)


def test_search_terms_removes_question_prefix_and_punctuation():
    assert web_search._search_terms("c'est quoi Kandorya ?") == "c'est quoi Kandorya"
    assert (
        web_search._search_terms("je me demandais... c'est quoi Kandorya ?")
        == "je me demandais... c'est quoi Kandorya"
    )


def test_pre_search_uses_model_and_appends_only_three_sources(monkeypatch):
    class FixedResponseModel(FakeMessagesListChatModel):
        def bind_tools(self, tools, *, tool_choice=None, **kwargs):
            return self

    monkeypatch.setattr(web_search, "_get_ddgs", lambda: object)
    monkeypatch.setattr(
        web_search,
        "_fetch_results",
        lambda *_args, **_kwargs: [
            {
                "title": "Site officiel de Kandorya",
                "href": "https://www.kandorya.net/",
                "body": "Un jeu de rôle grandeur nature en France.",
            },
            {"title": "Source 2", "href": "https://example.com/2", "body": "Second excerpt."},
            {"title": "Source 3", "href": "https://example.com/3", "body": "Third excerpt."},
            {"title": "Source 4", "href": "https://example.com/4", "body": "Fourth excerpt."},
        ],
    )
    plugin = web_search.load_plugin({})
    model = FixedResponseModel(
        responses=[AIMessage(content="Kandorya est un jeu de rôle grandeur nature.")]
    )
    checkpointer = InMemorySaver()
    agent = build_agent(
        model,
        system_prompt="Réponds en français.",
        plugins=(plugin,),
        token_counter=lambda messages: sum(
            len(str(message.content).split()) for message in messages
        ),
        history_token_budget=1000,
        history_low_watermark=0.6,
        checkpointer=checkpointer,
    )
    settings = replace(load_config(Path("config.ini")), session_id="web-search-test")
    output = io.StringIO()
    app = ChatApp(
        agent,
        checkpointer,
        settings,
        Path("fake.gguf"),
        lambda messages: len(messages),
        plugin.tool_icons,
    )
    app.console = Console(file=output, force_terminal=False, width=120)

    app._answer("c'est quoi Kandorya ?")

    state = agent.get_state(app._config()).values
    answer = state["messages"][-1].content
    assert answer.startswith("Kandorya est un jeu de rôle grandeur nature.")
    assert "Sources :" in answer
    assert "Site officiel de Kandorya — https://www.kandorya.net/" in answer
    assert "Source 2 — https://example.com/2" in answer
    assert "Source 3 — https://example.com/3" in answer
    assert "Source 4" not in answer
    rendered = output.getvalue()
    assert "Kandorya est un jeu de rôle grandeur nature." in rendered
    assert "Sources :" in rendered
    assert "Un jeu de rôle grandeur nature en France." not in rendered
    assert "Second excerpt." not in rendered


def test_model_does_not_receive_search_tool_for_greetings_or_weather(monkeypatch):
    monkeypatch.setattr(web_search, "_get_ddgs", lambda: object)
    search_plugin = web_search.load_plugin({})
    weather_plugin = weather.load_plugin({})
    policy = search_plugin.middleware[1]
    tools = [*search_plugin.tools, *weather_plugin.tools]
    exposed_tool_names = []

    for query in (
        "salut toi !",
        "Quel temps fait-il à Couëron ?",
        "Qui est Ada Lovelace ?",
    ):
        message = HumanMessage(content=query)
        request = ModelRequest(
            model=FakeMessagesListChatModel(responses=[AIMessage(content="Réponse")]),
            messages=[message],
            tools=tools,
            state={"messages": [message]},
        )
        policy.wrap_model_call(
            request,
            lambda updated: (
                exposed_tool_names.append([item.name for item in updated.tools])
                or AIMessage(content="Réponse")
            ),
        )

    assert len(exposed_tool_names) == 3
    assert exposed_tool_names == [["get_weather"], ["get_weather"], ["get_weather"]]


def test_searched_answer_preserves_prior_messages(monkeypatch):
    monkeypatch.setattr(web_search, "_get_ddgs", lambda: object)
    plugin = web_search.load_plugin({})
    policy = plugin.middleware[1]
    prior_user = HumanMessage(content="ancienne question")
    prior_answer = AIMessage(content="Ancien contexte à ne pas utiliser")
    current_user = HumanMessage(content="Qui est cette personne ?", id="current-question")
    search_call = AIMessage(
        content=" ",
        tool_calls=[
            {"name": "search_web", "args": {"query": "Qui est cette personne"}, "id": "search-1"}
        ],
        additional_kwargs={"microchat_auto_search_for": "current-question"},
    )
    search_result = ToolMessage(
        content="Résultats frais uniquement", name="search_web", tool_call_id="search-1"
    )
    session_messages = [prior_user, prior_answer, current_user, search_call, search_result]
    system_prompt = "Instructions système conservées"
    request = ModelRequest(
        model=FakeMessagesListChatModel(responses=[AIMessage(content="Réponse")]),
        messages=session_messages,
        system_prompt=system_prompt,
        tools=list(plugin.tools),
        state={"messages": session_messages},
    )
    model_inputs = []

    policy.wrap_model_call(
        request,
        lambda updated: model_inputs.append(updated) or AIMessage(content="Réponse"),
    )

    assert model_inputs[0].messages == session_messages
    assert model_inputs[0].system_prompt == system_prompt
    assert request.state["messages"] == session_messages


def test_follow_up_reuses_search_context_without_old_tool_messages(monkeypatch):
    monkeypatch.setattr(web_search, "_get_ddgs", lambda: object)
    plugin = web_search.load_plugin({})
    policy = plugin.middleware[1]
    previous_user = HumanMessage(content="c'est quoi Kandorya ?", id="question-1")
    search_call = AIMessage(
        content=" ",
        tool_calls=[{"name": "search_web", "args": {"query": "Kandorya"}, "id": "search-1"}],
        additional_kwargs={"microchat_auto_search_for": "question-1"},
    )
    search_result = ToolMessage(
        content=(
            "Question : c'est quoi Kandorya ?\n\nExtraits de recherche :\n"
            "1. Kandorya\n   URL : https://example.com/kandorya\n"
            "   Extrait : Kandorya se déroule près d'Angers."
        ),
        name="search_web",
        tool_call_id="search-1",
    )
    previous_answer = AIMessage(
        content="Kandorya est un jeu de rôle.\n\nSources :\n- Kandorya — https://example.com/kandorya",
        additional_kwargs={"microchat_sources_for": "question-1"},
    )
    follow_up = HumanMessage(content="et c'est où ?", id="question-2")
    messages = [previous_user, search_call, search_result, previous_answer, follow_up]
    request = ModelRequest(
        model=FakeMessagesListChatModel(responses=[AIMessage(content="Réponse")]),
        messages=messages,
        tools=list(plugin.tools),
        state={"messages": messages},
    )
    captured = []

    policy.wrap_model_call(
        request,
        lambda updated: captured.append(updated.messages) or AIMessage(content="Réponse"),
    )

    model_messages = captured[0]
    assert not any(
        isinstance(message, (ToolMessage,))
        or (
            isinstance(message, AIMessage)
            and message.additional_kwargs.get("microchat_auto_search_for")
        )
        for message in model_messages
    )
    assert not any(
        isinstance(message, AIMessage) and "Sources :" in str(message.content)
        for message in model_messages
    )
    assert "Kandorya se déroule près d'Angers." in model_messages[-1].content
    assert "Question : et c'est où ?" in model_messages[-1].content
    assert request.messages == messages


def test_langgraph_preserves_conversation_history_during_search(monkeypatch):
    class CapturingModel(FakeMessagesListChatModel):
        _model_inputs: list = PrivateAttr(default_factory=list)

        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def bind_tools(self, tools, *, tool_choice=None, **kwargs):
            return self

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            self._model_inputs.append(messages)
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    monkeypatch.setattr(web_search, "_get_ddgs", lambda: object)
    monkeypatch.setattr(
        web_search,
        "_fetch_results",
        lambda *_args, **_kwargs: [
            {
                "title": "Ada Lovelace",
                "href": "https://example.com/ada",
                "body": "Ada Lovelace était une mathématicienne britannique.",
            }
        ],
    )
    plugin = web_search.load_plugin({})
    model = CapturingModel(
        responses=[
            AIMessage(content="Ancien contexte de conversation."),
            AIMessage(content="Ada Lovelace était une mathématicienne."),
        ]
    )
    agent = build_agent(
        model,
        system_prompt="Réponds en français.",
        plugins=(plugin,),
        token_counter=lambda messages: sum(
            len(str(message.content).split()) for message in messages
        ),
        history_token_budget=1000,
        history_low_watermark=0.6,
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "search-history-isolation"}}

    agent.invoke({"messages": [HumanMessage(content="Bonjour")]}, config=config)
    agent.invoke({"messages": [HumanMessage(content="Qui est Ada Lovelace ?")]}, config=config)

    search_prompt = model._model_inputs[1]
    assert any(
        isinstance(message, HumanMessage) and message.content == "Qui est Ada Lovelace ?"
        for message in search_prompt
    )
    assert any(
        isinstance(message, ToolMessage)
        and "Ada Lovelace était une mathématicienne britannique." in message.content
        for message in search_prompt
    )
    assert any(
        isinstance(message, AIMessage) and message.content == "Ancien contexte de conversation."
        for message in search_prompt
    )


def test_web_search_plugin_requires_optional_dependency(monkeypatch):
    def missing_dependency() -> Any:
        raise ValueError("Installez ddgs avec l'extra search.")

    monkeypatch.setattr(web_search, "_get_ddgs", missing_dependency)

    with pytest.raises(ValueError, match="extra search"):
        web_search.load_plugin({})


@pytest.mark.parametrize(
    ("configuration", "message"),
    [
        ({"timeout_seconds": "invalid"}, "doivent être entiers"),
        ({"timeout_seconds": "0"}, "timeout_seconds doit être supérieur"),
        ({"max_results": "4"}, "max_results doit être compris"),
        ({"region": " "}, "region ne peut pas être vide"),
    ],
)
def test_web_search_plugin_validates_configuration(configuration, message):
    with pytest.raises(ValueError, match=message):
        web_search.load_plugin(configuration)


def test_fetch_results_passes_search_settings_to_ddgs(monkeypatch):
    calls: dict[str, Any] = {}

    class FakeDDGS:
        def __init__(self, *, timeout: int):
            calls["timeout"] = timeout

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def text(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
            calls.update(query=query, **kwargs)
            return [{"title": "Result", "href": "https://example.com", "body": "Snippet"}]

    monkeypatch.setattr(web_search, "_get_ddgs", lambda: FakeDDGS)

    results = web_search._fetch_results(
        "current question", region="fr-fr", timeout_seconds=6, max_results=4
    )

    assert results[0]["title"] == "Result"
    assert calls == {
        "timeout": 6,
        "query": "current question",
        "region": "fr-fr",
        "safesearch": "moderate",
        "max_results": 3,
        "backend": "auto",
    }


def test_fetch_results_retries_once_after_transient_failure(monkeypatch):
    attempts = 0

    class RetryDDGS:
        def __init__(self, *, timeout: int):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def text(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("transient DDG failure")
            return [{"title": "Result", "href": "https://example.com", "body": "Snippet"}]

    monkeypatch.setattr(web_search, "_get_ddgs", lambda: RetryDDGS)

    results = web_search._fetch_results(
        "Kandorya", region="wt-wt", timeout_seconds=2, max_results=2
    )

    assert attempts == 2
    assert results[0]["title"] == "Result"


def test_fetch_results_treats_no_results_as_empty(monkeypatch):
    class EmptyDDGS:
        def __init__(self, *, timeout: int):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def text(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
            raise RuntimeError("No results found.")

    monkeypatch.setattr(web_search, "_get_ddgs", lambda: EmptyDDGS)

    results = web_search._fetch_results(
        "question rare", region="wt-wt", timeout_seconds=3, max_results=2
    )

    assert results == []


def test_search_tool_returns_citable_results(monkeypatch):
    search_configuration: dict[str, Any] = {}

    def fake_fetch(
        query: str, *, region: str, timeout_seconds: int, max_results: int
    ) -> list[dict[str, Any]]:
        search_configuration.update(
            query=query,
            region=region,
            timeout_seconds=timeout_seconds,
            max_results=max_results,
        )
        return [
            {
                "title": "Source title",
                "href": "https://example.com/source",
                "body": "A useful excerpt.",
            },
            {"title": "Unsafe URL", "href": "javascript:alert(1)", "body": "Ignore me."},
        ]

    monkeypatch.setattr(web_search, "_fetch_results", fake_fetch)
    tool = web_search.make_search_tool(region="fr-fr", timeout_seconds=9, max_results=2)

    result = tool.invoke({"query": "question inconnue"})

    assert "Source title" in result
    assert "https://example.com/source" in result
    assert "A useful excerpt." in result
    assert "Unsafe URL" not in result
    assert search_configuration == {
        "query": "question inconnue",
        "region": "fr-fr",
        "timeout_seconds": 9,
        "max_results": 2,
    }


def test_search_tool_handles_empty_results(monkeypatch):
    monkeypatch.setattr(web_search, "_fetch_results", lambda *_args, **_kwargs: [])
    tool = web_search.make_search_tool(region="wt-wt", timeout_seconds=10, max_results=5)

    assert "Aucun résultat pertinent" in tool.invoke({"query": "question rare"})


def test_format_results_accepts_alternate_result_keys():
    result = web_search._format_results(
        [{"title": "Title", "link": "https://example.com", "snippet": "Text"}]
    )

    assert "https://example.com" in result
    assert "Text" in result


def test_format_sources_keeps_malformed_long_titles_from_merging_results():
    results = [
        {
            "title": "Premier titre — Deuxième titre — " + "Autre résultat " * 20,
            "href": "https://www.example.com/one",
        },
        {"title": "Source distincte", "href": "https://second.example.org/page"},
    ]

    sources = web_search._format_sources(results)

    assert sources == [
        {"title": "example.com", "url": "https://www.example.com/one"},
        {"title": "Source distincte", "url": "https://second.example.org/page"},
    ]
