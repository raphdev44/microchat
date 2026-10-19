from __future__ import annotations

from typing import Any

import pytest
from langchain.agents.middleware.types import ModelRequest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from microchat.agents import weather, web_search
from microchat.agents.runtime import build_agent


def test_web_search_plugin_declares_tool_and_icon(monkeypatch):
    monkeypatch.setattr(web_search, "_get_ddgs", lambda: object)

    plugin = web_search.load_plugin({"region": "us-en", "timeout_seconds": "8", "max_results": "3"})

    assert plugin.name == "web_search"
    assert plugin.tools[0].name == "search_web"
    assert plugin.tool_icons == {"search_web": web_search.ICON}
    assert len(plugin.middleware) == 2


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


def test_pre_search_returns_sources_without_calling_model(monkeypatch):
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
            }
        ],
    )
    plugin = web_search.load_plugin({})
    model = FixedResponseModel(responses=[AIMessage(content="Réponse inventée du modèle")])
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

    state = agent.invoke(
        {"messages": [HumanMessage(content="c'est quoi Kandorya ?")]},
        config={"configurable": {"thread_id": "web-search-test"}},
    )

    assert "Site officiel de Kandorya" in state["messages"][-1].content
    assert "https://www.kandorya.net/" in state["messages"][-1].content
    assert "Réponse inventée" not in state["messages"][-1].content


def test_model_does_not_receive_search_tool_for_greetings_or_weather(monkeypatch):
    monkeypatch.setattr(web_search, "_get_ddgs", lambda: object)
    search_plugin = web_search.load_plugin({})
    weather_plugin = weather.load_plugin({})
    policy = search_plugin.middleware[-1]
    tools = [*search_plugin.tools, *weather_plugin.tools]
    exposed_tool_names = []

    for query in (
        "salut toi !",
        "Quel temps fait-il à Couëron ?",
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

    assert len(exposed_tool_names) == 2
    assert exposed_tool_names == [["get_weather"], ["get_weather"]]


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
        ({"max_results": "11"}, "max_results doit être compris"),
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
        "max_results": 4,
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
