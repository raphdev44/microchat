"""Assemblage partagé du chatbot et compatibilité des modèles locaux."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from typing import Any
from uuid import uuid4

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import after_model, before_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, RemoveMessage, ToolCall
from langchain_core.messages.utils import trim_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime

from microchat.plugins import AgentPlugin

_TOOL_CALL_PATTERN = re.compile(r"<tool_call>(.*?)(?:</tool_call>|$)", re.DOTALL)
_THINK_PATTERN = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL)


def visible_text(text: str) -> str:
    """Retire les balises de réflexion et d'appel d'outil du texte affiché."""
    return _TOOL_CALL_PATTERN.sub("", _THINK_PATTERN.sub("", text)).strip()


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


def _parse_text_tool_calls(text: str) -> list[ToolCall]:
    """Extrait les appels d'outil écrits en texte par les modèles locaux."""
    calls: list[ToolCall] = []
    for block in _TOOL_CALL_PATTERN.findall(text):
        try:
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


def build_agent(
    llm: BaseChatModel,
    *,
    system_prompt: str,
    plugins: Sequence[AgentPlugin],
    token_counter: Any,
    history_token_budget: int,
    history_low_watermark: float,
    checkpointer: InMemorySaver,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Assemble le chatbot de base avec les outils et middleware des plugins actifs."""

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
        """Convertit les appels d'outil textuels des modèles locaux en appels structurés."""
        message = state["messages"][-1]
        if not isinstance(message, AIMessage):
            return None
        text = "".join(text_from_content(message.content))
        if message.tool_calls:
            if not text:
                return {
                    "messages": [
                        AIMessage(
                            content=" ",
                            tool_calls=message.tool_calls,
                            additional_kwargs=message.additional_kwargs,
                            response_metadata=message.response_metadata,
                            id=message.id,
                        )
                    ]
                }
            return None
        calls = _parse_text_tool_calls(text)
        if not calls:
            return None
        return {
            "messages": [
                AIMessage(content=visible_text(text) or " ", tool_calls=calls, id=message.id)
            ]
        }

    tools = [tool for plugin in plugins for tool in plugin.tools]
    names = [tool.name for tool in tools]
    if len(set(names)) != len(names):
        raise ValueError("Plusieurs plugins déclarent le même nom d'outil.")
    instructions = [plugin.instructions for plugin in plugins if plugin.instructions]
    middleware = [trim_history]
    middleware.extend(item for plugin in plugins for item in plugin.middleware)
    middleware.append(extract_text_tool_calls)
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt="\n\n".join([system_prompt, *instructions]),
        middleware=middleware,
        checkpointer=checkpointer,
        name="microchat_agent",
    )
