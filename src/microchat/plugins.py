"""Contrat et découverte des plugins-agent installés."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import Any

from langchain_core.tools import BaseTool


@dataclass(frozen=True)
class AgentPlugin:
    """Contribution d'un plugin au chatbot principal."""

    name: str
    instructions: str
    tools: Sequence[BaseTool]
    tool_icons: Mapping[str, str]
    middleware: Sequence[Any] = ()
    commands: Mapping[str, PluginCommand] | None = None
    context_provider: Callable[[], Sequence[tuple[str, str]]] | None = None


PluginFactory = Callable[[Mapping[str, str]], AgentPlugin]


@dataclass(frozen=True)
class PluginCommand:
    """Commande slash fournie par un plugin actif."""

    usage: str
    description: str
    handler: Callable[[str], str]
    icon: str = ""


def load_plugins(
    names: Sequence[str], configurations: Mapping[str, Mapping[str, str]]
) -> tuple[AgentPlugin, ...]:
    """Charge les plugins demandés depuis le groupe d'entry points Microchat."""
    if len(set(names)) != len(names):
        raise ValueError("La liste des plugins contient un nom en double.")

    available = {point.name: point for point in entry_points(group="microchat.plugins")}
    missing = sorted(set(names) - available.keys())
    if missing:
        raise ValueError(
            f"Plugins non installés : {', '.join(missing)}. "
            "Installez leur paquet ou retirez-les de plugins dans config.ini."
        )

    plugins: list[AgentPlugin] = []
    for name in names:
        plugin = available[name].load()(configurations.get(name, {}))
        if plugin.name != name:
            raise ValueError(f"Le plugin {name} a déclaré le nom inattendu {plugin.name}.")
        tool_names = {tool.name for tool in plugin.tools}
        unknown_icons = set(plugin.tool_icons) - tool_names
        if unknown_icons:
            raise ValueError(
                f"Le plugin {name} déclare des icônes pour des outils inconnus : "
                f"{', '.join(sorted(unknown_icons))}."
            )
        plugins.append(plugin)
    return tuple(plugins)
