"""Agents autonomes : un fichier par agent, outils et graphe regroupés."""

from microchat.agents import weather

# Chaque sous-agent déclare ICON et TOOL_NAMES ; ajoutez-le à ce tuple.
_AGENTS = (weather,)

TOOL_ICONS: dict[str, str] = {
    tool_name: agent.ICON for agent in _AGENTS for tool_name in agent.TOOL_NAMES
}
