"""Plugin de recherche DuckDuckGo avec extraits et liens de sources."""

from __future__ import annotations

import logging
import re
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

from langchain.agents import AgentState
from langchain.agents.middleware import after_model, before_model, wrap_model_call
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolCall, ToolMessage
from langchain_core.tools import BaseTool, ToolException, tool
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field

from microchat.agents.runtime import text_from_content
from microchat.plugins import AgentPlugin

ICON = "🌐"
TOOL_NAME = "search_web"
MAX_SEARCH_RESULTS = 3
MAX_SOURCE_TITLE_LENGTH = 120
MAX_SNIPPET_LENGTH = 350
_STOPWORDS = frozenset(
    {
        "alors",
        "au",
        "aux",
        "avec",
        "ce",
        "ces",
        "cette",
        "comment",
        "dans",
        "de",
        "des",
        "dire",
        "du",
        "elle",
        "en",
        "est",
        "et",
        "explique",
        "fait",
        "fonctionne",
        "il",
        "je",
        "la",
        "le",
        "les",
        "leur",
        "lui",
        "ma",
        "mais",
        "me",
        "meme",
        "mes",
        "moi",
        "mon",
        "ne",
        "nos",
        "notre",
        "nous",
        "ou",
        "par",
        "pas",
        "plus",
        "pour",
        "quand",
        "que",
        "quel",
        "quelle",
        "qui",
        "quoi",
        "se",
        "signifie",
        "son",
        "sont",
        "sur",
        "ta",
        "te",
        "tes",
        "toi",
        "ton",
        "trouve",
        "tu",
        "un",
        "une",
        "veut",
        "vos",
        "votre",
        "vous",
        "about",
        "did",
        "does",
        "is",
        "tell",
        "the",
        "what",
        "when",
        "where",
        "who",
    }
)
_LOOKUP_PHRASES = (
    r"c['’]est quoi|c['’]est qui|qu['’]est-ce que|qu['’]est-ce qu|qui est|"
    r"que signifie|que veut dire|"
    r"où est|où se trouve|quand a|qui a|comment fonctionne|explique(?:-moi)?|"
    r"what is|who is|where is|when did|what does|tell me about"
)
_SEARCH_REQUEST = re.compile(rf"\b(?:{_LOOKUP_PHRASES})\b", re.IGNORECASE)
_WEATHER_REQUEST = re.compile(
    r"\b(?:m[eé]t[eé]o|quel temps (?:fait|fera|va faire)|temps qu'il fait|"
    r"temp[eé]rature (?:à|de)|pleut(?:-il)?|va-t-il pleuvoir|pr[eé]visions m[eé]t[eé]o)\b",
    re.IGNORECASE,
)
INSTRUCTIONS = (
    "Règle d'or : Réponds directement à la dernière question de l'utilisateur en langage naturel, "
    "en 2 phrases maximum.\n"
    "Surtout ne liste pas les résultats de recherche, n'écris pas de section Sources et n'utilise "
    "pas la première personne (« je », « mon », « notre »).\n"
    "Ne confonds pas des homonymes ou des noms proches et surtout n'invente rien si tu ne sais pas."
)


class SearchQuery(BaseModel):
    """Requête bornée pour la recherche web."""

    query: str = Field(
        min_length=2,
        max_length=500,
        description="Question ou mots-clés précis à rechercher sur le web.",
    )


def _get_ddgs() -> Any:
    """Retourne le client DDGS ou explique comment installer l'extra optionnel."""
    try:
        from ddgs import DDGS
    except ImportError as exc:
        raise ValueError(
            "Le plugin web_search nécessite ddgs. Installez-le avec "
            "`python -m pip install -e '.[search]'`."
        ) from exc
    return DDGS


def _plain_tokens(text: str) -> list[str]:
    """Découpe un texte en mots minuscules sans accents pour comparer requête et extraits."""
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    plain = "".join(char for char in decomposed if not unicodedata.combining(char))
    return re.findall(r"[a-z0-9]{2,}", plain)


def _subject_tokens(query: str) -> list[str]:
    """Retourne les mots significatifs de la question, hors mots-outils."""
    return [token for token in _plain_tokens(query) if token not in _STOPWORDS and len(token) >= 3]


def _trim_snippet(body: str, max_len: int = MAX_SNIPPET_LENGTH) -> str:
    """Tronque proprement un extrait sur une fin de phrase complète."""
    if " : -" in body and body.endswith("..."):
        body = body.split(" : -", 1)[0].strip() + "."
    if len(body) > max_len:
        chunk = body[:max_len]
        last_dot = max(chunk.rfind(". "), chunk.rfind("."), chunk.rfind("! "), chunk.rfind("? "))
        if last_dot >= 120:
            return chunk[: last_dot + 1].strip()
        return chunk.rsplit(" ", 1)[0] + "…"
    return body


def _clean_and_rank_results(
    query: str, results: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Désimbrique les extraits fusionnés, les tronque et écarte les quasi-homonymes."""
    raw_titles = [" ".join(str(item.get("title") or "").split()) for item in results]
    raw_bodies = [
        " ".join(str(item.get("body") or item.get("snippet") or "").split()) for item in results
    ]
    subject = _subject_tokens(query)
    def_pattern = (
        re.compile(
            rf"\b{re.escape(subject[-1])}\b\s+(?:est|sont|fut|désigne|consiste|se\s+situe|se\s+déroule)\b",
            re.IGNORECASE,
        )
        if subject
        else None
    )

    cleaned: list[dict[str, Any]] = []
    for index, item in enumerate(results):
        title = raw_titles[index]
        body = raw_bodies[index]
        for other_index, other_title in enumerate(raw_titles):
            if index != other_index and len(other_title) >= 10:
                pos = title.find(other_title)
                if pos > 0:
                    title = title[:pos].strip()
        for other_index, other_body in enumerate(raw_bodies):
            if index != other_index and len(other_body) >= 20:
                pos = body.find(other_body)
                if pos > 0:
                    body = body[:pos].strip()
        merged_title = (
            title != raw_titles[index] or len(raw_titles[index]) > MAX_SOURCE_TITLE_LENGTH
        )
        def_match = def_pattern.search(body) if def_pattern is not None else None
        if len(body) > MAX_SNIPPET_LENGTH and def_match is not None:
            body = body[def_match.start() :]
        elif merged_title and "..." in body:
            ellipsis_pos = body.find("...")
            if ellipsis_pos > 0:
                body = body[: ellipsis_pos + 3].strip()
        body = _trim_snippet(body)
        updated = dict(item)
        if title:
            updated["title"] = title
        if "body" in updated or "snippet" not in updated:
            updated["body"] = body
        else:
            updated["snippet"] = body
        cleaned.append(updated)

    if not subject:
        return cleaned

    scored: list[tuple[int, int, int, dict[str, Any]]] = []
    for index, item in enumerate(cleaned):
        url = str(item.get("href") or item.get("url") or item.get("link") or "")
        body = str(item.get("body") or item.get("snippet") or "")
        haystack = set(_plain_tokens(f"{item.get('title', '')} {url} {body}"))
        score = sum(1 for token in subject if token in haystack)
        has_definition = int(bool(def_pattern and def_pattern.search(body)))
        scored.append((score, has_definition, -index, item))

    best_score = max((score for score, _, _, _ in scored), default=0)
    if best_score > 0:
        kept = [entry for entry in scored if entry[0] == best_score]
        kept.sort(reverse=True)
        return [item for _, _, _, item in kept]
    return cleaned


def _fetch_results(
    query: str, *, region: str, timeout_seconds: int, max_results: int
) -> list[dict[str, Any]]:
    """Interroge DuckDuckGo en utilisant le paquet optionnel ddgs."""
    ddgs = _get_ddgs()
    result_limit = min(max_results, MAX_SEARCH_RESULTS)
    for attempt in range(2):
        try:
            with ddgs(timeout=timeout_seconds) as search:
                results = list(
                    search.text(
                        query,
                        region=region,
                        safesearch="moderate",
                        max_results=result_limit,
                        backend="auto",
                    )
                )
            if results:
                return _clean_and_rank_results(query, results[:MAX_SEARCH_RESULTS])
        except Exception as exc:
            if str(exc).strip().casefold() in {"no results found", "no results found."}:
                return []
            if attempt == 1:
                raise
    return []


def _format_results(results: Sequence[Mapping[str, Any]]) -> str:
    """Formate des résultats en extraits citables par le modèle."""
    if not results:
        return "Aucun résultat pertinent trouvé. Essaie une autre formulation."

    formatted: list[str] = []
    for index, result in enumerate(results[:MAX_SEARCH_RESULTS], start=1):
        title = " ".join(str(result.get("title") or "Sans titre").split())
        url = str(result.get("href") or result.get("url") or result.get("link") or "").strip()
        snippet = " ".join(str(result.get("body") or result.get("snippet") or "").split())
        if not url.startswith(("https://", "http://")):
            continue
        if len(title) > MAX_SOURCE_TITLE_LENGTH:
            title = (urlsplit(url).hostname or "Source web").removeprefix("www.")
        if len(snippet) > MAX_SNIPPET_LENGTH:
            snippet = snippet[:MAX_SNIPPET_LENGTH].rsplit(" ", 1)[0] + "…"
        formatted.append(f"{index}. {title}\n   URL : {url}\n   Extrait : {snippet}")

    return "\n\n".join(formatted) or "Aucun résultat avec une URL valide n'a été trouvé."


def _format_sources(results: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    """Extrait au plus trois titres et URL valides pour l'affichage final."""
    sources: list[dict[str, str]] = []
    for result in results[:MAX_SEARCH_RESULTS]:
        title = " ".join(str(result.get("title") or "Sans titre").split())
        url = str(result.get("href") or result.get("url") or result.get("link") or "").strip()
        if url.startswith(("https://", "http://")):
            if len(title) > MAX_SOURCE_TITLE_LENGTH:
                title = urlsplit(url).hostname or "Source web"
                title = title.removeprefix("www.")
            sources.append({"title": title, "url": url})
    return sources


def _requires_lookup(query: str) -> bool:
    """Repère les demandes factuelles explicites qui doivent être recherchées avant réponse."""
    return not _WEATHER_REQUEST.search(query) and bool(_SEARCH_REQUEST.search(query))


def _strip_sources(content: str) -> str:
    """Retire le bloc Sources ajouté pour l'affichage afin de ne pas polluer le contexte LLM."""
    marker = "\n\nSources :\n"
    if marker in content:
        return content.split(marker, 1)[0].rstrip()
    if content.startswith("Sources :\n"):
        return ""
    return content


def _make_search_policy_middleware() -> Any:
    """Masque l'outil de recherche au modèle et nettoie les recherches obsolètes de l'historique."""

    @wrap_model_call
    def restrict_search_tool(request: Any, handler: Any) -> Any:
        user_message = next(
            (item for item in reversed(request.messages) if isinstance(item, HumanMessage)),
            None,
        )
        query = " ".join(text_from_content(user_message.content)).strip() if user_message else ""
        current_key = (user_message.id or query) if user_message is not None else None
        latest_search = next(
            (
                item
                for item in reversed(request.messages)
                if isinstance(item, AIMessage)
                and item.additional_kwargs.get("microchat_auto_search_for")
            ),
            None,
        )
        active_search = (
            latest_search
            if latest_search is not None
            and current_key is not None
            and latest_search.additional_kwargs.get("microchat_auto_search_for") == current_key
            else None
        )
        latest_call_ids = {
            call["id"]
            for call in (latest_search.tool_calls if latest_search is not None else ())
            if "id" in call
        }
        latest_excerpts = ""
        if active_search is None and latest_call_ids:
            latest_tool = next(
                (
                    item
                    for item in reversed(request.messages)
                    if isinstance(item, ToolMessage) and item.tool_call_id in latest_call_ids
                ),
                None,
            )
            if latest_tool is not None and isinstance(latest_tool.content, str):
                first_block = latest_tool.content.split("\n\n2. ", 1)[0]
                if "Extrait : " in first_block:
                    latest_excerpts = first_block.split("Extrait : ", 1)[1].strip()

        obsolete_call_ids = {
            call["id"]
            for item in request.messages
            if isinstance(item, AIMessage)
            and item.additional_kwargs.get("microchat_auto_search_for")
            and item is not active_search
            for call in item.tool_calls
            if "id" in call
        }
        cleaned_messages: list[BaseMessage] = []
        changed = False
        for item in request.messages:
            if (
                isinstance(item, AIMessage)
                and item.additional_kwargs.get("microchat_auto_search_for")
                and item is not active_search
            ):
                changed = True
                continue
            if isinstance(item, ToolMessage) and item.tool_call_id in obsolete_call_ids:
                changed = True
                continue
            if (
                isinstance(item, AIMessage)
                and item.additional_kwargs.get("microchat_sources_for")
                and isinstance(item.content, str)
            ):
                stripped = _strip_sources(item.content)
                if stripped != item.content:
                    item = item.model_copy(update={"content": stripped})
                    changed = True
            if (
                user_message is not None
                and item is user_message
                and latest_excerpts
                and isinstance(item.content, str)
            ):
                item = item.model_copy(
                    update={"content": f"Contexte : {latest_excerpts}\n\nQuestion : {item.content}"}
                )
                changed = True
            cleaned_messages.append(item)

        overrides: dict[str, Any] = {
            "tools": [item for item in request.tools if item.name != TOOL_NAME],
        }
        if changed:
            overrides["messages"] = cleaned_messages
        if latest_search is not None:
            overrides["model_settings"] = {**request.model_settings, "stop": ["\n\n"]}
        request = request.override(**overrides)
        return handler(request)

    return restrict_search_tool


def _search_terms(query: str) -> str:
    """Nettoie la ponctuation finale sans perdre le contexte de la question."""
    return " ".join(query.split()).strip(" \t\r\n?.!,;:«»\"'") or query.strip()


def _search_data(
    query: str, *, region: str, timeout_seconds: int, max_results: int
) -> tuple[str, list[dict[str, str]]]:
    """Retourne les extraits pour le modèle et les sources pour l'interface."""
    try:
        results = _fetch_results(
            query,
            region=region,
            timeout_seconds=timeout_seconds,
            max_results=max_results,
        )
    except ImportError as exc:
        raise ToolException(
            "Le plugin web_search requiert ddgs. Installez-le avec "
            "`python -m pip install -e '.[search]'`."
        ) from exc
    except Exception as exc:
        raise ToolException(f"La recherche DuckDuckGo a échoué ({type(exc).__name__}).") from exc
    return _format_results(results), _format_sources(results)


def make_search_tool(*, region: str, timeout_seconds: int, max_results: int) -> BaseTool:
    """Construit l'outil de recherche avec les paramètres de configuration du plugin."""

    @tool(args_schema=SearchQuery)
    def search_web(query: str) -> str:
        """Recherche des pages web pertinentes et retourne leurs extraits et URL."""
        result, _sources = _search_data(
            query,
            region=region,
            timeout_seconds=timeout_seconds,
            max_results=max_results,
        )
        return result

    search_web.handle_tool_error = True
    return search_web


def _make_lookup_middleware(search: Callable[[str], tuple[str, list[dict[str, str]]]]) -> Any:
    """Ajoute les extraits et les sources à l'historique avant l'appel du modèle."""

    @before_model(can_jump_to=["end"])
    def search_factual_question(state: AgentState, runtime: Runtime[None]) -> dict[str, Any] | None:
        messages = state["messages"]
        user_message = next(
            (message for message in reversed(messages) if isinstance(message, HumanMessage)),
            None,
        )
        if user_message is None:
            return None

        query = " ".join(text_from_content(user_message.content)).strip()
        if not query or not _requires_lookup(query):
            return None

        search_key = user_message.id or query
        if any(
            isinstance(message, AIMessage)
            and message.additional_kwargs.get("microchat_auto_search_for") == search_key
            for message in messages
        ):
            return None

        tool_call_id = f"call_{uuid4().hex}"
        search_query = _search_terms(query)[:500]
        try:
            result, sources = search(search_query)
        except ToolException as exc:
            result, sources = str(exc), []
        tool_content = (
            f"Question : {query}\n\nExtraits de recherche :\n{result}" if sources else result
        )
        return {
            "messages": [
                AIMessage(
                    content=" ",
                    tool_calls=[
                        ToolCall(
                            name=TOOL_NAME,
                            args={"query": search_query},
                            id=tool_call_id,
                            type="tool_call",
                        )
                    ],
                    additional_kwargs={
                        "microchat_auto_search_for": search_key,
                        "microchat_search_sources": sources,
                    },
                ),
                ToolMessage(
                    content=tool_content,
                    name=TOOL_NAME,
                    tool_call_id=tool_call_id,
                    additional_kwargs={"microchat_hide_output": True},
                ),
            ],
        }

    return search_factual_question


def _make_sources_middleware() -> Any:
    """Ajoute les sources à la réponse sans les faire rédiger par le modèle."""

    @after_model
    def append_sources(
        state: AgentState, runtime: Runtime[None]
    ) -> dict[str, list[AIMessage]] | None:
        messages = state["messages"]
        message = messages[-1]
        if not isinstance(message, AIMessage) or message.tool_calls:
            return None
        user_message = next(
            (item for item in reversed(messages) if isinstance(item, HumanMessage)), None
        )
        if user_message is None:
            return None
        query = " ".join(text_from_content(user_message.content)).strip()
        search_key = user_message.id or query
        if message.additional_kwargs.get("microchat_sources_for") == search_key:
            return None
        search_message = next(
            (
                item
                for item in reversed(messages[:-1])
                if isinstance(item, AIMessage)
                and item.additional_kwargs.get("microchat_auto_search_for") == search_key
            ),
            None,
        )
        sources = (
            search_message.additional_kwargs.get("microchat_search_sources", [])
            if search_message is not None
            else []
        )
        if not sources:
            return None
        source_lines = ["Sources :"]
        source_lines.extend(f"- {source['title']} — {source['url']}" for source in sources)
        answer = "".join(text_from_content(message.content)).rstrip()
        content = f"{answer}\n\n{'\n'.join(source_lines)}" if answer else "\n".join(source_lines)
        updated = message.model_copy(
            update={
                "content": content,
                "additional_kwargs": {
                    **message.additional_kwargs,
                    "microchat_sources_for": search_key,
                },
            }
        )
        return {"messages": [updated]}

    return append_sources


def load_plugin(config: Mapping[str, str]) -> AgentPlugin:
    """Construit le plugin web-search depuis la section [plugin:web_search]."""
    try:
        timeout_seconds = int(config.get("timeout_seconds", "5"))
        max_results = int(config.get("max_results", str(MAX_SEARCH_RESULTS)))
    except ValueError as exc:
        raise ValueError(
            "plugin:web_search timeout_seconds et max_results doivent être entiers."
        ) from exc

    region = config.get("region", "wt-wt").strip()
    if timeout_seconds < 1:
        raise ValueError("plugin:web_search timeout_seconds doit être supérieur à zéro.")
    if not 1 <= max_results <= MAX_SEARCH_RESULTS:
        raise ValueError(
            f"plugin:web_search max_results doit être compris entre 1 et {MAX_SEARCH_RESULTS}."
        )
    if not region:
        raise ValueError("plugin:web_search region ne peut pas être vide.")
    _get_ddgs()
    for logger_name in ("ddgs", "ddgs.ddgs", "ddgs.http_client", "primp"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    search_tool = make_search_tool(
        region=region,
        timeout_seconds=timeout_seconds,
        max_results=max_results,
    )

    def search(query: str) -> tuple[str, list[dict[str, str]]]:
        return _search_data(
            query,
            region=region,
            timeout_seconds=timeout_seconds,
            max_results=max_results,
        )

    return AgentPlugin(
        name="web_search",
        instructions=INSTRUCTIONS,
        tools=(search_tool,),
        tool_icons={TOOL_NAME: ICON},
        middleware=(
            _make_lookup_middleware(search),
            _make_search_policy_middleware(),
            _make_sources_middleware(),
        ),
    )
