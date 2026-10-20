"""Application microchat : configuration INI, modèle local, interface et sessions."""

from __future__ import annotations

import argparse
import configparser
import logging
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Any

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.graph.state import CompiledStateGraph
from rich.align import Align
from rich.columns import Columns
from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from microchat import __version__
from microchat.agents.runtime import build_agent as build_plugin_agent
from microchat.agents.runtime import text_from_content, visible_text
from microchat.plugins import load_plugins

logger = logging.getLogger(__name__)
CONFIG_PATH = Path.cwd() / "config.ini"


@dataclass(frozen=True)
class AppConfig:
    """Valeurs de fonctionnement lues dans ``config.ini``."""

    model_repo_id: str
    model_path: Path | None
    models_dir: Path
    n_threads: int
    n_ctx: int
    n_batch: int
    use_mmap: bool
    use_mlock: bool
    warm_up: bool
    flash_attn: bool
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    repeat_penalty: float
    system_prompt: str
    plugins: tuple[str, ...]
    plugin_configurations: Mapping[str, Mapping[str, str]]
    prompt_reserve_tokens: int
    history_low_watermark: float
    session_id: str
    render_refresh_hz: int
    render_interval_seconds: float
    verbose: bool

    @property
    def history_token_budget(self) -> int:
        """Budget restant pour les échanges après la réponse et la marge prompt."""
        return self.n_ctx - self.max_tokens - self.prompt_reserve_tokens


def load_config(path: Path = CONFIG_PATH) -> AppConfig:
    """Charge et valide les paramètres de l'application depuis un fichier INI."""
    parser = configparser.ConfigParser(interpolation=None)
    if not path.is_file():
        raise ValueError(f"Fichier de configuration introuvable : {path}")
    with path.open(encoding="utf-8") as config_file:
        parser.read_file(config_file)
    if not parser.has_section("microchat"):
        raise ValueError(f"La section [microchat] manque dans {path}.")

    section = parser["microchat"]
    required = {
        "model_repo_id",
        "model_path",
        "models_dir",
        "n_threads",
        "n_ctx",
        "n_batch",
        "use_mmap",
        "use_mlock",
        "warm_up",
        "flash_attn",
        "max_tokens",
        "temperature",
        "top_p",
        "top_k",
        "repeat_penalty",
        "system_prompt",
        "plugins",
        "prompt_reserve_tokens",
        "history_low_watermark",
        "session_id",
        "render_refresh_hz",
        "render_interval_seconds",
        "verbose",
    }
    missing = sorted(required - set(section))
    unknown = sorted(set(section) - required)
    if missing:
        raise ValueError(f"Paramètres manquants dans [microchat] : {', '.join(missing)}")
    if unknown:
        raise ValueError(f"Paramètres inconnus dans [microchat] : {', '.join(unknown)}")
    plugin_sections = {
        name for name in parser.sections() if name.startswith("plugin:") and len(name) > 7
    }
    unknown_sections = set(parser.sections()) - {"microchat"} - plugin_sections
    if unknown_sections:
        raise ValueError(f"Sections inconnues : {', '.join(sorted(unknown_sections))}")

    def read_int(name: str) -> int:
        value = section.getint(name)
        if value is None:
            raise ValueError(f"Le paramètre {name} est obligatoire.")
        return value

    def read_float(name: str) -> float:
        value = section.getfloat(name)
        if value is None:
            raise ValueError(f"Le paramètre {name} est obligatoire.")
        return value

    def read_bool(name: str) -> bool:
        value = section.getboolean(name)
        if value is None:
            raise ValueError(f"Le paramètre {name} est obligatoire.")
        return value

    model_path = section["model_path"].strip()
    settings = AppConfig(
        model_repo_id=section["model_repo_id"].strip(),
        model_path=Path(model_path).expanduser() if model_path else None,
        models_dir=Path(section["models_dir"]).expanduser(),
        n_threads=read_int("n_threads"),
        n_ctx=read_int("n_ctx"),
        n_batch=read_int("n_batch"),
        use_mmap=read_bool("use_mmap"),
        use_mlock=read_bool("use_mlock"),
        warm_up=read_bool("warm_up"),
        flash_attn=read_bool("flash_attn"),
        max_tokens=read_int("max_tokens"),
        temperature=read_float("temperature"),
        top_p=read_float("top_p"),
        top_k=read_int("top_k"),
        repeat_penalty=read_float("repeat_penalty"),
        system_prompt=section["system_prompt"],
        plugins=tuple(name.strip() for name in section["plugins"].split(",") if name.strip()),
        plugin_configurations={
            name.removeprefix("plugin:"): dict(parser[name]) for name in plugin_sections
        },
        prompt_reserve_tokens=read_int("prompt_reserve_tokens"),
        history_low_watermark=read_float("history_low_watermark"),
        session_id=section["session_id"].strip(),
        render_refresh_hz=read_int("render_refresh_hz"),
        render_interval_seconds=read_float("render_interval_seconds"),
        verbose=read_bool("verbose"),
    )
    if not settings.model_repo_id:
        raise ValueError("model_repo_id ne peut pas être vide.")
    if settings.n_threads < 0 or settings.n_ctx < 512 or settings.n_batch < 1:
        raise ValueError("n_threads doit être >= 0, n_ctx >= 512 et n_batch >= 1.")
    if settings.max_tokens < 1 or settings.history_token_budget < 1:
        raise ValueError("Le budget du contexte doit laisser au moins un token d'historique.")
    if not 0 <= settings.temperature <= 2:
        raise ValueError("temperature doit être comprise entre 0 et 2.")
    if not 0 < settings.top_p <= 1 or settings.top_k < 0 or settings.repeat_penalty < 0:
        raise ValueError("top_p, top_k ou repeat_penalty invalide.")
    if not 0 < settings.history_low_watermark <= 1:
        raise ValueError("history_low_watermark doit être dans ]0, 1].")
    if settings.render_refresh_hz < 1 or settings.render_interval_seconds < 0:
        raise ValueError("Les réglages de rafraîchissement Rich sont invalides.")
    return settings


def _select_gguf(repo_id: str, filenames: Sequence[str]) -> str:
    """Prend le GGUF le plus léger courant, ou l'unique GGUF du dépôt."""
    files = sorted(name for name in filenames if name.lower().endswith(".gguf"))
    if not files:
        raise ValueError(f"Aucun fichier GGUF trouvé dans le dépôt {repo_id}.")
    for quantization in ("Q4_K_M", "Q4_K_S", "Q4_0", "Q5_K_M", "Q5_K_S", "Q6_K", "Q8_0", "F16"):
        for filename in files:
            if quantization.lower() in filename.lower():
                return filename
    return files[0]


def resolve_model_path(settings: AppConfig) -> Path:
    """Renvoie un modèle local ou découvre puis télécharge un GGUF du Hub."""
    if settings.model_path is not None:
        if not settings.model_path.is_file():
            raise ValueError(f"Modèle local introuvable : {settings.model_path}")
        return settings.model_path

    repo_dir = settings.models_dir / settings.model_repo_id.replace("/", "--")
    cached = sorted(repo_dir.glob("*.gguf"))
    if cached:
        return repo_dir / _select_gguf(settings.model_repo_id, [path.name for path in cached])

    from huggingface_hub import HfApi, hf_hub_download

    try:
        filename = _select_gguf(
            settings.model_repo_id,
            HfApi().list_repo_files(settings.model_repo_id, repo_type="model"),
        )
        logger.info("Téléchargement de %s/%s", settings.model_repo_id, filename)
        return Path(
            hf_hub_download(
                repo_id=settings.model_repo_id,
                filename=filename,
                local_dir=repo_dir,
            )
        )
    except Exception as exc:
        raise RuntimeError(f"Impossible de récupérer un GGUF depuis le Hub : {exc}") from exc


def _disable_chat_thinking(llm: Any) -> None:
    """Désactive le mode réflexion des templates Jinja qui supportent enable_thinking."""
    client = getattr(llm, "client", None)
    handlers = getattr(client, "_chat_handlers", None)
    chat_format = getattr(client, "chat_format", None)
    handler = handlers.get(chat_format) if isinstance(handlers, Mapping) else None
    for cell in getattr(handler, "__closure__", None) or ():
        template = getattr(cell.cell_contents, "_environment", None)
        for target in (template, getattr(template, "environment", None)):
            globals_map = getattr(target, "globals", None)
            if isinstance(globals_map, dict):
                globals_map["enable_thinking"] = False


def build_llm(settings: AppConfig, model_path: Path) -> Any:
    """Construit le modèle llama.cpp à partir des paramètres INI."""
    import psutil
    from langchain_community.chat_models import ChatLlamaCpp

    threads = settings.n_threads or psutil.cpu_count(logical=False) or os.cpu_count() or 1
    llm = ChatLlamaCpp(
        model_path=str(model_path),
        n_gpu_layers=0,
        n_threads=threads,
        n_ctx=settings.n_ctx,
        n_batch=settings.n_batch,
        use_mmap=settings.use_mmap,
        use_mlock=settings.use_mlock,
        max_tokens=settings.max_tokens,
        temperature=settings.temperature,
        top_p=settings.top_p,
        top_k=settings.top_k,
        repeat_penalty=settings.repeat_penalty,
        verbose=settings.verbose,
        disable_streaming="tool_calling",
        model_kwargs={"n_threads_batch": threads, "flash_attn": settings.flash_attn},
    )
    if "/no_think" in settings.system_prompt:
        _disable_chat_thinking(llm)
    return llm


def make_token_counter(llm: Any) -> Callable[[Sequence[BaseMessage]], int]:
    """Compte les tokens avec le tokenizer natif de llama.cpp."""
    tokenize = llm.client.tokenize

    @lru_cache(maxsize=2048)
    def count_text(text: str) -> int:
        return len(tokenize(text.encode("utf-8"), add_bos=False, special=False))

    def count(messages: Sequence[BaseMessage]) -> int:
        return 3 + sum(
            count_text(
                message.content if isinstance(message.content, str) else str(message.content)
            )
            + 4
            for message in messages
        )

    return count


def _parse_command(line: str) -> str | None:
    """Retourne une commande connue, ``unknown`` ou ``None`` si c'est un message."""
    if not line.startswith("/"):
        return None
    name = line[1:].split(maxsplit=1)[0].lower() if len(line) > 1 else ""
    aliases = {
        "help": "help",
        "h": "help",
        "?": "help",
        "clear": "clear",
        "reset": "clear",
        "info": "info",
        "quit": "quit",
        "exit": "quit",
        "q": "quit",
    }
    return aliases.get(name, "unknown")


class ChatApp:
    """Boucle terminal, rendu Rich et adaptateur de l'agent LangGraph."""

    def __init__(
        self,
        agent: CompiledStateGraph[Any, Any, Any, Any],
        checkpointer: InMemorySaver,
        settings: AppConfig,
        model_path: Path,
        token_counter: Callable[[Sequence[BaseMessage]], int],
        tool_icons: Mapping[str, str] | None = None,
    ) -> None:
        self.agent = agent
        self.checkpointer = checkpointer
        self.settings = settings
        self.model_path = model_path
        self.token_counter = token_counter
        self.tool_icons = dict(tool_icons or {})
        self.console = Console()
        self.session_id = settings.session_id

    def _config(self) -> RunnableConfig:
        return RunnableConfig(configurable={"thread_id": self.session_id})

    def run(self) -> None:
        """Lance la session interactive jusqu'à /quit, Ctrl+D ou fin de flux."""
        info = {
            "Modèle": self.model_path.name,
            "Threads CPU": str(self.settings.n_threads or "auto"),
            "Contexte": str(self.settings.n_ctx),
            "Plugins": ", ".join(self.settings.plugins) or "aucun",
        }
        banner = Group(
            Text(f"microchat v{__version__} — chatbot local 100 % CPU"),
            Text("Tapez /help pour l'aide, /quit pour quitter.", style="dim"),
            Text(f"Modèle : {info['Modèle']}", style="dim"),
        )
        self.console.print(Panel(banner, border_style="cyan", expand=False))

        while True:
            try:
                line = self.console.input("[bold green]Vous ›[/] ").strip()
            except EOFError:
                break
            except KeyboardInterrupt:
                self.console.print()
                self.console.print("[dim]Ctrl+C ignoré : utilisez /quit ou Ctrl+D pour quitter.[/]")
                continue
            if not line:
                continue

            command = _parse_command(line)
            if command is None:
                self._answer(line)
            elif command == "quit":
                break
            elif command == "help":
                self._show_help()
            elif command == "clear":
                self.checkpointer.delete_thread(self.session_id)
                self.console.print("[dim]Historique effacé.[/]")
            elif command == "info":
                self._show_info(info)
            else:
                self.console.print(
                    f"[yellow]Commande inconnue : {line.split()[0]} (essayez /help)[/]"
                )
        self.console.print("[dim]À bientôt ![/]")

    def _show_help(self) -> None:
        table = Table(show_header=False, box=None, padding=(0, 2))
        for usage, description in (
            ("/help", "Affiche cette aide"),
            ("/clear", "Efface l'historique de la conversation"),
            ("/info", "Affiche les informations du modèle"),
            ("/quit", "Quitte l'application (ou Ctrl+D)"),
        ):
            table.add_row(f"[cyan]{usage}[/]", description)
        self.console.print(Panel(table, title="Aide", border_style="blue", expand=False))

    def _show_info(self, info: dict[str, str]) -> None:
        table = Table(show_header=False, box=None, padding=(0, 2))
        for label, value in info.items():
            table.add_row(f"[bold]{label}[/]", value)
        table.add_row("[bold]Messages en mémoire[/]", str(self._message_count()))
        self.console.print(Panel(table, title="Configuration", border_style="blue", expand=False))

    def _message_count(self) -> int:
        messages = self.agent.get_state(self._config()).values.get("messages", [])
        return sum(
            isinstance(message, (HumanMessage, ToolMessage))
            or (isinstance(message, AIMessage) and not message.tool_calls)
            for message in messages
        )

    def _answer(self, question: str) -> None:
        try:
            self._show_answer(question)
        except KeyboardInterrupt:
            self.console.print("[yellow]Génération interrompue (ce tour n'est pas mémorisé).[/]")
        except Exception as exc:
            logger.exception("Échec de génération")
            self.console.print(Text(f"Erreur pendant la génération : {exc}", style="bold red"))

    def _show_answer(self, question: str) -> None:
        config = self._config()
        previous = list(self.agent.get_state(config).values.get("messages", []))
        self.console.print("[bold cyan]Assistant ›[/]")
        parts: list[str] = []
        chunks = self.agent.stream(
            {"messages": [HumanMessage(content=question)]},
            config=config,
            stream_mode="messages",
        )
        start = perf_counter()
        first_token_at = start
        last_render = 0.0
        spinner = Spinner("dots", text=Text("Réflexion…", style="dim"))
        tool_waiting = False
        icons: list[str] = []

        def render(body: RenderableType) -> RenderableType:
            return Group(Text(" ".join(icons)), body) if icons else body

        def tool_result(message: ToolMessage, text: str) -> RenderableType:
            artifact = message.artifact
            panel = artifact.get("microchat_panel") if isinstance(artifact, dict) else None
            if isinstance(panel, dict) and panel.get("kind") in ("card", "panel"):
                tiles = panel.get("tiles")
                body: RenderableType = Markdown(panel["body"])
                panel_width = None
                if tiles:
                    weekly = panel["kind"] == "panel"
                    tile_renderables: list[RenderableType] = [
                        Align.center(Text.from_markup(tile), vertical="middle", height=3)
                        for tile in tiles
                    ]
                    if weekly:
                        tile_renderables = [Padding(tile, (1, 1)) for tile in tile_renderables]
                        tile_width = max(
                            self.console.measure(tile).maximum for tile in tile_renderables
                        )
                        column_count = min(
                            len(tiles), max(1, (self.console.width - 3) // (tile_width + 1))
                        )
                        content_width = column_count * (tile_width + 1) - 1
                        content_width = max(
                            content_width,
                            self.console.measure(Text(panel["title"])).maximum,
                            *(self.console.measure(Text(note)).maximum
                              for note in panel.get("notes", [])),
                        )
                        panel_width = min(self.console.width, content_width + 4)
                    else:
                        panel_width = min(self.console.width, 64)
                    body = Group(
                        Columns(
                            tile_renderables,
                            equal=weekly,
                            expand=not weekly,
                            padding=(0, 1),
                        ),
                        *(Text(note, style="dim") for note in panel.get("notes", [])),
                    )
                return Panel(
                    body,
                    title=Text(panel["title"]),
                    subtitle=Text(panel.get("subtitle", "°C")) if tiles else None,
                    border_style="cyan",
                    expand=not tiles and panel["kind"] == "panel",
                    padding=(0, 1) if tiles and panel["kind"] == "panel" else (1, 1),
                    width=panel_width,
                )
            return Markdown(text) if text else spinner

        try:
            with Live(
                spinner,
                console=self.console,
                refresh_per_second=self.settings.render_refresh_hz,
                vertical_overflow="ellipsis",
            ) as live:
                for chunk, _metadata in chunks:
                    if isinstance(chunk, ToolMessage):
                        icon = self.tool_icons.get(chunk.name or "")
                        if icon and icon not in icons:
                            icons.append(icon)
                        if icon:
                            spinner = Spinner(
                                "dots", text=Text("Traitement du résultat…", style="dim")
                            )
                            live.update(render(spinner))
                        if chunk.additional_kwargs.get("microchat_hide_output"):
                            spinner = Spinner("dots", text=Text("Recherche web…", style="dim"))
                            live.update(render(spinner))
                            continue
                        result = visible_text("".join(text_from_content(chunk.content)))
                        live.update(render(tool_result(chunk, result)))
                        continue
                    if not isinstance(chunk, (AIMessage, AIMessageChunk)):
                        continue
                    text = "".join(text_from_content(chunk.content))
                    tool_call_chunks = getattr(chunk, "tool_call_chunks", [])
                    tool_calls = getattr(chunk, "tool_calls", [])
                    if not tool_waiting and (
                        "<tool_call>" in "".join(parts) + text
                        or any(
                            tool_call.get("name") in self.tool_icons
                            for tool_call in [*tool_call_chunks, *tool_calls]
                        )
                    ):
                        tool_waiting = True
                        spinner = Spinner("dots", text=Text("Appel d'un outil…", style="dim"))
                        live.update(render(spinner))
                    if tool_call_chunks or tool_calls:
                        continue
                    if not text:
                        continue
                    now = perf_counter()
                    if not parts:
                        first_token_at = now
                    parts.append(text)
                    if now - last_render >= self.settings.render_interval_seconds:
                        answer = visible_text("".join(parts))
                        live.update(render(Markdown(answer) if answer else spinner))
                        last_render = now
                # Le message final peut être un résultat d'outil direct ou réécrit par un hook.
                final = self.agent.get_state(config).values.get("messages", [])
                last = final[-1] if final else None
                answer = (
                    visible_text("".join(text_from_content(last.content)))
                    if isinstance(last, (AIMessage, ToolMessage))
                    else ""
                )
                body = (
                    tool_result(last, answer) if isinstance(last, ToolMessage)
                    else Markdown(answer) if answer else Text("(réponse vide)")
                )
                live.update(render(body))
        except (Exception, GeneratorExit, KeyboardInterrupt):
            current = self.agent.get_state(config).values.get("messages", [])
            if current != previous:
                self.agent.update_state(
                    config,
                    {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *previous]},
                )
            raise
        finally:
            close = getattr(chunks, "close", None)
            if callable(close):
                close()

        if not self.console.is_terminal:
            self.console.print()
        elapsed = perf_counter() - start
        if isinstance(last, AIMessage) and last.additional_kwargs.get("microchat_auto_search_for"):
            self.console.print(f"[dim]Recherche web · total {elapsed:.1f}s[/]\n")
            return
        empty_message_tokens = self.token_counter([AIMessage(content="")])
        tokens = max(
            0,
            self.token_counter([AIMessage(content=answer)]) - empty_message_tokens,
        )
        first_token_s = first_token_at - start
        decode_time = elapsed - first_token_s
        rate = (tokens - 1) / decode_time if tokens > 1 and decode_time > 0 else 0
        self.console.print(
            f"[dim]{tokens} tokens · {rate:.1f} tok/s · "
            f"1er token {first_token_s:.2f}s · total {elapsed:.1f}s[/]\n"
        )


def run() -> int:
    """Charge config.ini, prépare le modèle et lance la boucle."""
    parser = argparse.ArgumentParser(description="Microchat local extensible par plugins-agent.")
    parser.add_argument("--version", action="version", version=f"microchat {__version__}")
    parser.parse_args()

    try:
        settings = load_config()
        logging.basicConfig(
            level=logging.DEBUG if settings.verbose else logging.INFO,
            format="%(levelname)s %(message)s",
        )
        model_path = resolve_model_path(settings)
        console = Console()
        with console.status("Chargement du modèle…", spinner="dots"):
            llm = build_llm(settings, model_path)
            if settings.warm_up:
                try:
                    llm.client.create_chat_completion(
                        messages=[
                            {"role": "system", "content": settings.system_prompt},
                            {"role": "user", "content": "Bonjour"},
                        ],
                        max_tokens=1,
                        temperature=0.0,
                    )
                except Exception:
                    logger.warning("Pré-chauffage du modèle impossible", exc_info=True)
        checkpointer = InMemorySaver()
        plugins = load_plugins(settings.plugins, settings.plugin_configurations)
        token_counter = make_token_counter(llm)
        agent = build_plugin_agent(
            llm,
            system_prompt=settings.system_prompt,
            plugins=plugins,
            token_counter=token_counter,
            history_token_budget=settings.history_token_budget,
            history_low_watermark=settings.history_low_watermark,
            checkpointer=checkpointer,
        )
        tool_icons = {
            tool_name: icon for plugin in plugins for tool_name, icon in plugin.tool_icons.items()
        }
        ChatApp(agent, checkpointer, settings, model_path, token_counter, tool_icons).run()
    except (ValueError, RuntimeError, OSError, configparser.Error) as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 1
    return 0
