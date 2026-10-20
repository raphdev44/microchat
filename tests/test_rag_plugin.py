from __future__ import annotations

import json

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from microchat.agents import rag
from microchat.agents.runtime import build_agent


def test_index_and_retrieve_relevant_chunks(tmp_path, monkeypatch):
    monkeypatch.setattr(
        rag,
        "_embed",
        lambda texts, model_name=rag.EMBEDDING_MODEL, on_progress=None: [
            [1.0, 0.0] if "océan" in text else [0.0, 1.0] for text in texts
        ],
    )
    source = tmp_path / "notes.md"
    source.write_text("L'océan Atlantique est vaste.\n\nLes volcans sont actifs.", encoding="utf-8")
    index = rag.RagIndex(tmp_path / "index.json", max_results=1)

    result = index.index(str(source))

    assert "indexés" in result
    assert (
        index.search("océan")[0]["text"] == "L'océan Atlantique est vaste. Les volcans sont actifs."
    )


def test_supported_text_html_and_folder_indexing(tmp_path, monkeypatch):
    monkeypatch.setattr(
        rag,
        "_embed",
        lambda texts, model_name=rag.EMBEDDING_MODEL, on_progress=None: [[1.0, 0.0] for _ in texts],
    )
    (tmp_path / "one.html").write_text(
        "<h1>Document</h1><script>ignore me</script><p>Contenu utile</p>", encoding="utf-8"
    )
    (tmp_path / "two.txt").write_text("Autre contenu", encoding="utf-8")
    (tmp_path / "ignored.csv").write_text("hors format", encoding="utf-8")
    index = rag.RagIndex(tmp_path / "store" / "index.json", max_results=4)

    index.index(str(tmp_path))
    records = json.loads(index.path.read_text(encoding="utf-8"))["chunks"]

    assert len(records) == 2
    assert all("ignore me" not in item["text"] for item in records)


def test_rag_commands_switch_mode_and_reinitialize(tmp_path, monkeypatch):
    monkeypatch.setattr(
        rag,
        "_embed",
        lambda texts, model_name=rag.EMBEDDING_MODEL, on_progress=None: [[1.0, 0.0] for _ in texts],
    )
    plugin = rag.load_plugin({"index_path": str(tmp_path / "index.json")})
    commands = plugin.commands
    assert commands is not None
    path = tmp_path / "document.txt"
    path.write_text("contenu", encoding="utf-8")
    commands["rag_index"].handler(str(path))

    assert "activé" in commands["rag_start"].handler("")
    assert "désactivé" in commands["rag_stop"].handler("")
    assert "réinitialisé" in commands["rag_init"].handler("")
    assert json.loads((tmp_path / "index.json").read_text(encoding="utf-8"))["chunks"] == []


def test_index_requires_a_source(tmp_path):
    index = rag.RagIndex(tmp_path / "index.json", max_results=4)

    with pytest.raises(ValueError, match="Usage"):
        index.index(" ")


def test_configured_chunk_size_limits_indexed_fragment_length(tmp_path, monkeypatch):
    monkeypatch.setattr(
        rag,
        "_embed",
        lambda texts, model_name=rag.EMBEDDING_MODEL, on_progress=None: [[1.0, 0.0] for _ in texts],
    )
    plugin = rag.load_plugin(
        {
            "index_path": str(tmp_path / "index.json"),
            "chunk_size": "200",
        }
    )
    assert plugin.commands is not None
    source = tmp_path / "long.txt"
    source.write_text("mot " * 200, encoding="utf-8")

    plugin.commands["rag_index"].handler(str(source))
    records = json.loads((tmp_path / "index.json").read_text(encoding="utf-8"))["chunks"]

    assert len(records) > 1
    assert all(len(record["text"]) <= 200 for record in records)


@pytest.mark.parametrize("chunk_size", ["0", "150", "invalid"])
def test_rejects_invalid_chunk_size(tmp_path, chunk_size):
    with pytest.raises(ValueError, match="chunk_size"):
        rag.load_plugin({"chunk_size": chunk_size, "index_path": str(tmp_path / "index.json")})


@pytest.mark.parametrize("min_similarity", ["invalid", "-0.1", "1.01"])
def test_rejects_invalid_min_similarity(tmp_path, min_similarity):
    with pytest.raises(ValueError, match="min_similarity"):
        rag.load_plugin(
            {
                "min_similarity": min_similarity,
                "index_path": str(tmp_path / "index.json"),
            }
        )


def test_search_filters_below_min_similarity(tmp_path, monkeypatch):
    monkeypatch.setattr(
        rag,
        "_embed",
        lambda texts, model_name=rag.EMBEDDING_MODEL, on_progress=None: [[1.0, 0.0]],
    )
    index_path = tmp_path / "index.json"
    index_path.write_text(
        json.dumps(
            {
                "chunks": [
                    {"source": "doc", "text": "threshold match", "vector": [0.8, 0.6]},
                    {"source": "doc", "text": "below threshold", "vector": [0.6, 0.8]},
                ]
            }
        ),
        encoding="utf-8",
    )
    index = rag.RagIndex(index_path, max_results=2, min_similarity=0.8)

    results = index.search("query")

    assert [(item["text"], item["similarity"]) for item in results] == [
        ("threshold match", 0.8)
    ]


def test_injects_only_best_matches_up_to_configured_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(
        rag,
        "_embed",
        lambda texts, model_name=rag.EMBEDDING_MODEL, on_progress=None: [[1.0, 0.0]],
    )
    index_path = tmp_path / "index.json"
    index_path.write_text(
        json.dumps(
            {
                "chunks": [
                    {"source": "doc", "text": "best", "vector": [0.9, 0.435889894]},
                    {"source": "doc", "text": "second", "vector": [0.8, 0.6]},
                    {"source": "doc", "text": "third", "vector": [0.7, 0.714142843]},
                    {"source": "doc", "text": "fourth", "vector": [0.6, 0.8]},
                ]
            }
        ),
        encoding="utf-8",
    )
    plugin = rag.load_plugin(
        {
            "index_path": str(index_path),
            "max_results": "4",
            "max_injected_results": "2",
            "min_similarity": "0.0",
        }
    )
    assert plugin.commands is not None
    plugin.commands["rag_start"].handler("")
    agent = build_agent(
        FakeMessagesListChatModel(responses=[AIMessage(content="Réponse")]),
        system_prompt="Réponds en français.",
        plugins=(plugin,),
        token_counter=lambda messages: len(messages),
        history_token_budget=100,
        history_low_watermark=0.6,
        checkpointer=InMemorySaver(),
    )

    agent.invoke(
        {"messages": [HumanMessage(content="Question")]},
        config={"configurable": {"thread_id": "rag-injection-limit"}},
    )

    assert plugin.context_provider is not None
    injected = plugin.context_provider()
    assert [text for _, text, _ in injected] == ["best", "second"]
    assert [score for _, _, score in injected] == pytest.approx([0.9, 0.8])


@pytest.mark.parametrize("injected", ["0", "5", "invalid"])
def test_rejects_invalid_max_injected_results(tmp_path, injected):
    with pytest.raises(ValueError, match="max_injected_results"):
        rag.load_plugin(
            {
                "max_results": "4",
                "max_injected_results": injected,
                "index_path": str(tmp_path / "index.json"),
            }
        )


def test_embedding_reports_progress_for_each_vector(monkeypatch):
    received_batch_sizes: list[int] = []

    class FakeEmbeddingModel:
        def embed(self, texts, *, batch_size):
            received_batch_sizes.append(batch_size)
            yield from ([1.0, 0.0] for _ in texts)

    monkeypatch.setattr(rag, "_embedding_model", lambda _model_name: FakeEmbeddingModel())
    updates: list[bool] = []

    vectors = rag._embed(["first", "second"], on_progress=lambda: updates.append(True))

    assert vectors == [[1.0, 0.0], [1.0, 0.0]]
    assert len(updates) == 2
    assert received_batch_sizes == [rag.EMBEDDING_BATCH_SIZE]


def test_index_explains_permission_denied_for_cloud_documents(tmp_path, monkeypatch):
    source = tmp_path / "document.docx"
    source.touch()
    index = rag.RagIndex(tmp_path / "index.json", max_results=4)
    monkeypatch.setattr(
        rag,
        "_read_document",
        lambda _path: (_ for _ in ()).throw(PermissionError("access denied")),
    )

    with pytest.raises(ValueError, match="disponible sur cet appareil"):
        index.index(str(source))


def test_plugin_configures_embedding_model_for_index_and_search(tmp_path, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(
        rag,
        "_embed",
        lambda texts, model_name=rag.EMBEDDING_MODEL, on_progress=None: (
            calls.append(model_name) or [[1.0, 0.0] for _ in texts]
        ),
    )
    plugin = rag.load_plugin(
        {"index_path": str(tmp_path / "index.json"), "embedding_model": "custom/embedding-model"}
    )
    assert plugin.commands is not None
    source = tmp_path / "document.md"
    source.write_text("Contenu indexé.", encoding="utf-8")
    plugin.commands["rag_index"].handler(str(source))
    plugin.commands["rag_start"].handler("")
    agent = build_agent(
        FakeMessagesListChatModel(responses=[AIMessage(content="Réponse")]),
        system_prompt="Réponds en français.",
        plugins=(plugin,),
        token_counter=lambda messages: len(messages),
        history_token_budget=100,
        history_low_watermark=0.6,
        checkpointer=InMemorySaver(),
    )
    agent.invoke(
        {"messages": [HumanMessage(content="Question")]},
        config={"configurable": {"thread_id": "rag-custom-model"}},
    )

    assert calls == ["custom/embedding-model", "custom/embedding-model"]
    assert plugin.context_provider is not None
    assert plugin.context_provider() == ((str(source.resolve()), "Contenu indexé.", 1.0),)


def test_rag_middleware_injects_retrieved_context_only_when_started(tmp_path, monkeypatch):
    monkeypatch.setattr(
        rag,
        "_embed",
        lambda texts, model_name=rag.EMBEDDING_MODEL, on_progress=None: [[1.0, 0.0] for _ in texts],
    )
    source = tmp_path / "doc.txt"
    source.write_text("La réponse vient du document.", encoding="utf-8")
    plugin = rag.load_plugin({"index_path": str(tmp_path / "index.json")})
    assert plugin.commands is not None
    plugin.commands["rag_index"].handler(str(source))
    seen_messages: list[str] = []

    class RecordingModel(FakeMessagesListChatModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            seen_messages.append(str(messages[-1].content))
            return super()._generate(messages, stop, run_manager, **kwargs)

    agent = build_agent(
        RecordingModel(responses=[AIMessage(content="Réponse"), AIMessage(content="Réponse RAG")]),
        system_prompt="Réponds en français.",
        plugins=(plugin,),
        token_counter=lambda messages: len(messages),
        history_token_budget=100,
        history_low_watermark=0.6,
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "rag-test"}}
    agent.invoke({"messages": [HumanMessage(content="Question")]}, config=config)
    plugin.commands["rag_start"].handler("")
    agent.invoke({"messages": [HumanMessage(content="Question")]}, config=config)

    assert "Contexte documentaire" not in seen_messages[0]
    assert "La réponse vient du document." in seen_messages[1]
    assert rag.RAG_GOLDEN_RULE in seen_messages[1]
    assert rag.RAG_GOLDEN_RULE not in seen_messages[0]
