from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from src.graph.repository_graph import RepositoryGraph
from src.ingestion.chunking import chunk_documents
from src.ingestion.loader import load_repository
from src.llm.model import get_llm
from src.prompts.repository_agent_prompt import REPOSITORY_AGENT_PROMPT
from src.repository.source import RepositoryContext, prepare_repository
from src.retrieval.pipeline import RetrievalPipeline
from src.retrieval.vector_store import (
    create_vector_store,
    load_vector_store,
    vector_store_exists,
)
from src.tools.repository_tools import (
    create_repository_read_tool,
    create_repository_search_tool,
)


DEFAULT_RECURSION_LIMIT = 30


@dataclass
class RepositoryRuntime:
    repository: RepositoryContext
    graph: object
    checkpointer: object
    source_file_count: int


def extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("text")
        )
    return str(content)


def get_interrupt_payload(result):
    interrupts = result.get("__interrupt__", []) if result else []
    if not interrupts:
        return None
    return interrupts[0].value


def build_repository_graph(
    repo_path: str,
    repository_id: str,
    checkpointer=None,
    llm=None,
):
    """Build a repository-scoped retrieval pipeline, tools, and LangGraph workflow."""
    documents = None

    if vector_store_exists(repository_id):
        vector_store = load_vector_store(repository_id)
    else:
        documents = load_repository(repo_path)
        if not documents:
            raise ValueError(
                "No supported source files were found in the selected repository."
            )

        chunks = chunk_documents(documents)
        if not chunks:
            raise ValueError(
                "Repository files were loaded but produced no searchable chunks."
            )

        vector_store = create_vector_store(chunks, repository_id)

    retrieval_pipeline = RetrievalPipeline(vector_store)
    repository_search = create_repository_search_tool(retrieval_pipeline)
    repository_read = create_repository_read_tool(repo_path)

    return RepositoryGraph(
        llm=llm or get_llm(),
        tools=[repository_search, repository_read],
    ).build(checkpointer=checkpointer)


def prepare_repository_runtime(
    repository_source: str,
    source_type: str = "auto",
    checkpointer=None,
) -> RepositoryRuntime:
    """Resolve a repository source and prepare everything needed to run the workflow."""
    repository = prepare_repository(repository_source, source_type=source_type)
    documents = load_repository(str(repository.path))
    if not documents:
        raise ValueError(
            "No supported source files were found in the selected repository."
        )

    active_checkpointer = checkpointer or InMemorySaver()
    graph = build_repository_graph(
        repo_path=str(repository.path),
        repository_id=repository.repository_id,
        checkpointer=active_checkpointer,
    )

    return RepositoryRuntime(
        repository=repository,
        graph=graph,
        checkpointer=active_checkpointer,
        source_file_count=len(documents),
    )


def make_config(thread_id: str | None = None, recursion_limit: int = DEFAULT_RECURSION_LIMIT):
    return {
        "configurable": {
            "thread_id": thread_id or f"issue2impact-{uuid4()}",
        },
        "recursion_limit": recursion_limit,
    }


def start_workflow(graph, query: str, config=None):
    query = query.strip()
    if not query:
        raise ValueError("Describe an issue or ask a repository question first.")

    active_config = config or make_config()
    result = graph.invoke(
        {
            "user_query": query,
            "retry_count": 0,
            "messages": [
                SystemMessage(content=REPOSITORY_AGENT_PROMPT),
                HumanMessage(content=query),
            ],
        },
        config=active_config,
    )
    return result, active_config


def resume_workflow(graph, config, approved: bool, feedback: str = ""):
    if not config:
        raise ValueError("A workflow configuration is required to resume execution.")

    clean_feedback = feedback.strip()
    if not approved and not clean_feedback:
        raise ValueError("Add feedback before rejecting the implementation plan.")
    if approved and not clean_feedback:
        clean_feedback = "Approved by human reviewer."

    return graph.invoke(
        Command(
            resume={
                "approved": approved,
                "feedback": clean_feedback,
            }
        ),
        config=config,
    )
