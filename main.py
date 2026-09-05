import argparse
from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from src.graph.repository_graph import RepositoryGraph
from src.ingestion.chunking import chunk_documents
from src.ingestion.loader import load_repository
from src.llm.model import get_llm
from src.prompts.repository_agent_prompt import REPOSITORY_AGENT_PROMPT
from src.repository.source import prepare_repository
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


DEFAULT_QUERY = """
Users sometimes remain authenticated when an invalid token is supplied.
Investigate how token validation and logout work in this repository and give me
an implementation plan to make authentication handling safer.
""".strip()


def build_repository_graph(
    repo_path: str,
    repository_id: str,
    checkpointer=None,
):
    """Build the repository-specific graph, retrieval index, and tools."""
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
            raise ValueError("Repository files were loaded but produced no searchable chunks.")
        vector_store = create_vector_store(chunks, repository_id)

    retrieval_pipeline = RetrievalPipeline(vector_store)
    repository_search = create_repository_search_tool(retrieval_pipeline)
    repository_read = create_repository_read_tool(repo_path)

    return RepositoryGraph(
        llm=get_llm(),
        tools=[repository_search, repository_read],
    ).build(checkpointer=checkpointer)


def extract_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict)
        )
    return str(content)


def print_trace(result, repository=None):
    if repository is not None:
        print("\n=== REPOSITORY ===")
        print("Name:", repository.name)
        print("Source type:", repository.source_type)
        print("Path:", repository.path)
        print("Repository ID:", repository.repository_id)

    print("\n=== ROUTING ===")
    print("Route:", result.get("route"))
    print("Reason:", result.get("route_reason"))

    print("\n=== INVESTIGATION ===")
    print(result.get("investigation", "N/A"))

    print("\n=== PLAN ===")
    print(result.get("plan", "N/A"))

    print("\n=== CRITIC ===")
    print("Approved:", result.get("plan_approved"))
    print("Needs more evidence:", result.get("needs_more_evidence"))
    print("Feedback:", result.get("critic_feedback", "N/A"))
    print("Retries:", result.get("retry_count", 0))

    print("\n=== HUMAN REVIEW ===")
    print("Approved:", result.get("human_approved"))
    print("Feedback:", result.get("human_feedback", "N/A"))

    print("\n=== MESSAGE TRACE ===")
    for message in result.get("messages", []):
        print(f"\n--- {type(message).__name__} ---")
        print(message)


def run(
    query: str,
    repository_source: str = "demo_repo",
    source_type: str = "auto",
    show_trace: bool = False,
    input_fn=input,
    thread_id: str | None = None,
):
    """Prepare a repository, run the workflow, and handle human approval interrupts."""
    repository = prepare_repository(repository_source, source_type=source_type)

    checkpointer = InMemorySaver()
    graph = build_repository_graph(
        repo_path=str(repository.path),
        repository_id=repository.repository_id,
        checkpointer=checkpointer,
    )

    config = {
        "configurable": {
            "thread_id": thread_id or f"issue2impact-{uuid4()}",
        },
        "recursion_limit": 30,
    }

    result = graph.invoke(
        {
            "user_query": query,
            "retry_count": 0,
            "messages": [
                SystemMessage(content=REPOSITORY_AGENT_PROMPT),
                HumanMessage(content=query),
            ],
        },
        config=config,
    )

    while result.get("__interrupt__"):
        interrupt_info = result["__interrupt__"][0].value

        print("\n=== HUMAN APPROVAL REQUIRED ===")
        print(interrupt_info.get("message", "Review the proposed plan."))
        print("\nPlan:\n")
        print(interrupt_info.get("plan", "N/A"))
        print("\nCritic feedback:\n")
        print(interrupt_info.get("critic_feedback", "N/A"))

        answer = input_fn("\nApprove plan? [y/n]: ").strip().lower()
        approved = answer in {"y", "yes"}

        if approved:
            feedback = "Approved by human reviewer."
        else:
            feedback = input_fn("Why are you rejecting it? ").strip()

        result = graph.invoke(
            Command(
                resume={
                    "approved": approved,
                    "feedback": feedback,
                }
            ),
            config=config,
        )

    if show_trace:
        print_trace(result, repository=repository)

    final_content = result["messages"][-1].content
    return extract_text(final_content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Investigate a local, GitHub, or ZIP source-code repository."
    )
    parser.add_argument("query", nargs="?", default=DEFAULT_QUERY)
    parser.add_argument(
        "--repo",
        default="demo_repo",
        help=(
            "Repository source: local folder path, public GitHub repository URL, "
            "or path to a ZIP archive."
        ),
    )
    parser.add_argument(
        "--source-type",
        choices=["auto", "local", "github", "zip"],
        default="auto",
        help="Repository source type. 'auto' detects the type from --repo.",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help=(
            "Print repository metadata, routing, investigation, plan, critic state, "
            "human-review state, and every message in the workflow."
        ),
    )
    args = parser.parse_args()

    print(
        run(
            args.query,
            repository_source=args.repo,
            source_type=args.source_type,
            show_trace=args.trace,
        )
    )
