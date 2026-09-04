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
from src.retrieval.pipeline import RetrievalPipeline
from src.retrieval.vector_store import (
    create_vector_store,
    load_vector_store,
    vector_store_exists,
)
from src.tools.repository_tools import (
    create_repository_search_tool,
    read_repository_file,
)


DEFAULT_QUERY = """
Users sometimes remain authenticated when an invalid token is supplied.
Investigate how token validation and logout work in this repository and give me
an implementation plan to make authentication handling safer.
""".strip()


def build_repository_graph(repo_path: str = "demo_repo", checkpointer=None):
    """Build the Phase 9 graph with retrieval dependencies and persistence."""
    if vector_store_exists():
        vector_store = load_vector_store()
    else:
        documents = load_repository(repo_path)
        chunks = chunk_documents(documents)
        vector_store = create_vector_store(chunks)

    retrieval_pipeline = RetrievalPipeline(vector_store)
    repository_search = create_repository_search_tool(retrieval_pipeline)

    return RepositoryGraph(
        llm=get_llm(),
        tools=[repository_search, read_repository_file],
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


def print_trace(result):
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


def run(query: str, show_trace: bool = False, input_fn=input, thread_id: str | None = None):
    """Run the graph and handle any human-approval interrupts in the terminal."""
    checkpointer = InMemorySaver()
    graph = build_repository_graph(checkpointer=checkpointer)

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

        feedback = ""
        if not approved:
            feedback = input_fn("Why are you rejecting it? ").strip()
        else:
            feedback = "Approved by human reviewer."

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
        print_trace(result)

    final_content = result["messages"][-1].content
    return extract_text(final_content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Issue2Impact agent graph.")
    parser.add_argument("query", nargs="?", default=DEFAULT_QUERY)
    parser.add_argument(
        "--trace",
        action="store_true",
        help=(
            "Print routing, investigation, plan, critic state, human-review state, "
            "and every message in the workflow."
        ),
    )
    args = parser.parse_args()

    print(run(args.query, show_trace=args.trace))
