import argparse

from langchain_core.messages import HumanMessage, SystemMessage

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
what is Software Engineering?
""".strip()


def build_repository_graph(repo_path: str = "demo_repo"):
    """Build the complete Phase 6 routed graph and retrieval dependencies."""
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
    ).build()


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


def run(query: str, show_trace: bool = False):
    graph = build_repository_graph()

    result = graph.invoke(
        {
            "user_query": query,
            "messages": [
                SystemMessage(content=REPOSITORY_AGENT_PROMPT),
                HumanMessage(content=query),
            ],
        },
        config={"recursion_limit": 10},
    )

    if show_trace:
        print("\n=== ROUTING ===")
        print("Route:", result.get("route"))
        print("Reason:", result.get("route_reason"))

        for message in result["messages"]:
            print(f"\n--- {type(message).__name__} ---")
            print(message)

    final_content = result["messages"][-1].content
    return extract_text(final_content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Issue2Impact agent graph.")
    parser.add_argument("query", nargs="?", default=DEFAULT_QUERY)
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Print the routing decision and every message in the agent/tool loop.",
    )
    args = parser.parse_args()

    print(run(args.query, show_trace=args.trace))
