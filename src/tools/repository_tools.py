from pathlib import Path
from langchain_core.tools import tool


def create_repository_search_tool(retrieval_pipeline):

    @tool
    def search_repository(query: str) -> str:
        """
        Search the source-code repository for files and code
        relevant to a question.

        Use this tool whenever repository evidence is required.
        """

        results = retrieval_pipeline.retrieve(query)

        if not results:
            return "No relevant files found in the repository."

        formatted_results = []

        for index, (document, score) in enumerate(results, start=1):

            file_path = document.metadata.get(
                "file_path",
                "Unknown file path",
            )

            chunk_index = document.metadata.get(
                "chunk_index",
                "Unknown chunk index",
            )

            formatted_results.append(
                f"""
Reranked candidate {index}
Score: {score}
Chunk index: {chunk_index}
File Path: {file_path}
Code:
{document.page_content[:300]}
""".strip()
            )

        return "\n\n".join(formatted_results)

    return search_repository


@tool
def read_repository_file(file_path: str) -> str:
    """
    Read the complete content of a specific file
    from the demo repository.

    Use this after search_repository identifies
    a file that needs closer inspection.
    """

    repo_root = Path("demo_repo").resolve()

    requested_path = (
        repo_root / file_path
    ).resolve()

    # Prevent access outside demo_repo
    if (
        requested_path != repo_root
        and repo_root not in requested_path.parents
    ):
        return "Invalid file path."

    if not requested_path.exists():
        return f"File does not exist: {file_path}"

    if not requested_path.is_file():
        return f"Not a file: {file_path}"

    try:
        return requested_path.read_text(
            encoding="utf-8"
        )

    except UnicodeDecodeError:
        return "Unable to read file as UTF-8."

    except OSError as error:
        return f"Unable to read file: {error}"