from pathlib import Path
from langchain_core.documents import Document


SUPPORTED_EXTENSIONS = {
    ".py": "python",
    ".md": "markdown",
    ".txt": "text",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
}


IGNORED_DIRECTORIES = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
}


def load_repository(repo_path: str) -> list[Document]:

    repo_root = Path(repo_path)

    documents = []

# it will caryy both files and directories, so we need to filter out directories
    for file_path in repo_root.rglob("*"):


# if the file is not a file, we skip it
        if not file_path.is_file():
            continue

# if the file is in an ignored directory, we skip it
        if any(
            ignored in file_path.parts
            for ignored in IGNORED_DIRECTORIES
        ):
            continue
        # get the file extension and check if it is supported

        extension = file_path.suffix.lower()

        if extension not in SUPPORTED_EXTENSIONS:
            continue

# try to read the file content, if it fails, we skip it
        try:
            content = file_path.read_text(
                encoding="utf-8"
            )

        except UnicodeDecodeError:
            continue

        relative_path = file_path.relative_to(
            repo_root
        )

        document = Document(
            page_content=content,
            metadata={
                "file_path": str(relative_path),
                "extension": extension,
                "language": SUPPORTED_EXTENSIONS[
                    extension
                ],
            },
        )

        documents.append(document)

    return documents