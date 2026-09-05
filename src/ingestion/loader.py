from pathlib import Path

from langchain_core.documents import Document


SUPPORTED_EXTENSIONS = {
    ".py": "python",
    ".md": "markdown",
    ".txt": "text",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".go": "go",
    ".rs": "rust",
    ".php": "php",
    ".rb": "ruby",
    ".swift": "swift",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".scala": "scala",
    ".html": "html",
    ".css": "css",
    ".scss": "css",
    ".sql": "sql",
    ".toml": "toml",
    ".ini": "text",
    ".cfg": "text",
    ".xml": "xml",
    ".sh": "shell",
    ".bat": "shell",
    ".ps1": "powershell",
}

SUPPORTED_FILENAMES = {
    "Dockerfile": "dockerfile",
    "Makefile": "makefile",
    ".gitignore": "text",
}

IGNORED_DIRECTORIES = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
    ".idea",
    ".vscode",
    "target",
    "coverage",
    ".pytest_cache",
    ".mypy_cache",
    ".next",
}

MAX_FILE_SIZE_BYTES = 1_000_000


def is_supported_repository_file(file_path: Path) -> bool:
    return (
        file_path.suffix.lower() in SUPPORTED_EXTENSIONS
        or file_path.name in SUPPORTED_FILENAMES
    )


def _language_for_file(file_path: Path) -> str:
    if file_path.name in SUPPORTED_FILENAMES:
        return SUPPORTED_FILENAMES[file_path.name]
    return SUPPORTED_EXTENSIONS[file_path.suffix.lower()]


def load_repository(repo_path: str) -> list[Document]:
    repo_root = Path(repo_path).expanduser().resolve()

    if not repo_root.exists() or not repo_root.is_dir():
        raise ValueError(f"Repository folder does not exist: {repo_path}")

    documents = []

    for file_path in repo_root.rglob("*"):
        if not file_path.is_file():
            continue

        if any(ignored in file_path.parts for ignored in IGNORED_DIRECTORIES):
            continue

        if not is_supported_repository_file(file_path):
            continue

        try:
            if file_path.stat().st_size > MAX_FILE_SIZE_BYTES:
                continue
            content = file_path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue

        relative_path = file_path.relative_to(repo_root)

        documents.append(
            Document(
                page_content=content,
                metadata={
                    "file_path": str(relative_path),
                    "extension": file_path.suffix.lower(),
                    "language": _language_for_file(file_path),
                },
            )
        )

    return documents
