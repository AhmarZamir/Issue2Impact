from __future__ import annotations

import hashlib
import shutil
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from src.ingestion.loader import IGNORED_DIRECTORIES, is_supported_repository_file


RepositorySourceType = Literal["local", "github", "zip"]
DEFAULT_WORKSPACE_ROOT = Path("workspace/repos")


@dataclass(frozen=True)
class RepositoryContext:
    path: Path
    name: str
    source_type: RepositorySourceType
    repository_id: str
    source: str


def infer_repository_source(source: str) -> RepositorySourceType:
    candidate = Path(source).expanduser()

    if candidate.exists() and candidate.is_dir():
        return "local"

    if candidate.exists() and candidate.is_file() and candidate.suffix.lower() == ".zip":
        return "zip"

    parsed = urlparse(source)
    if parsed.scheme in {"http", "https"} and parsed.netloc.lower() == "github.com":
        return "github"

    raise ValueError(
        "Unable to determine repository source. Provide a local folder, a .zip file, "
        "or a public GitHub repository URL."
    )


def prepare_repository(
    source: str,
    source_type: str = "auto",
    workspace_root: str | Path = DEFAULT_WORKSPACE_ROOT,
) -> RepositoryContext:
    resolved_type = infer_repository_source(source) if source_type == "auto" else source_type

    if resolved_type not in {"local", "github", "zip"}:
        raise ValueError("source_type must be one of: auto, local, github, zip")

    workspace = Path(workspace_root).expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    if resolved_type == "local":
        return _prepare_local_repository(source)

    if resolved_type == "github":
        return _prepare_github_repository(source, workspace)

    return _prepare_zip_repository(source, workspace)


def _prepare_local_repository(source: str) -> RepositoryContext:
    repo_path = Path(source).expanduser().resolve()

    if not repo_path.exists() or not repo_path.is_dir():
        raise ValueError(f"Local repository folder does not exist: {source}")

    fingerprint = _directory_fingerprint(repo_path)
    repository_id = _stable_id(f"local:{repo_path}:{fingerprint}")

    return RepositoryContext(
        path=repo_path,
        name=repo_path.name,
        source_type="local",
        repository_id=repository_id,
        source=str(repo_path),
    )


def _prepare_github_repository(source: str, workspace: Path) -> RepositoryContext:
    clone_url, repo_name = _normalize_github_url(source)
    source_key = _stable_id(clone_url)
    clone_path = workspace / f"github-{source_key}-{repo_name}"

    try:
        if (clone_path / ".git").exists():
            _run_git(["-C", str(clone_path), "pull", "--ff-only"])
        else:
            if clone_path.exists():
                shutil.rmtree(clone_path)
            _run_git(["clone", "--depth", "1", clone_url, str(clone_path)])
    except RuntimeError:
        if clone_path.exists() and (clone_path / ".git").exists():
            pass
        else:
            raise

    commit_sha = _run_git(["-C", str(clone_path), "rev-parse", "HEAD"]).strip()
    repository_id = _stable_id(f"github:{clone_url}:{commit_sha}")

    return RepositoryContext(
        path=clone_path.resolve(),
        name=repo_name,
        source_type="github",
        repository_id=repository_id,
        source=clone_url,
    )


def _prepare_zip_repository(source: str, workspace: Path) -> RepositoryContext:
    zip_path = Path(source).expanduser().resolve()

    if not zip_path.exists() or not zip_path.is_file() or zip_path.suffix.lower() != ".zip":
        raise ValueError(f"ZIP repository file does not exist: {source}")

    archive_hash = _file_hash(zip_path)
    extraction_root = workspace / f"zip-{archive_hash[:16]}"

    if extraction_root.exists():
        shutil.rmtree(extraction_root)
    extraction_root.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as archive:
        _safe_extract_zip(archive, extraction_root)

    repo_path = _single_root_directory(extraction_root)
    repository_id = _stable_id(f"zip:{archive_hash}")

    return RepositoryContext(
        path=repo_path.resolve(),
        name=repo_path.name or zip_path.stem,
        source_type="zip",
        repository_id=repository_id,
        source=str(zip_path),
    )


def _normalize_github_url(source: str) -> tuple[str, str]:
    parsed = urlparse(source.strip())

    if parsed.scheme not in {"http", "https"} or parsed.netloc.lower() != "github.com":
        raise ValueError("Only public https://github.com/<owner>/<repo> repository URLs are supported.")

    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) != 2:
        raise ValueError("GitHub URL must point to a repository root: https://github.com/<owner>/<repo>")

    owner, repo = parts
    repo_name = repo[:-4] if repo.endswith(".git") else repo
    if not owner or not repo_name:
        raise ValueError("Invalid GitHub repository URL.")

    return f"https://github.com/{owner}/{repo_name}.git", repo_name


def _run_git(arguments: list[str]) -> str:
    try:
        result = subprocess.run(
            ["git", *arguments],
            check=True,
            capture_output=True,
            text=True,
            timeout=180,
        )
    except FileNotFoundError as error:
        raise RuntimeError("Git is required for GitHub repository URLs but was not found.") from error
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("Git operation timed out while preparing the repository.") from error
    except subprocess.CalledProcessError as error:
        message = (error.stderr or error.stdout or "Git command failed.").strip()
        raise RuntimeError(message) from error

    return result.stdout


def _directory_fingerprint(repo_path: Path) -> str:
    digest = hashlib.sha256()
    files = []

    for file_path in repo_path.rglob("*"):
        if not file_path.is_file():
            continue
        if any(ignored in file_path.parts for ignored in IGNORED_DIRECTORIES):
            continue
        if not is_supported_repository_file(file_path):
            continue
        files.append(file_path)

    for file_path in sorted(files, key=lambda path: str(path.relative_to(repo_path)).lower()):
        relative = file_path.relative_to(repo_path).as_posix()
        digest.update(relative.encode("utf-8"))
        try:
            with file_path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        except OSError:
            continue

    return digest.hexdigest()


def _file_hash(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_id(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]


def _safe_extract_zip(archive: zipfile.ZipFile, destination: Path) -> None:
    destination = destination.resolve()

    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        if target != destination and destination not in target.parents:
            raise ValueError("ZIP archive contains an unsafe path outside the extraction directory.")

    archive.extractall(destination)


def _single_root_directory(extraction_root: Path) -> Path:
    children = [child for child in extraction_root.iterdir() if child.name != "__MACOSX"]
    directories = [child for child in children if child.is_dir()]
    files = [child for child in children if child.is_file()]

    if len(directories) == 1 and not files:
        return directories[0]

    return extraction_root
