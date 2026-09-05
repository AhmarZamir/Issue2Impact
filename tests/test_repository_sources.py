import zipfile

import pytest

from src.repository.source import infer_repository_source, prepare_repository


def test_local_repository_context_changes_when_source_changes(tmp_path):
    repo = tmp_path / "sample_repo"
    repo.mkdir()
    source_file = repo / "app.py"
    source_file.write_text("print('one')\n", encoding="utf-8")

    first = prepare_repository(str(repo), source_type="local", workspace_root=tmp_path / "workspace")
    source_file.write_text("print('two')\n", encoding="utf-8")
    second = prepare_repository(str(repo), source_type="local", workspace_root=tmp_path / "workspace")

    assert first.path == repo.resolve()
    assert first.source_type == "local"
    assert first.repository_id != second.repository_id


def test_zip_repository_is_extracted_and_resolved(tmp_path):
    archive_path = tmp_path / "repo.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("project/app.py", "print('hello')\n")
        archive.writestr("project/README.md", "# Demo\n")

    context = prepare_repository(
        str(archive_path),
        source_type="zip",
        workspace_root=tmp_path / "workspace",
    )

    assert context.source_type == "zip"
    assert context.name == "project"
    assert (context.path / "app.py").exists()
    assert (context.path / "README.md").exists()


def test_zip_repository_blocks_path_traversal(tmp_path):
    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../outside.py", "print('bad')\n")

    with pytest.raises(ValueError, match="unsafe path"):
        prepare_repository(
            str(archive_path),
            source_type="zip",
            workspace_root=tmp_path / "workspace",
        )


def test_source_type_auto_detection(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()

    assert infer_repository_source(str(repo)) == "local"
    assert infer_repository_source("https://github.com/example/project") == "github"
