from src.tools.repository_tools import create_repository_read_tool


def test_read_repository_file_reads_inside_selected_repository(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "auth.py").write_text("def login():\n    return True\n", encoding="utf-8")

    read_repository_file = create_repository_read_tool(str(repo))
    content = read_repository_file.invoke({"file_path": "auth.py"})

    assert "def login" in content


def test_read_repository_file_blocks_path_traversal(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("secret", encoding="utf-8")

    read_repository_file = create_repository_read_tool(str(repo))
    result = read_repository_file.invoke({"file_path": "../secret.txt"})

    assert result == "Invalid file path."
