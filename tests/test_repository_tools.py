from src.tools.repository_tools import read_repository_file


def test_read_repository_file_reads_inside_demo_repository():
    content = read_repository_file.invoke({"file_path": "auth.py"})

    assert "def login" in content


def test_read_repository_file_blocks_path_traversal():
    result = read_repository_file.invoke({"file_path": "../main.py"})

    assert result == "Invalid file path."
