from src.ingestion.chunking import chunk_documents
from src.ingestion.loader import load_repository


def test_demo_repository_loads_supported_source_files():
    documents = load_repository("demo_repo")
    paths = {document.metadata["file_path"].replace("\\", "/") for document in documents}

    assert "auth.py" in paths
    assert "tests/test_auth.py" in paths
    assert not any("__pycache__" in path for path in paths)


def test_chunks_keep_file_and_chunk_metadata():
    chunks = chunk_documents(load_repository("demo_repo"))

    assert chunks
    assert all("file_path" in chunk.metadata for chunk in chunks)
    assert all("chunk_index" in chunk.metadata for chunk in chunks)
