from pathlib import Path

from langchain_chroma import Chroma

from src.retrieval.embeddings import get_embedding_model


VECTOR_STORE_ROOT = Path("data/vector_stores")


def get_vector_store_path(repository_id: str) -> Path:
    if not repository_id:
        raise ValueError("repository_id is required for repository-specific vector storage.")
    return VECTOR_STORE_ROOT / repository_id


def _collection_name(repository_id: str) -> str:
    return f"issue2impact_{repository_id[:12]}"


def create_vector_store(chunks, repository_id: str):
    embeddings = get_embedding_model()
    persist_directory = get_vector_store_path(repository_id)
    persist_directory.mkdir(parents=True, exist_ok=True)

    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=_collection_name(repository_id),
        persist_directory=str(persist_directory),
    )


def load_vector_store(repository_id: str):
    embeddings = get_embedding_model()
    persist_directory = get_vector_store_path(repository_id)

    return Chroma(
        collection_name=_collection_name(repository_id),
        embedding_function=embeddings,
        persist_directory=str(persist_directory),
    )


def vector_store_exists(repository_id: str) -> bool:
    persist_directory = get_vector_store_path(repository_id)
    return persist_directory.exists() and any(persist_directory.iterdir())
