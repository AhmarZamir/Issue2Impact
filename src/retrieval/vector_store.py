from langchain_chroma import Chroma

from pathlib import Path



from src.retrieval.embeddings import (
    get_embedding_model,
)


DB_PATH = "./chroma_db"

COLLECTION_NAME = "issue2impact"


def create_vector_store(chunks):

    embeddings = get_embedding_model()

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=DB_PATH,
    )

    return vector_store


def load_vector_store():

    embeddings = get_embedding_model()

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=DB_PATH,
    )

    return vector_store


def vector_store_exists():

    return Path(DB_PATH).exists()
