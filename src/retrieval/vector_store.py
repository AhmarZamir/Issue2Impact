from langchain_community.vectorstores import Chroma
from src.retrieval.embeddings import get_embedding_model



def create_vector_store(chunks):

    embeddings = get_embedding_model()

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding =embeddings,
        collection_name ="issue2impact",
        persist_directory ="./chroma_db",
    )

    return vector_store


