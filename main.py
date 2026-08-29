from src.ingestion.loader import load_repository
from src.ingestion.chunking import chunk_documents
from src.retrieval.vector_store import create_vector_store



# Loading Repository
documents = load_repository("demo_repo")



print(f"Loaded {len(documents)} documents from the repository.")

print("\n--------------------")


# chunking that documents
chunks = chunk_documents(documents)

print(f"Created {len(chunks)} chunks from the documents.")



# Storing that embeddings of chunking inot the vector store
vector_store = create_vector_store(chunks)

query = "How to authenticate a user?"

results = vector_store.similarity_search(query, k=3)


for index , result in enumerate(results):

    print("=================")

    print(f"file path : {result.metadata["file_path"]}")

    print(f"chunk index : {result.metadata["chunk_index"]}")

    print(f"Content: {result.page_content}")



    









# for chunk in chunks:

#     print("\n--------------------")

#     print(
#         "File:",
#         chunk.metadata["file_path"]
#     )

#     print(
#         "chuked index:",
#         chunk.metadata["chunk_index"]
#     )

#     print("\nContent:")

#     print(len(chunk.page_content[:200]))


