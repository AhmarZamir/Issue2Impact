from src.ingestion.loader import load_repository
from src.ingestion.chunking import chunk_documents
from src.retrieval.vector_store import create_vector_store
from src.retrieval.retriever import retrieve_candidates
from src.retrieval.reranker import DocumetReranker
from src.retrieval.pipeline import RetrievalPipeline







# Loading Repository
documents = load_repository("demo_repo")



print(f"Loaded {len(documents)} documents from the repository.")

print("\n--------------------")


# chunking that documents
chunks = chunk_documents(documents)

print(f"Created {len(chunks)} chunks from the documents.")


# Storing that embeddings of chunking inot the vector store
vector_store = create_vector_store(chunks)


query = "How does user login work?"



results = RetrievalPipeline.retrieve(query)



print("\n--------------------")
print("\n--------------------")
print("\n--------------------")


print("\nRERANKED RESULTS")
print("================")

for index , (document , score) in enumerate(results):

    print(f"Reranked candidate {index + 1}:")
    print(f"Score : {score}")
    print(f"file path : {document.metadata["file_path"]}")

    print(f"Content : {document.page_content[:300]}")  # Print first 300 characters of the content



    









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


