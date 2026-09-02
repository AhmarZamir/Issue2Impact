from src.ingestion.loader import load_repository
from src.ingestion.chunking import chunk_documents
from src.retrieval.vector_store import create_vector_store
from src.retrieval.pipeline import RetrievalPipeline 
from evaluation.evaluate_retrieval import evaluate_retrieval

from src.tools.repository_tools import create_repository_search_tool, read_repository_file
from src.agents.repository_agent import RepositoryAgent


from src.retrieval.retriever import retrieve_candidates
from src.retrieval.reranker import DocumetReranker










# Loading Repository
documents = load_repository("demo_repo")



print(f"Loaded {len(documents)} documents from the repository.")

print("\n--------------------")


# chunking that documents
chunks = chunk_documents(documents)

print(f"Created {len(chunks)} chunks from the documents.")


# Storing that embeddings of chunking inot the vector store
vector_store = create_vector_store(chunks)


query = """
There seems to be an issue with authentication.

Where is token validation implemented and
what part of the repository should I inspect?
"""








# candidates = retrieve_candidates(vector_store , query , top_k=8)

# print("Retrieved Candidates")
# print("\n--------------------")

# for index , document in enumerate(candidates):
#     print(f"Candidate {index + 1}:")
#     print(f"chunk index: {document.metadata['chunk_index']}")
#     print(f"File Path: {document.metadata['file_path']}")
#     print(f"Content: {document.page_content[:300]}")  


# reranked = DocumetReranker().rerank(query , candidates , top_k=3)


# print("\nReranked Results")
# print("================")

# for reranked_index , (document , score) in enumerate(reranked):
#     print(f"Reranked candidate {reranked_index + 1}:")
#     print(f"Score : {score}")
#     print(f"chunk index: {document.metadata['chunk_index']}")
#     print(f"File Path: {document.metadata['file_path']}")
#     print(f"Content: {document.page_content[:300]}")  # Print first 300 characters of the content

    

RetrievalPipe = RetrievalPipeline(vector_store)

tool = create_repository_search_tool(RetrievalPipe )

agent = RepositoryAgent([tool , read_repository_file])



results = agent.inspect_decision(query)
# results = agent.run(query)



print(f"Results Content {results}")

# print(f"tool called {results.tool_calls}")







print("\n--------------------")
print("\n--------------------")
print("\n--------------------")


print("\nRERANKED RESULTS")
print("================")


# for index , (document , score) in enumerate(results):

#     print(f"Reranked candidate {index + 1}:")
#     print(f"Score : {score}")
#     print(f"file path : {document.metadata["file_path"]}")

#     print(f"Content : {document.page_content[:300]}")  # Print first 300 characters of the content




    









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







# Evaluation of the retrieval pipeline using predefined test cases


# eval_result = evaluate_retrieval(RetrievalPipe)

# print(f"Retrieval Evaluation Accuracy: {eval_result:.2f}%")
