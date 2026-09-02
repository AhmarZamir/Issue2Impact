from langchain_core.documents import Document

from src.retrieval.pipeline import RetrievalPipeline


class FakeVectorStore:
    def __init__(self, documents):
        self.documents = documents
        self.requested_k = None

    def similarity_search(self, query, k):
        self.requested_k = k
        return self.documents


class FakeReranker:
    def __init__(self):
        self.requested_k = None

    def rerank(self, query, documents, top_k):
        self.requested_k = top_k
        return [(document, 1.0) for document in documents[:top_k]]


def test_pipeline_retrieves_candidates_then_reranks_top_results():
    documents = [Document(page_content=f"chunk {index}") for index in range(5)]
    vector_store = FakeVectorStore(documents)
    reranker = FakeReranker()
    pipeline = RetrievalPipeline(
        vector_store,
        candidate_k=5,
        rerank_k=2,
        reranker=reranker,
    )

    results = pipeline.retrieve("authentication")

    assert vector_store.requested_k == 5
    assert reranker.requested_k == 2
    assert len(results) == 2
