from src.retrieval.retriever import retrieve_candidates
from src.retrieval.reranker import DocumentReranker

class RetrievalPipeline:

    def __init__(
        self,
        vector_store,
        candidate_k: int = 8,
        rerank_k: int = 3,
        reranker=None,
    ):

        self.vector_store = vector_store
        self.candidate_k = candidate_k
        self.rerank_k = rerank_k
        self.reranker = reranker or DocumentReranker()

    def retrieve(self, query: str):
        candidates = retrieve_candidates(
            self.vector_store,
            query,
            top_k=self.candidate_k,
        )
        return self.reranker.rerank(
            query,
            candidates,
            top_k=self.rerank_k,
        )
