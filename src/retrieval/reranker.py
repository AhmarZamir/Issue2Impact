class DocumentReranker:

    def __init__(self, model=None):
        if model is None:
            from sentence_transformers import CrossEncoder

            model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        self.model = model



    def rerank(self , query:str , documents, top_k:int =3):
        if not documents:
            return []    

        pairs = [
            [query , document.page_content] 
                 for document in documents
                ]

        scores = self.model.predict(pairs)


        scored_documents = list(zip(documents, scores))

        scored_documents.sort(key=lambda item: item[1], reverse=True)

        return scored_documents[:top_k]

# Backwards-compatible alias for code written during Phase 3.
DocumetReranker = DocumentReranker
