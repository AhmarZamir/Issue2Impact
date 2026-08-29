from sentence_transformers import CrossEncoder


class DocumetReranker:


    def __init__(self):
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")



    def rerank(self , query:str , documents, top_k:int =3):
        if not documents:
            return []    

        pairs = [
            [query , document.page_content] 
                 for document in documents
                ]

        scores = self.model.predict(pairs)


        scored_docuemts = list(zip(documents , scores))

        scored_docuemts.sort(key=lambda x: x[1] , reverse=True)


        return scored_docuemts[:top_k]




        