def retrieve_candidates(vectorStore, query, top_k=8):

    results =vectorStore.similarity_search(query , k=top_k,)


    return results