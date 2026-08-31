from langchain_core.tools import tool 


def create_repository_search_tool(retrieval_pipeline):



    @tool
    def search_repository(query:str):

        
        """
        Search the source-code repository for
        files and code relevant to a question.

        Use this tool whenever repository
        evidence is required.
        """


        results = retrieval_pipeline.retrieve(query)

        if not results:
            return "No relevant files found in the repository."


        formatted_results = []


        for index , (document , score) in enumerate(results , start =1,):

            file_path = document.metadata.get("file_path", "Unknown file path")

            chunk_index = document.metadata.get("chunk_index", "Unknown chunk index")

            formatted_results.append(
                f"""

reranked candidate {index}
Score: {score}
chunk index: {chunk_index}
File Path: {file_path}
Code : {document.page_content[:300]}

"""
)

            return "\n".join(formatted_results)


    return search_repository

