from src.llm.model import get_llm

from langchain_core.messages import SystemMessage , HumanMessage



class RepositoryAgent:

    def __init__(self , repository_search_tool):

        self.llm = get_llm()

        self.tools =[repository_search_tool]


        self.llm_with_tools = (self.llm.bind_tools(self.tools))


    def inspect_decision(self , query:str):


        messsages =[
            SystemMessage(
                content ="""
You are an AI software repository analyst.

Your job is to answer questions about
a source-code repository.

If answering the question requires
information from the repository,
use the search_repository tool.

Do not invent repository details.


"""
            ),

            HumanMessage(
                content = query
            ),
        ]    


        response = self.llm_with_tools.invoke( messsages)



        return response 

    


