from src.llm.model import get_llm

from langchain_core.messages import SystemMessage , HumanMessage , ToolMessage



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


    def run (self , query:str):

        messages =[

            SystemMessage(
                content ="""You are Issue2Impact, an AI software
repository analyst.

Your job is to investigate software
issues using repository evidence.

Rules:

1. Use search_repository whenever
   repository-specific evidence is needed.

2. Never invent file names or code.

3. When repository evidence is available,
   mention the relevant file paths.

4. Base conclusions on retrieved evidence.
"""
            ),

            HumanMessage(
                content = query

            )
        ]


        response = self.llm_with_tools.invoke(messages)


        messages.append(response)


        if response.tool_calls:

            
            for tool_call in response.tool_calls:

                if tool_call["name"] == "search_repository":

                    tool_result = self.tools[0].invoke(tool_call["args"])

                    messages.append(
                        ToolMessage(
                            content = tool_result,
                            tool_call_id = tool_call["id"],
                    ))


                    final_response =self.llm_with_tools.invoke(messages)


                    return final_response.content



                return response.content






