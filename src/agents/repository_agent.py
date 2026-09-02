from src.llm.model import get_llm

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    ToolMessage,
)


class RepositoryAgent:

    def __init__(self, tools):

        self.llm = get_llm()

        self.tools = tools

        self.llm_with_tools = self.llm.bind_tools(self.tools)

    def inspect_decision(self, query: str):

        messages = [
            SystemMessage(
                content="""
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
                content=query
            ),
        ]

        response = self.llm_with_tools.invoke(messages)

        return response

    def run(self, query: str):

        messages = [
            SystemMessage(
                content="""
You are Issue2Impact, an AI software repository analyst.

Use repository evidence when needed.
Never invent file names or code.
"""
            ),
            HumanMessage(content=query),

        ]


        # First Gemini LLM call to decide which tools to use
        response = self.llm_with_tools.invoke(messages)
        print("\nFIRST CONTENT:", response.content)
        print("TOOL CALLS:", response.tool_calls)


        if not response.tool_calls:
            return response.content


        tool_results = []

        for tool_call in response.tool_calls:
            if tool_call["name"] == "search_repository":
                tool_result = self.tools[0].invoke(
    tool_call["args"]
)
                print("\nTOOL RESULT:", tool_result)


                tool_results.append(str(tool_result))


        repository_evidence = "\n\n".join(tool_results)


        final_messages = [
            SystemMessage(
                content="""  You are Issue2Impact, an AI software repository analyst.

Answer the user's question using only the
repository evidence provided below.

Mention relevant file paths.
Do not invent repository details.
"""),

            HumanMessage(
                content=f"""
                        User question:{query} Repository evidence: 

{repository_evidence}

Now answer the user's original question directly.
"""
        ),
    ]

        final_response = self.llm.invoke(final_messages)


        print("\nFINAL CONTENT:", final_response.content)


        return final_response.content
                