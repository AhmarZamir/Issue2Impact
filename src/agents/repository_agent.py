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

Your job is to investigate software issues using repository evidence.

Rules:

1. Use search_repository when repository-specific evidence is needed.
2. Never invent file names or code.
3. When repository evidence is available, mention relevant file paths.
4. Base conclusions on retrieved evidence.
5. After a tool result is provided, answer the user's question directly.
"""
            ),

            HumanMessage(content=query),
        ]

        # First call: model can choose a tool
        response = self.llm_with_tools.invoke(messages)

        print("\nFIRST CONTENT:", response.content)
        print("TOOL CALLS:", response.tool_calls)

        # If no tool is needed
        if not response.tool_calls:
            return response.content

        messages.append(response)

        # Execute tool calls
        for tool_call in response.tool_calls:

            if tool_call["name"] == "search_repository":

                tool_result = self.tools[0].invoke(
                    tool_call["args"]
                )

                print("\nTOOL RESULT:")
                print(tool_result)

                messages.append(
                    ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tool_call["id"],
                    )
                )

        # IMPORTANT:
        # Second call uses plain LLM, NOT llm_with_tools.
        final_response = self.llm.invoke(messages)

        return final_response.content