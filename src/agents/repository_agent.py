from src.llm.model import get_llm
from src.prompts.repository_agent_prompt import REPOSITORY_AGENT_PROMPT

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    ToolMessage,
)


class RepositoryAgent:
    """Phase 4 manual tool loop kept as a learning reference."""

    def __init__(self, tools, llm=None, max_tool_rounds: int = 4):
        self.llm = llm or get_llm()
        self.tools = tools
        self.tools_by_name = {tool.name: tool for tool in tools}
        self.max_tool_rounds = max_tool_rounds
        self.llm_with_tools = self.llm.bind_tools(self.tools)

    def inspect_decision(self, query: str):

        messages = [
            SystemMessage(
                content=REPOSITORY_AGENT_PROMPT
            ),

            HumanMessage(
                content=query
            ),
        ]

        response = self.llm_with_tools.invoke(messages)

        return response

    def run(self, query: str):
        messages = [
            SystemMessage(content=REPOSITORY_AGENT_PROMPT),
            HumanMessage(content=query),
        ]

        for _ in range(self.max_tool_rounds + 1):
            response = self.llm_with_tools.invoke(messages)
            messages.append(response)

            if not response.tool_calls:
                return response.content

            if _ == self.max_tool_rounds:
                break

            for tool_call in response.tool_calls:
                tool = self.tools_by_name.get(tool_call["name"])
                if tool is None:
                    tool_result = f"Unknown tool: {tool_call['name']}"
                else:
                    tool_result = tool.invoke(tool_call["args"])

                messages.append(
                    ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tool_call["id"],
                    )
                )

        raise RuntimeError(
            f"Agent exceeded the maximum of {self.max_tool_rounds} tool rounds."
        )
