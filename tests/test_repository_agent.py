from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from src.agents.repository_agent import RepositoryAgent


class RecordingLLM:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.calls.append(list(messages))
        return next(self.responses)


@tool
def search_repository(query: str) -> str:
    """Search repository evidence."""
    return f"found {query} in auth.py"


def test_phase_4_loop_returns_tool_output_to_the_model():
    llm = RecordingLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_repository",
                        "args": {"query": "login"},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="Login is in auth.py."),
        ]
    )
    agent = RepositoryAgent([search_repository], llm=llm)

    answer = agent.run("Where is login?")

    assert answer == "Login is in auth.py."
    assert any(isinstance(message, ToolMessage) for message in llm.calls[1])
