from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from src.graph.repository_graph import RepositoryGraph


class ScriptedLLM:
    """Small deterministic model used to test routing without an API call."""

    def __init__(self, responses):
        self.responses = iter(responses)

    def bind_tools(self, tools):
        self.tools = tools
        return self

    def invoke(self, messages):
        return next(self.responses)


@tool
def search_repository(query: str) -> str:
    """Search test repository evidence."""
    return f"auth.py matches {query}"


@tool
def read_repository_file(file_path: str) -> str:
    """Read a test repository file."""
    return f"contents of {file_path}"


def tool_call(name, args, call_id):
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def test_graph_answers_general_question_without_a_tool():
    llm = ScriptedLLM([AIMessage(content="Unit tests verify behavior.")])
    graph = RepositoryGraph(llm, [search_repository]).build()

    result = graph.invoke({"messages": [HumanMessage(content="What is testing?")]})

    assert result["messages"][-1].content == "Unit tests verify behavior."
    assert not any(isinstance(message, ToolMessage) for message in result["messages"])


def test_graph_executes_search_and_returns_to_agent():
    llm = ScriptedLLM(
        [
            AIMessage(
                content="",
                tool_calls=[tool_call("search_repository", {"query": "login"}, "1")],
            ),
            AIMessage(content="Login is implemented in auth.py."),
        ]
    )
    graph = RepositoryGraph(llm, [search_repository]).build()

    result = graph.invoke({"messages": [HumanMessage(content="Where is login?")]})

    assert [type(message).__name__ for message in result["messages"]] == [
        "HumanMessage",
        "AIMessage",
        "ToolMessage",
        "AIMessage",
    ]
    assert "auth.py" in result["messages"][2].content


def test_graph_supports_multi_step_tool_use():
    llm = ScriptedLLM(
        [
            AIMessage(
                content="",
                tool_calls=[tool_call("search_repository", {"query": "tokens"}, "1")],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    tool_call(
                        "read_repository_file",
                        {"file_path": "auth.py"},
                        "2",
                    )
                ],
            ),
            AIMessage(content="Token validation is in auth.py."),
        ]
    )
    graph = RepositoryGraph(
        llm,
        [search_repository, read_repository_file],
    ).build()

    result = graph.invoke(
        {"messages": [HumanMessage(content="Investigate token validation.")]},
        config={"recursion_limit": 10},
    )

    tool_messages = [
        message for message in result["messages"] if isinstance(message, ToolMessage)
    ]
    assert len(tool_messages) == 2
    assert "contents of auth.py" in tool_messages[-1].content
    assert result["messages"][-1].content == "Token validation is in auth.py."
