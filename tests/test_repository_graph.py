from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from src.graph.repository_graph import RepositoryGraph


class ScriptedLLM:
    """Deterministic model used to test graph behavior without an API call."""

    def __init__(self, responses):
        self.responses = iter(responses)

    def bind_tools(self, tools):
        self.tools = tools
        return self

    def invoke(self, messages):
        return next(self.responses)


class StaticRouter:
    """Deterministic router used to isolate graph routing in tests."""

    def __init__(self, route, reason="test route"):
        self.route_name = route
        self.reason = reason

    def route(self, query):
        return SimpleNamespace(route=self.route_name, reason=self.reason)


class StaticPlanner:
    """Deterministic planner used to test investigator-to-planner handoff."""

    def __init__(self):
        self.calls = []

    def create_plan(self, issue, investigation):
        self.calls.append((issue, investigation))
        return SimpleNamespace(
            summary="Tighten token validation and logout behavior.",
            files_to_change=["auth.py"],
            steps=[
                "Update token validation behavior.",
                "Keep logout behavior consistent with validation.",
            ],
            tests=["Add invalid-token logout coverage."],
            risks=["Authentication behavior may regress for valid tokens."],
        )


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


def invoke(graph, query):
    return graph.invoke(
        {
            "user_query": query,
            "messages": [HumanMessage(content=query)],
        },
        config={"recursion_limit": 12},
    )


def test_router_sends_general_question_to_general_node():
    llm = ScriptedLLM([AIMessage(content="Unit tests verify behavior.")])
    graph = RepositoryGraph(
        llm,
        [search_repository],
        router=StaticRouter("general", "general programming question"),
        planner=StaticPlanner(),
    ).build()

    result = invoke(graph, "What is testing?")

    assert result["route"] == "general"
    assert result["route_reason"] == "general programming question"
    assert result["messages"][-1].content == "Unit tests verify behavior."
    assert "investigation" not in result
    assert "plan" not in result
    assert not any(isinstance(message, ToolMessage) for message in result["messages"])


def test_router_sends_unsupported_question_to_deterministic_response():
    llm = ScriptedLLM([])
    graph = RepositoryGraph(
        llm,
        [search_repository],
        router=StaticRouter("unsupported"),
        planner=StaticPlanner(),
    ).build()

    result = invoke(graph, "Write a romantic poem.")

    assert result["route"] == "unsupported"
    assert "outside Issue2Impact" in result["messages"][-1].content
    assert "investigation" not in result
    assert "plan" not in result
    assert not any(isinstance(message, ToolMessage) for message in result["messages"])


def test_repository_route_hands_investigation_to_planner():
    planner = StaticPlanner()
    llm = ScriptedLLM(
        [
            AIMessage(
                content="",
                tool_calls=[tool_call("search_repository", {"query": "login"}, "1")],
            ),
            AIMessage(content="auth.py contains the login implementation."),
        ]
    )
    graph = RepositoryGraph(
        llm,
        [search_repository],
        router=StaticRouter("repository"),
        planner=planner,
    ).build()

    result = invoke(graph, "Investigate login behavior and propose a fix.")

    assert result["route"] == "repository"
    assert result["investigation"] == "auth.py contains the login implementation."
    assert planner.calls == [
        (
            "Investigate login behavior and propose a fix.",
            "auth.py contains the login implementation.",
        )
    ]
    assert "Implementation Plan" in result["plan"]
    assert "auth.py" in result["plan"]
    assert result["messages"][-1].content == result["plan"]


def test_repository_route_supports_multi_step_tool_use_before_planning():
    planner = StaticPlanner()
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
            AIMessage(
                content=(
                    "auth.py contains validate_token() and logout(); logout depends "
                    "on token validation."
                )
            ),
        ]
    )
    graph = RepositoryGraph(
        llm,
        [search_repository, read_repository_file],
        router=StaticRouter("repository"),
        planner=planner,
    ).build()

    result = invoke(graph, "Investigate token validation and logout safety.")

    tool_messages = [
        message for message in result["messages"] if isinstance(message, ToolMessage)
    ]
    assert len(tool_messages) == 2
    assert "contents of auth.py" in tool_messages[-1].content
    assert "validate_token()" in result["investigation"]
    assert result["plan"].startswith("Implementation Plan")
    assert len(planner.calls) == 1
