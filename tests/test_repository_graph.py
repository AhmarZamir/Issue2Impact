from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from src.graph.repository_graph import RepositoryGraph


class ScriptedLLM:
    def __init__(self, responses):
        self.responses = iter(responses)

    def bind_tools(self, tools):
        self.tools = tools
        return self

    def invoke(self, messages):
        return next(self.responses)


class StaticRouter:
    def __init__(self, route, reason="test route"):
        self.route_name = route
        self.reason = reason

    def route(self, query):
        return SimpleNamespace(route=self.route_name, reason=self.reason)


class StaticPlanner:
    def __init__(self):
        self.calls = []

    def create_plan(self, issue, investigation, critic_feedback=None):
        self.calls.append((issue, investigation, critic_feedback))
        return SimpleNamespace(
            summary="Tighten authentication handling.",
            files_to_change=["auth.py"],
            steps=["Update token validation behavior."],
            tests=["Add invalid-token coverage."],
            risks=["Valid-token behavior could regress."],
        )


class SequenceCritic:
    def __init__(self, decisions):
        self.decisions = iter(decisions)
        self.calls = []

    def review(self, issue, investigation, plan):
        self.calls.append((issue, investigation, plan))
        return next(self.decisions)


@tool
def search_repository(query: str) -> str:
    """Search test repository evidence."""
    return f"auth.py matches {query}"


def tool_call(name, args, call_id):
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def invoke(graph, query):
    return graph.invoke(
        {
            "user_query": query,
            "retry_count": 0,
            "messages": [HumanMessage(content=query)],
        },
        config={"recursion_limit": 30},
    )


def approved_decision(feedback="Plan is grounded and complete."):
    return SimpleNamespace(
        approved=True,
        feedback=feedback,
        needs_more_evidence=False,
    )


def rejected_plan_decision(feedback="Plan needs revision."):
    return SimpleNamespace(
        approved=False,
        feedback=feedback,
        needs_more_evidence=False,
    )


def test_general_route_still_ends_without_planner_or_critic():
    llm = ScriptedLLM([AIMessage(content="Unit tests verify behavior.")])
    graph = RepositoryGraph(
        llm,
        [search_repository],
        router=StaticRouter("general"),
        planner=StaticPlanner(),
        critic=SequenceCritic([]),
    ).build()

    result = invoke(graph, "What is unit testing?")

    assert result["route"] == "general"
    assert result["messages"][-1].content == "Unit tests verify behavior."
    assert "investigation" not in result
    assert "plan" not in result


def test_critic_can_approve_first_plan_without_retry():
    planner = StaticPlanner()
    critic = SequenceCritic([approved_decision()])
    llm = ScriptedLLM([AIMessage(content="auth.py contains validate_token().")])
    graph = RepositoryGraph(
        llm,
        [search_repository],
        router=StaticRouter("repository"),
        planner=planner,
        critic=critic,
    ).build()

    result = invoke(graph, "Investigate token validation and propose a fix.")

    assert result["plan_approved"] is True
    assert result["retry_count"] == 0
    assert len(planner.calls) == 1
    assert len(critic.calls) == 1
    assert result["messages"][-1].content.startswith("Critic-approved implementation plan")


def test_plan_revision_loop_stops_after_critic_approval():
    planner = StaticPlanner()
    critic = SequenceCritic(
        [
            rejected_plan_decision("Add a stronger test step."),
            approved_decision("Revised plan is acceptable."),
        ]
    )
    llm = ScriptedLLM([AIMessage(content="auth.py contains validate_token().")])
    graph = RepositoryGraph(
        llm,
        [search_repository],
        router=StaticRouter("repository"),
        planner=planner,
        critic=critic,
    ).build()

    result = invoke(graph, "Investigate token validation and propose a fix.")

    assert result["plan_approved"] is True
    assert result["retry_count"] == 1
    assert len(planner.calls) == 2
    assert planner.calls[1][2] == "Add a stronger test step."
    assert len(critic.calls) == 2


def test_repeated_rejection_hits_retry_limit_instead_of_recursion_error():
    planner = StaticPlanner()
    critic = SequenceCritic(
        [
            rejected_plan_decision("First rejection."),
            rejected_plan_decision("Second rejection."),
            rejected_plan_decision("Third rejection."),
        ]
    )
    llm = ScriptedLLM([AIMessage(content="auth.py contains validate_token().")])
    graph = RepositoryGraph(
        llm,
        [search_repository],
        router=StaticRouter("repository"),
        planner=planner,
        critic=critic,
    ).build()

    result = invoke(graph, "Investigate token validation and propose a fix.")

    assert result["plan_approved"] is False
    assert result["retry_count"] == 2
    assert len(planner.calls) == 3
    assert len(critic.calls) == 3
    assert "within the retry limit" in result["messages"][-1].content
