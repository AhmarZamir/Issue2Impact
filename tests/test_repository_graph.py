from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

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


def build_graph(llm, planner, critic, route="repository"):
    return RepositoryGraph(
        llm,
        [search_repository],
        router=StaticRouter(route),
        planner=planner,
        critic=critic,
    ).build(checkpointer=InMemorySaver())


def config(thread_id):
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 30,
    }


def initial_state(query):
    return {
        "user_query": query,
        "retry_count": 0,
        "messages": [HumanMessage(content=query)],
    }


def test_critic_approval_pauses_for_human_instead_of_finishing():
    planner = StaticPlanner()
    critic = SequenceCritic([approved_decision()])
    llm = ScriptedLLM([AIMessage(content="auth.py contains validate_token().")])
    graph = build_graph(llm, planner, critic)

    result = graph.invoke(initial_state("Investigate token validation."), config("pause-test"))

    assert result["plan_approved"] is True
    assert result.get("__interrupt__")
    payload = result["__interrupt__"][0].value
    assert payload["type"] == "human_approval"
    assert "Implementation Plan" in payload["plan"]


def test_human_can_approve_and_finalize_same_thread():
    planner = StaticPlanner()
    critic = SequenceCritic([approved_decision()])
    llm = ScriptedLLM([AIMessage(content="auth.py contains validate_token().")])
    graph = build_graph(llm, planner, critic)
    cfg = config("approve-test")

    paused = graph.invoke(initial_state("Investigate token validation."), cfg)
    assert paused.get("__interrupt__")

    result = graph.invoke(
        Command(resume={"approved": True, "feedback": "Looks good."}),
        config=cfg,
    )

    assert result["human_approved"] is True
    assert result["human_feedback"] == "Looks good."
    assert not result.get("__interrupt__")
    assert result["messages"][-1].content.startswith("Human-approved implementation plan")


def test_human_rejection_revises_plan_then_requests_approval_again():
    planner = StaticPlanner()
    critic = SequenceCritic([approved_decision(), approved_decision("Revision is sound.")])
    llm = ScriptedLLM([AIMessage(content="auth.py contains validate_token().")])
    graph = build_graph(llm, planner, critic)
    cfg = config("reject-revise-test")

    first_pause = graph.invoke(initial_state("Investigate token validation."), cfg)
    assert first_pause.get("__interrupt__")

    second_pause = graph.invoke(
        Command(
            resume={
                "approved": False,
                "feedback": "Add explicit malformed-token test coverage.",
            }
        ),
        config=cfg,
    )

    assert second_pause.get("__interrupt__")
    assert second_pause["retry_count"] == 1
    assert len(planner.calls) == 2
    assert "Human review rejected" in planner.calls[1][2]
    assert "malformed-token" in planner.calls[1][2]

    result = graph.invoke(
        Command(resume={"approved": True, "feedback": "Approved revision."}),
        config=cfg,
    )
    assert result["human_approved"] is True
    assert result["retry_count"] == 1


def test_repeated_human_rejection_hits_retry_limit():
    planner = StaticPlanner()
    critic = SequenceCritic([approved_decision(), approved_decision()])
    llm = ScriptedLLM([AIMessage(content="auth.py contains validate_token().")])
    graph = build_graph(llm, planner, critic)
    cfg = config("human-limit-test")

    paused = graph.invoke(initial_state("Investigate token validation."), cfg)
    assert paused.get("__interrupt__")

    paused_again = graph.invoke(
        Command(resume={"approved": False, "feedback": "Revise once."}),
        config=cfg,
    )
    assert paused_again.get("__interrupt__")
    assert paused_again["retry_count"] == 1

    result = graph.invoke(
        Command(resume={"approved": False, "feedback": "Still not acceptable."}),
        config=cfg,
    )

    assert result["retry_count"] == 2
    assert result["human_approved"] is False
    assert not result.get("__interrupt__")
    assert "within the retry limit" in result["messages"][-1].content


def test_general_route_does_not_require_human_approval():
    planner = StaticPlanner()
    critic = SequenceCritic([])
    llm = ScriptedLLM([AIMessage(content="Unit tests verify behavior.")])
    graph = build_graph(llm, planner, critic, route="general")

    result = graph.invoke(initial_state("What is unit testing?"), config("general-test"))

    assert result["route"] == "general"
    assert not result.get("__interrupt__")
    assert result["messages"][-1].content == "Unit tests verify behavior."
