from typing import Annotated, Literal
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


RouteType = Literal[
    "repository",
    "general",
    "unsupported",
]


class AgentState(TypedDict, total=False):
    """Shared state for one Issue2Impact workflow execution."""

    messages: Annotated[list[BaseMessage], add_messages]

    # Original request and routing metadata
    user_query: str
    route: RouteType
    route_reason: str

    # Investigator/planner handoff artifacts
    investigation: str
    plan: str

    # Critic + reflection state
    critic_feedback: str
    plan_approved: bool
    needs_more_evidence: bool
    retry_count: int

    # Phase 9 human-in-the-loop state
    human_approved: bool
    human_feedback: str
