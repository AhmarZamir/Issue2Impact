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
    user_query: str
    route: RouteType
    route_reason: str
