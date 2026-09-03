from typing import Annotated , Literal
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


RouteType = Literal[
    "Repository",
    "General",
    "Unsupported",
]

class AgentState(TypedDict , total=False):
    """Shared message history for one repository investigation."""

    messages: Annotated[list[BaseMessage], add_messages]

    user_query: str
    route: RouteType
    route_reason: str


