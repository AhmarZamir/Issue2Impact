from typing import Annotated
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Shared message history for one repository investigation."""

    messages: Annotated[list[BaseMessage], add_messages]
