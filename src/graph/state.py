from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph.state import add_messages


class AgentState(TypedDict):
    """
    Represents the state of an agent.
    """

    messages = Annotated[list , add_messages]