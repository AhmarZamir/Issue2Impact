from typing import Literal

from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from src.prompts.router_prompt import ROUTER_PROMPT

class RouteDecision(BaseModel):

    route: Literal[
        "repository",
        "general",
        "unsupported",
    ] = Field(
        description="The workflow that should handle the request."
    )

    reason: str = Field(
        description="Short reason for selecting this route."
    )


class RouterAgent:

    def __init__(self, llm):
        self.llm = llm
        self.router_llm = self.llm.with_structured_output(RouteDecision)


    def route(self , query: str):
        """Classify the user query into one of the three routes."""
        messages =[
            SystemMessage(
                content = ROUTER_PROMPT
            ),
            HumanMessage(
                content = query
            ),
        ]    


        response = self.router_llm.invoke(messages)

        return response




    
