from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, SystemMessage

from src.prompts.critic_prompt import CRITIC_PROMPT


class CriticDecision(BaseModel):
    approved: bool = Field(
        description="Whether the implementation plan is sufficiently grounded and complete."
    )
    feedback: str = Field(
        description="Specific explanation of why the plan is acceptable or what must improve."
    )
    needs_more_evidence: bool = Field(
        description="Whether the repository investigation needs more evidence before replanning."
    )


class CriticAgent:
    def __init__(self, llm):
        self.llm = llm
        self.critic_llm = self.llm.with_structured_output(CriticDecision)

    def review(self, issue: str, investigation: str, plan: str):
        return self.critic_llm.invoke(
            [
                SystemMessage(content=CRITIC_PROMPT),
                HumanMessage(
                    content=f"""
Original issue:

{issue}

Repository investigation:

{investigation}

Proposed implementation plan:

{plan}
"""
                ),
            ]
        )
