from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, SystemMessage

from src.prompts.planner_prompt import PLANNER_PROMPT


class ImplementationPlan(BaseModel):
    summary: str = Field(description="Short summary of the proposed fix.")
    files_to_change: list[str] = Field(
        description=(
            "Repository files likely requiring changes. "
            "Only include files supported by evidence."
        )
    )
    steps: list[str] = Field(description="Ordered implementation steps.")
    tests: list[str] = Field(description="Tests that should be added or updated.")
    risks: list[str] = Field(description="Potential risks or side effects.")


class PlannerAgent:
    def __init__(self, llm):
        self.llm = llm
        self.planner_llm = self.llm.with_structured_output(ImplementationPlan)

    def create_plan(
        self,
        issue: str,
        investigation: str,
        critic_feedback: str | None = None,
    ):
        feedback_section = ""
        if critic_feedback:
            feedback_section = f"""

Previous critic feedback:

{critic_feedback}

Revise the plan to address this feedback.
"""

        return self.planner_llm.invoke(
            [
                SystemMessage(content=PLANNER_PROMPT),
                HumanMessage(
                    content=f"""
Original issue:

{issue}

Repository investigation:

{investigation}
{feedback_section}
"""
                ),
            ]
        )
