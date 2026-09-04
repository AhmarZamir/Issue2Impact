from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt

from src.agents.critic import CriticAgent
from src.agents.planner import PlannerAgent
from src.agents.router import RouterAgent
from src.graph.state import AgentState


MAX_RETRIES = 2


class RepositoryGraph:
    """Build the Phase 9 routed workflow with reflection and human approval."""

    def __init__(self, llm, tools, router=None, planner=None, critic=None):
        self.llm = llm
        self.tools = tools
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.router = router or RouterAgent(self.llm)
        self.planner = planner
        self.critic = critic

    def router_node(self, state: AgentState):
        query = state.get("user_query")
        if not query:
            for message in reversed(state.get("messages", [])):
                if getattr(message, "type", None) == "human":
                    query = message.content
                    break
        if not query:
            raise ValueError("No user query was provided to the graph.")

        decision = self.router.route(query)
        return {
            "user_query": query,
            "route": decision.route,
            "route_reason": decision.reason,
            "retry_count": state.get("retry_count", 0),
        }

    @staticmethod
    def route_request(state: AgentState):
        return state["route"]

    def general_node(self, state: AgentState):
        response = self.llm.invoke(
            [
                {
                    "role": "system",
                    "content": (
                        "You are Issue2Impact. Answer general software-engineering "
                        "and programming questions clearly and concisely. Do not "
                        "claim to have inspected repository code on this route."
                    ),
                },
                {"role": "user", "content": state["user_query"]},
            ]
        )
        return {"messages": [response]}

    @staticmethod
    def unsupported_node(state: AgentState):
        return {
            "messages": [
                AIMessage(
                    content=(
                        "This request is outside Issue2Impact's software repository "
                        "analysis scope."
                    )
                )
            ]
        }

    def agent_node(self, state: AgentState):
        response = self.llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    @staticmethod
    def should_continue(state: AgentState):
        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", None):
            return "tools"
        return "investigation_complete"

    @staticmethod
    def _content_to_text(content):
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict):
                    text = block.get("text")
                    if text:
                        parts.append(text)
                else:
                    parts.append(str(block))
            return "\n".join(parts)
        return str(content)

    def capture_investigation_node(self, state: AgentState):
        last_message = state["messages"][-1]
        investigation = self._content_to_text(last_message.content).strip()
        if not investigation:
            investigation = "Repository investigation produced no textual summary."
        return {"investigation": investigation}

    @staticmethod
    def _format_plan(plan):
        files = "\n".join(f"- {item}" for item in plan.files_to_change) or "- None"
        steps = "\n".join(
            f"{index}. {step}" for index, step in enumerate(plan.steps, start=1)
        ) or "1. No implementation step could be recommended from the evidence."
        tests = "\n".join(f"- {item}" for item in plan.tests) or "- None"
        risks = "\n".join(f"- {item}" for item in plan.risks) or "- None"
        return (
            "Implementation Plan\n\n"
            f"Summary:\n{plan.summary}\n\n"
            f"Files to change:\n{files}\n\n"
            f"Steps:\n{steps}\n\n"
            f"Tests:\n{tests}\n\n"
            f"Risks:\n{risks}"
        )

    def planner_node(self, state: AgentState):
        planner = self.planner or PlannerAgent(self.llm)
        plan = planner.create_plan(
            issue=state["user_query"],
            investigation=state["investigation"],
            critic_feedback=state.get("critic_feedback"),
        )
        formatted_plan = self._format_plan(plan)
        return {
            "plan": formatted_plan,
            "messages": [AIMessage(content=formatted_plan)],
        }

    def critic_node(self, state: AgentState):
        critic = self.critic or CriticAgent(self.llm)
        decision = critic.review(
            issue=state["user_query"],
            investigation=state["investigation"],
            plan=state["plan"],
        )
        return {
            "critic_feedback": decision.feedback,
            "plan_approved": decision.approved,
            "needs_more_evidence": decision.needs_more_evidence,
        }

    @staticmethod
    def after_critic(state: AgentState):
        if state.get("plan_approved", False):
            return "approved"
        if state.get("retry_count", 0) >= MAX_RETRIES:
            return "max_retries"
        if state.get("needs_more_evidence", False):
            return "reinvestigate"
        return "revise_plan"

    @staticmethod
    def plan_retry_node(state: AgentState):
        return {"retry_count": state.get("retry_count", 0) + 1}

    @staticmethod
    def evidence_retry_node(state: AgentState):
        return {"retry_count": state.get("retry_count", 0) + 1}

    @staticmethod
    def reinvestigation_context_node(state: AgentState):
        return {
            "messages": [
                HumanMessage(
                    content=(
                        "The Critic rejected the current plan because more repository "
                        "evidence is required.\n\nCritic feedback:\n"
                        f"{state.get('critic_feedback', 'No feedback provided.')}\n\n"
                        "Continue investigating the repository and focus on the missing "
                        "evidence. Do not create an implementation plan."
                    )
                )
            ]
        }

    @staticmethod
    def human_approval_node(state: AgentState):
        decision = interrupt(
            {
                "type": "human_approval",
                "issue": state["user_query"],
                "investigation": state["investigation"],
                "plan": state["plan"],
                "critic_feedback": state.get("critic_feedback", "Approved."),
                "message": "The Critic approved this plan. Approve or reject it.",
            }
        )

        if not isinstance(decision, dict):
            decision = {"approved": bool(decision), "feedback": ""}

        return {
            "human_approved": bool(decision.get("approved", False)),
            "human_feedback": str(decision.get("feedback", "")).strip(),
        }

    @staticmethod
    def after_human_approval(state: AgentState):
        return "approved" if state.get("human_approved", False) else "rejected"

    @staticmethod
    def human_rejection_node(state: AgentState):
        feedback = state.get("human_feedback") or "The human rejected the plan."
        return {
            "critic_feedback": (
                "Human review rejected the previous implementation plan.\n\n"
                f"Human feedback:\n{feedback}"
            )
        }

    @staticmethod
    def human_retry_node(state: AgentState):
        return {"retry_count": state.get("retry_count", 0) + 1}

    @staticmethod
    def after_human_retry(state: AgentState):
        if state.get("retry_count", 0) >= MAX_RETRIES:
            return "max_retries"
        return "revise"

    @staticmethod
    def finalize_node(state: AgentState):
        human_feedback = state.get("human_feedback") or "Approved without additional feedback."
        return {
            "messages": [
                AIMessage(
                    content=(
                        "Human-approved implementation plan\n\n"
                        f"{state['plan']}\n\n"
                        "Critic review:\n"
                        f"{state.get('critic_feedback', 'Approved.')}\n\n"
                        "Human review:\n"
                        f"{human_feedback}"
                    )
                )
            ]
        }

    @staticmethod
    def retry_exhausted_node(state: AgentState):
        return {
            "messages": [
                AIMessage(
                    content=(
                        "The workflow could not reach an approved implementation plan "
                        "within the retry limit.\n\n"
                        "Latest review feedback:\n"
                        f"{state.get('critic_feedback', 'N/A')}\n\n"
                        "Latest plan:\n"
                        f"{state.get('plan', 'N/A')}"
                    )
                )
            ]
        }

    def build(self, checkpointer=None):
        graph = StateGraph(AgentState)

        graph.add_node("router", self.router_node)
        graph.add_node("agent", self.agent_node)
        graph.add_node("tools", ToolNode(self.tools))
        graph.add_node("investigation_complete", self.capture_investigation_node)
        graph.add_node("planner", self.planner_node)
        graph.add_node("critic", self.critic_node)
        graph.add_node("plan_retry", self.plan_retry_node)
        graph.add_node("evidence_retry", self.evidence_retry_node)
        graph.add_node("reinvestigation_context", self.reinvestigation_context_node)
        graph.add_node("human_approval", self.human_approval_node)
        graph.add_node("human_rejection", self.human_rejection_node)
        graph.add_node("human_retry", self.human_retry_node)
        graph.add_node("finalize", self.finalize_node)
        graph.add_node("retry_exhausted", self.retry_exhausted_node)
        graph.add_node("general", self.general_node)
        graph.add_node("unsupported", self.unsupported_node)

        graph.add_edge(START, "router")
        graph.add_conditional_edges(
            "router",
            self.route_request,
            {
                "repository": "agent",
                "general": "general",
                "unsupported": "unsupported",
            },
        )

        graph.add_conditional_edges(
            "agent",
            self.should_continue,
            {
                "tools": "tools",
                "investigation_complete": "investigation_complete",
            },
        )

        graph.add_edge("tools", "agent")
        graph.add_edge("investigation_complete", "planner")
        graph.add_edge("planner", "critic")

        graph.add_conditional_edges(
            "critic",
            self.after_critic,
            {
                "approved": "human_approval",
                "revise_plan": "plan_retry",
                "reinvestigate": "evidence_retry",
                "max_retries": "retry_exhausted",
            },
        )

        graph.add_edge("plan_retry", "planner")
        graph.add_edge("evidence_retry", "reinvestigation_context")
        graph.add_edge("reinvestigation_context", "agent")

        graph.add_conditional_edges(
            "human_approval",
            self.after_human_approval,
            {
                "approved": "finalize",
                "rejected": "human_rejection",
            },
        )
        graph.add_edge("human_rejection", "human_retry")
        graph.add_conditional_edges(
            "human_retry",
            self.after_human_retry,
            {
                "revise": "planner",
                "max_retries": "retry_exhausted",
            },
        )

        graph.add_edge("finalize", END)
        graph.add_edge("retry_exhausted", END)
        graph.add_edge("general", END)
        graph.add_edge("unsupported", END)

        return graph.compile(checkpointer=checkpointer)
