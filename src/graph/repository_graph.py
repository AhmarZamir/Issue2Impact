from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from src.agents.planner import PlannerAgent
from src.agents.router import RouterAgent
from src.graph.state import AgentState


class RepositoryGraph:
    """Build the Phase 7 routed investigator -> planner workflow."""

    def __init__(self, llm, tools, router=None, planner=None):
        self.llm = llm
        self.tools = tools
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.router = router or RouterAgent(self.llm)
        self.planner = planner

    def router_node(self, state: AgentState):
        """Classify the request and store the routing decision in graph state."""
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
        }

    @staticmethod
    def route_request(state: AgentState):
        """Select the workflow chosen by the router."""
        return state["route"]

    def general_node(self, state: AgentState):
        """Answer general software-engineering questions without repository tools."""
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
                {
                    "role": "user",
                    "content": state["user_query"],
                },
            ]
        )
        return {"messages": [response]}

    @staticmethod
    def unsupported_node(state: AgentState):
        """Return a deterministic response for requests outside project scope."""
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
        """Repository investigator: reason over messages and optionally call tools."""
        response = self.llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    @staticmethod
    def should_continue(state: AgentState):
        """Use tools while requested; otherwise hand the investigation to the planner."""
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
        """Persist the investigator's final evidence summary for downstream agents."""
        last_message = state["messages"][-1]
        investigation = self._content_to_text(last_message.content).strip()

        if not investigation:
            investigation = "Repository investigation produced no textual summary."

        return {"investigation": investigation}

    @staticmethod
    def _format_plan(plan):
        files = "\n".join(f"- {item}" for item in plan.files_to_change) or "- None"
        steps = "\n".join(
            f"{index}. {step}"
            for index, step in enumerate(plan.steps, start=1)
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
        """Create an evidence-grounded implementation plan from the investigation."""
        planner = self.planner or PlannerAgent(self.llm)
        plan = planner.create_plan(
            issue=state["user_query"],
            investigation=state["investigation"],
        )
        formatted_plan = self._format_plan(plan)

        return {
            "plan": formatted_plan,
            "messages": [AIMessage(content=formatted_plan)],
        }

    def build(self):
        graph = StateGraph(AgentState)

        graph.add_node("router", self.router_node)
        graph.add_node("agent", self.agent_node)
        graph.add_node("tools", ToolNode(self.tools))
        graph.add_node("investigation_complete", self.capture_investigation_node)
        graph.add_node("planner", self.planner_node)
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
        graph.add_edge("planner", END)
        graph.add_edge("general", END)
        graph.add_edge("unsupported", END)

        return graph.compile()
