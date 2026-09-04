from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from src.agents.router import RouterAgent
from src.graph.state import AgentState
from src.agents.planner import PlannerAgent

class RepositoryGraph:
    """Build the Phase 6 routed Issue2Impact workflow."""

    def __init__(self, llm, tools, router=None):
        self.llm = llm
        self.tools = tools
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.router = router or RouterAgent(self.llm)
        self.planner =PlannerAgent(self.llm)

        
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
        """Route repository tool requests to ToolNode; otherwise finish."""
        last_message = state["messages"][-1]
        return "tools" if getattr(last_message, "tool_calls", None) else "end"

    def build(self):
        graph = StateGraph(AgentState)

        graph.add_node("router", self.router_node)
        graph.add_node("agent", self.agent_node)
        graph.add_node("tools", ToolNode(self.tools))
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
                "end": END,
            },
        )

        graph.add_edge("tools", "agent")
        graph.add_edge("general", END)
        graph.add_edge("unsupported", END)

        return graph.compile()
