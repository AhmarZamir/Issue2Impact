from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from src.graph.state import AgentState


class RepositoryGraph:
    """Build the Phase 5 agent -> tools -> agent workflow."""

    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools
        self.llm_with_tools = self.llm.bind_tools(self.tools)

    def agent_node(self, state: AgentState):
        response = self.llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    @staticmethod
    def should_continue(state: AgentState):
        """Route tool requests to ToolNode; otherwise finish the run."""
        last_message = state["messages"][-1]
        return "tools" if getattr(last_message, "tool_calls", None) else "end"

    def build(self):
        graph = StateGraph(AgentState)
        graph.add_node("agent", self.agent_node)
        graph.add_node("tools", ToolNode(self.tools))

        graph.add_edge(START, "agent")
        graph.add_conditional_edges(
            "agent",
            self.should_continue,
            {"tools": "tools", "end": END},
        )
        graph.add_edge("tools", "agent")

        return graph.compile()
