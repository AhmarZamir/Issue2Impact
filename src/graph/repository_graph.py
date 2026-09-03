from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from src.graph.state import AgentState
from langchain_core.messages import AIMessage
from src.agents.router import RouterAgent



class RepositoryGraph:
    """Build the Phase 5 agent -> tools -> agent workflow."""

    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.router = RouterAgent(
        self.llm
    )

    @staticmethod
    def route_request(state: AgentState,):

        return state["route"]


    def general_node(self, state: AgentState):

        query = state["user_query"]

        response = self.llm.invoke(
        [
            {
                "role": "system",
                "content": (
                    "You are Issue2Impact. "
                    "Answer general software "
                    "engineering questions "
                    "clearly and concisely."
                ),
            },
            {
                "role": "user",
                "content": query,
            },
        ]
    )

        return {"messages": response}

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


                       # ToolNode(self.tools)
                       # It automatically:
       # Reads the requested tool name.
       # Finds the matching Python tool.
       # Passes the arguments.
       # Executes the tool.
       # Creates a ToolMessage.
       # Adds the result to state.


        graph.add_node("tools", ToolNode(self.tools))

        graph.add_edge(START, "agent")
        graph.add_conditional_edges(
            "agent",
            self.should_continue,
            {"tools": "tools", "end": END},
        )
        graph.add_edge("tools", "agent")

        return graph.compile()


    def router_node(self, state: AgentState):
        # Use the router agent to classify the query into one of three routes.

        query = state.get("user_query")

        if not query:
            for message in reversed(state["messages"]):
                if message.type =="Human":
                    query = message.content
                    break


        decision = self.router.route(query)

        return {
        "user_query": query,
        "route": decision.route,
        "route_reason": decision.reason,
    }



        