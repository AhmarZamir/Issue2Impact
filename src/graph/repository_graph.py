from langgraph.graph import Start , End , StateGraph
from langgraph.prebuilt import ToolNode

from src.graph.state import AgentState


class RepositoryGraph:
    
    def __init__(self, llm , tools):
        self.llm = llm
        self.tools = tools

        self.llm_with_tools = self.llm.bind_tools(self.tools)


    def agent_node(self , state: AgentState):

        response = self.llm_with_tools.invoke(state["messages"]) 


        return {
           " messages": [response]
        }


        def build(self):

           graph = StateGraph(
            AgentState
        )


           graph.add_node(
            "agent",
            self.agent_node,
        )


           graph.add_node(
            "tools",
            ToolNode(
                self.tools
            ),
        )


           graph.add_edge(
            START,
            "agent",
        )


           graph.add_conditional_edges(
            "agent",
            self.should_continue,
            {
                "tools": "tools",
                "end": END,
            },
        )


           graph.add_edge(
            "tools",
            "agent",
        )


           return graph.compile()    