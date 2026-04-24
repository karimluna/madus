"""LangGraph graph construction.
Flow: START -> extract -> orchestrator -> [text, image, table] -> critic
       -> (summarizer | orchestrator on retry) -> END

Fan-in to critic uses LangGraph's barrier sync: the critic only fires
when all three predecessor agents have committed their state updates.
This is why Optional[str] fields matter — during parallel execution
the state is partially populated until all agents return.

ref: LangGraph branching: https://langchain-ai.github.io/langgraph/how-tos/branching/
ref: MDocAgent: https://arxiv.org/abs/2503.13964
"""

from langgraph.graph import StateGraph, START, END

from core.models import DocumentState
from services.reasoning.graph.nodes.extraction import extraction_node
from services.reasoning.graph.nodes.orchestrator import orchestrator_node
from services.reasoning.graph.nodes.text_agent import text_agent_node
from services.reasoning.graph.nodes.image_agent import image_agent_node
from services.reasoning.graph.nodes.table_agent import table_agent_node
from services.reasoning.graph.nodes.critic import (
    critic_node,
    route_after_critic,
)
from services.reasoning.graph.nodes.summarizer import summarizer_node


def build_graph():
    """Construct and compile the MADUS reasoning graph."""
    g = StateGraph(DocumentState)

    # Register nodes
    g.add_node("extract", extraction_node)
    g.add_node("orchestrator", orchestrator_node)
    g.add_node("text_agent", text_agent_node)
    g.add_node("image_agent", image_agent_node)
    g.add_node("table_agent", table_agent_node)
    g.add_node("critic", critic_node)
    g.add_node("summarizer", summarizer_node)

    # Linear path: START -> extract -> orchestrator
    g.add_edge(START, "extract")
    g.add_edge("extract", "orchestrator")

    # Fan-out: orchestrator dispatches to all three agents
    # Each agent checks state.active_agents to decide if it should work
    g.add_edge("orchestrator", "text_agent")
    g.add_edge("orchestrator", "image_agent")
    g.add_edge("orchestrator", "table_agent")

    # Fan-in: barrier sync, critic waits for all three agents
    g.add_edge(["text_agent", "image_agent", "table_agent"], "critic")

    # Conditional: critic -> retry or finalize
    g.add_conditional_edges(
        "critic",
        route_after_critic,
        {"orchestrator": "orchestrator", "summarizer": "summarizer"},
    )

    # End
    g.add_edge("summarizer", END)

    return g.compile()
