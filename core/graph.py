from langgraph.graph import StateGraph, END
from core.state import AgentState

from agents.triage_agent import triage_agent_node
from agents.medical_rag_agent import medical_rag_agent_node
from agents.patient_history_agent import patient_history_agent_node
from agents.guardrail_agent import guardrail_agent_node


def route_triage(state: AgentState) -> str:
    """
    Conditional edge routing function based on triage agent's decision.
    """
    next_agent = state.get("next_agent")
    if next_agent == "patient_history_agent":
        return "patient_history_agent"
    elif next_agent == "medical_rag_agent":
        return "medical_rag_agent"
    return END

def build_graph():
    """
    Builds and compiles the Multi-Agent routing graph.
    """
    workflow = StateGraph(AgentState)

    # Add agent nodes
    workflow.add_node("triage_agent", triage_agent_node)
    workflow.add_node("medical_rag_agent", medical_rag_agent_node)
    workflow.add_node("patient_history_agent", patient_history_agent_node)
    workflow.add_node("guardrail_agent", guardrail_agent_node)

    # Set the entry point
    workflow.set_entry_point("triage_agent")

    # Add conditional edges from the triage agent
    workflow.add_conditional_edges(
        "triage_agent",
        route_triage,
        {
            "patient_history_agent": "patient_history_agent",
            "medical_rag_agent": "medical_rag_agent",
            END: END
        }
    )

    # Set finish points - Route through guardrail
    workflow.add_edge("medical_rag_agent", "guardrail_agent")
    workflow.add_edge("patient_history_agent", "guardrail_agent")
    workflow.add_edge("guardrail_agent", END)

    # Compile the graph
    app = workflow.compile()
    return app

# Initialize graph instance
app = build_graph()
