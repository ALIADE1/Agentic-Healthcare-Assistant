from langgraph.graph import StateGraph, END
from core.state import AgentState
from langchain_core.messages import AIMessage
from core.cache import SemanticCache

from agents.triage_agent import triage_agent_node
from agents.medical_rag_agent import medical_rag_agent_node
from agents.patient_history_agent import patient_history_agent_node
from agents.guardrail_agent import guardrail_agent_node

def cache_check_node(state: AgentState) -> dict:
    """
    Checks the semantic cache for a pre-calculated safe response.
    If a hit occurs, bypasses RAG and History pipelines and routes straight to the guardrail.
    """
    current_input = state.get("current_input", "")
    
    # Instantiate the semantic cache
    cache = SemanticCache()
    cached_response = cache.lookup(current_input)
    
    if cached_response:
        return {
            "messages": [AIMessage(content=cached_response)],
            "next_agent": "guardrail_agent"  # Skip straight to guardrails to append fresh disclaimer
        }
    
    return {
        "next_agent": "triage_agent"
    }

def route_cache(state: AgentState) -> str:
    """
    Conditional edge routing function for Cache check results.
    """
    next_agent = state.get("next_agent")
    if next_agent == "guardrail_agent":
        return "guardrail_agent"
    return "triage_agent"

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
    Builds and compiles the Multi-Agent routing graph with Semantic Caching.
    """
    workflow = StateGraph(AgentState)

    # Add agent nodes
    workflow.add_node("cache_check_node", cache_check_node)
    workflow.add_node("triage_agent", triage_agent_node)
    workflow.add_node("medical_rag_agent", medical_rag_agent_node)
    workflow.add_node("patient_history_agent", patient_history_agent_node)
    workflow.add_node("guardrail_agent", guardrail_agent_node)

    # Set the entry point of the entire workflow to the Cache Check
    workflow.set_entry_point("cache_check_node")

    # Add conditional routing from the Cache check
    workflow.add_conditional_edges(
        "cache_check_node",
        route_cache,
        {
            "guardrail_agent": "guardrail_agent",
            "triage_agent": "triage_agent"
        }
    )

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
