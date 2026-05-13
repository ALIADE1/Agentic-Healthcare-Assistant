import os
from pydantic import BaseModel, Field
from core.state import AgentState
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

class TriageOutput(BaseModel):
    next_agent: str = Field(
        description="The next agent to route to. Must be strictly 'patient_history_agent' or 'medical_rag_agent'."
    )
    reasoning: str = Field(
        description="The reasoning behind the routing decision."
    )

def triage_agent_node(state: AgentState) -> dict:
    """
    Intelligent triage using ChatGroq to route based on intent and severity.
    """
    current_input = state.get("current_input", "")
    
    # Using Llama 3 70B for fast, accurate structured output
    llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile")
    
    # Structured output routing
    structured_llm = llm.with_structured_output(TriageOutput)
    
    system_prompt = """You are the Triage Agent in an intelligent healthcare assistant system.
Analyze the user's input and decide the most appropriate next step:
- If the user asks about their own medical records, past visits, or updating their medical history, route to 'patient_history_agent'.
- If the user asks for general medical advice, symptoms, standard first-aid protocols, or is experiencing a medical issue, route to 'medical_rag_agent'.
Output your decision strictly matching the schema."""

    response = structured_llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=current_input)
    ])
    
    # Save reasoning as a message
    message = AIMessage(content=f"Triage Decision: Routed to {response.next_agent} because {response.reasoning}")
    
    return {
        "messages": [message],
        "next_agent": response.next_agent
    }
