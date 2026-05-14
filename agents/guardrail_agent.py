from core.state import AgentState
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

def guardrail_agent_node(state: AgentState) -> dict:
    """
    Evaluates the response to ensure no definitive medical diagnoses or prescription drug recommendations are made.
    Softens language if too definitive, and appends a medical disclaimer.
    """
    if not state.get("messages"):
        return state
        
    last_message_content = state["messages"][-1].content
    
    llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile")
    
    system_prompt = (
        "You are a Medical Guardrail Agent. Evaluate the provided response from a medical AI assistant. "
        "Ensure no definitive medical diagnoses or prescription drug recommendations are made. "
        "If the language is too definitive, slightly soften it (e.g., change 'You have X' to "
        "'Your symptoms may be consistent with X', or 'Take Y' to 'A doctor may recommend Y'). "
        "If it is already safe, leave it as is. Do not add any conversational filler or introductions. "
        "Just output the final filtered text."
    )

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=last_message_content)
    ])
    
    disclaimer = "\n\n⚠️ **Disclaimer:** I am an AI, not a doctor. Please consult a healthcare professional for medical advice."
    final_content = response.content.strip() + disclaimer
    
    return {
        "messages": [AIMessage(content=final_content)]
    }
