from core.state import AgentState
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from core.cache import SemanticCache

def guardrail_agent_node(state: AgentState) -> dict:
    """
    Evaluates the response to ensure no definitive medical diagnoses or prescription drug recommendations are made.
    Softens language if too definitive, appends a medical disclaimer, and updates the semantic cache for RAG queries.
    """
    if not state.get("messages"):
        return state
        
    last_message = state["messages"][-1]
    last_message_content = last_message.content
    
    # Bypass guardrails for security refusal messages or system triage logs
    if state.get("security_flagged") or last_message_content.startswith("Triage Decision:") or last_message_content.startswith("⚠️"):
        return state

    # Instantiate LLM to review and soften language
    llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile")
    
    system_prompt = (
        "You are a Medical Guardrail Agent. Evaluate the provided response from a medical AI assistant. "
        "Ensure no definitive medical diagnoses or prescription drug recommendations are made. "
        "If the language is too definitive, slightly soften it (e.g., change 'You have X' to "
        "'Your symptoms may be consistent with X', or 'Take Y' to 'A doctor may recommend Y'). "
        "If it is already safe, leave it as is. Do not add any conversational filler or introductions. "
        "Just output the final filtered text."
    )

    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=last_message_content)
        ])
        softened_response = response.content.strip()
    except Exception as e:
        print(f"[Guardrail Agent] LLM exception, falling back to original: {e}")
        softened_response = last_message_content

    # Cache-Augmented Generation (CAG) Integration
    # Crucial Privacy Guard: ONLY cache general medical inquiries handled by medical_rag_agent.
    # NEVER cache patient_history_agent answers to strictly preserve patient HIPAA and medical privacy.
    if state.get("next_agent") == "medical_rag_agent":
        try:
            cache = SemanticCache()
            # Store the pre-disclaimer, softened response for semantic reuse
            cache.update(state.get("current_input", ""), softened_response)
        except Exception as e:
            print(f"[Guardrail Agent] Semantic caching failed: {e}")

    disclaimer = "\n\n⚠️ **Disclaimer:** I am an AI, not a doctor. Please consult a healthcare professional for medical advice."
    
    # Secure disclaimer appending, avoiding duplicates
    if disclaimer.strip() in softened_response:
        final_content = softened_response
    else:
        final_content = softened_response + disclaimer
    
    return {
        "messages": [AIMessage(content=final_content)]
    }
