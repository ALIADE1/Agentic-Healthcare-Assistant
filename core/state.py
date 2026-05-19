from typing import TypedDict, Annotated, List, Optional
import operator
from langchain_core.messages import BaseMessage

def merge_unique_lists(existing: List[str], new_vals: List[str]) -> List[str]:
    """
    Helper reducer to merge lists of medical entities while retaining uniqueness and order.
    Ensures that as the multi-agent workflow processes new requests, entities are accumulated
    without duplicates rather than overwritten.
    """
    if existing is None:
        existing = []
    if new_vals is None:
        return existing
    
    # Maintain order and enforce uniqueness
    combined = list(existing)
    for val in new_vals:
        # Normalize casing to prevent near-duplicates (e.g. 'Fever' and 'fever')
        val_clean = val.strip()
        if not any(x.lower() == val_clean.lower() for x in combined):
            combined.append(val_clean)
    return combined

class AgentState(TypedDict):
    """
    The state of the multi-agent graph.
    Enriched with advanced entity-based medical memory and security parameters.
    """
    # Conversation history, appending new messages
    messages: Annotated[List[BaseMessage], operator.add]
    
    # Current active user input
    current_input: str
    
    # Patient ID if identified (e.g. "P12345")
    patient_id: Optional[str]
    
    # The next agent to route to ("medical_rag_agent", "patient_history_agent", "guardrail_agent", "END")
    next_agent: Optional[str]
    
    # Intermediate results or thoughts from agents
    intermediate_results: Optional[str]
    
    # Advanced Entity-Based Memory (uses the merge_unique_lists reducer for cumulative memory)
    extracted_symptoms: Annotated[List[str], merge_unique_lists]
    past_medications: Annotated[List[str], merge_unique_lists]
    chronic_conditions: Annotated[List[str], merge_unique_lists]
    
    # Security flag for prompt injection or malicious activities
    security_flagged: Optional[bool]
