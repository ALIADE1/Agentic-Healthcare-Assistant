from typing import TypedDict, Annotated, List, Optional
import operator
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    """
    The state of the multi-agent graph.
    """
    # Conversation history, appending new messages
    messages: Annotated[List[BaseMessage], operator.add]
    
    # Current user input
    current_input: str
    
    # Patient ID if identified
    patient_id: Optional[str]
    
    # The next agent to route to
    next_agent: Optional[str]
    
    # Intermediate results or thoughts from agents
    intermediate_results: Optional[str]
