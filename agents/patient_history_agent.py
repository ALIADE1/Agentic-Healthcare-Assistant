from core.state import AgentState
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import create_react_agent
from tools.db_tool import get_patient_history, update_patient_record

def patient_history_agent_node(state: AgentState) -> dict:
    """
    Patient History Agent using ChatGroq bound to the DB tools.
    """
    llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile")
    tools = [get_patient_history, update_patient_record]
    
    agent = create_react_agent(llm, tools=tools)
    
    patient_id = state.get("patient_id", "P12345")
    
    system_prompt = f"""You are the Patient History Agent.
You have access to tools to retrieve or update the electronic health records for a patient.
The current active patient_id is '{patient_id}'. You MUST use this ID for tool calls unless specified otherwise by the user.
Answer the user's query based strictly on the information you retrieve or confirm updates."""
    
    response = agent.invoke({
        "messages": [
            SystemMessage(content=system_prompt),
            HumanMessage(content=state.get("current_input", ""))
        ]
    })
    
    return {
        "messages": [response["messages"][-1]]
    }
