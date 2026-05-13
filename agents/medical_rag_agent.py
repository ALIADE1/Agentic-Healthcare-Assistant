from core.state import AgentState
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import create_react_agent
from tools.rag_tool import retrieve_medical_guidelines

def medical_rag_agent_node(state: AgentState) -> dict:
    """
    Medical RAG agent using ChatGroq bound to the local HuggingFace RAG tool.
    """
    # Using Mixtral or Llama3 for general medical reasoning
    llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile")
    tools = [retrieve_medical_guidelines]
    
    agent = create_react_agent(llm, tools=tools)
    
    system_prompt = """You are the Medical RAG Agent.
Use the `retrieve_medical_guidelines` tool to answer general medical queries, symptoms, and first-aid questions.
Do NOT hallucinate. If the tool does not provide the answer, say that you don't have enough context, and provide general safe advice."""
    
    response = agent.invoke({
        "messages": [
            SystemMessage(content=system_prompt),
            HumanMessage(content=state.get("current_input", ""))
        ]
    })
    
    return {
        "messages": [response["messages"][-1]]
    }
