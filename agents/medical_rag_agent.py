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

    system_prompt = """You are a helpful and expert Medical Assistant. 
                        You have access to a retrieval tool to search medical documents.
                        Always use the tool first to find relevant information.
                        If the retrieved context contains the answer, use it. 
                        If the retrieved context DOES NOT contain the answer, OR if the user asks a general medical question, USE YOUR OWN GENERAL MEDICAL KNOWLEDGE to provide a safe, accurate, and helpful response.
                        Do not say 'I don't have enough context' if you know the answer from your general knowledge."""
    response = agent.invoke(
        {
            "messages": [
                SystemMessage(content=system_prompt),
                HumanMessage(content=state.get("current_input", "")),
            ]
        }
    )

    return {"messages": [response["messages"][-1]]}
