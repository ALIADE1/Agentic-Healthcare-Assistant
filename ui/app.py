import sys
import os
import streamlit as st
from dotenv import load_dotenv

# Ensure the root project directory is in the sys path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load environment variables (GROQ_API_KEY)
load_dotenv()

# Important: Import graph after environment is loaded
from core.graph import app as graph_app

def main():
    st.set_page_config(page_title="Agentic Health Care Assistant", page_icon="🩺", layout="wide")
    
    st.title("🩺 Agentic Health Care Assistant & Triage Copilot")
    st.markdown("Powered by LangGraph, ChatGroq (`llama3-70b-8192`), and Local HuggingFace Embeddings.")

    # Sidebar settings
    with st.sidebar:
        st.header("Settings")
        patient_id = st.text_input("Patient ID", value="P12345")
        st.info("The system uses this ID to fetch or update EHR memory.")
        
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("How can I help you today?"):
        
        # Show user input
        with st.chat_message("user"):
            st.markdown(prompt)
            
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Process via LangGraph
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            with st.spinner("Processing request..."):
                initial_state = {
                    "messages": [], 
                    "current_input": prompt,
                    "patient_id": patient_id,
                    "next_agent": None,
                    "intermediate_results": None
                }
                
                final_answer = ""
                
                # Stream events from the graph
                for event in graph_app.stream(initial_state):
                    for node_name, node_state in event.items():
                        if "messages" in node_state and len(node_state["messages"]) > 0:
                            latest_msg = node_state["messages"][-1].content
                            
                            if node_name == "triage_agent":
                                # Show triage reasoning as a toast notification
                                st.toast(f"Triage Decision: {latest_msg}", icon="🚦")
                            else:
                                # Show the final response from either the Medical RAG or Patient History agent
                                final_answer = latest_msg
                                response_placeholder.markdown(final_answer)
                
                # Append to session state
                if final_answer:
                    st.session_state.messages.append({"role": "assistant", "content": final_answer})

if __name__ == "__main__":
    main()
