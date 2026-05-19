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
    st.set_page_config(
        page_title="Agentic Health Care Assistant", 
        page_icon="🩺", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🩺 Agentic Health Care Assistant & Triage Copilot")
    st.markdown("Powered by LangGraph, ChatGroq (`llama3.3-70b`), Semantic CAG Caching, and Guardrails.")

    # Initialize session state for conversation
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    # Initialize session state for advanced clinical memory
    if "extracted_symptoms" not in st.session_state:
        st.session_state.extracted_symptoms = []
    if "past_medications" not in st.session_state:
        st.session_state.past_medications = []
    if "chronic_conditions" not in st.session_state:
        st.session_state.chronic_conditions = []

    # Sidebar settings & Clinical Memory Panel
    with st.sidebar:
        st.header("⚙️ Configuration")
        patient_id = st.text_input("Patient ID", value="P12345")
        st.info("The system uses this ID to fetch or update clinical records in EHR.")
        
        st.write("---")
        
        # Display Advanced Entity-Based Memory
        st.subheader("🧠 Clinical Memory & Entities")
        st.markdown("Extracted in real-time from your queries to accumulate clinical context.")
        
        # Symptoms Section
        st.markdown("**🤒 Extracted Symptoms**")
        if st.session_state.extracted_symptoms:
            for symptom in st.session_state.extracted_symptoms:
                st.markdown(f"- `{symptom}`")
        else:
            st.caption("No symptoms extracted yet.")
            
        st.write("")
        
        # Medications Section
        st.markdown("**💊 Active/Past Medications**")
        if st.session_state.past_medications:
            for med in st.session_state.past_medications:
                st.markdown(f"- `{med}`")
        else:
            st.caption("No medications extracted yet.")
            
        st.write("")
        
        # Chronic Conditions Section
        st.markdown("**🩺 Chronic Conditions**")
        if st.session_state.chronic_conditions:
            for cond in st.session_state.chronic_conditions:
                st.markdown(f"- `{cond}`")
        else:
            st.caption("No conditions extracted yet.")
            
        st.write("---")
        
        if st.button("🔄 Clear Chat & Memory", use_container_width=True):
            st.session_state.messages = []
            st.session_state.extracted_symptoms = []
            st.session_state.past_medications = []
            st.session_state.chronic_conditions = []
            st.toast("Chat history and clinical memory cleared!", icon="🧹")
            st.rerun()

    # Display chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("How can I help you today? (e.g. 'I am suffering from severe migraine and fever', 'What medications am I on?')"):
        
        # Show user input
        with st.chat_message("user"):
            st.markdown(prompt)
            
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Process via LangGraph
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            with st.spinner("Analyzing request & checking history..."):
                initial_state = {
                    "messages": [], 
                    "current_input": prompt,
                    "patient_id": patient_id,
                    "next_agent": None,
                    "intermediate_results": None,
                    "extracted_symptoms": st.session_state.extracted_symptoms,
                    "past_medications": st.session_state.past_medications,
                    "chronic_conditions": st.session_state.chronic_conditions,
                    "security_flagged": False
                }
                
                final_answer = ""
                
                # Stream events from the graph
                for event in graph_app.stream(initial_state):
                    for node_name, node_state in event.items():
                        # Extract clinical state to preserve across chat turns
                        if "extracted_symptoms" in node_state:
                            st.session_state.extracted_symptoms = node_state["extracted_symptoms"]
                        if "past_medications" in node_state:
                            st.session_state.past_medications = node_state["past_medications"]
                        if "chronic_conditions" in node_state:
                            st.session_state.chronic_conditions = node_state["chronic_conditions"]

                        if "messages" in node_state and len(node_state["messages"]) > 0:
                            latest_msg = node_state["messages"][-1].content
                            
                            if node_name == "triage_agent":
                                # Show triage reasoning in a toast
                                st.toast(f"{latest_msg}", icon="🚦")
                            elif node_name == "cache_check_node":
                                # Notify if semantic cache hit occurred
                                if "next_agent" in node_state and node_state["next_agent"] == "guardrail_agent":
                                    st.toast("⚡ Semantic Cache Hit! Retrieving instantly...", icon="⚡")
                            elif node_name == "guardrail_agent":
                                # Guardrail is the final node, capturing final audited text
                                final_answer = latest_msg
                                response_placeholder.markdown(final_answer)
                
                # If guardrail failed to generate (e.g. prompt injection block)
                if not final_answer:
                    # Look at intermediate messages for the refusal text
                    for event in graph_app.stream(initial_state):
                        for node_name, node_state in event.items():
                            if "messages" in node_state and len(node_state["messages"]) > 0:
                                latest_msg = node_state["messages"][-1].content
                                if "⚠️ **Security Alert:**" in latest_msg:
                                    final_answer = latest_msg
                                    response_placeholder.markdown(final_answer)

                # Append assistant response to session state
                if final_answer:
                    st.session_state.messages.append({"role": "assistant", "content": final_answer})
                    st.rerun()

if __name__ == "__main__":
    main()
