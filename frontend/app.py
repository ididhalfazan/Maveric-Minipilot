import streamlit as st
import requests
import uuid

BACKEND_URL = "http://localhost:8000"

MODULES = {
    "General / Full Workflow": None,
    "Digital Twin": "digital_twin",
    "RF Prediction": "rf_prediction",
    "UE Tracks Generation": "ue_tracks",
    "Orchestration": "orchestration",
}

st.set_page_config(
    page_title="Maveric MiniPilot",
    page_icon="🤖",
    layout="wide"
)

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "module_focus" not in st.session_state:
    st.session_state.module_focus = None

with st.sidebar:
    st.title("🤖 Maveric MiniPilot")
    st.markdown("AI-powered guide for the Maveric RIC Algorithm Development Platform")
    st.divider()

    st.subheader("Module Focus")
    selected_module_label = st.radio(
        "Select a module to focus on:",
        options=list(MODULES.keys()),
        index=0
    )
    st.session_state.module_focus = MODULES[selected_module_label]

    st.divider()

    st.subheader("Quick Questions")
    quick_qs = [
        "Walk me through the full Maveric workflow",
        "What are the inputs and outputs of Digital Twin?",
        "How does RF Prediction work?",
        "What microservices does Orchestration use?",
        "How do I generate UE Tracks?",
    ]
    for q in quick_qs:
        if st.button(q, use_container_width=True):
            st.session_state.pending_message = q

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear chat", use_container_width=True):
            requests.post(f"{BACKEND_URL}/clear",
                          json={"session_id": st.session_state.session_id})
            st.session_state.messages = []
            st.session_state.session_id = str(uuid.uuid4())
            st.rerun()
    with col2:
        st.caption(f"Session:\n`{st.session_state.session_id[:8]}...`")

st.title(f"Maveric MiniPilot — {selected_module_label}")

try:
    health = requests.get(f"{BACKEND_URL}/health", timeout=2)
    if health.status_code != 200:
        st.error("Backend is not responding. Make sure uvicorn is running.")
        st.stop()
except Exception:
    st.error("Cannot reach backend at http://localhost:8000. Start the backend first.")
    st.stop()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📄 Sources from Maveric repo"):
                for src in msg["sources"]:
                    st.code(src)

pending = st.session_state.pop("pending_message", None)
user_input = st.chat_input("Ask anything about Maveric...") or pending

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Searching Maveric docs..."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/chat",
                    json={
                        "message": user_input,
                        "session_id": st.session_state.session_id,
                        "module_focus": st.session_state.module_focus,
                    },
                    timeout=30
                )
                data = response.json()

                if response.status_code == 200:
                    answer = data["answer"]
                    sources = data.get("sources", [])
                    st.markdown(answer)
                    if sources:
                        with st.expander("📄 Sources from Maveric repo"):
                            for src in sources:
                                st.code(src)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                else:
                    st.error(f"Error: {data.get('detail', 'Unknown error')}")

            except requests.exceptions.Timeout:
                st.error("Request timed out.")
            except Exception as e:
                st.error(f"Something went wrong: {str(e)}")