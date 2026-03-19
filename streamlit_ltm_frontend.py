import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command

from langgraph_ltm_backend import (
    chatbot,
    ingest_pdf,
    retrieve_all_threads,
    thread_document_metadata,
)


# =========================== Utilities ===========================
def generate_thread_id():
    return uuid.uuid4()


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []
    st.session_state["pending_interrupt"] = None


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])


# ======================= Session Initialization ===================
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}

# ✅ HITL state — stores interrupt prompt if graph is paused
if "pending_interrupt" not in st.session_state:
    st.session_state["pending_interrupt"] = None

add_thread(st.session_state["thread_id"])

thread_key = str(st.session_state["thread_id"])
thread_docs = st.session_state["ingested_docs"].setdefault(thread_key, {})
threads = st.session_state["chat_threads"][::-1]
selected_thread = None

CONFIG = {
    "configurable": {"thread_id": thread_key},
    "metadata": {"thread_id": thread_key},
    "run_name": "chat_turn",
}

# ============================ Sidebar ============================
st.sidebar.title("LangGraph PDF Chatbot")
st.sidebar.markdown(f"**Thread ID:** `{thread_key}`")

if st.sidebar.button("New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

if thread_docs:
    latest_doc = list(thread_docs.values())[-1]
    st.sidebar.success(
        f"Using `{latest_doc.get('filename')}` "
        f"({latest_doc.get('chunks')} chunks from {latest_doc.get('documents')} pages)"
    )
else:
    st.sidebar.info("No PDF indexed yet.")

uploaded_pdf = st.sidebar.file_uploader("Upload a PDF for this chat", type=["pdf"])
if uploaded_pdf:
    if uploaded_pdf.name in thread_docs:
        st.sidebar.info(f"`{uploaded_pdf.name}` already processed for this chat.")
    else:
        with st.sidebar.status("Indexing PDF…", expanded=True) as status_box:
            summary = ingest_pdf(
                uploaded_pdf.getvalue(),
                thread_id=thread_key,
                filename=uploaded_pdf.name,
            )
            thread_docs[uploaded_pdf.name] = summary
            status_box.update(label="✅ PDF indexed", state="complete", expanded=False)

st.sidebar.subheader("Past conversations")
if not threads:
    st.sidebar.write("No past conversations yet.")
else:
    for thread_id in threads:
        if st.sidebar.button(str(thread_id), key=f"side-thread-{thread_id}"):
            selected_thread = thread_id

# ============================ Main Layout ========================
st.title("Multi Utility Chatbot")

# Chat history dikhao
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])


# ============================ HITL Block ============================
# ✅ Agar graph interrupt pe ruka hai toh approval maango
if st.session_state["pending_interrupt"]:
    interrupt_prompt = st.session_state["pending_interrupt"]

    st.warning("⏸️ **Human Approval Required**")
    st.info(f"🤖 {interrupt_prompt}")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("✅ Yes, Approve", use_container_width=True):
            result = chatbot.invoke(
                Command(resume="yes"),
                config=CONFIG,
            )
            st.session_state["pending_interrupt"] = None

            # Final AI response nikalo
            final_msg = ""
            for msg in reversed(result.get("messages", [])):
                if isinstance(msg, AIMessage) and msg.content:
                    final_msg = msg.content
                    break

            st.session_state["message_history"].append(
                {"role": "assistant", "content": final_msg}
            )
            st.rerun()

    with col2:
        if st.button("❌ No, Cancel", use_container_width=True):
            result = chatbot.invoke(
                Command(resume="no"),
                config=CONFIG,
            )
            st.session_state["pending_interrupt"] = None

            final_msg = ""
            for msg in reversed(result.get("messages", [])):
                if isinstance(msg, AIMessage) and msg.content:
                    final_msg = msg.content
                    break

            st.session_state["message_history"].append(
                {"role": "assistant", "content": final_msg}
            )
            st.rerun()


# ============================ Chat Input ============================
user_input = st.chat_input(
    "Ask about your document or use tools",
    disabled=st.session_state["pending_interrupt"] is not None,  # ✅ interrupt pe input disable
)

if user_input:
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    with st.chat_message("assistant"):
        status_holder = {"box": None}
        collected_content = []

        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                node_name = metadata.get("langgraph_node", "")
                if node_name == "summarize":
                    continue  # ❌ summarize node ka output skip karo

                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                if isinstance(message_chunk, AIMessage) and message_chunk.content:
                    collected_content.append(message_chunk.content)
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())

        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )

    # ✅ Check karo graph interrupt pe ruka hai ya nahi
    graph_state = chatbot.get_state(config=CONFIG)
    interrupts = graph_state.tasks[0].interrupts if graph_state.tasks else []

    if interrupts:
        # Interrupt prompt save karo aur HITL block dikhao
        st.session_state["pending_interrupt"] = interrupts[0].value
        st.rerun()
    else:
        # Normal response history mein save karo
        final_content = ai_message if isinstance(ai_message, str) else "".join(collected_content)
        st.session_state["message_history"].append(
            {"role": "assistant", "content": final_content}
        )

    doc_meta = thread_document_metadata(thread_key)
    if doc_meta:
        st.caption(
            f"Document indexed: {doc_meta.get('filename')} "
            f"(chunks: {doc_meta.get('chunks')}, pages: {doc_meta.get('documents')})"
        )

st.divider()

if selected_thread:
    st.session_state["thread_id"] = selected_thread
    messages = load_conversation(selected_thread)

    temp_messages = []
    for msg in messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        content = msg.content if isinstance(msg.content, str) else ""
        temp_messages.append({"role": role, "content": content})
    st.session_state["message_history"] = temp_messages
    st.session_state["ingested_docs"].setdefault(str(selected_thread), {})
    st.rerun()