from __future__ import annotations

import os
import tempfile
from typing import Annotated, Any, Dict, Optional, TypedDict

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, RemoveMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg import Connection
import requests

load_dotenv()

# -------------------
# 1. LLM + Embeddings
# -------------------
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant"
)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# -------------------
# 2. PostgreSQL Checkpointer
# -------------------
DB_URI = "postgresql://admin:admin123@localhost:5432/chatbot"

conn = Connection.connect(DB_URI, autocommit=True)
checkpointer = PostgresSaver(conn)
checkpointer.setup()  # tables banata hai automatically

# -------------------
# 3. STM Settings
# -------------------
MAX_MESSAGES = 6  # last 6 messages rakho

# -------------------
# 4. PDF Retriever Store (per thread)
# -------------------
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}


def _get_retriever(thread_id: Optional[str]):
    if thread_id and thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]
    return None


def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )

        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

        return {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


# -------------------
# 5. Tools
# -------------------
search_tool = DuckDuckGoSearchRun(region="us-en")


@tool
def calculator(first_num: float, second_num: float, operation: str) -> str:
    """Calculate two numbers. Operations: add, sub, mul, div."""
    if operation == "add":
        return str(first_num + second_num)
    elif operation == "sub":
        return str(first_num - second_num)
    elif operation == "mul":
        return str(first_num * second_num)
    elif operation == "div":
        if second_num == 0:
            return "Error: Division by zero"
        return str(first_num / second_num)
    return f"Error: Unsupported operation '{operation}'"


@tool
def get_stock_price(symbol: str) -> str:
    """Get latest stock price for a symbol like AAPL or TSLA."""
    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    )
    r = requests.get(url)
    return str(r.json())


@tool
def purchase_stock(symbol: str, quantity: int) -> dict:
    """Simulate purchasing a given quantity of a stock symbol. Requires human approval."""
    decision = interrupt(f"Approve buying {quantity} shares of {symbol}? (yes/no)")

    if isinstance(decision, str) and decision.lower() == "yes":
        return {
            "status": "success",
            "message": f"Purchase order placed for {quantity} shares of {symbol}.",
            "symbol": symbol,
            "quantity": quantity,
        }
    else:
        return {
            "status": "cancelled",
            "message": f"Purchase of {quantity} shares of {symbol} was declined.",
            "symbol": symbol,
            "quantity": quantity,
        }


@tool
def rag_tool(query: str, thread_id: str) -> dict:
    """Retrieve relevant information from uploaded PDF. Requires query and thread_id."""
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {
            "error": "No document indexed for this chat. Upload a PDF first.",
            "query": query,
        }

    result = retriever.invoke(query)
    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {
        "query": query,
        "context": context,
        "metadata": metadata,
        "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename"),
    }


tools = [search_tool, get_stock_price, calculator, purchase_stock, rag_tool]
llm_with_tools = llm.bind_tools(tools)

# -------------------
# 6. State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    summary: str  # ✅ summary store karne ke liye


# -------------------
# 7. STM — Summarize + Trim
# -------------------
def summarize_and_trim(state: ChatState) -> dict:
    """
    Jab messages > MAX_MESSAGES ho jayein:
    1. Purane messages ka summary banao
    2. Sirf last MAX_MESSAGES rakho
    3. Summary ko state mein save karo
    """
    messages = state["messages"]
    existing_summary = state.get("summary", "")

    # Summary banao purane messages ki
    if existing_summary:
        summary_prompt = (
            f"Existing summary:\n{existing_summary}\n\n"
            f"Extend this summary with the new messages above."
        )
    else:
        summary_prompt = "Create a concise summary of the conversation above."

    # Purane messages + summary request bhejo LLM ko
    messages_to_summarize = messages[:-MAX_MESSAGES]  # purane messages
    summary_messages = messages_to_summarize + [HumanMessage(content=summary_prompt)]
    summary_response = llm.invoke(summary_messages)
    new_summary = summary_response.content
    
    print(f"✅ Summary generated: {new_summary[:100]}")  # debug ke liye
    # Purane messages remove karo — sirf last MAX_MESSAGES rakho
    messages_to_delete = messages[:-MAX_MESSAGES]
    delete_messages = [RemoveMessage(id=m.id) for m in messages_to_delete]

    return {
        "summary": new_summary,
        "messages": delete_messages,
    }


# -------------------
# 8. Nodes
# -------------------
def chat_node(state: ChatState, config=None):
    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")

    # ✅ Summary ko system message mein daalo agar hai
    summary = state.get("summary", "")
    # ✅ Console mein dikhega
    print(f"📝 Messages count: {len(state['messages'])}")
    print(f"📝 Messages count: {state['messages']}")
    print(f"📋 Summary: '{summary[:100]}...' " if summary else "📋 Summary: None yet")
    
    summary_text = f"\n\nConversation summary so far:\n{summary}" if summary else ""

    system_message = SystemMessage(
        content=(
            "You are a helpful assistant. For questions about the uploaded PDF, call "
            f"the `rag_tool` and include the thread_id `{thread_id}`. "
            "You can also use web search, stock price, calculator, and purchase_stock tools."
            f"{summary_text}"
        )
    )

    messages = [system_message, *state["messages"]]
    response = llm_with_tools.invoke(messages, config=config)
    return {"messages": [response]}


tool_node = ToolNode(tools)


# ✅ Single routing function — tool, summarize, ya end
def tools_or_summarize(state: ChatState):
    messages = state["messages"]
    last_message = messages[-1]

    # Tool call hai?
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    # Messages zyada hain?
    if len(messages) > MAX_MESSAGES and isinstance(last_message, AIMessage):
        return "summarize"

    return END


# -------------------
# 9. Graph
# -------------------
graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)
graph.add_node("summarize", summarize_and_trim)

graph.add_edge(START, "chat_node")

# ✅ Ek hi conditional edge — teeno cases handle karta hai
graph.add_conditional_edges(
    "chat_node",
    tools_or_summarize,
    {
        "tools": "tools",
        "summarize": "summarize",
        END: END,
    }
)
graph.add_edge("tools", "chat_node")
graph.add_edge("summarize", END)

chatbot = graph.compile(checkpointer=checkpointer)

# -------------------
# 10. Helpers
# -------------------
def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)


def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_RETRIEVERS


def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(str(thread_id), {})