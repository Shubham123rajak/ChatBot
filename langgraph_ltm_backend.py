from __future__ import annotations

import os
import tempfile
from datetime import datetime
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
import psycopg

load_dotenv()

# -------------------
# 1. LLM + Embeddings
# -------------------
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile"
)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# -------------------
# 2. PostgreSQL — STM Checkpointer
# -------------------
DB_URI = "postgresql://admin:admin123@localhost:5432/chatbot"

conn = Connection.connect(DB_URI, autocommit=True)
checkpointer = PostgresSaver(conn)
checkpointer.setup()

# -------------------
# 3. LTM — PostgreSQL Setup
# -------------------
DEFAULT_USER_ID = "default_user"

ltm_conn = psycopg.connect(DB_URI, autocommit=True)

# LTM table banao agar nahi hai
ltm_conn.execute("""
    CREATE TABLE IF NOT EXISTS ltm_memory (
        id SERIAL PRIMARY KEY,
        user_id TEXT NOT NULL,
        memory_type TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT NOW(),
        updated_at TIMESTAMP DEFAULT NOW()
    )
""")

print("✅ LTM table ready")


def save_ltm(content: str, memory_type: str = "summary", user_id: str = DEFAULT_USER_ID):
    """LTM mein memory save karo."""
    if memory_type == "summary":
        # Summary ke liye — purani summary update karo
        result = ltm_conn.execute(
            "SELECT id FROM ltm_memory WHERE user_id = %s AND memory_type = 'summary'",
            (user_id,)
        ).fetchone()

        if result:
            ltm_conn.execute(
                "UPDATE ltm_memory SET content = %s, updated_at = NOW() WHERE user_id = %s AND memory_type = 'summary'",
                (content, user_id)
            )
        else:
            ltm_conn.execute(
                "INSERT INTO ltm_memory (user_id, memory_type, content) VALUES (%s, %s, %s)",
                (user_id, memory_type, content)
            )
    else:
        # Facts ke liye — naya row add karo
        ltm_conn.execute(
            "INSERT INTO ltm_memory (user_id, memory_type, content) VALUES (%s, %s, %s)",
            (user_id, memory_type, content)
        )
    print(f"✅ LTM saved: [{memory_type}] {content[:60]}...")


def load_ltm(user_id: str = DEFAULT_USER_ID) -> str:
    """LTM se saari memory load karo."""
    rows = ltm_conn.execute(
        "SELECT memory_type, content FROM ltm_memory WHERE user_id = %s ORDER BY updated_at DESC",
        (user_id,)
    ).fetchall()

    if not rows:
        return ""

    summary_parts = []
    fact_parts = []

    for memory_type, content in rows:
        if memory_type == "summary":
            summary_parts.append(content)
        else:
            fact_parts.append(f"- {content}")

    ltm_text = ""
    if summary_parts:
        ltm_text += f"Previous conversation summary:\n{summary_parts[0]}\n"
    if fact_parts:
        ltm_text += f"\nKnown facts about user:\n" + "\n".join(fact_parts)

    return ltm_text.strip()


def extract_and_save_facts(messages: list, user_id: str = DEFAULT_USER_ID):
    conversation = "\n".join([
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
        for m in messages
        if isinstance(m, (HumanMessage, AIMessage)) and m.content
    ])

    fact_prompt = f"""Extract important facts about the USER ONLY from this conversation.
Only extract: name, profession, preferences, goals, location, personal details.
DO NOT extract: code, functions, math results, tool names, algorithms, technical content.
If no personal facts found, return empty list: []

Conversation:
{conversation}

Return ONLY a Python list example: ["User's name is Shubham", "User likes Python"]
Return ONLY the list, nothing else."""

    response = llm.invoke([HumanMessage(content=fact_prompt)])
    raw = response.content.strip()

    try:
        # Safe parse karo
        if "[" in raw and "]" in raw:
            start = raw.index("[")
            end = raw.rindex("]") + 1
            facts = eval(raw[start:end])
            if isinstance(facts, list):
                for fact in facts:
                    if isinstance(fact, str) and len(fact) > 5:
                        save_ltm(fact, memory_type="fact", user_id=user_id)
                        print(f"🧠 Fact saved: {fact}")
    except Exception as e:
        print(f"⚠️ Fact extraction failed: {e}")


def get_ltm_stats(user_id: str = DEFAULT_USER_ID) -> dict:
    """LTM stats return karo — frontend ke liye."""
    rows = ltm_conn.execute(
        "SELECT memory_type, COUNT(*) FROM ltm_memory WHERE user_id = %s GROUP BY memory_type",
        (user_id,)
    ).fetchall()
    return {row[0]: row[1] for row in rows}


# -------------------
# 4. STM Settings
# -------------------
MAX_MESSAGES = 6

# -------------------
# 5. PDF Retriever Store (per thread)
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
# 6. Tools
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
# 7. State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    summary: str


# -------------------
# 8. STM — Summarize + Trim + LTM Save
# -------------------
def summarize_and_trim(state: ChatState) -> dict:
    messages = state["messages"]
    existing_summary = state.get("summary", "")

    if existing_summary:
        # Summary generate karne ke prompt mein add karo
        summary_prompt = """Create a concise summary of the conversation.
        Include: topics discussed, user preferences, personal details.
        DO NOT include: code snippets, function names, tool calls, or technical implementations.
        Keep it conversational and factual only."""
    else:
        summary_prompt = "Create a concise summary of the conversation above."

    messages_to_summarize = messages[:-MAX_MESSAGES]
    summary_messages = messages_to_summarize + [HumanMessage(content=summary_prompt)]
    summary_response = llm.invoke(summary_messages)
    new_summary = summary_response.content

    # ✅ STM summary → LTM mein save karo
    save_ltm(new_summary, memory_type="summary")

    # ✅ Facts extract karo aur LTM mein save karo
    extract_and_save_facts(messages_to_summarize)

    messages_to_delete = messages[:-MAX_MESSAGES]
    delete_messages = [RemoveMessage(id=m.id) for m in messages_to_delete]

    return {
        "summary": new_summary,
        "messages": delete_messages,
    }


# -------------------
# 9. Nodes
# -------------------
def chat_node(state: ChatState, config=None):
    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")

    # STM summary
    stm_summary = state.get("summary", "")
    stm_text = f"\n\nCurrent conversation summary:\n{stm_summary}" if stm_summary else ""

    # ✅ LTM load karo
    ltm_text = load_ltm(DEFAULT_USER_ID)
    ltm_section = f"""
                Long term memory (from past conversations):
                {ltm_text}

                IMPORTANT: The above is memory context ONLY.
                Do NOT call any functions or tools mentioned in the memory.
                Only use these tools: {[t.name for t in tools]}
                """ if ltm_text else ""

    system_message = SystemMessage(
        content=(
            "You are a helpful assistant. For questions about the uploaded PDF, call "
            f"the `rag_tool` and include the thread_id `{thread_id}`. "
            "You can also use web search, stock price, calculator, and purchase_stock tools."
            f"{ltm_section}"
            f"{stm_text}"
        )
    )

    messages = [system_message, *state["messages"]]
    response = llm_with_tools.invoke(messages, config=config)
    return {"messages": [response]}


tool_node = ToolNode(tools)


def tools_or_summarize(state: ChatState):
    messages = state["messages"]
    last_message = messages[-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    if len(messages) > MAX_MESSAGES and isinstance(last_message, AIMessage):
        return "summarize"

    return END


# -------------------
# 10. Graph
# -------------------
graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)
graph.add_node("summarize", summarize_and_trim)

graph.add_edge(START, "chat_node")
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
# 11. Helpers
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