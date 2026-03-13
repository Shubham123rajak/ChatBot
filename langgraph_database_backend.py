from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph,START,END
from typing import TypedDict,Annotated
from pydantic import BaseModel,Field
from langchain_core.messages import SystemMessage, HumanMessage,BaseMessage
import operator
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3


load_dotenv()

llm = ChatGoogleGenerativeAI(model='gemini-3-flash-preview')


class chatState(TypedDict):

    messages :Annotated[list[BaseMessage],add_messages]

def chat_node(state: chatState):
    messages = state['messages']

    response = llm.invoke(messages)

    return {'messages' : [response]}    


conn = sqlite3.Connection(database='chatbot.db',check_same_thread=False)
checkpointer = SqliteSaver(conn = conn)
graph = StateGraph(chatState)

graph.add_node('chat_node',chat_node)

graph.add_edge(START,'chat_node')
graph.add_edge('chat_node',END)

# checkpoint = InMemorySaver()
chatbot = graph.compile(checkpointer = checkpointer)    

def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])

    return list(all_threads)