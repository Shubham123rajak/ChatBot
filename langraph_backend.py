from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph,START,END
from typing import TypedDict,Annotated
from pydantic import BaseModel,Field
from langchain_core.messages import SystemMessage, HumanMessage,BaseMessage
import operator
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages


load_dotenv()

llm = ChatGoogleGenerativeAI(model='gemini-3-flash-preview')


class chatState(TypedDict):

    messages :Annotated[list[BaseMessage],add_messages]

def chat_node(state: chatState):
    messages = state['messages']

    response = llm.invoke(messages)

    return {'messages' : [response]}    


graph = StateGraph(chatState)

graph.add_node('chat_node',chat_node)

graph.add_edge(START,'chat_node')
graph.add_edge('chat_node',END)

checkpoint = InMemorySaver()
chatbot = graph.compile(checkpointer = checkpoint)    