from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage,BaseMessage,AIMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from langgraph.checkpoint.sqlite import SqliteSaver
from typing import TypedDict,Annotated
load_dotenv(dotenv_path=".env", override=True)
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
model = ChatGroq(
    model="llama-3.3-70b-versatile",  
    api_key=GROQ_API_KEY  
)
class Chatstate(TypedDict):

    message : Annotated[list[BaseMessage],add_messages]
graph=StateGraph(Chatstate)

def message_node(state:Chatstate):
    message = state['message']
    
    # FIX: Handle empty messages
    if not message or len(message) == 0:
        # Return a default greeting
        default_response = AIMessage(content="Hello! I'm your AI assistant. How can I help you today?")
        return {"message": [default_response]}
    
    try:
        response = model.invoke(message)
        return {"message": [response]}
    except Exception as e:
        # Handle API errors gracefully
        error_response = AIMessage(content=f"Sorry, I encountered an error: {str(e)}")
        return {"message": [error_response]}

graph.add_node('message_node',message_node)
graph.add_edge(START,'message_node')
graph.add_edge('message_node',END)
import sqlite3
connector=sqlite3.connect(database="checkpoint.db",check_same_thread=False)
checkpointer=SqliteSaver(conn=connector)
chatbot=graph.compile(checkpointer=checkpointer)

def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])

    return list(all_threads)
