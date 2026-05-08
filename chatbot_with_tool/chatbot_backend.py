from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage,AIMessage,HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from typing import TypedDict,Annotated
from dotenv import load_dotenv
from langgraph.prebuilt import tools_condition,ToolNode
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
import sqlite3
import requests
import os

load_dotenv(dotenv_path=".env", override=True)
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
model = ChatGroq(
    model="llama-3.3-70b-versatile",  
    api_key=GROQ_API_KEY  
)
search_tool=DuckDuckGoSearchRun('en-us')
@tool