from langchain_groq import ChatGroq
from langgraph.graph import StateGraph,START,END
from typing import TypedDict
from dotenv import load_dotenv
import os

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

from langchain_groq import ChatGroq

llm = ChatGroq(
    model="groq/compound",  
    api_key=GROQ_API_KEY  
)
class Query(TypedDict):
    question: str
    answer:str

graph=StateGraph(Query)
def Question(state:Query) -> Query:
    question=state['question']
    prompt=f"Answer the following question in pointwise format:{question}"
    answer=llm.invoke(prompt).content
    state['answer']=answer
    return state
graph.add_node('Question',Question)
graph.add_edge(START,'Question')
graph.add_edge('Question',END)
workflow = graph.compile()
llm.invoke('How far is moon from the earth?').content


