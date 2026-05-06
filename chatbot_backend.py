from langgraph.graph import StateGraph,START,END
from langchain_core.messages import HumanMessage,AIMessage,BaseMessage
from langchain_groq import ChatGroq
from langgraph.graph.message import add_messages
from typing import TypedDict,Annotated
from dotenv import load_dotenv
import os
from langgraph.checkpoint.memory import MemorySaver
load_dotenv(dotenv_path=".env", override=True)
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
model = ChatGroq(
    model="llama-3.3-70b-versatile",  
    api_key=GROQ_API_KEY  
)
class chatstate(TypedDict):
    message : Annotated[list[BaseMessage],add_messages]
    
def chatnode(state:chatstate):
    message=state['message']
    response=model.invoke(message)
    return{'message':[response]}
checkpointer=MemorySaver()
graph=StateGraph(chatstate)
graph.add_node('chat_node',chatnode)
graph.add_edge(START,'chat_node')
graph.add_edge('chat_node',END)
chatbot=graph.compile(checkpointer=checkpointer)

chatbot


thread_id='1'
while True:
    user_input = input("User: ")
    if user_input.lower() == "exit":
        break
    config={'configurable':{'thread_id':thread_id}}
    state = {
        'message': [HumanMessage(content=user_input)]
    }
    response = chatbot.invoke(state, config=config)['message'][-1].content
    print(f"Chatbot: {response}")


