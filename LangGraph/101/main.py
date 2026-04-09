from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain.messages import AIMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-flash-lite-latest")

def llm_call(state: MessagesState):
    response = llm.invoke(state["messages"])
    print("LLM Response", response)
    return {"messages":[response]}

def print_state(state: MessagesState):
    print("\nState in print_state", state)
    print("\nLast message text:", state['messages'][-1].text)
    return {}

graph = StateGraph(MessagesState)
graph.add_node(llm_call)
graph.add_node(print_state)
graph.add_edge(START, "llm_call")
graph.add_edge("llm_call", "print_state")
graph.add_edge("print_state", END)
graph = graph.compile()

output = graph.invoke({"messages": HumanMessage("Hi")})

print("\n\nGraph output", output)