from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import HumanMessage 
from langchain_core.tools import tool
from dotenv import load_dotenv
from langgraph.prebuilt import ToolNode

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-flash-lite-latest")



@tool
def sum_of_two(a: int, b: int) -> int:
    """
    Take two integers as an argumant and return the sum of those two
    Args: 
        a: integer
        b: integer
    Returns:
        sum: integer
    """
    print("\n\nIntegers to ADD:", a,b)
    return a+b

@tool
def triple(a: int) -> int:
    """Takes a number and returns the triple of it"""
    print("\n\nNumber to TRIPLE", a)
    return a*3

tools = [sum_of_two, triple]
llm_with_tools = llm.bind_tools(tools)

tools_by_name = {tool.name: tool for tool in tools}

def llm_call(state: MessagesState):
    response = llm_with_tools.invoke(state["messages"])
    print("\nLLM Response", response)
    return {"messages":[response]}

def should_call_tool(state: MessagesState):
    if state["messages"][-1].tool_calls:
        return "call_tool"
    return END

graph = StateGraph(MessagesState)
graph.add_node(llm_call)
graph.add_node("call_tool", ToolNode(tools))
graph.add_edge(START, "llm_call")
graph.add_conditional_edges("llm_call",should_call_tool)
graph.add_edge("call_tool", "llm_call")
graph = graph.compile()

output = graph.invoke({"messages": [HumanMessage("Find the sum of 3 and 5 and then triple it. and explain it")]})

print("\n\nGraph output")
for msg in output["messages"]:
    print("\n\n")
    print(msg)

print("\n\nLast Graph output", output['messages'][-1].text)