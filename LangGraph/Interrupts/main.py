from dotenv import load_dotenv
from langgraph.types import Command, interrupt
load_dotenv()
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver


@tool
def add(a: int, b:int):
    """
    Takes two numbers and returns its sum
    """
    return a + b

tools = [add]

llm = ChatGoogleGenerativeAI(model="gemini-flash-lite-latest").bind_tools(tools)


def chatbot(state: MessagesState):
    messages = state['messages']

    response = llm.invoke(messages)

    print("\n\nLLM Response", response)

    return {"messages":[response]}

def tool_call(state: MessagesState):
    tool_to_call = state["messages"][-1]
    print("\n\nTool to call", tool_to_call)
    approved = interrupt("Do you approve this action?")
    print("\n\nApproved", approved)
    return {"messages": [ToolMessage("3", tool_call_id=tool_to_call.tool_calls[0]["id"])]}

graph = StateGraph(MessagesState)

graph.add_node("chatbot", chatbot)
graph.add_node("tools", tool_call)

graph.set_entry_point("chatbot")

graph.add_conditional_edges("chatbot", tools_condition)
graph.add_edge("tools", "chatbot")

graph.add_edge("chatbot", END)
graph = graph.compile(
    checkpointer=MemorySaver()
)
config = {"configurable": {"thread_id": "1"}}

output = graph.invoke({
    "messages": [HumanMessage("Add 1 and 2")]
}, config)

print("\n\nOutput",output)
state = graph.get_state(config)
print("\n\nState after interrupt", state)
pending = state.next
print("\n\nPending node", pending)
interrupt_val = state.tasks[0].interrupts[0].value
print("\n\nInterrupts",state.tasks[0].interrupts)

output = graph.invoke(Command(resume=True), config)

state = graph.get_state(config)
print("\n\nState before final output", state)

print("\n\nOutput",output)

