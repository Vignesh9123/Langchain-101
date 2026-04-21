from dotenv import load_dotenv
load_dotenv()
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition

SYSTEM_PROMPT = (
    "You are a helpful assistant that can get user information from the database. "
    "Only get user information if the user asks for it explicitly. "
    "The database is the source of truth for user information. "
    "Try to figure out the user's name from the conversation and then get the user information. "
    "If the user's name is not in the conversation, ask the user for their name first."
)

@tool
def get_user_info(name: str) -> str:
    """
    Get user information from the database
    Args:
        name: str
    Returns:
        str
    """
    print("\n\nGetting user information for", name)
    if name == "John Doe":
        return "John Doe is 30 years old and lives in New York City. He loves Thalapathy Vijay movies."
    elif name == "Jane Doe":
        return "Jane Doe is 25 years old and lives in Los Angeles. She likes Jeniffer Lawrence movies."
    else:
        return "User not found"

llm = ChatGoogleGenerativeAI(model="gemini-flash-lite-latest")
tools = [get_user_info]
llm_with_tools = llm.bind_tools(tools)

def llm_call(state: MessagesState):
    messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
    response = llm_with_tools.invoke(messages)
    print("\n\nLLM Response", response)
    return {"messages":[response]}

tool_node = ToolNode(tools=tools)

graph = StateGraph(MessagesState)
graph.add_node(llm_call)
graph.add_node("tools", tool_node)
graph.add_edge(START, "llm_call")
graph.add_conditional_edges("llm_call", tools_condition)
graph.add_edge("tools", "llm_call")
graph.add_edge("llm_call", END)

with SqliteSaver.from_conn_string(":memory:") as memory:
    graph = graph.compile(
        checkpointer=memory
    )
    config = {"configurable": {"thread_id": "1"}}

    output = graph.invoke({"messages": [
        HumanMessage(content="Hi I am John Doe")]}, config)
    print("\n\nGraph output", output)

    output = graph.invoke({"messages": [HumanMessage(content="What is my information?")]}, config)
    print("\n\nGraph output", output)

    config = {"configurable": {"thread_id": "2"}}
    output = graph.invoke({"messages": [HumanMessage(content="Hi I am Jane Doe")]}, config)
    print("\n\nGraph output", output)

    output = graph.invoke({"messages": [HumanMessage(content="What is my information?")]}, config)
    print("\n\nGraph output", output)

    config = {"configurable": {"thread_id": "3"}}
    output = graph.invoke({"messages": [HumanMessage(content="Hi What is my information?")]}, config)
    print("\n\nGraph output", output)

    config = {"configurable": {"thread_id": "1"}}
    output = graph.invoke({"messages": [HumanMessage(content="Hi Guess my favorite movie of my favorite actor? Hint: It starts with K and was released in 2014")]}, config)
    print("\n\nGraph output", output)