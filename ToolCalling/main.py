from langchain.agents import create_agent
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
@tool
def add(a: int, b: int):
    """
    A function that adds two numbers

    Args:
        a: First number to add
        b: Second number to add
    
    Returns:
        The sum of two numbers
    """
    print("Calling add agent tool")
    return a+b


llm = ChatGoogleGenerativeAI(model="gemini-flash-lite-latest")

agent = create_agent(model=llm, tools=[add])

response = agent.invoke(input={
    "messages":[{
        "role":"user",
        "content":"Hi Bro, what's the sum of 1 and 2"
    }]
})

print("AI RESPONSE")
print(response)