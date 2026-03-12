from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-flash-lite-latest"
)

template = """
You are a helpful assistant that can answer user questions in the following format:
POINT_1: Some point of answer
POINT_2: Other point of answer
...
The question is {question}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["question"]
)

chain = prompt | llm

response = chain.invoke({"question": " How to learn Python?"})
print(response)
