from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
llm = ChatGoogleGenerativeAI(
    model="gemini-flash-lite-latest"
)

print(llm.invoke("Hello, how are you?").content)