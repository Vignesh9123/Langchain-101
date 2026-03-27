from operator import itemgetter
import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

llm = ChatGoogleGenerativeAI(
    model="gemini-flash-lite-latest"
)

vector_store = PineconeVectorStore(
    embedding=embeddings,
    index_name=os.environ.get("INDEX_NAME")
)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:

{context}

Question: {question}

Provide a detailed answer:"""
)

def format_docs(relevant_docs):
    print("Relevant docs", relevant_docs)
    return "\n\n".join(doc.page_content for doc in relevant_docs)

def rag_without_lcel(query: str):   
    relevant_docs = retriever.invoke(query)
    print("\n\nRelevant docs are", relevant_docs)
    context = format_docs(relevant_docs)
    messages = prompt_template.format_messages(context=context, question=query)
    print("\n\nMessages sending to llm", messages)
    rag_response = llm.invoke(messages)
    print("\n\nRag implementation response", rag_response)

def rag_with_lcel():
    chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs
        )
        | prompt_template
        | llm
    )
    return chain

query = "what is Openclaw, explain in detail? and how to implement safety in this"

raw_response = llm.invoke(query)

print("\n\nRaw response", raw_response)

rag_without_lcel(query)

lcel_chain = rag_with_lcel()
print("\n\n")
print("RAG WITHOUT LCEL")
print("="*50)
response = lcel_chain.invoke({
    "question" : query
})
print(response.content)
