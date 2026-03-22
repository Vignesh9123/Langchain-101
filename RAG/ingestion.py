from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from dotenv import load_dotenv
import os
load_dotenv()



MEDIUM_BLOG_PATH = os.path.join(os.path.dirname(__file__), "mediumblog1.txt")

loader = TextLoader(MEDIUM_BLOG_PATH, encoding="utf-8")
document = loader.load()

splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

texts = splitter.split_documents(document)

print(f"Doc Chunked into {len(texts)} chunks")

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001" # This has 3072 dimensions acc to website so configure index accordingly
)


PineconeVectorStore.from_documents(
    documents=texts,
    embedding=embeddings,
    index_name=os.environ.get("INDEX_NAME")
)