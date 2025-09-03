from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from uuid import uuid4
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_openai.embeddings import OpenAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

api_key=os.getenv("OPENAI_API_KEY")

# embeddings = OllamaEmbeddings(model="mxbai-embed-large")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
db_path = "./chroma_qoreid_faqs_db"
add_docs = not os.path.exists(db_path)

chunks = []
ids =[]
file_path =os.path.join("./data", "qoreid_faqs.pdf")
if add_docs:
    
    loader = PyPDFLoader(file_path=file_path)
    loaded_docs = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n","\n"," ",""],
        chunk_size=1000, 
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(documents=loaded_docs)
    ids =[str(uuid4()) for _ in range(len(chunks))]

vector_store = Chroma(
    collection_name="qoreid_faqs",
    persist_directory=db_path,
    embedding_function=embeddings
)
if add_docs:
    print(f"Adding {len(chunks)} documents into vector store.")
    vector_store.add_documents(documents=chunks, ids=ids)
    print(f"finished injesting file: {file_path}")

retriever = vector_store.as_retriever(search_kwargs={'k': 5})



