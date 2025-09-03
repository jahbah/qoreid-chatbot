import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from langsmith import traceable
from vector_stores_qoreid_pdf import retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import os
from dotenv import load_dotenv
import time

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")


st.title("✅QoreID Support Bot")

# llm = ChatOllama(
#         model="mistral",
#         temperature=0.0,
#         verbose=True
#     )

llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key = api_key,
        verbose=True,
        streaming=True
    )

retriever = retriever

@traceable
def get_context_retriever_chain():
    # Create a prompt template to retrieve relevant context based on chat-history and user query   
    prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="messages"),
    ("user", "{input}"),
    ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    # Create a history aware retriever chain
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    
    return retriever_chain

# Create a conversational RAG Chain
@traceable
def get_conversational_rag_chain(retriever_chain): 
    
    prompt = ChatPromptTemplate.from_messages([
    ("system","""Your name is 'Caleb', You are an Agent support assistant for QoreID and you are expected to answer customers questions strictly based on the provided document.
    Use the information in the document to answer the question. Respond confidently and assuredly. Do not make up answers except the intent is salutations or pleasantaries.
    if the information is not in the document. say, 'i don't know the answer but i can redirct you to someone who knows, pls click this link and it will take them to that [Contact us form](https://verifyme.ng/about-us/contact.'
    Use the below context:\n\n{context}"""),
    MessagesPlaceholder(variable_name="messages"),
    ("user", "{input}")
    ])
    
    stuff_documents_chain = create_stuff_documents_chain(llm,prompt)
    
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

@traceable
def get_response(prompt):
    retriever_chain = get_context_retriever_chain()
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    
    response = conversation_rag_chain.invoke({
        "messages": st.session_state.messages,
        "input": prompt
    })
    return response['answer']

# initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [AIMessage("Hello! I'm Caleb, your QoreID Support AI Assitant. How can I help you today?")]

# display chat messages from history on app rerun
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# create the bar where we can type messages
prompt = st.chat_input("Ask your question.")


# did the user submit a prompt?
if prompt:

    # add the message from the user (prompt) to the screen with streamlit
    with st.chat_message("user"):
        st.markdown(prompt)
        
        st.session_state.messages.append(HumanMessage(prompt))

    
    with st.spinner("Processing..."):
        result = get_response(prompt)

    placeholder =st.empty()
    with st.chat_message("assistant"):
        # st.markdown(result)
        
        # Simulate typing effects
        for i in range(len(result)):
            placeholder.markdown(result[:i+1])
            time.sleep(0.01)
            
        st.session_state.messages.append(AIMessage(result))


