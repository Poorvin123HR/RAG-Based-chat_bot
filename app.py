import os
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Set Google API key
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# Load and process PDF
@st.cache_resource
def load_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    return text_splitter.split_documents(documents)

# Create vector store
@st.cache_resource
def create_vector_store(docs):
    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return FAISS.from_documents(docs, embedding)

# Create chat chain with memory
def create_chatbot_chain(vectorstore):
    retriever = vectorstore.as_retriever()
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
    )
    return qa_chain

# Streamlit UI
st.title("📄 PDF Chatbot with Memory (RAG + Gemini)")
pdf_path = st.file_uploader("mypaper.pdf", type=["pdf"])

if pdf_path:
    with st.spinner("Processing PDF..."):
        docs = load_pdf(pdf_path)
        vectorstore = create_vector_store(docs)
        qa_chain = create_chatbot_chain(vectorstore)

    # Chat interaction
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.chat_input("Ask me something about the PDF...")
    if user_input:
        response = qa_chain({"question": user_input, "chat_history": st.session_state.chat_history})
        st.session_state.chat_history.append(("user", user_input))
        st.session_state.chat_history.append(("bot", response["answer"]))

    # Display conversation
    for role, msg in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(msg)
