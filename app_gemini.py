import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.set_page_config(page_title="Gemini Multi-File Chat", layout="wide")
st.title("🚀 Gemini Multi-Format RAG")

# 1. Setup Sidebar for API Key
with st.sidebar:
    api_key = st.text_input("Enter Google API Key:", type="password")
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key

# 2. File Uploader Logic
uploaded_files = st.file_uploader("Upload PDF, CSV, or TXT", accept_multiple_files=True)

if uploaded_files and api_key:
    all_docs = []
    for file in uploaded_files:
        # Temporary save to read
        with open(file.name, "wb") as f:
            f.write(file.getbuffer())
        
        # Select Loader based on extension
        if file.name.endswith(".pdf"):
            loader = PyPDFLoader(file.name)
        elif file.name.endswith(".csv"):
            loader = CSVLoader(file.name)
        else:
            loader = TextLoader(file.name)
            
        all_docs.extend(loader.load())
        os.remove(file.name) # Clean up

    # 3. Chunking & Indexing
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=150)
    chunks = text_splitter.split_documents(all_docs)
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vector_store = FAISS.from_documents(chunks, embeddings)
    st.success(f"Indexed {len(uploaded_files)} files!")

    # 4. Chat Interface
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        # Retrieve context
        docs = vector_store.similarity_search(user_question, k=4)
        context = "\n".join([d.page_content for d in docs])
        
        # Generate Answer
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
        response = llm.invoke(f"Use this context: {context}\n\nQuestion: {user_question}")
        
        st.markdown("### Answer:")
        st.write(response.content)

elif not api_key:
    st.info("Please enter your Google API Key in the sidebar to begin.")