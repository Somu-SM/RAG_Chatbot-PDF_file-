import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. File Uploader UI
st.title("🤖 Multi-File Assistant")
uploaded_files = st.file_uploader("Upload files", accept_multiple_files=True, type=['pdf', 'csv', 'txt'])

if uploaded_files:
    documents = []
    for uploaded_file in uploaded_files:
        # Save file temporarily to load it
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # 2. Logic to handle different extensions
        if uploaded_file.name.endswith(".pdf"):
            loader = PyPDFLoader(uploaded_file.name)
        elif uploaded_file.name.endswith(".csv"):
            loader = CSVLoader(uploaded_file.name)
        else:
            loader = TextLoader(uploaded_file.name)
        
        documents.extend(loader.load())

    # 3. Process and Store
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    
    # Using Ollama for local embeddings
    vector_db = FAISS.from_documents(chunks, OllamaEmbeddings(model="llama3"))
    st.success("Documents processed!")

    # 4. Querying
    query = st.text_input("Ask a question about your files:")
    if query:
        docs = vector_db.similarity_search(query)
        llm = ChatOllama(model="llama3")
        response = llm.invoke(f"Context: {docs} \n\n Question: {query}")
        st.write(response.content)