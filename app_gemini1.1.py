import streamlit as st
import os
import json
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- 1. DIRECT API CONFIGURATION ---
os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY"  # enter your Google api key

st.set_page_config(page_title="Gemini Multi-File Chat", layout="wide")
st.title("🚀 Gemini Multi-Format RAG")

# Initialize Session States
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [] 
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# --- 2. SIDEBAR: CHAT HISTORY & OPTIONS ---
with st.sidebar:
    st.header("📜 Chat History")
    if st.button("➕ Clear All History"):
        st.session_state.chat_history = []
        st.rerun()

    st.divider()

    for idx, chat in enumerate(st.session_state.chat_history):
        col1, col2, col3, col4 = st.columns([4, 1, 1, 1])
        with col1:
            st.write(f"**{chat['title'][:15]}...**")
        with col2:
            if st.button("✏️", key=f"ren_{idx}"):
                chat['title'] = "Updated Query" # Simple rename logic
        with col3:
            if st.button("🗑️", key=f"del_{idx}"):
                st.session_state.chat_history.pop(idx)
                st.rerun()
        with col4:
            chat_data = json.dumps(chat)
            st.download_button("💾", data=chat_data, file_name=f"chat_{idx}.json", key=f"dl_{idx}")

# --- 3. FILE UPLOADER ---
uploaded_files = st.file_uploader("Upload PDF, CSV, or TXT", accept_multiple_files=True)

if uploaded_files and st.session_state.vector_store is None:
    with st.spinner("Indexing documents..."):
        all_docs = []
        for file in uploaded_files:
            temp_path = f"./{file.name}"
            with open(temp_path, "wb") as f:
                f.write(file.getbuffer())
            
            if file.name.endswith(".pdf"): loader = PyPDFLoader(temp_path)
            elif file.name.endswith(".csv"): loader = CSVLoader(temp_path)
            else: loader = TextLoader(temp_path)
                
            all_docs.extend(loader.load())
            os.remove(temp_path)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = text_splitter.split_documents(all_docs)
        
        # FIXED: Use the correct model name to avoid the 404 error
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        st.session_state.vector_store = FAISS.from_documents(chunks, embeddings)
        st.success("Files indexed successfully!")

# --- 4. CHAT INTERFACE & BOTTOM-UP HISTORY ---
user_question = st.chat_input("Ask a question about your documents...")

if user_question and st.session_state.vector_store:
    docs = st.session_state.vector_store.similarity_search(user_question, k=4)
    context = "\n".join([d.page_content for d in docs])
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    response = llm.invoke(f"Use this context: {context}\n\nQuestion: {user_question}")
    
    st.session_state.chat_history.append({"title": user_question, "answer": response.content})

# Display History in normal order (Newest at the bottom)
for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(chat["title"])
    with st.chat_message("assistant"):

        st.markdown(chat["answer"])
