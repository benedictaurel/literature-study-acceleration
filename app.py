import streamlit as st
import os
import shutil
from rag_engine import RAGEngine

# Page Config
st.set_page_config(page_title="AI Research Assistant", page_icon="📚", layout="wide")

# Title and Description
st.title("📚 AI Research Assistant")
st.markdown("""
This assistant uses **RAG (Retrieval-Augmented Generation)** to answer questions based on your uploaded academic papers (PDF). 
It can also **automatically find and download O-RAN Specifications** and search **OpenAlex** and **ArXiv** for broader research.
""")

# Session State Initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = None

if "docs_processed" not in st.session_state:
    st.session_state.docs_processed = False

# Sidebar
with st.sidebar:
    st.header("Document Upload")
    uploaded_files = st.file_uploader("Upload PDF Papers", type=["pdf"], accept_multiple_files=True)
    
    st.divider()
    st.header("Knowledge Base")
    files_placeholder = st.empty()
    
    def update_file_list():
        if os.path.exists("./papers"):
            files = [f for f in os.listdir("./papers") if f.endswith(".pdf")]
            if files:
                files_placeholder.markdown("\n".join([f"- {f}" for f in files]))
            else:
                files_placeholder.info("No documents found.")
        else:
            files_placeholder.info("No documents found.")

    # Initial load of file list
    update_file_list()

    st.divider()
    process_btn = st.button("Process Uploaded Documents")

    if process_btn:
        if not os.environ["GOOGLE_API_KEY"]:
            st.error("Please provide a Google API Key.")
        else:
            with st.spinner("Processing documents..."):
                # Ensure papers directory exists
                papers_dir = "./papers"
                if not os.path.exists(papers_dir):
                    os.makedirs(papers_dir)
                
                # Save uploaded files (APPEND, do not delete existing)
                if uploaded_files:
                    for uploaded_file in uploaded_files:
                        file_path = os.path.join(papers_dir, uploaded_file.name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                
                try:
                    # Initialize Engine if not exists
                    if st.session_state.rag_engine is None:
                         st.session_state.rag_engine = RAGEngine(
                             google_api_key=os.environ.get("GOOGLE_API_KEY", ""),
                             openalex_api_key=os.environ.get("OPENALEX_API_KEY")
                         )
                    
                    # Set Callback
                    st.session_state.rag_engine.set_file_callback(lambda path: update_file_list())

                    # Process Docs
                    st.session_state.rag_engine.initialize_vector_store(process_new=True)
                    st.session_state.rag_engine.create_agent()
                    st.session_state.docs_processed = True
                    
                    if uploaded_files:
                        st.success(f"Processed {len(uploaded_files)} documents successfully!")
                    else:
                        st.warning("Processed existing documents.")
                        
                except Exception as e:
                    st.error(f"Error processing documents: {e}")

    st.divider()
    
    if st.session_state.docs_processed:
        st.success("✅ System Ready")
    else:
        st.info("Waiting for documents...")

# Backend Initialization for existing session (if just refreshed but keys in env)
if st.session_state.rag_engine is None and os.getenv("GOOGLE_API_KEY"):
    try:
        st.session_state.rag_engine = RAGEngine()
        # Check if DB exists to load it
        if os.path.exists("./chroma_db") and os.listdir("./chroma_db"):
            with st.spinner("Loading existing knowledge base..."):
                # Register callback for this session too
                st.session_state.rag_engine.set_file_callback(lambda path: update_file_list())
                st.session_state.rag_engine.initialize_vector_store(process_new=False)
                st.session_state.rag_engine.create_agent()
                st.session_state.docs_processed = True
    except Exception:
        pass # Handle silently, user might need to input key

# Chat Interface
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question..."):
    if not st.session_state.rag_engine or not st.session_state.rag_engine.agent:
        st.error("Please initialize the system by providing an API Key and/or processing documents locally (or relying on external tools).")
        # Allow running without docs? The tools fallback to external.
        # But we need RAGEngine initialized.
        if st.session_state.rag_engine is None and os.environ.get("GOOGLE_API_KEY"):
             st.session_state.rag_engine = RAGEngine()
             st.session_state.rag_engine.initialize_vector_store(process_new=False)
             st.session_state.rag_engine.create_agent()
        elif st.session_state.rag_engine is None:
             st.stop()

    # Display user message
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    with st.chat_message("assistant"):
        with st.spinner("Agent is investigating..."):
            response = st.session_state.rag_engine.ask(prompt)
            st.markdown(response)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
