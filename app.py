import streamlit as st
from google_rag_pipeline import GoogleRAGPipeline
from rag_pipeline import RAGPipeline
from utils import process_pdf, process_text, process_url
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Intelligent Q&A",
    page_icon="🧠",
    layout="wide"
)

# --- RAG Pipeline Initialization Function ---
@st.cache_resource
def initialize_pipeline(pipeline_type):
    """Initializes and returns the selected RAG pipeline."""
    if pipeline_type == "Google AI":
        return GoogleRAGPipeline()
    elif pipeline_type == "Ollama":
        return RAGPipeline()
    return None

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pipeline_type" not in st.session_state:
    st.session_state.pipeline_type = "Google AI"

# Initialize the pipeline based on the session state
rag_pipeline = initialize_pipeline(st.session_state.pipeline_type)

if "sources" not in st.session_state:
    st.session_state.sources = rag_pipeline.get_sources() if rag_pipeline else []

# --- Main Application ---
st.title("🧠 Intelligent Document Q&A")
st.markdown("---")

# --- Tabs for Navigation ---
tab1, tab2 = st.tabs(["💬 Chat with Documents", "📚 Manage Knowledge Base"])

# --- Chat Tab ---
with tab1:
    st.header("Chat Interface")

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("View Sources"):
                    for source in message["sources"]:
                        st.info(source)

    # Accept user input
    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            with st.spinner("Thinking..."):
                try:
                    answer, sources = rag_pipeline.query_rag(prompt)
                    
                    # Simulate stream of response with milliseconds delay
                    for chunk in answer.split():
                        full_response += chunk + " "
                        time.sleep(0.05)
                        message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)

                    if sources:
                        with st.expander("View Sources"):
                            for source in sources:
                                st.info(source)
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": full_response,
                        "sources": sources
                    })

                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Sorry, I encountered an error: {e}"
                    })


# --- Knowledge Base Management Tab ---
with tab2:
    st.header("Knowledge Base Management")
    
    # Configuration section
    with st.container(border=True):
        st.subheader("⚙️ Configuration")
        new_pipeline_type = st.selectbox(
            "Choose a RAG Pipeline:",
            ("Google AI", "Ollama"),
            index=0 if st.session_state.pipeline_type == "Google AI" else 1
        )
        
        if new_pipeline_type != st.session_state.pipeline_type:
            st.session_state.pipeline_type = new_pipeline_type
            st.success(f"Switched to {new_pipeline_type} pipeline. The change will be applied on the next interaction or page reload.")
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    # Column 1: Add new documents
    with col1:
        with st.container(border=True):
            st.subheader("➕ Add New Document")
            input_method = st.radio(
                "Choose input method:",
                ("PDF", "URL", "Text"),
                horizontal=True
            )

            if input_method == "PDF":
                uploaded_file = st.file_uploader("Upload a PDF file", type="pdf", label_visibility="collapsed")
                if uploaded_file and st.button("Add PDF to Knowledge Base"):
                    with st.spinner("Processing PDF..."):
                        text = process_pdf(uploaded_file)
                        message = rag_pipeline.add_document(text, uploaded_file.name)
                        st.success(message)
                        st.session_state.sources = rag_pipeline.get_sources()
                        st.rerun()

            elif input_method == "URL":
                url = st.text_input("Enter a URL", placeholder="https://example.com")
                if url and st.button("Add URL to Knowledge Base"):
                    with st.spinner("Fetching and processing URL..."):
                        text = process_url(url)
                        message = rag_pipeline.add_document(text, url)
                        st.success(message)
                        st.session_state.sources = rag_pipeline.get_sources()
                        st.rerun()

            elif input_method == "Text":
                uploaded_file = st.file_uploader("Upload a text file", type="txt", label_visibility="collapsed")
                if uploaded_file and st.button("Add Text File to Knowledge Base"):
                    with st.spinner("Processing text file..."):
                        text = process_text(uploaded_file)
                        message = rag_pipeline.add_document(text, uploaded_file.name)
                        st.success(message)
                        st.session_state.sources = rag_pipeline.get_sources()
                        st.rerun()

    # Column 2: View and clear knowledge base
    with col2:
        with st.container(border=True):
            st.subheader("📖 Current Knowledge Base")
            if not st.session_state.sources:
                st.info("No documents in the knowledge base yet.")
            else:
                st.write("Here are the current sources:")
                for source in st.session_state.sources:
                    st.text(f"- {source}")

            if st.button("🗑️ Clear Knowledge Base", use_container_width=True):
                with st.spinner("Clearing knowledge base..."):
                    message = rag_pipeline.clear_knowledge_base()
                    st.success(message)
                    st.session_state.sources = []
                    st.rerun()