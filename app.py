"""
PDF RAG Web App - Streamlit entry point
Enables users to upload PDFs and chat with them via Groq API + FAISS retrieval.
"""

import os
import logging
from typing import Optional, cast

import streamlit as st
from dotenv import load_dotenv

from rag.pipeline import RAGPipeline
from rag.prompts import SYSTEM_PROMPT

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="PDF RAG Chat",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
def initialize_session_state() -> None:
    """Initialize Streamlit session state for RAG pipeline components."""
    # pipelines mapping pdf_name -> RAGPipeline
    if "pipelines" not in st.session_state:
        st.session_state["pipelines"] = {}  # type: dict[str, RAGPipeline]
    # chat history per document
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = {}  # type: dict[str, list]
    # currently selected PDF name
    if "selected_pdf" not in st.session_state:
        st.session_state["selected_pdf"] = None



def validate_api_key() -> bool:
    """Validate that GROQ_API_KEY is present."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error(
            "❌ Missing GROQ_API_KEY. Please set it in `.env` file or environment variables."
        )
        return False
    return True


def process_uploaded_pdf(uploaded_file) -> None:
    """
    Process uploaded PDF and initialize RAG pipeline.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
    """
    name = uploaded_file.name
    try:
        with st.spinner("🔄 Processing PDF..."):
            pipeline = RAGPipeline()
            pipeline.ingest_pdf(uploaded_file)
            # store pipeline and select
            st.session_state["pipelines"][name] = pipeline
            st.session_state["selected_pdf"] = name
            # reset history for this document
            st.session_state["chat_history"][name] = []
        st.success(f"✅ Successfully loaded: {name}")
        logger.info(f"PDF processed: {name}")
    except Exception as e:
        st.error(f"❌ Error processing PDF: {str(e)}")
        logger.error(f"PDF processing failed: {str(e)}", exc_info=True)
        # remove any partial pipeline
        st.session_state["pipelines"].pop(name, None)
        st.session_state["selected_pdf"] = None


def handle_chat_input(user_message: str) -> None:
    """Route input to currently selected pipeline and history."""
    """
    Process user message and stream RAG response.
    
    Args:
        user_message: User's input query
    """
    if not user_message.strip():
        return
    
    selected = st.session_state.get("selected_pdf")
    if not selected or selected not in st.session_state["pipelines"]:
        st.error("⚠️ Please upload and select a PDF first.")
        return
    pipeline = st.session_state["pipelines"][selected]
    # Add user message to history and UI
    st.session_state["chat_history"][selected].append({"role": "user", "content": user_message})
    with st.chat_message("user"):
        st.write(user_message)
    
    try:
        # Stream assistant response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            
            try:
                for token in pipeline.chat_stream(user_message):
                    full_response += token
                    response_placeholder.write(full_response)
            except AttributeError:
                # Fallback for when chat_stream is not available (testing, older versions)
                response = pipeline.chat(user_message)
                response_placeholder.write(response)
                full_response = response
        
        logger.info("Chat response generated successfully")
        # Save assistant response to session state so history and debug show it
        try:
            st.session_state["chat_history"][selected].append({"role": "assistant", "content": full_response})
        except Exception:
            pass
        else:
            try:
                logger.info("Assistant saved to session: %s", full_response[:200].replace("\n", " "))
            except Exception:
                logger.info("Assistant saved to session (could not preview content)")
    except Exception as e:
        st.error(f"❌ Error generating response: {str(e)}")
        logger.error(f"Chat failed: {str(e)}", exc_info=True)
        # Remove failed user message from history
        st.session_state["chat_history"].pop()


def render_sidebar() -> None:
    """Render sidebar with PDF upload and settings."""
    with st.sidebar:
        st.header("📚 Upload PDF")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help="Upload a PDF document to chat with"
        )
        
        if uploaded_file is not None:
            if uploaded_file.name not in st.session_state["pipelines"]:
                process_uploaded_pdf(uploaded_file)
        # selection dropdown
        if st.session_state["pipelines"]:
            keys = list(st.session_state["pipelines"].keys())
            sel = st.session_state.get("selected_pdf")
            try:
                default = keys.index(sel) if sel in keys else 0
            except Exception:
                default = 0
            choice = st.selectbox("Open PDF", keys, index=default)
            st.session_state["selected_pdf"] = choice
        
        st.divider()
        
        # Display current PDF status
        sel = st.session_state.get("selected_pdf")
        if sel:
            st.success(f"✅ Current PDF: {sel}")
        else:
            st.info("📤 Upload a PDF to begin")
        
        st.divider()
        
        # Settings
        st.subheader("⚙️ Settings")
        if st.button("🗑️ Clear Chat History"):
            sel = st.session_state.get("selected_pdf")
            if sel and sel in st.session_state["chat_history"]:
                st.session_state["chat_history"][sel] = []
            st.rerun()
        if st.button("🗑️ Remove PDF"):
            sel = st.session_state.get("selected_pdf")
            if sel:
                st.session_state["pipelines"].pop(sel, None)
                st.session_state["chat_history"].pop(sel, None)
                st.session_state["selected_pdf"] = None
            st.rerun()


def render_chat() -> None:
    """Render chat interface with message history."""
    st.header("💬 Chat with Your PDF")
    
    sel = st.session_state.get("selected_pdf")
    if not sel or sel not in st.session_state["pipelines"]:
        st.info("👈 Upload a PDF in the sidebar to start chatting")
        return
    # Display chat history for selected PDF
    for message in st.session_state["chat_history"].get(sel, []):
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Input area
    user_input = st.chat_input(
        "Ask a question about the PDF...",
        key="chat_input"
    )
    if user_input:
        handle_chat_input(user_input)
        st.rerun()

    # End of chat rendering


def main() -> None:
    """Main application entry point."""
    initialize_session_state()
    
    # Validate API key
    if not validate_api_key():
        st.stop()
    
    # Render UI
    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()
