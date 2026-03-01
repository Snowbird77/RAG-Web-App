"""RAG pipeline orchestration using LangChain and Groq."""

import logging
import os
from typing import Optional, Generator

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

from rag.retriever import PDFRetriever
from utils.pdf_processor import PDFProcessor
from rag.prompts import SYSTEM_PROMPT, RETRIEVAL_PROMPT, CHAT_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Orchestrates PDF ingestion, retrieval, and LLM generation."""
    
    def __init__(
        self,
        groq_model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        chunk_size: int = 1000,
        chunk_overlap: int = 150
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            groq_model: Groq model identifier (default: llama-3.3-70b-versatile)
            temperature: LLM temperature for response variation
            max_tokens: Maximum tokens in response
            chunk_size: PDF chunk size in characters
            chunk_overlap: Chunk overlap to prevent context loss
        """
        # Validate API key
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment")
        
        # Initialize components
        self.retriever = PDFRetriever(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.pdf_processor = PDFProcessor()
        
        # Initialize LLM
        self.llm = ChatGroq(
            model=groq_model,
            temperature=temperature,
            max_tokens=max_tokens,
            groq_api_key=groq_api_key
        )
        
        # Create prompt template
        self.prompt = PromptTemplate(
            template=RETRIEVAL_PROMPT,
            input_variables=["context", "question"]
        )
        
        self.chat_history = []
        logger.info(f"RAGPipeline initialized with model: {groq_model}")
    
    def ingest_pdf(self, uploaded_file) -> None:
        """
        Ingest PDF file into the pipeline.
        
        Args:
            uploaded_file: Streamlit UploadedFile object
        """
        try:
            logger.info(f"Ingesting PDF: {uploaded_file.name}")
            
            # Extract text from PDF
            documents = self.pdf_processor.extract_text(uploaded_file)
            
            # Ingest into retriever
            self.retriever.ingest_documents(documents)
            
            self.chat_history = []
            logger.info("PDF ingested successfully")
        except Exception as e:
            logger.error(f"Error ingesting PDF: {str(e)}", exc_info=True)
            raise
    
    def chat(self, user_message: str) -> str:
        """
        Process user message and generate response using RAG.
        
        Args:
            user_message: User's query
            
        Returns:
            Generated response from LLM
        """
        try:
            # Add to history
            self.chat_history.append({"role": "user", "content": user_message})
            
            # Retrieve relevant context
            context = self.retriever.retrieve(user_message, k=4)
            
            # Build conversation history string
            history_str = "\n".join(f"{m['role']}: {m['content']}" for m in self.chat_history)
            # Compose prompt with system instructions and chat template
            prompt_text = SYSTEM_PROMPT + "\n\n" + CHAT_PROMPT_TEMPLATE.format(
                context=context,
                history=history_str,
                question=user_message,
            )
            
            # Generate response
            response = self.llm.invoke(prompt_text)
            response_text = response.content
            
            # Add to history
            self.chat_history.append({"role": "assistant", "content": response_text})
            
            logger.info("Chat response generated successfully")
            return response_text
        except Exception as e:
            logger.error(f"Error in chat: {str(e)}", exc_info=True)
            raise
    
    def chat_stream(self, user_message: str) -> Generator[str, None, None]:
        """
        Process user message and stream response tokens using RAG.
        
        Args:
            user_message: User's query
            
        Yields:
            Response tokens as they arrive from LLM
        """
        try:
            # Add to history
            self.chat_history.append({"role": "user", "content": user_message})
            
            # Retrieve relevant context
            context = self.retriever.retrieve(user_message, k=4)
            
            # Build conversation history string
            history_str = "\n".join(f"{m['role']}: {m['content']}" for m in self.chat_history)
            # Compose prompt with system instructions and chat template
            prompt_text = SYSTEM_PROMPT + "\n\n" + CHAT_PROMPT_TEMPLATE.format(
                context=context,
                history=history_str,
                question=user_message,
            )
            
            # Stream response from LLM (robust to different stream APIs)
            full_response = ""
            try:
                stream_gen = self.llm.stream(prompt_text)
                import re
                for chunk in stream_gen:
                    # chunk may be a string or an object with content/delta
                    if isinstance(chunk, str):
                        token = chunk
                    else:
                        # Prefer explicit attributes
                        token = getattr(chunk, "content", None)
                        if not token:
                            # delta may be a dict-like with content
                            delta = getattr(chunk, "delta", None)
                            if isinstance(delta, dict):
                                token = delta.get("content") or delta.get("text")
                        if not token:
                            # Fallback: stringify and try to remove metadata suffixes
                            s = str(chunk)
                            # Remove known metadata suffixes to avoid leaking them
                            for sep in ("response_metadata", "usage_metadata", "additional_kwargs", "id="):
                                if sep in s:
                                    s = s.split(sep)[0]
                            # Try to extract quoted content
                            m = re.search(r"content='([^']*)'", s)
                            if m and m.group(1).strip():
                                token = m.group(1)
                            else:
                                # Try capture after id if present
                                m2 = re.search(r"id='[^']*'\s*([^\n]+)", s)
                                if m2:
                                    token = m2.group(1)
                                else:
                                    # Fallback: take the last reasonably long readable substring
                                    words = re.findall(r"[A-Za-z][A-Za-z0-9 ,\.\-']{2,}", s)
                                    token = words[-1] if words else s
                    # Clean token and yield with a trailing space for readability
                    token = (token or "").replace("\\n", " ")
                    # remove stray 'content' keyword and extra whitespace
                    token = re.sub(r"\bcontent\b", "", token, flags=re.IGNORECASE)
                    token = re.sub(r"\s{2,}", " ", token).strip()
                    if token:
                        full_response += token + " "
                        yield token + " "
            except Exception:
                # Streaming not supported by this LLM client; fall back to non-streaming
                logger.info("LLM streaming not available, falling back to chunked response")
                response = self.llm.invoke(prompt_text)
                response_text = getattr(response, "content", str(response))
                # Yield in small chunks to simulate streaming
                chunk_size = 20
                for i in range(0, len(response_text), chunk_size):
                    token = response_text[i : i + chunk_size]
                    full_response += token
                    yield token
            
            # Add complete response to history
            self.chat_history.append({"role": "assistant", "content": full_response})
            
            logger.info("Chat stream completed successfully")
        except Exception as e:
            logger.error(f"Error in chat stream: {str(e)}", exc_info=True)
            raise
    
    def reset(self) -> None:
        """Reset pipeline state."""
        self.retriever.reset()
        self.chat_history = []
        logger.info("RAG pipeline reset")
