"""FAISS-based vector retriever for PDF documents."""

import logging
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class PDFRetriever:
    """Manages FAISS vector store and document retrieval."""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """
        Initialize retriever with embeddings and chunking strategy.
        
        Args:
            chunk_size: Size of text chunks in characters
            chunk_overlap: Overlap between chunks to prevent context loss
            model_name: HuggingFace embedding model name
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model_name = model_name
        
        # Initialize embeddings (cached locally)
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " "],
            length_function=len
        )
        
        self.vectorstore: Optional[FAISS] = None
        logger.info(f"PDFRetriever initialized with model: {model_name}")
    
    def ingest_documents(self, documents: List[str]) -> None:
        """
        Ingest documents into FAISS vectorstore.
        
        Args:
            documents: List of document texts to index
        """
        try:
            # Split documents into chunks
            chunks = self.text_splitter.split_documents(documents)
            logger.info(f"Created {len(chunks)} chunks from documents")
            
            # Create FAISS index
            self.vectorstore = FAISS.from_documents(
                chunks,
                self.embeddings
            )
            logger.info("FAISS index created successfully")
        except Exception as e:
            logger.error(f"Error ingesting documents: {str(e)}", exc_info=True)
            raise
    
    def retrieve(self, query: str, k: int = 4) -> str:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: User query string
            k: Number of top-k results to retrieve
            
        Returns:
            Concatenated context from retrieved documents
        """
        if not self.vectorstore:
            raise ValueError("Vectorstore not initialized. Ingest documents first.")
        
        try:
            # Perform similarity search
            docs = self.vectorstore.similarity_search(query, k=k)
            
            # Concatenate retrieved documents
            context = "\n\n".join([doc.page_content for doc in docs])
            logger.info(f"Retrieved {len(docs)} documents for query")
            
            return context
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}", exc_info=True)
            raise
    
    def reset(self) -> None:
        """Clear the vectorstore."""
        self.vectorstore = None
        logger.info("Vectorstore reset")
