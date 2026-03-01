"""PDF document processing utilities."""

import logging
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
import tempfile
import os

logger = logging.getLogger(__name__)


class PDFProcessor:
    """Handles PDF extraction and text processing."""
    
    def extract_text(self, uploaded_file) -> List[Document]:
        """
        Extract text from uploaded PDF file.
        
        Args:
            uploaded_file: Streamlit UploadedFile object
            
        Returns:
            List of Document objects with extracted text
            
        Raises:
            ValueError: If PDF processing fails
        """
        try:
            # Create temporary file to store uploaded PDF
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                tmp_path = tmp_file.name
            
            try:
                # Load PDF using LangChain
                loader = PyPDFLoader(tmp_path)
                documents = loader.load()
                
                if not documents:
                    raise ValueError("No text extracted from PDF")
                
                logger.info(
                    f"Extracted {len(documents)} pages from PDF: {uploaded_file.name}"
                )
                
                return documents
            finally:
                # Clean up temporary file
                os.unlink(tmp_path)
        except Exception as e:
            logger.error(f"Error extracting PDF text: {str(e)}", exc_info=True)
            raise ValueError(f"Failed to process PDF: {str(e)}")
    
    @staticmethod
    def validate_pdf(uploaded_file) -> bool:
        """
        Validate PDF file before processing.
        
        Args:
            uploaded_file: Streamlit UploadedFile object
            
        Returns:
            True if valid, False otherwise
        """
        if not uploaded_file:
            return False
        
        # Check file type
        if uploaded_file.type != "application/pdf":
            logger.warning(f"Invalid file type: {uploaded_file.type}")
            return False
        
        # Check file size (max 50MB)
        max_size = 50 * 1024 * 1024
        if uploaded_file.size > max_size:
            logger.warning(f"File size exceeds limit: {uploaded_file.size}")
            return False
        
        return True
