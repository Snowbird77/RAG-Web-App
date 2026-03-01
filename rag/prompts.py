"""Prompt templates for RAG pipeline."""

SYSTEM_PROMPT = """You are a helpful PDF assistant. Answer questions based on the provided context from the PDF document. 
If the answer is not found in the context, say "I couldn't find information about this in the PDF" rather than making up an answer.
Be concise and accurate in your responses."""

RETRIEVAL_PROMPT = """Based on the following context from the PDF, answer the user's question:

Context:
{context}

Question: {question}

Answer:"""

CHAT_PROMPT_TEMPLATE = """You are a helpful assistant answering questions about a PDF document.
Use the provided context to answer accurately. If information is not in the context, say so.

Context from PDF:
{context}

Conversation history:
{history}

User question: {question}

Answer:"""
