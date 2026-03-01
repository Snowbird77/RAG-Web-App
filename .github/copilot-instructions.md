# PDF RAG Web App - AI Agent Instructions

## Project Overview
A Streamlit web application for Retrieval-Augmented Generation (RAG) on PDF documents. Users upload PDFs, which are chunked, embedded locally, and indexed with FAISS. LangChain coordinates LLM calls via Groq API for contextual chat responses.

**Tech Stack:** Python 3.11 | Streamlit | LangChain | Groq API | FAISS | HuggingFace Embeddings

## Architecture Principles

### Core Data Flow
1. **PDF Upload** → Streamlit file widget → PDF parsing (PyPDF2/pdfplumber)
2. **Document Chunking** → LangChain RecursiveCharacterTextSplitter (tunable chunk_size/overlap)
3. **Embedding** → HuggingFace `all-MiniLM-L6-v2` (runs locally, no API calls)
4. **Vector Storage** → FAISS index (in-memory or persisted)
5. **RAG Retrieval** → FAISS similarity search → top-k results
6. **LLM Response** → LangChain chain with Groq API (streaming enabled)

### Key Integration Points
- **LangChain RetrievalQA/Chat chains:** Abstracts retrieval + prompt templates
- **Groq API:** Fast LLM inference; requires `GROQ_API_KEY` in environment
- **HuggingFace Embeddings:** Single-call initialization; caches model locally
- **Streamlit Session State:** Persists PDF objects, FAISS index, chat history across reruns

## Critical Developer Patterns

### Environment & Secrets
- Store `GROQ_API_KEY` in `.env` (gitignored); load via `os.getenv()` or `streamlit.secrets`
- Implement early validation: check for missing API key before processing
- Never commit `.env` or API keys; use example `.env.example` for documentation

### PDF Processing Pattern
```python
# Chunking: control context window + retrieval quality
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,        # Adjust per domain (technical docs need larger chunks)
    chunk_overlap=150,      # Prevent context loss at boundaries
    separators=["\n\n", "\n", " "]
)
```

### Streamlit Session State Convention
```python
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
```
Prevents re-initializing expensive objects (embeddings, FAISS indices) on every rerun.

## Development Workflows

### Local Testing
1. Create `.env` with valid `GROQ_API_KEY`
2. Run: `streamlit run app.py`
3. Test with small PDFs first (< 10 pages) to iterate quickly

### Deployment
- Environment: Use `streamlit config.toml` for production settings
- Secrets: Inject `GROQ_API_KEY` via platform (Streamlit Cloud, Docker, etc.)
- FAISS Index: Consider persistent storage if users expect session continuity

## Code Organization
- `app.py` or `main.py`: Streamlit UI entry point (imports utilities)
- `rag/`: LangChain pipeline components (retrieval, chains, prompts)
- `utils/`: PDF parsing, text chunking, embeddings helpers
- `.env.example`: Document required env vars (never commit actual `.env`)

## Common Pitfalls to Avoid
1. **Chunking too small/large:** Test with actual user PDFs; default 1000 chars often works
2. **Synchronous Groq calls:** Use LangChain's async chains for better UX in Streamlit
3. **Memory leaks:** Regularly clear FAISS indices in session state if user uploads new PDFs
4. **Missing error handling:** Wrap Groq API calls; provide clear user feedback on failures

## Style Guidelines
- Use type hints for all functions
- Validate inputs early (file size, format, API keys)
- Log via `logging` module; use `st.error()` for user-facing errors only
- Keep Streamlit UI logic separate from RAG pipeline logic