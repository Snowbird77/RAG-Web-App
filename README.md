# PDF RAG Web App

Deployed app: https://rag-web-app-jyg6lmd59qqrxmuuuqvrlu.streamlit.app/

A Streamlit web application for chatting with PDF documents using Retrieval-Augmented Generation (RAG). Upload PDFs, ask questions, and get contextual responses powered by Groq's LLM API and local embeddings.

## Features

- 📄 Upload and process multiple PDF documents
- 💬 Interactive chat interface with streaming responses
- 🔍 Semantic search using FAISS vector store
- 🤖 Powered by Groq API (llama-3.3-70b-versatile)
- 🎯 Local embeddings with HuggingFace (no embedding API costs)
- 🐳 Docker support for easy deployment

## Tech Stack

- **Python 3.11**
- **Streamlit** - Web UI framework
- **LangChain** - RAG orchestration
- **Groq API** - Fast LLM inference
- **FAISS** - Vector similarity search
- **HuggingFace** - Local embeddings (all-MiniLM-L6-v2)

## Quick Start

### Option 1: Local Development

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd RAG-Web-App
   ```

2. **Create and activate a virtual environment**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Create `.env` and add your key**

   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

5. **Run the app**

   ```bash
   python3 -m streamlit run app.py
   ```

   Open: http://localhost:8501

### Option 2: Docker Setup

1. **Export your API key**

   ```bash
   export GROQ_API_KEY=your_groq_api_key_here
   ```

2. **Build and run**

   ```bash
   docker compose up --build
   ```

3. **Stop**

   ```bash
   docker compose down
   ```

   Open: http://localhost:8501

## Run Tests

```bash
python3 -m pytest -q
```

## Project Structure

- `app.py` — Streamlit UI and session state
- `rag/pipeline.py` — RAG orchestration + streaming generation
- `rag/retriever.py` — chunking, embeddings, FAISS retrieval
- `utils/pdf_processor.py` — PDF extraction/validation
- `tests/` — unit + integration tests
