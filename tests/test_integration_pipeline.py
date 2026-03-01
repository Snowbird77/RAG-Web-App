import os

from langchain_core.documents import Document

from rag.pipeline import RAGPipeline


class FakeRetriever:
    def __init__(self):
        self.docs = None

    def ingest_documents(self, documents):
        self.docs = documents

    def retrieve(self, query, k=4):
        return "\n\n".join([d.page_content for d in self.docs])


class FakePDFProcessor:
    def extract_text(self, uploaded_file):
        return [Document(page_content="This is a test document about Foobar.")]


class FakeLLM:
    def invoke(self, prompt_text):
        class R:
            def __init__(self, content):
                self.content = content

        return R("FAKE_RESPONSE based on prompt")


def test_rag_pipeline_integration(monkeypatch):
    # Ensure pipeline does not raise on missing real API access
    os.environ["GROQ_API_KEY"] = "test"

    pipeline = RAGPipeline()

    # Replace heavy components with fakes for integration test
    pipeline.retriever = FakeRetriever()
    pipeline.pdf_processor = FakePDFProcessor()
    pipeline.llm = FakeLLM()

    class DummyFile:
        name = "dummy.pdf"

    # Ingest and chat through the pipeline
    pipeline.ingest_pdf(DummyFile())
    response = pipeline.chat("What is Foobar?")

    assert "FAKE_RESPONSE" in response
    assert pipeline.chat_history[-1]["role"] == "assistant"
