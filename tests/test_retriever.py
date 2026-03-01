import rag.retriever as retriever_module
from rag.retriever import PDFRetriever
from langchain_core.documents import Document


class FakeSplitter:
    def __init__(self, chunk_size=None, chunk_overlap=None, separators=None, length_function=None):
        pass

    def split_documents(self, documents):
        # return documents as-is if they already have page_content
        out = []
        for d in documents:
            if hasattr(d, "page_content"):
                out.append(d)
            else:
                out.append(Document(page_content=str(d)))
        return out


class FakeIndex:
    def __init__(self, docs):
        self.docs = docs

    def similarity_search(self, query, k=4):
        return self.docs[:k]


class FakeFAISS:
    @classmethod
    def from_documents(cls, docs, embeddings):
        return FakeIndex(docs)


def test_ingest_and_retrieve(monkeypatch):
    # Patch heavy dependencies with lightweight fakes
    monkeypatch.setattr(retriever_module, "RecursiveCharacterTextSplitter", FakeSplitter)
    monkeypatch.setattr(retriever_module, "FAISS", FakeFAISS)
    monkeypatch.setattr(retriever_module, "HuggingFaceEmbeddings", lambda model_name=None: None)

    pr = PDFRetriever()
    docs = [Document(page_content="doc a"), Document(page_content="doc b")]
    pr.ingest_documents(docs)
    ctx = pr.retrieve("query", k=2)
    assert "doc a" in ctx and "doc b" in ctx


def test_reset(monkeypatch):
    monkeypatch.setattr(retriever_module, "RecursiveCharacterTextSplitter", FakeSplitter)
    monkeypatch.setattr(retriever_module, "FAISS", FakeFAISS)
    monkeypatch.setattr(retriever_module, "HuggingFaceEmbeddings", lambda model_name=None: None)

    pr = PDFRetriever()
    docs = [Document(page_content="doc x")]
    pr.ingest_documents(docs)
    pr.reset()
    assert pr.vectorstore is None
