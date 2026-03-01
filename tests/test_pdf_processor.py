from utils.pdf_processor import PDFProcessor


class DummyFile:
    def __init__(self, type: str, size: int):
        self.type = type
        self.size = size


def test_validate_pdf_none():
    assert not PDFProcessor.validate_pdf(None)


def test_validate_pdf_invalid_type():
    f = DummyFile("text/plain", 1024)
    assert not PDFProcessor.validate_pdf(f)


def test_validate_pdf_too_large():
    f = DummyFile("application/pdf", 60 * 1024 * 1024)
    assert not PDFProcessor.validate_pdf(f)


def test_validate_pdf_valid():
    f = DummyFile("application/pdf", 1024)
    assert PDFProcessor.validate_pdf(f)
