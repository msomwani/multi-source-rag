from pathlib import Path
from typing import List
from langchain_core.documents import Document

from ..chunking import split_text_to_docs
from .base import BaseIngestor


class PDFIngestor(BaseIngestor):
    """Reads PDF, extracts per-page text, attaches metadata, splits into chunks."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"PDF not found: {self.path}")

    def _read_pages(self) -> List[str]:
        try:
            from pypdf import PdfReader
        except Exception as e:
            raise ImportError("pypdf is required. Install with `pip install pypdf`.") from e

        reader = PdfReader(str(self.path))
        pages: List[str] = []

        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            pages.append(text)

        return pages

    def _pages_to_docs(self, pages: List[str]) -> List[Document]:
        docs: List[Document] = []
        for i, text in enumerate(pages, start=1):
            metadata = {
                "source": str(self.path.name),
                "type": "pdf",
                "page": i
            }
            docs.append(Document(page_content=text, metadata=metadata))
        return docs

    def ingest(self) -> List[Document]:
        pages = self._read_pages()
        page_docs = self._pages_to_docs(pages)

        chunked: List[Document] = []

        for doc in page_docs:
            chunked.extend(
                split_text_to_docs(doc.page_content, metadata=doc.metadata)
            )

        return chunked
