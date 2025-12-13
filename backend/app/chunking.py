from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)


def split_text_to_docs(text: str, metadata: dict | None = None) -> list[Document]:
    """Split raw text into Document chunks preserving metadata."""
    doc = Document(page_content=text, metadata=metadata or {})
    return splitter.split_documents([doc])