from typing import List
from langchain_core.documents import Document
class BaseIngestor:
    """Abstract class for all igestors"""
    def ingest(self)-> List[Document]:
        raise NotImplementedError
    