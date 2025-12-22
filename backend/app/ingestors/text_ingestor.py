from pathlib import Path
from typing import List
from langchain_core.documents import Document

from ..chunking import split_text_to_docs
from .base import BaseIngestor

class TextIngestor(BaseIngestor):
    """
    READS PLAIN TEXT AND CHUNKS THEM
    """

    def __init__(self,path:Path|str):
        self.path=Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Text file not found:{self.path}")
        

    def _read_text(self)->str:
        try:
            return self.path.read_text(encoding="utf-8",errors="ignore")
        except Exception as e:
            raise RuntimeError(f"Falied to read text file {self.path}:{e}")

    def ingest(self)-> List[Document]:
        text=self._read_text()
        docs=split_text_to_docs(
            text,
            metadata={
                "source":self.path.name,
                "type":"text"

            }
        )
        return docs