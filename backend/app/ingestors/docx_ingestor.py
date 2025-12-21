import os
from typing import List
from docx import Document as DocxDocument
from langchain_core.documents import Document

from ..chunking import split_text_to_docs
from .base import BaseIngestor


class DocxIngestor(BaseIngestor):
    def __init__(self,file_path:str):
        self.file_path=file_path

    def ingest(self)->List[Document]:
        doc=DocxDocument(self.file_path)
        docs=[]
        paragraphs=[p.text.strip() for p in doc.paragraphs if p.text.strip()]

        for i,para in enumerate(paragraphs):
            metadata={
                "source": os.path.basename(self.file_path),
                "type":"docx",
                "paragraph":i
            }

            chunked_docs=split_text_to_docs(para,metadata)
            docs.extend(chunked_docs)

        return docs