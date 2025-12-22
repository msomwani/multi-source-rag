from pathlib import Path
from typing import List
import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document


from ..chunking import split_text_to_docs
from .base import BaseIngestor

class WebIngestor(BaseIngestor):
    """Fetches and cleans website content ,then chunks it inot Documnet objects"""

    def __init__(self,url:str):
        if not url.startswith("http"):
            raise ValueError("URL must start with http or https.")
        self.url=url

    def _fetch_html(self) ->str:
        try:
            resp=requests.get(self.url,timeout=10)
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            raise RuntimeError(f"Failed to fetch url {self.url}:{e}")
        
    def _extract_text(self,html:str)-> str:
        soup=BeautifulSoup(html,"html.parser")#built in python html parser

        for tag in soup (["script","style","nonscript"]):
            tag.decompose()

        text=soup.get_text(separator="\n")
        cleaned="\n".join([line.strip() for line in text.splitlines() if line.strip()])#removing blank spaces
        return cleaned
    
    def ingest(self) -> List[Document]:
        html=self._fetch_html()
        text=self._extract_text(html)

        docs=split_text_to_docs(
            text,
            metadata={"source":self.url,"type":"web"}
        )

        return docs

