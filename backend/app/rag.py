from typing import List
from .embeddings import OpenAIEmbeddingsWrapper
from .vectorstore import FaissStore

class RAGService:
    def __init__(self, embedder: OpenAIEmbeddingsWrapper, store: FaissStore, llm=None):
        self.embedder = embedder
        self.store = store
        self.llm = llm # optional; we can plug in LLM calls when querying

    def ingest_documents(self, docs: List):
        texts = [d.page_content for d in docs]
        metadatas = [d.metadata for d in docs]
        embs = self.embedder.embed(texts)
        self.store.add(embs, metadatas, texts)

    def retrieve(self, question: str, k: int = 5):
        q_emb = self.embedder.embed([question])[0]
        return self.store.similarity_search(q_emb, k=k)

    def answer(self, question: str, k: int =5):
        # placeholder: will call LLM later; for now just return retrieved metadata
        retrieved = self.retrieve(question, k=k)
        return {"answer": "(LLM not wired yet)", "retrieved": retrieved}