from pathlib import Path
from typing import Iterable, List
import faiss
import numpy as np
from langchain_core.documents import Document


class FaissStore:
    """
    A minimal FAISS + local persistence vector store.
    Stores:
      - embeddings index
      - documents (content + metadata)
    """

    def __init__(self, dim: int, persist_path: Path):
        self.dim = dim
        self.persist_path = Path(persist_path)
        self.index_file = self.persist_path / "faiss.index"
        self.docs_file = self.persist_path / "docs.npy"

        self._ensure()

        # FAISS index
        self.index = faiss.IndexFlatL2(dim)

        # Python list of Document objects
        self.docstore: List[Document] = []

    def _ensure(self):
        self.persist_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------
    # ADD DOCUMENTS
    # ------------------------------
    def add(self, embeddings: Iterable[List[float]], docs: Iterable[Document]):
        embeddings = list(embeddings)
        docs = list(docs)

        # Add vectors to FAISS
        arr = np.array(embeddings).astype("float32")
        self.index.add(arr)

        # Store documents
        self.docstore.extend(docs)

        # Save to disk
        faiss.write_index(self.index, str(self.index_file))
        np.save(self.docs_file, np.array(self.docstore, dtype=object), allow_pickle=True)

    # ------------------------------
    # LOAD EXISTING INDEX + DOCS
    # ------------------------------
    def load(self):
        if self.index_file.exists():
            self.index = faiss.read_index(str(self.index_file))

        if self.docs_file.exists():
            self.docstore = np.load(self.docs_file, allow_pickle=True).tolist()

    # ------------------------------
    # SEARCH
    # ------------------------------
    def search(self, query_embedding: List[float], top_k: int = 4) -> List[Document]:
        """Return top_k Document objects."""
        if len(self.docstore) == 0:
            return []

        q = np.array([query_embedding]).astype("float32")
        D, I = self.index.search(q, top_k)
        # print("DEBUG: FAISS indices =", I[0])
        # print("DEBUG: docstore length =", len(self.docstore))


        results = []
        for idx in I[0]:
            if idx == -1:
                continue
            if idx < len(self.docstore):
                results.append(self.docstore[idx])

        return results
