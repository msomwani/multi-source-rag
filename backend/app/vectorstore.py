from pathlib import Path
from typing import Iterable, List
import faiss
import numpy as np

class FaissStore:
    def __init__(self, dim: int, persist_path: Path):
        self.dim = dim
        self.persist_path = Path(persist_path)
        self.index_file = self.persist_path / "faiss.index"
        self.meta_file = self.persist_path / "meta.npy"
        self._ensure()
        self.index = faiss.IndexFlatL2(dim)
        self.metadatas: List[dict] = []

    def _ensure(self):
        self.persist_path.mkdir(parents=True, exist_ok=True)

    def add(self, embeddings: Iterable[List[float]], metadatas: Iterable[dict], texts: Iterable[str]):
        arr = np.array(list(embeddings)).astype('float32')
        self.index.add(arr)
        self.metadatas.extend(list(metadatas))
        # persist
        faiss.write_index(self.index, str(self.index_file))
        np.save(self.meta_file, np.array(self.metadatas, dtype=object), allow_pickle=True)

    def load(self):
        if self.index_file.exists():
            self.index = faiss.read_index(str(self.index_file))
        if self.meta_file.exists():
            self.metadatas = np.load(self.meta_file, allow_pickle=True).tolist()

    def similarity_search(self, query_emb: List[float], k: int = 5) -> List[dict]:
        q = np.array([query_emb]).astype('float32')
        D, I = self.index.search(q, k)
        results = []
        for idx in I[0]:
            if idx < len(self.metadatas):
                results.append(self.metadatas[idx])
        return results