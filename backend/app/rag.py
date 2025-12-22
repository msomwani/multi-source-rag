from typing import List, Dict
from .embeddings import OpenAIEmbeddingsWrapper
from .vectorstore import FaissStore
from langchain_core.documents import Document
from openai import OpenAI


class RAGService:
    def __init__(self, embedder: OpenAIEmbeddingsWrapper, store: FaissStore):
        self.embedder = embedder
        self.store = store
        self.llm = OpenAI()  # Auto loads OPENAI_API_KEY

    # -----------------------------
    # INGEST DOCUMENTS
    # -----------------------------
    def ingest_documents(self, docs: List[Document]):
        texts = [d.page_content for d in docs]
        embs = self.embedder.embed(texts)
        print("DEBUG: Ingest embedding dim =", len(embs[0]))
        self.store.add(embs, docs)


    # -----------------------------
    # BUILD CONTEXT FROM DOCUMENTS
    # -----------------------------
    def _build_context(self, docs: List[Document]) -> str:
        parts = []
        for d in docs:
            meta = d.metadata
            src = meta.get("source", "unknown")
            page = meta.get("page")
            page_info = f" (page {page})" if page else ""

            parts.append(
                f"Source: {src}{page_info}\nContent:\n{d.page_content}\n---\n"
            )
        return "\n".join(parts)

    # -----------------------------
    # ANSWER QUESTION USING GPT-4.1
    # -----------------------------
    def answer(self, question: str, k: int = 4) -> Dict:
        # 1. Embed question

        q_embed = self.embedder.embed([question])[0]

        # 2. Retrieve top-k docs
        # print("DEBUG: Query embedding dim =", len(q_embed))

        docs = self.store.search(q_embed, top_k=k)

        # print("DEBUG: Retrieved docs =", docs)


        if not docs:
            return {
                "answer": "No relevant information found in the stored documents.",
                "sources": []
            }

        # 3. Build RAG context
        context = self._build_context(docs)

        # 4. Build prompt
        prompt = f"""
You are an assistant that must answer ONLY using the provided context.
If the answer is not present, reply: "I could not find the answer in the provided sources."

Context:
--------------------
{context}
--------------------

Question: {question}

Answer clearly and cite the sources (e.g. "from file.pdf page 3").
"""

        # 5. LLM call
        completion = self.llm.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "You answer strictly using the provided context."},
                {"role": "user", "content": prompt}
            ]
        )

        final_answer = completion.choices[0].message.content

        # 6. Prepare citations
        sources = [
            {k: v for k, v in doc.metadata.items()}
            for doc in docs
        ]

        return {
            "answer": final_answer,
            "sources": sources
        }
