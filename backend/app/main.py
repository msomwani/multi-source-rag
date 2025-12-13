import os
import shutil
from fastapi import FastAPI, UploadFile, File, Form
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()


from .config import settings
from .ingest import PDFIngestor
from .embeddings import OpenAIEmbeddingsWrapper
from .vectorstore import FaissStore
from .rag import RAGService

app = FastAPI(title=settings.APP_NAME)

# Ensure directories exist
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# bootstrap components
embedder = OpenAIEmbeddingsWrapper()
# derive dim by embedding a tiny sample
sample = embedder.embed(["hello world"]) if True else [[0.0]*1536]
dim = len(sample[0])
store = FaissStore(dim=dim, persist_path=settings.DATA_DIR)
store.load()
rag = RAGService(embedder=embedder, store=store)


@app.post('/ingest/pdf')
async def ingest_pdf(file: UploadFile = File(...)):
    dest = settings.UPLOAD_DIR / file.filename
    with open(dest, 'wb') as f:
        shutil.copyfileobj(file.file, f)
    ingestor = PDFIngestor(dest)
    docs = ingestor.ingest()
    rag.ingest_documents(docs)
    return {"status": "ok", "ingested_chunks": len(docs), "source": str(dest.name)}


@app.post('/query')
async def query(question: str = Form(...)):
    return rag.answer(question)