import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

from .config import settings
from .embeddings import OpenAIEmbeddingsWrapper
from .vectorstore import FaissStore
from .rag import RAGService

from .ingestors.pdf_ingestor import PDFIngestor
from .ingestors.docx_ingestor import DocxIngestor
from .ingestors.web_ingestor import WebIngestor
from .ingestors.text_ingestor import TextIngestor


app = FastAPI(title=settings.APP_NAME)

# Ensure directories exist
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------
# INITIALIZE EMBEDDINGS + FAISS STORE SAFELY
# ------------------------------------------------------

# 1. Embed a dummy text to detect true embedding dimension
embedder = OpenAIEmbeddingsWrapper()
sample = embedder.embed(["hello world"])
dim = len(sample[0])

# 2. Initialize vector store
store = FaissStore(dim=dim, persist_path=settings.DATA_DIR)

# 3. Load persisted FAISS + docs if any
store.load()

# 4. Check for dimension mismatch (critical fix)
if store.index.ntotal > 0:
    stored_dim = store.index.d
    if stored_dim != dim:
        raise ValueError(
            f"FAISS index dimension mismatch: stored {stored_dim} vs embedder {dim}.\n"
            f"DELETE the folder: {settings.DATA_DIR} and restart the server."
        )

# 5. Initialize RAG
rag = RAGService(embedder=embedder, store=store)



# ------------------------------------------------------
# FILE TYPE DETECTION
# ------------------------------------------------------

def detect_ingestor(file_path: str):
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return PDFIngestor
    elif ext == ".docx":
        return DocxIngestor
    elif ext in [".txt", ".md", ".log"]:
        return TextIngestor
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: .pdf, .docx, .txt"
        )



# ------------------------------------------------------
# INGEST FILE ENDPOINT
# ------------------------------------------------------

@app.post("/ingest/file")
async def ingest_file(file: UploadFile = File(...)):
    dest = settings.UPLOAD_DIR / file.filename

    # Save uploaded file
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Detect ingestor and ingest
    IngestorClass = detect_ingestor(str(dest))
    ingestor = IngestorClass(str(dest))
    docs = ingestor.ingest()

    # Store into vector DB
    rag.ingest_documents(docs)

    return {
        "status": "ok",
        "ingested_chunks": len(docs),
        "source": dest.name,
        "type": dest.suffix
    }



# ------------------------------------------------------
# INGEST URL ENDPOINT
# ------------------------------------------------------

@app.post("/ingest/url")
async def ingest_url(url: str = Form(...)):
    try:
        ingestor = WebIngestor(url)
        docs = ingestor.ingest()
        rag.ingest_documents(docs)

        return {
            "status": "ok",
            "ingested_chunks": len(docs),
            "source": url,
            "type": "web"
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))



# ------------------------------------------------------
# QUERY ENDPOINT
# ------------------------------------------------------
class QueryRequest(BaseModel):
    question:str

@app.post("/query")
async def query(payload:QueryRequest):
    return rag.answer(payload.question)