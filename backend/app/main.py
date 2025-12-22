import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException,Form
from pathlib import Path
from dotenv import load_dotenv

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


# Bootstrap components
embedder = OpenAIEmbeddingsWrapper()

# Derive embedding dimensions
sample = embedder.embed(["hello world"]) if True else [[0.0] * 1536]
dim = len(sample[0])

store = FaissStore(dim=dim, persist_path=settings.DATA_DIR)
store.load()

rag = RAGService(embedder=embedder, store=store)



# Auto-detect ingestor based on file extension

def detect_ingestor(file_path: str):
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return PDFIngestor
    elif ext == ".docx":
        return DocxIngestor
    elif ext in [".txt",".md",".log"]:
        return TextIngestor
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: .pdf, .docx"
        )



# Unified ingestion endpoint

@app.post("/ingest/file")
async def ingest_file(file: UploadFile = File(...)):
    # 1. Save uploaded file
    dest = settings.UPLOAD_DIR / file.filename
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # 2. Select appropriate ingestor
    IngestorClass = detect_ingestor(str(dest))
    ingestor = IngestorClass(str(dest))

    # 3. Extract documents
    docs = ingestor.ingest()

    # 4. Store into vector store
    rag.ingest_documents(docs)

    return {
        "status": "ok",
        "ingested_chunks": len(docs),
        "source": dest.name,
        "type": dest.suffix
    }

@app.post("/ingest/url")
async def ingest_url(url:str=Form(...)):
    try:
        ingestor=WebIngestor(url)
        docs=ingestor.ingest()
        rag.ingest_documents(docs)

        return{
            "status":"ok",
            "ingested_chunks":len(docs),
            "source":url,
            "type":"web"
        }
    
    except Exception as e:
        raise HTTPException(status_code=400,detail=str(e))



# ---------------------------------------------------------
# Query endpoint (unchanged)
# ---------------------------------------------------------
@app.post("/query")
async def query(question: str):
    return rag.answer(question)