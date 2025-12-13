# backend/app/config.py
from pydantic_settings import BaseSettings
from pathlib import Path
from dotenv import load_dotenv
import os

# Load .env early
load_dotenv()

class Settings(BaseSettings):
    # Project info
    APP_NAME: str = "pdf-first-rag"

    # File system paths
    DATA_DIR: Path = Path(os.getenv("DATA_DIR", "./data"))
    UPLOAD_DIR: Path = Path(os.getenv("UPLOAD_DIR", "./uploads"))

    # Models
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")

    # Secrets (optional at object creation; we validate later)
    OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")

    class Config:
        env_file = ".env"

# Instantiate settings (will read from the .env)
settings = Settings()
