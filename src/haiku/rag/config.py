import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, field_validator

from haiku.rag.utils import get_default_data_dir

load_dotenv()


class AppConfig(BaseModel):
    ENV: str = "development"

    DEFAULT_DATA_DIR: Path = get_default_data_dir()
    MONITOR_DIRECTORIES: list[Path] = []

    EMBEDDINGS_PROVIDER: str = "ollama"
    EMBEDDINGS_MODEL: str = "mxbai-embed-large"
    EMBEDDINGS_VECTOR_DIM: int = 1024

    QA_PROVIDER: str = "ollama"
    QA_MODEL: str = "qwen3"

    CHUNK_SIZE: int = 1024  # Much larger for better context retention
    CHUNK_OVERLAP: int = 256  # Larger overlap for better continuity
    
    # Financial document settings
    USE_FINANCIAL_CHUNKER: bool = False  # Enable for HKEx announcements
    FINANCIAL_CHUNK_SIZE: int = 500  # Target chunk size for financial docs
    FINANCIAL_CHUNK_OVERLAP: int = 100  # Overlap for context
    FINANCIAL_MIN_CHUNK_SIZE: int = 300  # Minimum chunk size
    FINANCIAL_MAX_CHUNK_SIZE: int = 500  # Maximum chunk size
    FINANCIAL_CHUNK_SIZE_VARIANCE: int = 100  # Chunk size variance
    PRESERVE_TABLES: bool = True  # Avoid splitting tables
    EXTRACT_METADATA: bool = True  # Extract doc metadata
    
    # Financial QA settings
    USE_FINANCIAL_QA: bool = False  # Enable financial-specific QA
    FINANCIAL_QA_MODEL: str = ""  # Override QA model for financial queries

    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # Provider keys
    VOYAGE_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    OPENAI_BASE_URL: str = ""
    ANTHROPIC_API_KEY: str = ""

    # SiliconFlow configuration
    SILICONFLOW_API_KEY: str = ""
    SILICONFLOW_BASE_URL: str = "https://api.siliconflow.cn/v1"

    @field_validator("MONITOR_DIRECTORIES", mode="before")
    @classmethod
    def parse_monitor_directories(cls, v):
        if isinstance(v, str):
            if not v.strip():
                return []
            return [
                Path(path.strip()).absolute() for path in v.split(",") if path.strip()
            ]
        return v


# Expose Config object for app to import
Config = AppConfig.model_validate(os.environ)
if Config.OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = Config.OPENAI_API_KEY
if Config.VOYAGE_API_KEY:
    os.environ["VOYAGE_API_KEY"] = Config.VOYAGE_API_KEY
if Config.ANTHROPIC_API_KEY:
    os.environ["ANTHROPIC_API_KEY"] = Config.ANTHROPIC_API_KEY
