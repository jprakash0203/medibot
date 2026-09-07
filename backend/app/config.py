"""Application configuration loaded from environment variables."""
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Always resolve relative to backend/, not the process cwd
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_ENV_FILE = _BACKEND_ROOT / ".env"


class Settings(BaseSettings):
    """Typed application settings. Values come from .env or environment."""

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM
    llm_provider: str = "groq"
    llm_model: str = "llama-3.3-70b-versatile"
    llm_api_key: str
    llm_base_url: str | None = None

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "medibot_chunks"

    # Auth
    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 1440

    # Paths
    data_dir: Path = Path("./data/mediassist_data")
    sqlite_db_path: Path = Path("./data/mediassist_data/db/mediassist.db")

    # Models
    dense_embedding_model: str = "BAAI/bge-small-en-v1.5"
    sparse_embedding_model: str = "Qdrant/bm25"
    reranker_model: str = "BAAI/bge-reranker-base"

    # Retrieval
    retrieval_top_k: int = Field(default=10, ge=1, le=50)
    rerank_top_k: int = Field(default=3, ge=1, le=10)

    @field_validator("data_dir", "sqlite_db_path", mode="after")
    @classmethod
    def _resolve_backend_paths(cls, value: Path) -> Path:
        """Make relative paths work even when cwd is not backend/."""
        if not value.is_absolute():
            return (_BACKEND_ROOT / value).resolve()
        return value


@lru_cache
def get_settings() -> Settings:
    """Cache settings — creating once is enough (they don't change at runtime)."""
    return Settings()
