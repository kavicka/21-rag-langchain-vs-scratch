"""All settings come from environment variables (or a local, git-ignored `.env`).

Why: secrets (Azure keys, AWS credentials) must never live in code or in git, and the same code has to run
on the Mac, in CI and later in AWS/Azure with nothing but a different environment.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore")

    # data / index
    data_dir: Path = ROOT / "data"
    chunking: Literal["section", "fixed"] = "section"
    qdrant_url: str | None = None  # e.g. http://localhost:6333 from docker compose; empty = embedded local mode
    qdrant_api_key: SecretStr | None = None

    # retrieval models (small English models: the corpus is English, so bge-m3 from P6 would be overkill)
    embed_model: str = "BAAI/bge-small-en-v1.5"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    device: str | None = None  # None = let sentence-transformers pick (mps on the Mac, cpu in the cloud)

    # generation
    llm_provider: Literal["ollama", "transformers", "bedrock", "azure_openai", "fake"] = "ollama"
    llm_model: str = "qwen3:4b"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 256
    ollama_url: str = "http://localhost:11434"
    # AWS: credentials come from the standard AWS chain (SSO profile, env, role), never from this file
    aws_region: str = "eu-central-1"
    # Azure OpenAI
    azure_openai_endpoint: str | None = None
    azure_openai_api_key: SecretStr | None = None
    azure_openai_api_version: str = "2024-10-21"

    # API
    allowed_origins: list[str] = ["http://localhost:5173"]


@lru_cache
def get_settings() -> Settings:
    return Settings()
