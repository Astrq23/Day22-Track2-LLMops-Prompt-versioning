"""Shared configuration for the Day 22 lab.

Run this file directly to verify the environment configuration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class LabConfig:
    langsmith_api_key: str
    langsmith_project: str
    ollama_base_url: str
    ollama_model: str
    ollama_embedding_model: str


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return

    load_dotenv(ROOT / ".env", override=False)


def _first_env(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


def load_config() -> LabConfig:
    _load_dotenv()

    langsmith_api_key = _first_env("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY", default="") or ""
    langsmith_project = _first_env("LANGCHAIN_PROJECT", "LANGSMITH_PROJECT", default="day22-langsmith-lab") or "day22-langsmith-lab"
    ollama_base_url = _first_env("OLLAMA_BASE_URL", default="http://localhost:11434") or "http://localhost:11434"
    ollama_model = _first_env("OLLAMA_MODEL", default="llama3.1:8b") or "llama3.1:8b"
    ollama_embedding_model = _first_env("OLLAMA_EMBEDDING_MODEL", default="nomic-embed-text") or "nomic-embed-text"

    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
    os.environ.setdefault("LANGCHAIN_PROJECT", langsmith_project)
    if langsmith_api_key:
        os.environ.setdefault("LANGCHAIN_API_KEY", langsmith_api_key)

    return LabConfig(
        langsmith_api_key=langsmith_api_key,
        langsmith_project=langsmith_project,
        ollama_base_url=ollama_base_url,
        ollama_model=ollama_model,
        ollama_embedding_model=ollama_embedding_model,
    )


def print_config() -> None:
    config = load_config()

    print("✅ Config loaded successfully")
    print(f"   LangSmith project : {config.langsmith_project}")
    print(f"   Ollama endpoint   : {config.ollama_base_url}")
    print(f"   Default LLM model : {config.ollama_model}")
    print(f"   Embedding model   : {config.ollama_embedding_model}")


if __name__ == "__main__":
    print_config()