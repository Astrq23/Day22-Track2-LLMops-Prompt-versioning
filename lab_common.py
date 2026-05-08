"""Shared helpers for the lab scripts."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from config import ROOT, load_config


DATA_DIR = ROOT / "data"
KNOWLEDGE_BASE_FILE = DATA_DIR / "knowledge_base.txt"

SYSTEM_V1 = (
    "You are a helpful AI assistant. Answer the user's question using ONLY the provided context. "
    "Keep your answer concise (2-4 sentences). If the context does not contain the answer, say: "
    "I don't have enough information.\n\n"
    "Context:\n{context}"
)

SYSTEM_V2 = (
    "You are an expert AI tutor. Provide a structured, accurate answer.\n\n"
    "Instructions:\n"
    "1. Read the context carefully.\n"
    "2. Identify the key facts relevant to the question.\n"
    "3. Write a clear, well-organized answer (3-5 sentences).\n"
    "4. State explicitly if the context lacks sufficient information.\n\n"
    "Context:\n{context}"
)


def ensure_environment() -> None:
    config = load_config()
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
    os.environ.setdefault("LANGCHAIN_PROJECT", config.langsmith_project)
    if config.langsmith_api_key:
        os.environ.setdefault("LANGCHAIN_API_KEY", config.langsmith_api_key)


def get_llm_settings():
    config = load_config()
    return config.ollama_base_url, config.ollama_model, config.ollama_embedding_model


@lru_cache(maxsize=1)
def make_llm():
    from langchain_ollama import ChatOllama

    base_url, model_name, _ = get_llm_settings()
    return ChatOllama(model=model_name, base_url=base_url)


def _embedding_model_candidates(preferred_model: str) -> list[str]:
    candidates = [
        preferred_model,
        "nomic-embed-text",
        "mxbai-embed-large",
    ]

    unique_candidates = []
    for candidate in candidates:
        if candidate and candidate not in unique_candidates:
            unique_candidates.append(candidate)
    return unique_candidates


@lru_cache(maxsize=1)
def make_embeddings():
    from langchain_ollama import OllamaEmbeddings

    base_url, _, embedding_model = get_llm_settings()
    last_error = None

    for model_name in _embedding_model_candidates(embedding_model):
        embeddings = OllamaEmbeddings(model=model_name, base_url=base_url)
        try:
            embeddings.embed_query("embedding healthcheck")
            if model_name != embedding_model:
                print(f"ℹ️  Embedding fallback: '{embedding_model}' -> '{model_name}'")
            return embeddings
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        "No compatible Ollama embedding model found. "
        "Pull one first, e.g. 'ollama pull nomic-embed-text', or set OLLAMA_EMBEDDING_MODEL."
    ) from last_error


def load_knowledge_base() -> str:
    if KNOWLEDGE_BASE_FILE.exists():
        return KNOWLEDGE_BASE_FILE.read_text(encoding="utf-8")

    from qa_pairs import QA_PAIRS

    return "\n\n".join(
        f"Question: {pair['question']}\nAnswer: {pair['reference']}"
        for pair in QA_PAIRS
    )


def build_vectorstore(chunk_size: int = 500, chunk_overlap: int = 50):
    from langchain_community.vectorstores import FAISS
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    text = load_knowledge_base()
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(text)
    embeddings = make_embeddings()
    return FAISS.from_texts(chunks, embeddings)


def build_prompt(version: str):
    from langchain_core.prompts import ChatPromptTemplate

    system_message = SYSTEM_V1 if version == "v1" else SYSTEM_V2
    return ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", "{question}"),
    ])


PROMPT_V1 = build_prompt("v1")
PROMPT_V2 = build_prompt("v2")


def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)