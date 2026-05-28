"""Central settings for the Legal RAG bot."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class PathSettings:
    base_dir: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    raw_data_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_data_dir: Path = PROJECT_ROOT / "data" / "processed"
    indexes_dir: Path = PROJECT_ROOT / "data" / "indexes"
    chroma_dir: Path = PROJECT_ROOT / "data" / "indexes" / "chroma_db"
    bm25_path: Path = PROJECT_ROOT / "data" / "indexes" / "bm25_retriever.pkl"


@dataclass(frozen=True)
class ModelSettings:
    llm_model: str = os.getenv("LEGAL_RAG_LLM_MODEL", "llama-3.1-8b-instant")
    judge_llm_model: str = os.getenv("LEGAL_RAG_JUDGE_LLM_MODEL", "llama-3.3-70b-versatile")
    embedding_model: str = os.getenv("LEGAL_RAG_EMBEDDING_MODEL", "keepitreal/vietnamese-sbert")
    reranker_model: str = os.getenv("LEGAL_RAG_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
    reranker_max_length: int = int(os.getenv("LEGAL_RAG_RERANKER_MAX_LENGTH", "512"))


@dataclass(frozen=True)
class RetrievalSettings:
    retrieval_k: int = int(os.getenv("LEGAL_RAG_RETRIEVAL_K", "30"))
    bm25_k: int = int(os.getenv("LEGAL_RAG_BM25_K", "10"))
    rerank_top_k: int = int(os.getenv("LEGAL_RAG_RERANK_TOP_K", "10"))


@dataclass(frozen=True)
class ChunkingSettings:
    chunk_size: int = int(os.getenv("LEGAL_RAG_CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("LEGAL_RAG_CHUNK_OVERLAP", "200"))


@dataclass(frozen=True)
class AppSettings:
    paths: PathSettings = field(default_factory=PathSettings)
    models: ModelSettings = field(default_factory=ModelSettings)
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    chunking: ChunkingSettings = field(default_factory=ChunkingSettings)


DEFAULT_CONFIG = AppSettings()
