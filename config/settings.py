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
    judge_llm_model: str = os.getenv("LEGAL_RAG_JUDGE_LLM_MODEL", "llama-3.1-8b-instant")
    embedding_model: str = os.getenv("LEGAL_RAG_EMBEDDING_MODEL", "keepitreal/vietnamese-sbert")
    reranker_model: str = os.getenv("LEGAL_RAG_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
    reranker_max_length: int = int(os.getenv("LEGAL_RAG_RERANKER_MAX_LENGTH", "512"))


def _env_float_optional(name: str) -> float | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return None
    return float(value)


@dataclass(frozen=True)
class RetrievalSettings:
    retrieval_k: int = int(os.getenv("LEGAL_RAG_RETRIEVAL_K", "15"))
    bm25_k: int = int(os.getenv("LEGAL_RAG_BM25_K", "10"))
    rerank_top_k: int = int(os.getenv("LEGAL_RAG_RERANK_TOP_K", "5"))
    rrf_k: int = int(os.getenv("LEGAL_RAG_RRF_K", "60"))
    semantic_weight: float = float(os.getenv("LEGAL_RAG_SEMANTIC_WEIGHT", "1.0"))
    bm25_weight: float = float(os.getenv("LEGAL_RAG_BM25_WEIGHT", "1.0"))
    rerank_min_score: float | None = _env_float_optional("LEGAL_RAG_RERANK_MIN_SCORE")
    graph_internal_ref_k: int = int(os.getenv("LEGAL_RAG_GRAPH_INTERNAL_REF_K", "20"))
    graph_external_scope_k: int = int(os.getenv("LEGAL_RAG_GRAPH_EXTERNAL_SCOPE_K", "10"))


@dataclass(frozen=True)
class ChunkingSettings:
    chunk_size: int = int(os.getenv("LEGAL_RAG_CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("LEGAL_RAG_CHUNK_OVERLAP", "200"))
    embedding_batch_size: int = int(os.getenv("LEGAL_RAG_EMBEDDING_BATCH_SIZE", "32"))
    chroma_batch_size: int = int(os.getenv("LEGAL_RAG_CHROMA_BATCH_SIZE", "256"))


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class GraphSettings:
    enabled: bool = _env_bool("LEGAL_RAG_GRAPH_ENABLED", True)
    neo4j_uri: str = os.getenv("LEGAL_RAG_NEO4J_URI", "bolt://localhost:7687")
    neo4j_user: str = os.getenv("LEGAL_RAG_NEO4J_USER", "neo4j")
    neo4j_password: str = os.getenv("LEGAL_RAG_NEO4J_PASSWORD", "")
    neo4j_database: str | None = os.getenv("LEGAL_RAG_NEO4J_DATABASE") or None
    extraction_mode: str = os.getenv("LEGAL_RAG_GRAPH_EXTRACTION_MODE", "hybrid")
    llm_provider: str = os.getenv("LEGAL_RAG_GRAPH_LLM_PROVIDER", "groq")
    llm_model: str = os.getenv("LEGAL_RAG_GRAPH_LLM_MODEL") or os.getenv(
        "LEGAL_RAG_LLM_MODEL",
        "llama-3.1-8b-instant",
    )
    llm_min_confidence: float = float(os.getenv("LEGAL_RAG_GRAPH_LLM_MIN_CONFIDENCE", "0.75"))
    llm_public_only: bool = _env_bool("LEGAL_RAG_GRAPH_LLM_PUBLIC_ONLY", True)


@dataclass(frozen=True)
class AppSettings:
    paths: PathSettings = field(default_factory=PathSettings)
    models: ModelSettings = field(default_factory=ModelSettings)
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    chunking: ChunkingSettings = field(default_factory=ChunkingSettings)
    graph: GraphSettings = field(default_factory=GraphSettings)


DEFAULT_CONFIG = AppSettings()
