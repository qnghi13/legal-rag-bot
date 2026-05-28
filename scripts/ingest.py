"""Build Chroma and BM25 indexes from raw documents."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import DEFAULT_CONFIG
from src.chunking.legal_chunker import load_and_chunk_folder
from src.embeddings.embedding_model import get_embedding_model


def create_vector_db(
    folder_path: str | Path,
    *,
    chunk_size: int = DEFAULT_CONFIG.chunking.chunk_size,
    chunk_overlap: int = DEFAULT_CONFIG.chunking.chunk_overlap,
    bm25_k: int = DEFAULT_CONFIG.retrieval.bm25_k,
    embedding_model: str = DEFAULT_CONFIG.models.embedding_model,
    chroma_path: str | Path | None = None,
    bm25_path: str | Path | None = None,
):
    chroma_dir = Path(chroma_path) if chroma_path else DEFAULT_CONFIG.paths.chroma_dir
    bm25_file = Path(bm25_path) if bm25_path else DEFAULT_CONFIG.paths.bm25_path

    chunks = load_and_chunk_folder(
        str(folder_path),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    if not chunks:
        print("[ingest] No chunks were created. Check the input folder.")
        return None

    print(f"[ingest] Building BM25 index (k={bm25_k})...")
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = bm25_k
    bm25_file.parent.mkdir(parents=True, exist_ok=True)
    with bm25_file.open("wb") as f:
        pickle.dump(bm25_retriever, f)
    print(f"[ingest] Saved BM25 index: {bm25_file}")

    print(f"[ingest] Loading embedding model: {embedding_model}")
    embedding = get_embedding_model(embedding_model)

    print(f"[ingest] Building Chroma DB: {chroma_dir}")
    chroma_dir.parent.mkdir(parents=True, exist_ok=True)
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=str(chroma_dir),
    )
    print(f"[ingest] Saved Chroma DB with {len(chunks)} chunks.")
    return db


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Legal RAG indexes")
    parser.add_argument("--data-dir", default=str(DEFAULT_CONFIG.paths.raw_data_dir))
    parser.add_argument("--chroma-path", default=None)
    parser.add_argument("--bm25-path", default=None)
    parser.add_argument("--embedding-model", default=DEFAULT_CONFIG.models.embedding_model)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CONFIG.chunking.chunk_size)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CONFIG.chunking.chunk_overlap)
    parser.add_argument("--bm25-k", type=int, default=DEFAULT_CONFIG.retrieval.bm25_k)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    create_vector_db(
        args.data_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        bm25_k=args.bm25_k,
        embedding_model=args.embedding_model,
        chroma_path=args.chroma_path,
        bm25_path=args.bm25_path,
    )


if __name__ == "__main__":
    main()
