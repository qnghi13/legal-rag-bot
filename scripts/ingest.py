"""Build Chroma and BM25 indexes from raw documents."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import DEFAULT_CONFIG
from src.chunking.legal_chunker import load_and_chunk_folder
from src.embeddings.embedding_model import get_embedding_model
from src.graph.extractor import GraphExtractionAudit, extract_graphs_from_vbpl_metadata
from src.graph.neo4j_store import Neo4jLegalGraphStore


def create_vector_db(
    folder_path: str | Path,
    *,
    chunk_size: int = DEFAULT_CONFIG.chunking.chunk_size,
    chunk_overlap: int = DEFAULT_CONFIG.chunking.chunk_overlap,
    bm25_k: int = DEFAULT_CONFIG.retrieval.bm25_k,
    embedding_model: str = DEFAULT_CONFIG.models.embedding_model,
    embedding_batch_size: int = DEFAULT_CONFIG.chunking.embedding_batch_size,
    chroma_batch_size: int = DEFAULT_CONFIG.chunking.chroma_batch_size,
    chroma_path: str | Path | None = None,
    bm25_path: str | Path | None = None,
    metadata_db: str | Path | None = None,
    build_graph: bool = False,
    reset_graph: bool = False,
    graph_extraction_mode: str = DEFAULT_CONFIG.graph.extraction_mode,
    allow_private_llm_extraction: bool = False,
    graph_audit_output: str | Path | None = None,
):
    if embedding_batch_size < 1:
        raise ValueError("embedding_batch_size must be >= 1")
    if chroma_batch_size < 1:
        raise ValueError("chroma_batch_size must be >= 1")

    chroma_dir = Path(chroma_path) if chroma_path else DEFAULT_CONFIG.paths.chroma_dir
    bm25_file = Path(bm25_path) if bm25_path else DEFAULT_CONFIG.paths.bm25_path

    chunks = load_and_chunk_folder(
        str(folder_path),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        metadata_db=str(metadata_db) if metadata_db else None,
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

    print(
        f"[ingest] Loading embedding model: {embedding_model} "
        f"(encode batch={embedding_batch_size})"
    )
    embedding = get_embedding_model(
        embedding_model,
        batch_size=embedding_batch_size,
        show_progress=True,
    )

    print(f"[ingest] Building Chroma DB: {chroma_dir}")
    chroma_dir.parent.mkdir(parents=True, exist_ok=True)
    db = Chroma(
        persist_directory=str(chroma_dir),
        embedding_function=embedding,
    )
    chunk_ids = [_make_chunk_id(chunk, index) for index, chunk in enumerate(chunks)]
    start_time = time.perf_counter()
    for start in range(0, len(chunks), chroma_batch_size):
        end = min(start + chroma_batch_size, len(chunks))
        db.add_documents(
            documents=chunks[start:end],
            ids=chunk_ids[start:end],
        )
        elapsed = time.perf_counter() - start_time
        rate = end / elapsed if elapsed else 0
        print(
            f"[ingest] Embedded/upserted {end}/{len(chunks)} chunks "
            f"({rate:.1f} chunks/s)"
        )
    print(f"[ingest] Saved Chroma DB with {len(chunks)} chunks.")
    if build_graph:
        _build_neo4j_graph(
            folder_path,
            metadata_db,
            extraction_mode=graph_extraction_mode,
            reset_graph=reset_graph,
            allow_private_llm_extraction=allow_private_llm_extraction,
            audit_output=graph_audit_output,
        )
    return db


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Legal RAG indexes")
    parser.add_argument("--data-dir", default=str(DEFAULT_CONFIG.paths.raw_data_dir))
    parser.add_argument("--chroma-path", default=None)
    parser.add_argument("--bm25-path", default=None)
    parser.add_argument(
        "--metadata-db",
        default=None,
        help=(
            "SQLite metadata registry. Defaults to metadata.sqlite in the data "
            "directory or its parent, e.g. data/raw/vbpl/metadata.sqlite."
        ),
    )
    parser.add_argument("--embedding-model", default=DEFAULT_CONFIG.models.embedding_model)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CONFIG.chunking.chunk_size)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CONFIG.chunking.chunk_overlap)
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=DEFAULT_CONFIG.chunking.embedding_batch_size,
        help="SentenceTransformer encode batch size.",
    )
    parser.add_argument(
        "--chroma-batch-size",
        type=int,
        default=DEFAULT_CONFIG.chunking.chroma_batch_size,
        help="Number of chunks to embed/upsert per Chroma add_documents call.",
    )
    parser.add_argument("--bm25-k", type=int, default=DEFAULT_CONFIG.retrieval.bm25_k)
    parser.add_argument(
        "--build-graph",
        action="store_true",
        help="Also extract legal graph facts and upsert them into Neo4j.",
    )
    parser.add_argument(
        "--reset-graph",
        action="store_true",
        help="Delete existing Neo4j legal graph nodes before rebuilding.",
    )
    parser.add_argument(
        "--graph-extraction-mode",
        choices=["rule", "hybrid", "llm-shadow"],
        default=DEFAULT_CONFIG.graph.extraction_mode,
        help="Graph extraction strategy.",
    )
    parser.add_argument(
        "--allow-private-llm-extraction",
        action="store_true",
        help="Allow LLM graph extraction for non-VBPL/private documents.",
    )
    parser.add_argument(
        "--graph-audit-output",
        default=None,
        help="Optional JSON path for graph extraction audit counters.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    create_vector_db(
        args.data_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        bm25_k=args.bm25_k,
        embedding_model=args.embedding_model,
        embedding_batch_size=args.embedding_batch_size,
        chroma_batch_size=args.chroma_batch_size,
        chroma_path=args.chroma_path,
        bm25_path=args.bm25_path,
        metadata_db=args.metadata_db,
        build_graph=args.build_graph,
        graph_extraction_mode=args.graph_extraction_mode,
        reset_graph=args.reset_graph,
        allow_private_llm_extraction=args.allow_private_llm_extraction,
        graph_audit_output=args.graph_audit_output,
    )


def _make_chunk_id(chunk, index: int) -> str:
    metadata = chunk.metadata
    clause_id = metadata.get("clause_id") or metadata.get("unit_id")
    if clause_id:
        chunk_part = metadata.get("chunk_part", 0)
        digest = hashlib.sha1(f"{clause_id}:{chunk_part}".encode("utf-8")).hexdigest()
        return f"chunk-{digest}"
    document_key = (
        metadata.get("document_id")
        or metadata.get("source_url")
        or metadata.get("markdown_path")
        or metadata.get("source")
        or "document"
    )
    chunk_key = metadata.get("chunk_index", index)
    digest = hashlib.sha1(f"{document_key}:{chunk_key}".encode("utf-8")).hexdigest()
    return f"chunk-{digest}"


def _build_neo4j_graph(
    folder_path: str | Path,
    metadata_db: str | Path | None,
    *,
    extraction_mode: str,
    reset_graph: bool,
    allow_private_llm_extraction: bool,
    audit_output: str | Path | None,
) -> None:
    metadata_db_path = _resolve_metadata_db(Path(folder_path), metadata_db)
    if not metadata_db_path:
        raise FileNotFoundError(
            "Cannot build graph without a VBPL metadata SQLite database. "
            "Pass --metadata-db data/raw/vbpl/metadata.sqlite."
        )

    print(f"[ingest] Extracting graph facts from: {metadata_db_path}")
    audit = GraphExtractionAudit()
    graphs = extract_graphs_from_vbpl_metadata(
        metadata_db_path,
        extraction_mode=extraction_mode,
        allow_private_llm_extraction=allow_private_llm_extraction,
        audit=audit,
    )
    print(f"[ingest] Extracted graph facts for {len(graphs)} documents.")
    _write_graph_audit(audit, audit_output)
    if not graphs:
        return

    print(f"[ingest] Upserting legal graph into Neo4j: {DEFAULT_CONFIG.graph.neo4j_uri}")
    with Neo4jLegalGraphStore(
        DEFAULT_CONFIG.graph.neo4j_uri,
        DEFAULT_CONFIG.graph.neo4j_user,
        DEFAULT_CONFIG.graph.neo4j_password,
        database=DEFAULT_CONFIG.graph.neo4j_database,
    ) as store:
        if reset_graph:
            print("[ingest] Clearing existing Neo4j legal graph nodes.")
            store.clear_graph()
        store.upsert_graphs(graphs)
    print("[ingest] Neo4j graph build complete.")


def _write_graph_audit(audit: GraphExtractionAudit, output: str | Path | None) -> None:
    payload = audit.to_dict()
    print("[ingest] Graph extraction audit:", payload)
    if not output:
        return
    audit_path = Path(output)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ingest] Saved graph audit: {audit_path}")


def _resolve_metadata_db(folder: Path, metadata_db: str | Path | None) -> Path | None:
    if metadata_db:
        path = Path(metadata_db)
        if path.exists():
            return path
        raise FileNotFoundError(f"Metadata DB not found: {path}")

    candidates = [folder / "metadata.sqlite", folder.parent / "metadata.sqlite"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


if __name__ == "__main__":
    main()
