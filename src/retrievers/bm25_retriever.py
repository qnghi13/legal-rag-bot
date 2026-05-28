"""BM25 retriever loading and adapter."""

from __future__ import annotations

import pickle
from pathlib import Path

from langchain_core.documents import Document

_BM25_CACHE: dict[str, object] = {}


def load_bm25_retriever(path: str | Path):
    bm25_path = Path(path)
    key = str(bm25_path.resolve())
    if key not in _BM25_CACHE:
        if not bm25_path.exists():
            raise FileNotFoundError(
                f"BM25 index not found at: {bm25_path}. Run `python scripts/ingest.py` first."
            )
        with bm25_path.open("rb") as f:
            _BM25_CACHE[key] = pickle.load(f)
    return _BM25_CACHE[key]


class BM25RetrieverAdapter:
    def __init__(self, retriever) -> None:
        self._retriever = retriever

    def retrieve(self, query: str, top_k: int) -> list[Document]:
        if hasattr(self._retriever, "k"):
            self._retriever.k = top_k
        return self._retriever.invoke(query)[:top_k]

