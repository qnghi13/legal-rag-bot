"""BM25 retriever loading and adapter."""

from __future__ import annotations

import pickle
from pathlib import Path

from langchain_core.documents import Document

from src.retrievers.scored import ScoredDocument, clone_document

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
        return [item.document for item in self.retrieve_with_scores(query, top_k)]

    def retrieve_with_scores(self, query: str, top_k: int) -> list[ScoredDocument]:
        scored = self._rank_with_scores(query, top_k)
        if scored is None:
            return self._fallback_rank(query, top_k)
        return scored

    def _rank_with_scores(self, query: str, top_k: int) -> list[ScoredDocument] | None:
        vectorizer = getattr(self._retriever, "vectorizer", None)
        docs = getattr(self._retriever, "docs", None)
        preprocess_func = getattr(self._retriever, "preprocess_func", None)
        if vectorizer is None or docs is None or preprocess_func is None:
            return None
        if not hasattr(vectorizer, "get_scores"):
            return None

        processed_query = preprocess_func(query)
        scores = vectorizer.get_scores(processed_query)
        ranked = sorted(
            enumerate(scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )[:top_k]
        scored_docs: list[ScoredDocument] = []
        for rank, (doc_index, score) in enumerate(ranked, start=1):
            raw_score = float(score)
            doc = docs[doc_index]
            scored_docs.append(
                ScoredDocument(
                    document=clone_document(
                        doc,
                        {
                            "source_retriever": "bm25",
                            "original_rank": rank,
                            "raw_score": raw_score,
                        },
                    ),
                    source_retriever="bm25",
                    original_rank=rank,
                    raw_score=raw_score,
                )
            )
        return scored_docs

    def _fallback_rank(self, query: str, top_k: int) -> list[ScoredDocument]:
        if hasattr(self._retriever, "k"):
            self._retriever.k = top_k
        docs = self._retriever.invoke(query)[:top_k]
        return [
            ScoredDocument(
                document=clone_document(
                    doc,
                    {
                        "source_retriever": "bm25",
                        "original_rank": rank,
                        "raw_score": None,
                    },
                ),
                source_retriever="bm25",
                original_rank=rank,
                raw_score=None,
            )
            for rank, doc in enumerate(docs, start=1)
        ]
