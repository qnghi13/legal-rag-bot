"""Hybrid retrieval with Reciprocal Rank Fusion."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document

from src.retrievers.base import BaseRetriever
from src.retrievers.scored import ScoredDocument, clone_document, document_key


class HybridRetriever:
    def __init__(
        self,
        semantic: BaseRetriever,
        keyword: BaseRetriever,
        *,
        rrf_k: int = 60,
        semantic_weight: float = 1.0,
        bm25_weight: float = 1.0,
    ) -> None:
        self.semantic = semantic
        self.keyword = keyword
        self.rrf_k = rrf_k
        self.weights = {
            "semantic": semantic_weight,
            "bm25": bm25_weight,
        }

    def retrieve(self, query: str, top_k: int) -> list[Document]:
        return [item.document for item in self.retrieve_with_scores(query, top_k)]

    def retrieve_with_scores(self, query: str, top_k: int) -> list[ScoredDocument]:
        candidates = (
            _retrieve_scored(self.semantic, query, top_k, source_retriever="semantic")
            + _retrieve_scored(self.keyword, query, top_k, source_retriever="bm25")
        )
        fused = self._fuse(candidates)
        return fused[:top_k]

    def _fuse(self, candidates: list[ScoredDocument]) -> list[ScoredDocument]:
        by_key: dict[str, _FusionBucket] = {}
        for candidate in candidates:
            key = document_key(candidate.document)
            bucket = by_key.setdefault(key, _FusionBucket(document=candidate.document))
            bucket.add(candidate, rrf_k=self.rrf_k, weight=self.weights.get(candidate.source_retriever, 1.0))

        ranked = sorted(by_key.values(), key=lambda bucket: bucket.fused_score, reverse=True)
        fused: list[ScoredDocument] = []
        for rank, bucket in enumerate(ranked, start=1):
            doc = clone_document(
                bucket.document,
                {
                    "retrieval_trace": bucket.trace,
                    "fused_score": bucket.fused_score,
                    "retrieval_rank": rank,
                },
            )
            fused.append(
                ScoredDocument(
                    document=doc,
                    source_retriever="hybrid",
                    original_rank=rank,
                    raw_score=bucket.fused_score,
                )
            )
        return fused


@dataclass
class _FusionBucket:
    document: Document
    fused_score: float = 0.0
    trace: list[dict] | None = None

    def add(self, candidate: ScoredDocument, *, rrf_k: int, weight: float) -> None:
        if self.trace is None:
            self.trace = []
        contribution = weight / (rrf_k + candidate.original_rank)
        self.fused_score += contribution
        entry = candidate.trace_entry()
        entry["rrf_contribution"] = contribution
        entry["weight"] = weight
        self.trace.append(entry)


def _retrieve_scored(
    retriever: BaseRetriever,
    query: str,
    top_k: int,
    *,
    source_retriever: str,
) -> list[ScoredDocument]:
    if hasattr(retriever, "retrieve_with_scores"):
        return retriever.retrieve_with_scores(query, top_k)

    docs = retriever.retrieve(query, top_k)
    return [
        ScoredDocument(
            document=clone_document(
                doc,
                {
                    "source_retriever": source_retriever,
                    "original_rank": rank,
                    "raw_score": None,
                },
            ),
            source_retriever=source_retriever,
            original_rank=rank,
            raw_score=None,
        )
        for rank, doc in enumerate(docs, start=1)
    ]
