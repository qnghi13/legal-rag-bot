"""Retriever interfaces."""

from __future__ import annotations

from typing import Protocol

from langchain_core.documents import Document

from src.retrievers.scored import ScoredDocument


class BaseRetriever(Protocol):
    def retrieve(self, query: str, top_k: int) -> list[Document]:
        """Return relevant documents for a query."""

    def retrieve_with_scores(self, query: str, top_k: int) -> list[ScoredDocument]:
        """Return relevant documents with retriever-local scores and ranks."""
