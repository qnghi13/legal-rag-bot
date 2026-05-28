"""Reranker interfaces."""

from __future__ import annotations

from typing import Protocol

from langchain_core.documents import Document


class BaseReranker(Protocol):
    def rerank(self, query: str, documents: list[Document], top_k: int) -> list[Document]:
        """Return the top documents after reranking."""

