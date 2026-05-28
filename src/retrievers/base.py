"""Retriever interfaces."""

from __future__ import annotations

from typing import Protocol

from langchain_core.documents import Document


class BaseRetriever(Protocol):
    def retrieve(self, query: str, top_k: int) -> list[Document]:
        """Return relevant documents for a query."""

