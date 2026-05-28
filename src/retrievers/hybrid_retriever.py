"""Hybrid retrieval that merges dense and sparse results."""

from __future__ import annotations

from langchain_core.documents import Document

from src.retrievers.base import BaseRetriever


class HybridRetriever:
    def __init__(self, semantic: BaseRetriever, keyword: BaseRetriever) -> None:
        self.semantic = semantic
        self.keyword = keyword

    def retrieve(self, query: str, top_k: int) -> list[Document]:
        docs = self.semantic.retrieve(query, top_k) + self.keyword.retrieve(query, top_k)
        return _deduplicate_documents(docs)


def _deduplicate_documents(documents: list[Document]) -> list[Document]:
    seen: set[int] = set()
    unique_docs: list[Document] = []
    for doc in documents:
        marker = hash(doc.page_content)
        if marker not in seen:
            seen.add(marker)
            unique_docs.append(doc)
    return unique_docs

