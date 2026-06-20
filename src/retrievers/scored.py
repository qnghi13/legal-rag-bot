"""Shared scored retrieval helpers."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document


@dataclass(frozen=True)
class ScoredDocument:
    document: Document
    source_retriever: str
    original_rank: int
    raw_score: float | None = None

    def trace_entry(self) -> dict[str, Any]:
        return {
            "source_retriever": self.source_retriever,
            "original_rank": self.original_rank,
            "raw_score": self.raw_score,
        }


def clone_document(document: Document, metadata_updates: dict[str, Any] | None = None) -> Document:
    metadata = dict(document.metadata or {})
    if metadata_updates:
        metadata.update(metadata_updates)
    return Document(
        page_content=document.page_content,
        metadata=metadata,
        id=getattr(document, "id", None),
    )


def document_key(document: Document) -> str:
    metadata = document.metadata or {}
    chunk_id = metadata.get("chunk_id") or metadata.get("id")
    if chunk_id:
        return f"chunk:{chunk_id}"

    document_id = metadata.get("document_id")
    chunk_index = metadata.get("chunk_index")
    if document_id not in (None, "") and chunk_index not in (None, ""):
        return f"document-chunk:{document_id}:{chunk_index}"

    digest = hashlib.sha1(document.page_content.encode("utf-8")).hexdigest()
    return f"content:{digest}"
