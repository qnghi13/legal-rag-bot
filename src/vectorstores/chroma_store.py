"""Chroma vector store loading."""

from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma


def load_chroma_store(path: str | Path, embedding) -> Chroma:
    return Chroma(persist_directory=str(path), embedding_function=embedding)

