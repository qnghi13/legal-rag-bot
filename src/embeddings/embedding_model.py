"""Cached embedding model factory."""

from __future__ import annotations

from langchain_huggingface import HuggingFaceEmbeddings

_MODEL_CACHE: dict[str, object] = {}


def get_embedding_model(model_name: str) -> HuggingFaceEmbeddings:
    key = f"embed:{model_name}"
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = HuggingFaceEmbeddings(model_name=model_name)
    return _MODEL_CACHE[key]

