"""Cached embedding model factory."""

from __future__ import annotations

import os

from langchain_huggingface import HuggingFaceEmbeddings

_MODEL_CACHE: dict[str, object] = {}


def get_embedding_model(
    model_name: str,
    *,
    batch_size: int | None = None,
    show_progress: bool = False,
) -> HuggingFaceEmbeddings:
    batch_size = batch_size or 32
    device = os.getenv("LEGAL_RAG_EMBEDDING_DEVICE")
    key = f"embed:{model_name}:batch={batch_size}:device={device or 'auto'}"
    if key not in _MODEL_CACHE:
        model_kwargs = {"device": device} if device else {}
        _MODEL_CACHE[key] = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs={"batch_size": batch_size},
            query_encode_kwargs={"batch_size": batch_size},
            show_progress=show_progress,
        )
    return _MODEL_CACHE[key]
