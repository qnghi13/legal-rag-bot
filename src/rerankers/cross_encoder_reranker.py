"""Cross-encoder reranker."""

from __future__ import annotations

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

_RERANKER_CACHE: dict[str, CrossEncoder] = {}


def get_cross_encoder(model_name: str, max_length: int = 512) -> CrossEncoder:
    key = f"{model_name}:{max_length}"
    if key not in _RERANKER_CACHE:
        _RERANKER_CACHE[key] = CrossEncoder(model_name, max_length=max_length)
    return _RERANKER_CACHE[key]


class CrossEncoderReranker:
    def __init__(self, model_name: str, max_length: int = 512) -> None:
        self.model = get_cross_encoder(model_name, max_length=max_length)

    def rerank(self, query: str, documents: list[Document], top_k: int) -> list[Document]:
        if not documents:
            return []
        pairs = [[query, doc.page_content] for doc in documents]
        scores = self.model.predict(pairs)
        scored = list(zip(documents, scores))
        scored.sort(key=lambda item: item[1], reverse=True)
        return [doc for doc, _ in scored[:top_k]]

