"""Semantic retriever backed by a LangChain vector store retriever."""

from __future__ import annotations

from langchain_core.documents import Document

from src.retrievers.scored import ScoredDocument, clone_document


class SemanticRetriever:
    def __init__(self, vectorstore, search_k: int = 10) -> None:
        self.search_k = search_k
        self._vectorstore = vectorstore
        self._retriever = vectorstore.as_retriever(search_kwargs={"k": search_k})

    def retrieve(self, query: str, top_k: int | None = None) -> list[Document]:
        return [item.document for item in self.retrieve_with_scores(query, top_k or self.search_k)]

    def retrieve_with_scores(self, query: str, top_k: int | None = None) -> list[ScoredDocument]:
        limit = top_k or self.search_k
        docs_and_scores = self._similarity_search_with_scores(query, limit)
        scored_docs: list[ScoredDocument] = []
        for rank, (doc, score) in enumerate(docs_and_scores, start=1):
            scored_docs.append(
                ScoredDocument(
                    document=clone_document(
                        doc,
                        {
                            "source_retriever": "semantic",
                            "original_rank": rank,
                            "raw_score": score,
                        },
                    ),
                    source_retriever="semantic",
                    original_rank=rank,
                    raw_score=score,
                )
            )
        return scored_docs

    def _similarity_search_with_scores(self, query: str, limit: int) -> list[tuple[Document, float | None]]:
        if hasattr(self._vectorstore, "similarity_search_with_score"):
            return [
                (doc, float(score))
                for doc, score in self._vectorstore.similarity_search_with_score(query, k=limit)
            ]

        docs = self._retriever.invoke(query)[:limit]
        return [(doc, None) for doc in docs]
