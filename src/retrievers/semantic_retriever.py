"""Semantic retriever backed by a LangChain vector store retriever."""

from __future__ import annotations

from langchain_core.documents import Document


class SemanticRetriever:
    def __init__(self, vectorstore, search_k: int = 10) -> None:
        self.search_k = search_k
        self._retriever = vectorstore.as_retriever(search_kwargs={"k": search_k})

    def retrieve(self, query: str, top_k: int | None = None) -> list[Document]:
        docs = self._retriever.invoke(query)
        return docs[: top_k or self.search_k]

