"""Graph expansion retriever for pinned legal context."""

from __future__ import annotations

import re
from typing import Any

from langchain_core.documents import Document

from src.graph.query_entities import QueryAnchor, parse_query_anchors


ARTICLE_NO_RE = re.compile(r"Điều\s+(\d+[a-z]?)", re.I)
CLAUSE_NO_RE = re.compile(r"Khoản\s+(\d+[a-z]?)", re.I)


class GraphRetriever:
    """Expand top reranked seed chunks with direct graph neighbors."""

    def __init__(
        self,
        store,
        *,
        vectorstore=None,
        internal_ref_k: int = 4,
        external_scope_k: int = 2,
        query_anchor_article_k: int = 3,
    ) -> None:
        self.store = store
        self.vectorstore = vectorstore
        self.internal_ref_k = internal_ref_k
        self.external_scope_k = external_scope_k
        self.query_anchor_article_k = query_anchor_article_k

    def retrieve(self, query: str, seed_docs: list[Document]) -> list[Document]:
        seed_refs = [_seed_ref_from_doc(doc) for doc in seed_docs]
        seed_refs = [seed for seed in seed_refs if seed]
        anchors = parse_query_anchors(query)
        anchor_docs = self._retrieve_query_anchor_docs(anchors)
        anchor_seed_refs = [_seed_ref_from_doc(doc) for doc in anchor_docs]
        anchor_seed_refs = [seed for seed in anchor_seed_refs if seed]
        related_seed_refs = _dedupe_seed_refs(seed_refs + anchor_seed_refs)

        graph_docs = list(anchor_docs)
        if related_seed_refs:
            graph_docs.extend(
                _document_from_payload(payload)
                for payload in self.store.get_related_clauses(
                    related_seed_refs,
                    limit=self.internal_ref_k,
                )
            )
            graph_docs.extend(self._retrieve_external_scoped(query, related_seed_refs))
        graph_docs.extend(self._retrieve_query_anchor_scoped(query, anchors))
        return _dedupe_documents(graph_docs)

    def _retrieve_query_anchor_docs(self, anchors: list[QueryAnchor]) -> list[Document]:
        unit_anchors = [anchor for anchor in anchors if anchor.article_no]
        if not unit_anchors or not hasattr(self.store, "resolve_query_anchors"):
            return []
        payloads = self.store.resolve_query_anchors(
            [anchor.to_seed() for anchor in unit_anchors],
            article_clause_limit=self.query_anchor_article_k,
        )
        return [_document_from_payload(payload) for payload in payloads]

    def _retrieve_query_anchor_scoped(
        self,
        query: str,
        anchors: list[QueryAnchor],
    ) -> list[Document]:
        if not self.vectorstore:
            return []
        doc_only_anchors = [anchor for anchor in anchors if not anchor.article_no]
        if not doc_only_anchors:
            return []
        scopes = [
            {
                "doc_number": anchor.doc_number,
                "title_hint": "",
                **anchor.trace_metadata(),
            }
            for anchor in doc_only_anchors
        ]
        if hasattr(self.store, "get_query_anchor_document_scopes"):
            scopes = self.store.get_query_anchor_document_scopes(
                [anchor.to_seed() for anchor in doc_only_anchors],
                limit=self.external_scope_k,
            )
        return self._retrieve_scoped_documents(
            query=query,
            scopes=scopes,
            graph_source="query_anchor_scoped",
            limit=self.external_scope_k,
        )

    def _retrieve_external_scoped(
        self,
        query: str,
        seed_refs: list[dict[str, str]],
    ) -> list[Document]:
        if not self.vectorstore or self.external_scope_k <= 0:
            return []
        scopes = self.store.get_external_document_scopes(
            seed_refs,
            limit=self.external_scope_k,
        )
        return self._retrieve_scoped_documents(
            query=query,
            scopes=scopes,
            graph_source="graph_external_scoped",
            limit=self.external_scope_k,
        )

    def _retrieve_scoped_documents(
        self,
        *,
        query: str,
        scopes: list[dict[str, str]],
        graph_source: str,
        limit: int,
    ) -> list[Document]:
        scoped_docs: list[Document] = []
        for scope in scopes:
            doc_number = scope.get("doc_number") or ""
            title_hint = scope.get("title_hint") or ""
            remaining = limit - len(scoped_docs)
            docs = self._similarity_search_scope(
                query=query,
                doc_number=doc_number,
                title_hint=title_hint,
                remaining=remaining,
            )
            for doc in docs:
                doc.metadata.update(
                    {
                        "graph_source": graph_source,
                        "query_anchor_raw": scope.get("query_anchor_raw", ""),
                        "query_anchor_doc_number": scope.get("query_anchor_doc_number", ""),
                        "query_anchor_article_no": scope.get("query_anchor_article_no", ""),
                        "query_anchor_clause_no": scope.get("query_anchor_clause_no", ""),
                    }
                )
            scoped_docs.extend(docs)
            if len(scoped_docs) >= limit:
                break
        return scoped_docs[:limit]

    def _similarity_search_scope(
        self,
        *,
        query: str,
        doc_number: str,
        title_hint: str,
        remaining: int,
    ) -> list[Document]:
        if remaining <= 0:
            return []
        filters = []
        if doc_number:
            filters.append({"doc_number": doc_number})
        if title_hint:
            filters.append({"title": {"$contains": title_hint}})

        docs: list[Document] = []
        for filter_payload in filters:
            try:
                docs = self.vectorstore.similarity_search(query, k=remaining, filter=filter_payload)
            except Exception:
                docs = []
            if docs:
                break
        for doc in docs:
            doc.metadata = dict(doc.metadata)
        return docs


def _seed_ref_from_doc(doc: Document) -> dict[str, str] | None:
    metadata = doc.metadata or {}
    clause_id = str(metadata.get("clause_id") or metadata.get("unit_id") or metadata.get("id") or "")
    document_id = str(metadata.get("document_id") or "")
    if clause_id:
        return {"clause_id": clause_id, "document_id": document_id}
    if not document_id:
        return None
    article_no = str(metadata.get("article_no") or _number_from_heading(metadata.get("Dieu"), ARTICLE_NO_RE))
    clause_no = str(metadata.get("clause_no") or _number_from_heading(metadata.get("Khoan"), CLAUSE_NO_RE))
    if not article_no or not clause_no:
        return None
    return {"document_id": document_id, "article_no": article_no, "clause_no": clause_no}


def _number_from_heading(value: Any, pattern: re.Pattern[str]) -> str:
    if not value:
        return ""
    match = pattern.search(str(value))
    return match.group(1) if match else ""


def _document_from_payload(payload: dict[str, Any]) -> Document:
    return Document(
        page_content=payload.get("page_content", ""),
        metadata=payload.get("metadata", {}),
    )


def _dedupe_documents(documents: list[Document]) -> list[Document]:
    seen: set[tuple[str, str]] = set()
    unique: list[Document] = []
    for doc in documents:
        clause_id = str(doc.metadata.get("clause_id") or doc.metadata.get("unit_id") or "")
        marker = (
            clause_id or str(doc.metadata.get("document_id", "")),
            "" if clause_id else doc.page_content,
        )
        if doc.page_content and marker not in seen:
            seen.add(marker)
            unique.append(doc)
    return unique


def _dedupe_seed_refs(seed_refs: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str, str, str]] = set()
    unique: list[dict[str, str]] = []
    for seed in seed_refs:
        clause_id = seed.get("clause_id", "")
        key = ("clause_id", clause_id, "", "") if clause_id else (
            "unit",
            seed.get("document_id", ""),
            seed.get("article_no", ""),
            seed.get("clause_no", ""),
        )
        if key not in seen:
            seen.add(key)
            unique.append(seed)
    return unique
