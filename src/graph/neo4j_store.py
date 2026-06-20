"""Neo4j persistence and query helpers for retrieval-only Legal GraphRAG."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Iterable

from src.graph.extractor import article_id, clause_id, normalize_doc_number, normalize_ref_key
from src.graph.schema import (
    AmendmentOperation,
    DocumentRelation,
    ExtractedLegalGraph,
    ReferenceMention,
)


RELATION_TYPES = {
    "AMENDS",
    "SUPPLEMENTS",
    "REPEALS",
    "REPLACES",
    "GUIDES",
    "DETAILS",
    "BASED_ON",
    "CITES",
    "REFERS_TO",
}


class Neo4jLegalGraphStore:
    """Neo4j adapter for a flat retrieval graph of legal units and relations."""

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        *,
        database: str | None = None,
    ) -> None:
        try:
            from neo4j import GraphDatabase
        except ImportError as exc:  # pragma: no cover - depends on local env
            raise ImportError("Install the `neo4j` package to use graph features.") from exc

        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._database = database

    def close(self) -> None:
        self._driver.close()

    def __enter__(self) -> "Neo4jLegalGraphStore":
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def create_constraints(self) -> None:
        statements = [
            "CREATE CONSTRAINT legal_unit_id IF NOT EXISTS FOR (n:LegalUnit) REQUIRE n.unit_id IS UNIQUE",
        ]
        with self._session() as session:
            for statement in statements:
                session.run(statement)

    def clear_graph(self) -> None:
        """Delete previous graph nodes before rebuilding the retrieval-only graph."""

        labels = [
            "LegalUnit",
            "LegalDocument",
            "Article",
            "Clause",
            "LegalDocumentRef",
            "ReferenceMention",
            "AmendmentOperation",
        ]
        legacy_constraints = [
            "legal_document_id",
            "article_id",
            "clause_id",
            "legal_document_ref_key",
            "reference_mention_id",
            "amendment_operation_id",
        ]
        with self._session() as session:
            for constraint in legacy_constraints:
                session.run(f"DROP CONSTRAINT {constraint} IF EXISTS")
            session.run(
                """
                MATCH (node)
                WHERE any(label IN labels(node) WHERE label IN $labels)
                DETACH DELETE node
                """,
                {"labels": labels},
            )

    def upsert_graphs(self, graphs: Iterable[ExtractedLegalGraph]) -> None:
        graph_list = list(graphs)
        known_docs = {
            normalize_doc_number(graph.document.doc_number): _document_unit_props(graph)
            for graph in graph_list
            if graph.document.doc_number
        }
        self.create_constraints()
        with self._session() as session:
            for graph in graph_list:
                self._upsert_graph(session, graph, known_docs)

    def get_related_clauses(
        self,
        seed_refs: list[dict[str, str]],
        *,
        limit: int,
    ) -> list[dict[str, Any]]:
        if limit <= 0 or not seed_refs:
            return []
        rows: list[dict[str, Any]] = []
        with self._session() as session:
            for seed in seed_refs:
                source_match, source_params = _source_unit_match(seed)
                result = session.run(
                    source_match
                    + """
                    CALL (source) {
                        MATCH (source)-[rel]->(target:LegalUnit {unit_type: 'clause'})
                        WHERE type(rel) IN $relation_types
                        RETURN target, 'graph_direct_outgoing' AS graph_source,
                            'outgoing' AS graph_direction, type(rel) AS graph_relation_type
                        UNION
                        MATCH (source)-[rel]->(article:LegalUnit {unit_type: 'article'})
                        WHERE type(rel) IN $relation_types
                        MATCH (target:LegalUnit {
                            unit_type: 'clause',
                            document_id: article.document_id,
                            article_no: article.article_no
                        })
                        RETURN target, 'graph_direct_outgoing' AS graph_source,
                            'outgoing' AS graph_direction, type(rel) AS graph_relation_type
                        UNION
                        MATCH (target:LegalUnit {unit_type: 'clause'})-[rel]->(source)
                        WHERE type(rel) IN $relation_types
                        RETURN target, 'graph_direct_incoming' AS graph_source,
                            'incoming' AS graph_direction, type(rel) AS graph_relation_type
                        UNION
                        MATCH (target:LegalUnit {unit_type: 'clause'})
                            -[rel]->(article:LegalUnit {unit_type: 'article'})
                        WHERE type(rel) IN $relation_types
                            AND article.document_id = source.document_id
                            AND article.article_no = source.article_no
                        RETURN target, 'graph_direct_incoming' AS graph_source,
                            'incoming' AS graph_direction, type(rel) AS graph_relation_type
                    }
                    RETURN DISTINCT target, graph_source, graph_direction, graph_relation_type
                    LIMIT $limit
                    """,
                    source_params
                    | {
                        "limit": limit,
                        "relation_types": sorted(RELATION_TYPES),
                    },
                )
                rows.extend(
                    _row_to_clause_payload(
                        record["target"],
                        record["graph_source"],
                        record["graph_direction"],
                        record["graph_relation_type"],
                    )
                    for record in result
                )
                if len(rows) >= limit:
                    break
        return _dedupe_clause_payloads(rows)[:limit]

    def get_external_document_scopes(
        self,
        seed_refs: list[dict[str, str]],
        *,
        limit: int,
    ) -> list[dict[str, str]]:
        if limit <= 0 or not seed_refs:
            return []
        scopes: list[dict[str, str]] = []
        with self._session() as session:
            for seed in seed_refs:
                source_match, source_params = _source_unit_match(seed)
                result = session.run(
                    source_match
                    + """
                    MATCH (source)-[:REFERS_TO]->(doc:LegalUnit {unit_type: 'document'})
                    RETURN doc.doc_number AS doc_number, doc.title AS title_hint
                    LIMIT $limit
                    """,
                    source_params | {"limit": limit},
                )
                scopes.extend(dict(record) for record in result)
                if len(scopes) >= limit:
                    break
        return _dedupe_scopes(scopes)[:limit]

    def resolve_query_anchors(
        self,
        anchors: list[dict[str, str]],
        *,
        article_clause_limit: int,
    ) -> list[dict[str, Any]]:
        if article_clause_limit <= 0 or not anchors:
            return []
        rows: list[dict[str, Any]] = []
        with self._session() as session:
            for anchor in anchors:
                doc_number = normalize_doc_number(anchor.get("doc_number", ""))
                article_no = anchor.get("article_no", "")
                clause_no = anchor.get("clause_no", "")
                if not doc_number or not article_no:
                    continue
                if clause_no:
                    result = session.run(
                        """
                        MATCH (clause:LegalUnit {
                            unit_type: 'clause',
                            doc_number: $doc_number,
                            article_no: $article_no,
                            clause_no: $clause_no
                        })
                        RETURN clause
                        LIMIT 1
                        """,
                        {
                            "doc_number": doc_number,
                            "article_no": article_no,
                            "clause_no": clause_no,
                        },
                    )
                else:
                    result = session.run(
                        """
                        MATCH (clause:LegalUnit {
                            unit_type: 'clause',
                            doc_number: $doc_number,
                            article_no: $article_no
                        })
                        RETURN clause
                        ORDER BY clause.clause_no
                        LIMIT $limit
                        """,
                        {
                            "doc_number": doc_number,
                            "article_no": article_no,
                            "limit": article_clause_limit,
                        },
                    )
                rows.extend(
                    _row_to_clause_payload(
                        record["clause"],
                        "query_anchor",
                        extra_metadata=_query_anchor_metadata(anchor),
                    )
                    for record in result
                )
        return _dedupe_clause_payloads(rows)

    def get_query_anchor_document_scopes(
        self,
        anchors: list[dict[str, str]],
        *,
        limit: int,
    ) -> list[dict[str, str]]:
        if limit <= 0 or not anchors:
            return []
        scopes: list[dict[str, str]] = []
        with self._session() as session:
            for anchor in anchors:
                doc_number = normalize_doc_number(anchor.get("doc_number", ""))
                if not doc_number:
                    continue
                result = session.run(
                    """
                    MATCH (doc:LegalUnit {unit_type: 'document', doc_number: $doc_number})
                    RETURN doc.doc_number AS doc_number, doc.title AS title_hint
                    LIMIT 1
                    """,
                    {"doc_number": doc_number},
                )
                found = False
                for record in result:
                    found = True
                    scopes.append(dict(record) | _query_anchor_metadata(anchor))
                if not found:
                    scopes.append(
                        {
                            "doc_number": doc_number,
                            "title_hint": "",
                            **_query_anchor_metadata(anchor),
                        }
                    )
                if len(scopes) >= limit:
                    break
        return _dedupe_scopes(scopes)[:limit]

    def _session(self):
        kwargs = {"database": self._database} if self._database else {}
        return self._driver.session(**kwargs)

    def _upsert_graph(
        self,
        session,
        graph: ExtractedLegalGraph,
        known_docs: dict[str, dict[str, Any]],
    ) -> None:
        document_props = _document_unit_props(graph)
        _merge_unit(session, document_props)
        for article in graph.articles:
            _merge_unit(session, _article_unit_props(article, graph))
        for clause in graph.clauses:
            _merge_unit(session, _clause_unit_props(clause, graph))
        for reference in graph.references:
            self._upsert_reference(session, reference, graph, known_docs)
        for amendment in graph.amendments:
            self._upsert_amendment(session, amendment, graph, known_docs)
        for relation in graph.document_relations:
            self._upsert_document_relation(session, relation, known_docs)

    def _upsert_reference(
        self,
        session,
        reference: ReferenceMention,
        graph: ExtractedLegalGraph,
        known_docs: dict[str, dict[str, Any]],
    ) -> None:
        if "ambiguous" in reference.scope:
            return
        target = _resolve_reference_target(reference, graph, known_docs)
        _merge_relation(
            session,
            source_unit_id=reference.source_clause_id,
            relationship="REFERS_TO",
            target=target,
            relationship_props=_reference_relationship_props(reference),
        )

    def _upsert_amendment(
        self,
        session,
        amendment: AmendmentOperation,
        graph: ExtractedLegalGraph,
        known_docs: dict[str, dict[str, Any]],
    ) -> None:
        target = _resolve_amendment_target(amendment, graph, known_docs)
        _merge_relation(
            session,
            source_unit_id=amendment.source_clause_id,
            relationship=_operation_relation_type(amendment.operation_type),
            target=target,
            relationship_props=_amendment_relationship_props(amendment),
        )

    def _upsert_document_relation(
        self,
        session,
        relation: DocumentRelation,
        known_docs: dict[str, dict[str, Any]],
    ) -> None:
        target = _resolve_document_target(
            relation.target_document_number,
            relation.target_title_hint,
            known_docs,
        )
        relationship = relation.relation_type if relation.relation_type in RELATION_TYPES else "CITES"
        _merge_relation(
            session,
            source_unit_id=relation.source_document_id,
            relationship=relationship,
            target=target,
            relationship_props=_document_relation_props(relation),
        )


def _merge_unit(session, props: dict[str, Any]) -> None:
    session.run(
        """
        MERGE (unit:LegalUnit {unit_id: $unit_id})
        SET unit += $props
        """,
        {"unit_id": props["unit_id"], "props": props},
    )


def _merge_relation(
    session,
    *,
    source_unit_id: str,
    relationship: str,
    target: dict[str, Any],
    relationship_props: dict[str, Any],
) -> None:
    _validate_relationship(relationship)
    session.run(
        f"""
        MATCH (source:LegalUnit {{unit_id: $source_unit_id}})
        MERGE (target:LegalUnit {{unit_id: $target_unit_id}})
        SET target += $target_props
        MERGE (source)-[rel:{relationship}]->(target)
        SET rel += $relationship_props
        """,
        {
            "source_unit_id": source_unit_id,
            "target_unit_id": target["unit_id"],
            "target_props": target["props"],
            "relationship_props": relationship_props,
        },
    )


def _document_unit_props(graph: ExtractedLegalGraph) -> dict[str, Any]:
    data = asdict(graph.document)
    unit_id = graph.document.document_id
    data.update(
        {
            "unit_id": unit_id,
            "id": unit_id,
            "unit_type": "document",
            "document_id": graph.document.document_id,
            "text": graph.document.title,
            "resolved": True,
        }
    )
    return data


def _article_unit_props(article, graph: ExtractedLegalGraph) -> dict[str, Any]:
    data = asdict(article)
    data.update(
        {
            "unit_id": article.id,
            "id": article.id,
            "unit_type": "article",
            "doc_number": graph.document.doc_number,
            "document_type": graph.document.document_type,
            "document_title": graph.document.title,
            "markdown_path": graph.document.markdown_path,
            "text": article.title,
            "resolved": True,
        }
    )
    return data


def _clause_unit_props(clause, graph: ExtractedLegalGraph) -> dict[str, Any]:
    data = asdict(clause)
    data.update(
        {
            "unit_id": clause.id,
            "id": clause.id,
            "unit_type": "clause",
            "doc_number": graph.document.doc_number,
            "document_type": graph.document.document_type,
            "document_title": graph.document.title,
            "title": clause.title or graph.document.title,
            "source_url": graph.document.source_url,
            "markdown_path": clause.markdown_path or graph.document.markdown_path,
            "resolved": True,
        }
    )
    return data


def _minimal_unit_props(
    *,
    unit_id: str,
    unit_type: str,
    document_props: dict[str, Any],
    article_no: str = "",
    clause_no: str = "",
    title_hint: str = "",
) -> dict[str, Any]:
    title = document_props.get("title") or title_hint
    return {
        "unit_id": unit_id,
        "id": unit_id,
        "unit_type": unit_type,
        "document_id": document_props.get("document_id", ""),
        "doc_number": document_props.get("doc_number", ""),
        "title": title,
        "document_title": title,
        "document_type": document_props.get("document_type", ""),
        "article_no": article_no,
        "clause_no": clause_no,
        "text": title if unit_type == "document" else "",
        "markdown_path": document_props.get("markdown_path", ""),
        "source_url": document_props.get("source_url", ""),
        "resolved": bool(document_props.get("resolved", False)),
    }


def _resolve_reference_target(
    reference: ReferenceMention,
    graph: ExtractedLegalGraph,
    known_docs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if reference.target_document_number and normalize_doc_number(reference.target_document_number) not in known_docs:
        return _resolve_document_target(
            reference.target_document_number,
            reference.target_title_hint,
            known_docs,
        )
    document_props = _target_document_props(reference.target_document_number, graph, known_docs)
    if reference.target_article_no and reference.target_clause_no:
        unit_id = clause_id(
            document_props["document_id"],
            reference.target_article_no,
            reference.target_clause_no,
        )
        return {
            "unit_id": unit_id,
            "props": _minimal_unit_props(
                unit_id=unit_id,
                unit_type="clause",
                document_props=document_props,
                article_no=reference.target_article_no,
                clause_no=reference.target_clause_no,
                title_hint=reference.target_title_hint,
            ),
        }
    if reference.target_article_no:
        unit_id = article_id(document_props["document_id"], reference.target_article_no)
        return {
            "unit_id": unit_id,
            "props": _minimal_unit_props(
                unit_id=unit_id,
                unit_type="article",
                document_props=document_props,
                article_no=reference.target_article_no,
                title_hint=reference.target_title_hint,
            ),
        }
    return _resolve_document_target(
        reference.target_document_number,
        reference.target_title_hint,
        known_docs,
        fallback_graph=graph,
    )


def _resolve_amendment_target(
    amendment: AmendmentOperation,
    graph: ExtractedLegalGraph,
    known_docs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if amendment.target_document_number and normalize_doc_number(amendment.target_document_number) not in known_docs:
        return _resolve_document_target(amendment.target_document_number, "", known_docs)
    document_props = _target_document_props(amendment.target_document_number, graph, known_docs)
    if amendment.target_article_no and amendment.target_clause_no:
        unit_id = clause_id(
            document_props["document_id"],
            amendment.target_article_no,
            amendment.target_clause_no,
        )
        return {
            "unit_id": unit_id,
            "props": _minimal_unit_props(
                unit_id=unit_id,
                unit_type="clause",
                document_props=document_props,
                article_no=amendment.target_article_no,
                clause_no=amendment.target_clause_no,
            ),
        }
    if amendment.target_article_no:
        unit_id = article_id(document_props["document_id"], amendment.target_article_no)
        return {
            "unit_id": unit_id,
            "props": _minimal_unit_props(
                unit_id=unit_id,
                unit_type="article",
                document_props=document_props,
                article_no=amendment.target_article_no,
            ),
        }
    return _resolve_document_target(amendment.target_document_number, "", known_docs, fallback_graph=graph)


def _resolve_document_target(
    target_document_number: str,
    target_title_hint: str,
    known_docs: dict[str, dict[str, Any]],
    *,
    fallback_graph: ExtractedLegalGraph | None = None,
) -> dict[str, Any]:
    doc_number = normalize_doc_number(target_document_number)
    if doc_number in known_docs:
        props = dict(known_docs[doc_number])
        return {"unit_id": props["unit_id"], "props": props}
    if not doc_number and fallback_graph and fallback_graph.document.document_id:
        props = _document_unit_props(fallback_graph)
        return {"unit_id": props["unit_id"], "props": props}
    ref_key = doc_number or normalize_ref_key(target_title_hint)
    unit_id = f"docref:{ref_key}"
    props = {
        "unit_id": unit_id,
        "id": unit_id,
        "unit_type": "document",
        "document_id": unit_id,
        "doc_number": doc_number,
        "title": target_title_hint,
        "document_title": target_title_hint,
        "text": target_title_hint,
        "resolved": False,
    }
    return {"unit_id": unit_id, "props": props}


def _target_document_props(
    target_document_number: str,
    graph: ExtractedLegalGraph,
    known_docs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    doc_number = normalize_doc_number(target_document_number)
    if doc_number and doc_number in known_docs:
        return dict(known_docs[doc_number])
    return _document_unit_props(graph)


def _source_unit_match(seed: dict[str, str]) -> tuple[str, dict[str, str]]:
    unit_id = seed.get("clause_id") or seed.get("unit_id") or ""
    if unit_id:
        return "MATCH (source:LegalUnit {unit_id: $unit_id})", {"unit_id": unit_id}
    return (
        """
        MATCH (source:LegalUnit {
            unit_type: 'clause',
            document_id: $document_id,
            article_no: $article_no,
            clause_no: $clause_no
        })
        """,
        {
            "document_id": seed.get("document_id", ""),
            "article_no": seed.get("article_no", ""),
            "clause_no": seed.get("clause_no", ""),
        },
    )


def _row_to_clause_payload(
    node,
    graph_source: str,
    graph_direction: str = "",
    graph_relation_type: str = "",
    extra_metadata: dict[str, str] | None = None,
    document=None,
) -> dict[str, Any]:
    data = dict(node)
    document_data = dict(document) if document else {}
    unit_id = data.get("unit_id") or data.get("id", "")
    metadata = {
        "clause_id": unit_id if data.get("unit_type", "clause") == "clause" else "",
        "unit_id": unit_id,
        "document_id": data.get("document_id", ""),
        "doc_number": data.get("doc_number") or document_data.get("doc_number", ""),
        "title": data.get("document_title") or data.get("title") or document_data.get("title", ""),
        "document_type": data.get("document_type") or document_data.get("document_type", ""),
        "article_no": data.get("article_no", ""),
        "clause_no": data.get("clause_no", ""),
        "structural_path": data.get("structural_path", ""),
        "markdown_path": data.get("markdown_path") or document_data.get("markdown_path", ""),
        "graph_source": graph_source,
        "graph_direction": graph_direction,
        "graph_relation_type": graph_relation_type,
        "source": data.get("markdown_path") or document_data.get("markdown_path") or "Neo4j graph",
    }
    metadata.update(extra_metadata or {})
    return {
        "page_content": data.get("text", ""),
        "metadata": metadata,
    }


def _dedupe_clause_payloads(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str, str]] = set()
    unique: list[dict[str, Any]] = []
    for row in rows:
        metadata = row.get("metadata", {})
        clause_id_value = metadata.get("clause_id", "")
        unit_id_value = metadata.get("unit_id", "")
        key = (
            clause_id_value or unit_id_value or metadata.get("document_id", ""),
            "" if clause_id_value or unit_id_value else metadata.get("article_no", ""),
            "" if clause_id_value or unit_id_value else metadata.get("clause_no", ""),
            "" if clause_id_value or unit_id_value else row.get("page_content", ""),
        )
        if key not in seen:
            seen.add(key)
            unique.append(row)
    return unique


def _dedupe_scopes(scopes: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str]] = set()
    unique: list[dict[str, str]] = []
    for scope in scopes:
        key = (normalize_doc_number(scope.get("doc_number", "")), normalize_ref_key(scope.get("title_hint", "")))
        if key != ("", "") and key not in seen:
            seen.add(key)
            unique.append(scope)
    return unique


def _query_anchor_metadata(anchor: dict[str, str]) -> dict[str, str]:
    return {
        "query_anchor_raw": anchor.get("query_anchor_raw", ""),
        "query_anchor_doc_number": normalize_doc_number(anchor.get("doc_number", "")),
        "query_anchor_article_no": anchor.get("article_no", ""),
        "query_anchor_clause_no": anchor.get("clause_no", ""),
    }


def _reference_relationship_props(reference: ReferenceMention) -> dict[str, Any]:
    return asdict(reference)


def _amendment_relationship_props(amendment: AmendmentOperation) -> dict[str, Any]:
    return asdict(amendment)


def _document_relation_props(relation: DocumentRelation) -> dict[str, Any]:
    return asdict(relation)


def _validate_relationship(relationship: str) -> None:
    if relationship not in RELATION_TYPES:
        raise ValueError(f"Unsupported Neo4j relationship: {relationship}")


def _operation_relation_type(operation_type: str) -> str:
    normalized = (operation_type or "").lower()
    if normalized == "supplement":
        return "SUPPLEMENTS"
    if normalized == "repeal":
        return "REPEALS"
    if normalized == "replace":
        return "REPLACES"
    return "AMENDS"
