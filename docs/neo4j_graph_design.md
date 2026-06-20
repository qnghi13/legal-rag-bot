# Current Neo4j Graph Design

The current graph is retrieval-focused, not a complete legal knowledge graph.

## 1. Node Model

Neo4j stores a single active label:

```text
(:LegalUnit)
```

A `LegalUnit` can represent:

- `unit_type = "document"`
- `unit_type = "article"`
- `unit_type = "clause"`

The unique key is:

```text
unit_id
```

The store creates this constraint:

```cypher
CREATE CONSTRAINT legal_unit_id IF NOT EXISTS
FOR (n:LegalUnit) REQUIRE n.unit_id IS UNIQUE
```

Legacy labels such as `LegalDocument`, `Article`, `Clause`, `LegalDocumentRef`, `ReferenceMention`, and `AmendmentOperation` are cleared by `clear_graph()` but are no longer the active retrieval schema.

## 2. Relationship Allow-List

Only these relationship types are materialized for retrieval:

- `AMENDS`
- `SUPPLEMENTS`
- `REPEALS`
- `REPLACES`
- `GUIDES`
- `DETAILS`
- `BASED_ON`
- `CITES`
- `REFERS_TO`

Structural containment edges such as document-has-article or article-has-clause are not materialized as active retrieval relations. Article/clause membership is represented through node properties such as `document_id`, `article_no`, and `clause_no`.

## 3. Edge Metadata

Relation details live on the edge. Depending on relation source, useful properties include:

- `raw_text`
- `confidence`
- `evidence_text`
- `resolution_reason`
- `extraction_method`
- `source_span`
- `effective_date`
- `target_document_number`
- `target_article_no`
- `target_clause_no`
- `target_point`
- `target_title_hint`

This keeps the graph flat while preserving enough evidence for retrieval trace and debugging.

## 4. Upsert Flow

Entry point:

- `Neo4jLegalGraphStore.upsert_graphs(...)`

For each extracted document graph:

1. Merge document, article, and clause `LegalUnit` nodes.
2. Resolve references/amendments/document relations to target legal units.
3. Create placeholder document units for unresolved external document references.
4. Merge allowed legal relation edges.

If an amendment operation type is `supplement`, `repeal`, or `replace`, it maps to `SUPPLEMENTS`, `REPEALS`, or `REPLACES`; otherwise it maps to `AMENDS`.

## 5. Retrieval Queries

`get_related_clauses(...)` retrieves 1-hop related clauses from seed clauses or units.

It checks:

- outgoing relation from seed clause to target clause,
- outgoing relation from seed clause to target article, expanded to target clauses,
- incoming relation from another clause to seed clause,
- incoming relation from another clause to seed article, expanded to source clauses.

Returned graph context metadata includes:

- `graph_source`: `graph_direct_outgoing` or `graph_direct_incoming`
- `graph_direction`: `outgoing` or `incoming`
- `graph_relation_type`: the Neo4j relationship type

`get_external_document_scopes(...)` returns document scopes from outgoing `REFERS_TO` relations so `GraphRetriever` can run scoped Chroma searches for related external documents.

`resolve_query_anchors(...)` resolves explicit document/article/clause references from the user query directly into clause payloads.

`get_query_anchor_document_scopes(...)` returns scoped document filters for document-only query anchors.

## 6. Runtime Graph Expansion

`GraphRetriever` combines several sources:

- query-anchor clauses from Neo4j,
- direct related clauses from Neo4j,
- scoped Chroma results from externally referenced documents,
- scoped Chroma results for document-only query anchors.

It deduplicates by `clause_id` or `unit_id`, then `rag_chain.py` packs graph docs after the main reranked text docs.

## 7. Reset Behavior

`Neo4jLegalGraphStore.clear_graph()`:

- drops known legacy constraints,
- deletes nodes with labels used by old and current graph schemas,
- leaves unrelated Neo4j nodes untouched unless they use one of those labels.

`scripts.ingest --reset-graph` calls this method only when `--build-graph` is also part of the run.

Important: this reset is not a graph-only command. Text indexing still runs first.
