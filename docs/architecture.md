# Current Architecture

This document describes the implementation in the current checkout and expands the high-level architecture shown in `README.md`.

## 1. Offline Data Flow

```text
scripts.crawl_vbpl
  -> VBPL public API / public document pages
  -> Markdown files
  -> SQLite metadata registry

scripts.ingest
  -> load Markdown files
  -> merge SQLite metadata
  -> normalize legal headings
  -> split by article/clause graph units
  -> build BM25 pickle
  -> build Chroma vector store
  -> optionally extract graph facts
  -> optionally upsert graph facts into Neo4j
```

### 1.1. Crawl

Entry points:

- `scripts/crawl_vbpl.py`
- `src/ingestion/vbpl_crawler.py`

The crawler searches VBPL, filters document types/statuses, fetches document HTML, cleans the content node, converts the content tree to Markdown, and upserts metadata into `data/raw/vbpl/metadata.sqlite`.

Default outputs:

- `data/raw/vbpl/markdown/*.md`
- `data/raw/vbpl/metadata.sqlite`
- `data/raw/vbpl/json/*.json` only when `--write-json` is used

### 1.2. Chunking

Entry point:

- `src/chunking/legal_chunker.py`

The chunker:

- loads Markdown files recursively,
- merges document metadata from SQLite,
- normalizes headings with `normalize_legal_headings(...)`,
- strips HTML comments and preamble before the first legal heading,
- extracts rule-based graph units from Markdown,
- creates chunks primarily from clause-level units,
- stores linking metadata such as `clause_id`, `unit_id`, `document_id`, `article_no`, `clause_no`, and `markdown_path`.

If a file has no legal headings or no graph clauses, it falls back to Markdown-header or recursive character splitting.

### 1.3. Indexing

Entry point:

- `scripts/ingest.py`

The command always builds text retrieval artifacts first:

- BM25: `data/indexes/bm25_retriever.pkl`
- Chroma: `data/indexes/chroma_db/`

When `--build-graph` is passed, it also reads the VBPL SQLite registry, extracts graph facts, writes an optional audit JSON, and upserts the graph into Neo4j.

## 2. Runtime Query Flow

Entry point:

- `src/chains/rag_chain.py`

```text
user input + chat history
  -> query rewriter
  -> semantic retriever over Chroma
  -> BM25 retriever
  -> HybridRetriever fusion/dedupe
  -> CrossEncoderReranker
  -> optional GraphRetriever expansion
  -> split documents into Dq and Gq
  -> format prompt context
  -> Groq LLM
  -> verify_answer(...)
  -> answer or fallback
```

### 2.1. Query Rewrite

`build_query_rewriter(...)` rewrites follow-up questions into standalone legal questions using chat history. It does not answer the question.

### 2.2. Hybrid Retrieval

Dense retrieval uses Chroma with the configured embedding model. Sparse retrieval uses the pickled BM25 retriever. `HybridRetriever` fuses the result lists and keeps trace metadata such as retrieval rank and fused score.

### 2.3. Reranking

`CrossEncoderReranker` reranks the fused candidates with the configured cross-encoder model. `LEGAL_RAG_RERANK_MIN_SCORE` can be used to drop weak candidates if set.

### 2.4. Graph Expansion

If graph retrieval is enabled and Neo4j is reachable, `GraphRetriever` expands top reranked chunks by:

- resolving explicit query anchors such as document/article/clause references,
- retrieving direct incoming/outgoing graph neighbors,
- retrieving scoped Chroma chunks from externally referenced documents.

Graph failures at startup disable graph retrieval for that chain instance instead of breaking vector retrieval.

### 2.5. Generation And Verification

The prompt receives two context sections:

- `Dq`: direct textual evidence from hybrid retrieval and query anchors.
- `Gq`: graph relation evidence from Neo4j or graph-scoped vector search.

The LLM answer is passed to `verify_answer(...)`. If it lacks citations or cites unsupported legal units, the final answer becomes `NO_CONTEXT_ANSWER`.

## 3. Configuration

Configuration lives in `config/settings.py` and is environment-driven.

Important model settings:

- `LEGAL_RAG_LLM_MODEL`
- `LEGAL_RAG_JUDGE_LLM_MODEL`
- `LEGAL_RAG_EMBEDDING_MODEL`
- `LEGAL_RAG_RERANKER_MODEL`
- `LEGAL_RAG_RERANKER_MAX_LENGTH`

Important retrieval settings:

- `LEGAL_RAG_RETRIEVAL_K`
- `LEGAL_RAG_BM25_K`
- `LEGAL_RAG_RERANK_TOP_K`
- `LEGAL_RAG_RRF_K`
- `LEGAL_RAG_SEMANTIC_WEIGHT`
- `LEGAL_RAG_BM25_WEIGHT`
- `LEGAL_RAG_RERANK_MIN_SCORE`
- `LEGAL_RAG_GRAPH_INTERNAL_REF_K`
- `LEGAL_RAG_GRAPH_EXTERNAL_SCOPE_K`

Important graph settings:

- `LEGAL_RAG_GRAPH_ENABLED`
- `LEGAL_RAG_NEO4J_URI`
- `LEGAL_RAG_NEO4J_USER`
- `LEGAL_RAG_NEO4J_PASSWORD`
- `LEGAL_RAG_NEO4J_DATABASE`
- `LEGAL_RAG_GRAPH_EXTRACTION_MODE`
- `LEGAL_RAG_GRAPH_LLM_PROVIDER`
- `LEGAL_RAG_GRAPH_LLM_MODEL`
- `LEGAL_RAG_GRAPH_LLM_MIN_CONFIDENCE`
- `LEGAL_RAG_GRAPH_LLM_PUBLIC_ONLY`

