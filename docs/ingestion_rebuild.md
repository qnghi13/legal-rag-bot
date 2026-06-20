# Current Ingestion And Rebuild Flow

This document captures the commands and behavior of the current ingest pipeline.

## 1. Crawl Or Refresh VBPL Markdown

```powershell
.\venv\Scripts\python.exe -m scripts.crawl_vbpl
```

Useful options:

```powershell
.\venv\Scripts\python.exe -m scripts.crawl_vbpl --max-docs 10
.\venv\Scripts\python.exe -m scripts.crawl_vbpl --keyword "lao động"
.\venv\Scripts\python.exe -m scripts.crawl_vbpl --keyword-scope all
.\venv\Scripts\python.exe -m scripts.crawl_vbpl --write-json
.\venv\Scripts\python.exe -m scripts.crawl_vbpl --write-json --include-content-json
```

The crawler writes:

- Markdown files to `data/raw/vbpl/markdown/`
- crawl metadata to `data/raw/vbpl/metadata.sqlite`
- JSON files only when `--write-json` is set

## 2. Build Text Retrieval Indexes

```powershell
.\venv\Scripts\python.exe -m scripts.ingest `
  --data-dir data/raw/vbpl/markdown `
  --metadata-db data/raw/vbpl/metadata.sqlite
```

This builds:

- `data/indexes/bm25_retriever.pkl`
- `data/indexes/chroma_db/`

The ingest command consumes existing Markdown. It does not crawl VBPL and does not regenerate Markdown from older JSON.

## 3. Build Text Indexes And Neo4j Graph

Rule-only graph build:

```powershell
.\venv\Scripts\python.exe -m scripts.ingest `
  --data-dir data/raw/vbpl/markdown `
  --metadata-db data/raw/vbpl/metadata.sqlite `
  --build-graph `
  --graph-extraction-mode rule `
  --graph-audit-output data/indexes/graph_extraction_audit.json
```

Hybrid graph build:

```powershell
.\venv\Scripts\python.exe -m scripts.ingest `
  --data-dir data/raw/vbpl/markdown `
  --metadata-db data/raw/vbpl/metadata.sqlite `
  --build-graph `
  --graph-extraction-mode hybrid `
  --graph-audit-output data/indexes/graph_extraction_audit.json
```

`hybrid` runs rule extraction first and calls the LLM only for documents that the extractor marks as ambiguous or hard. LLM facts must satisfy evidence and confidence checks before being merged.

Use `rule` when you want a deterministic rebuild without LLM extraction.

## 4. Reset Neo4j Before Rebuild

```powershell
.\venv\Scripts\python.exe -m scripts.ingest `
  --data-dir data/raw/vbpl/markdown `
  --metadata-db data/raw/vbpl/metadata.sqlite `
  --build-graph `
  --reset-graph `
  --graph-extraction-mode rule `
  --graph-audit-output data/indexes/graph_extraction_audit.json
```

Important behavior:

- `--reset-graph` clears Neo4j legal graph nodes and legacy constraints through `Neo4jLegalGraphStore.clear_graph()`.
- It does not make the ingest command graph-only.
- BM25 and Chroma work still runs before graph extraction/upsert.
- The current implementation does not have `--graph-only` or `--skip-vector`.

## 5. Full Clean Local Rebuild

Use this when parser/chunking changes require rebuilding local indexes from the current Markdown corpus.

```powershell
Remove-Item -LiteralPath .\data\indexes\chroma_db -Recurse -Force
Remove-Item -LiteralPath .\data\indexes\bm25_retriever.pkl -Force
```

Then run ingest again:

```powershell
.\venv\Scripts\python.exe -m scripts.ingest `
  --data-dir data/raw/vbpl/markdown `
  --metadata-db data/raw/vbpl/metadata.sqlite `
  --build-graph `
  --reset-graph `
  --graph-extraction-mode rule `
  --graph-audit-output data/indexes/graph_extraction_audit.json
```

Only use `--reset-graph` when the Neo4j database is dedicated to this project or you accept that the project graph labels will be cleared.

## 6. Output And Audit

`scripts.ingest` prints:

- chunk count and sample metadata,
- BM25 save path,
- embedding/chroma progress,
- graph extraction audit counters when `--build-graph` is used.

If `--graph-audit-output` is set, the audit is also written as JSON.

Important audit fields include:

- `documents_seen`
- `documents_llm_called`
- `documents_llm_skipped_private`
- `llm_facts_accepted`
- `llm_facts_rejected`
- `documents_without_clauses`
- `documents_with_duplicate_clause_keys`
- `ambiguous_references`
- `incoming_outgoing_edges_materialized`
- `relations_by_type`

## 7. Fast Verification

Focused tests:

```powershell
.\venv\Scripts\python.exe -m pytest tests/test_answer_verifier.py tests/test_legal_retrieval.py tests/test_legal_chunking.py tests/graph -q
```

Compile check:

```powershell
.\venv\Scripts\python.exe -m compileall src scripts tests
```

Full suite:

```powershell
.\venv\Scripts\python.exe -m pytest tests -q
```

