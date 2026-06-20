# Legal RAG Bot - Vietnamese Labor Law Assistant

An AI-powered conversational assistant for Vietnamese labor law. The system is built around **Retrieval-Augmented Generation (RAG)**, but the current implementation is no longer a plain vector-only RAG: it combines Chroma semantic retrieval, BM25 keyword retrieval, cross-encoder reranking, optional Neo4j graph expansion, Groq generation, and rule-based answer verification.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-Enabled-green)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)
![Groq](https://img.shields.io/badge/Groq-LLM-orange)
![Neo4j](https://img.shields.io/badge/Neo4j-Graph%20Retrieval-blue)

---

## 1. Project Overview

Understanding labor law can be difficult for students, new employees, and people who need quick answers from long legal documents. Topics such as employment contracts, probation, overtime, salaries, insurance, vocational training, employee rights, and employer obligations are often spread across laws, decrees, circulars, amendments, and guidance documents.

The **Legal RAG Bot** makes Vietnamese labor law easier to search and understand. It crawls labor-related legal documents from VBPL, converts them into structured Markdown, chunks them by legal units such as `Điều` and `Khoản`, builds text retrieval indexes, and can build a retrieval-oriented Neo4j graph for legal relations such as amendment, repeal, replacement, guidance, detailed regulation, legal basis, citation, and reference.

When a user asks a question, the system rewrites the query if chat history is relevant, retrieves direct legal text, optionally expands through Neo4j legal relations, separates evidence into `Dq` textual context and `Gq` graph context, asks the LLM to answer only from the retrieved context, and verifies that generated citations are supported before returning the final answer.

## 2. Current Features

- **VBPL crawling and Markdown export:** Crawls labor-related documents from `vbpl.vn`, writes Markdown to `data/raw/vbpl/markdown/`, and stores crawl metadata in `data/raw/vbpl/metadata.sqlite`.
- **Legal-unit chunking:** Normalizes Vietnamese legal headings, strips preamble before the first legal heading, and chunks documents by article/clause-level legal units when possible.
- **Hybrid text retrieval:** Combines Chroma semantic retrieval using `keepitreal/vietnamese-sbert` with BM25 keyword retrieval, then fuses and deduplicates candidates.
- **Cross-encoder reranking:** Uses `BAAI/bge-reranker-v2-m3` to rerank retrieved candidates before generation.
- **Neo4j graph retrieval:** Optionally builds and queries a retrieval-only legal graph with `LegalUnit` nodes and relation types such as `AMENDS`, `SUPPLEMENTS`, `REPEALS`, `REPLACES`, `GUIDES`, `DETAILS`, `BASED_ON`, `CITES`, and `REFERS_TO`.
- **Query-anchor resolution:** Detects explicit legal references in the user query, such as document/article/clause anchors, and retrieves matching legal units when Neo4j is available.
- **Dq/Gq context split:** Sends direct textual evidence as `Dq` and graph relation evidence as `Gq`, so the prompt can distinguish answer evidence from relationship context.
- **Conversational query rewriting:** Rewrites follow-up questions into standalone legal questions using chat history.
- **Strict prompt guardrails:** The prompt instructs the model to answer from `Dq`, use `Gq` only as relation/status support, and return the fallback answer when evidence is missing.
- **Post-generation answer verification:** Rejects answers without citations or with unsupported citations, then falls back to `NO_CONTEXT_ANSWER`.
- **Streamlit UI with trace inspection:** Shows the final answer, retrieved `Dq/Gq` context, and retrieval trace JSON.
- **Ragas evaluation entrypoint:** Provides `scripts.evaluate` for evaluation with configurable retrieval, reranking, and judge model settings.

## 3. Architecture

The system architecture has two main phases: **Data Preparation (Offline)** and **RAG Query Pipeline (Online)**.

### Phase 1: Data Preparation (Offline)

This pipeline is executed when the legal corpus or indexes need to be refreshed.

```text
===================================================================
           PHASE 1: INGESTION AND GRAPH BUILD
===================================================================

          [ VBPL public legal documents ]
                         |
                         v
              [ scripts.crawl_vbpl ]
                         |
          +--------------+---------------+
          |                              |
          v                              v
[ data/raw/vbpl/markdown/*.md ]  [ metadata.sqlite ]
          |                              |
          +--------------+---------------+
                         |
                         v
          [ normalize headings + merge metadata ]
                         |
                         v
          [ legal-unit chunking by Điều/Khoản ]
                         |
          +--------------+-----------------------------+
          |                                            |
          v                                            v
 [ BM25 keyword index ]                    [ Chroma semantic index ]
 data/indexes/bm25_retriever.pkl          data/indexes/chroma_db/
          |
          | optional: --build-graph
          v
 [ rule/hybrid graph extraction + audit ]
          |
          v
 [ Neo4j retrieval graph: LegalUnit + legal relations ]
```

Important implementation notes:

- `scripts.ingest` consumes existing Markdown; it does not crawl VBPL and does not regenerate Markdown.
- `--build-graph` adds graph extraction and Neo4j upsert after BM25/Chroma indexing.
- `--reset-graph` clears Neo4j legal graph nodes before graph rebuild, but the ingest command still rebuilds BM25 and writes Chroma chunks. There is no graph-only ingest mode yet.
- `--graph-extraction-mode rule` avoids LLM extraction and is the recommended deterministic rebuild mode.

### Phase 2: RAG Query Pipeline (Online)

This pipeline runs when a user interacts with the Streamlit chatbot.

```text
===================================================================
           PHASE 2: QUERY PIPELINE
===================================================================

              [ User Input + Chat History ]
                         |
                         v
                 [ Query Rewriter ]
                         |
                 (standalone query)
                         |
          +--------------+---------------+
          |                              |
          v                              v
 [ Chroma semantic search ]       [ BM25 keyword search ]
          |                              |
          +--------------+---------------+
                         |
                         v
          [ Hybrid fusion + deduplication ]
                         |
                         v
          [ Cross-encoder reranker ]
                         |
                         v
          [ Optional Neo4j graph expansion ]
                         |
                         v
          [ Split context into Dq and Gq ]
                         |
                         v
          [ Groq LLM with strict legal prompt ]
                         |
                         v
          [ Citation verifier ]
                         |
                         v
          [ Final answer + context trace in Streamlit ]
```

Detailed design documents:

- [Current Architecture](docs/current_architecture.md)
- [Ingestion And Rebuild](docs/current_ingestion_rebuild.md)
- [Retrieval Context Contract](docs/current_retrieval_contract.md)
- [Neo4j Graph Design](docs/current_neo4j_graph_design.md)

## 4. Future Improvements

The current system already supports hybrid retrieval, graph expansion, context tracing, and citation verification. The next improvements should focus on robustness, deployment, and stronger evaluation:

- **Local LLM support:** Add a local structured-output model for graph extraction and/or answer generation, such as a Qwen instruct model served through Ollama or vLLM, to reduce API dependency and rate-limit risk.
- **Production web app:** Move beyond the current Streamlit prototype toward a full web application with authentication, saved conversations, admin corpus controls, and richer citation browsing.
- **Graph-only maintenance mode:** Add `--graph-only` or `--skip-vector` so Neo4j can be rebuilt without rewriting BM25/Chroma artifacts.
- **Graph relation ranking:** Score graph evidence by relation type, confidence, direction, recency, and whether the target document is resolved.
- **Better graph extraction evaluation:** Build hand-labeled relation fixtures and report relation precision/recall for amendment, repeal, replacement, guidance, and reference cases.
- **Ablation evaluation:** Compare BM25-only, dense-only, hybrid, hybrid + rerank, hybrid + graph, and hybrid + graph + verifier.
- **Domain-specific embeddings/rerankers:** Evaluate Vietnamese legal embedding and reranking models against the current `keepitreal/vietnamese-sbert` and `BAAI/bge-reranker-v2-m3`.
- **Improved citation UI:** Highlight the exact `Điều`, `Khoản`, document, and graph relation supporting each answer sentence.
- **Agentic/legal workflow routing:** Add routing for query types such as direct legal lookup, relation-heavy lookup, corpus gap detection, and future web search for out-of-corpus recent updates.

## 5. Demo

| Chat Interface | Context Verification |
| :---: | :---: |
| <img src="./image/QA.png"> | <img src="./image/citation.png" alt="Context UI"> |
| *Conversational UI with Groq LLM* | *Users can inspect retrieved legal context* |

## 6. Setup

### Prerequisites

Install Python dependencies in a virtual environment. Tesseract OCR is only needed when using OCR/PDF scan extraction paths; the current VBPL Markdown pipeline does not require OCR for normal crawl/index runs.

Optional OCR dependency:

1. **Tesseract OCR**
   - Windows: install from the UB Mannheim Tesseract build and ensure `tesseract.exe` is available.
   - Linux: `sudo apt-get install tesseract-ocr`
   - macOS: `brew install tesseract`

Optional graph dependency:

1. **Neo4j**
   - Default URI: `bolt://localhost:7687`
   - Configure credentials through `.env`.

### Installation

**Step 1: Clone the repository**

```bash
git clone <your-repo-url>
cd <your-project-folder>
```

**Step 2: Create a virtual environment and install dependencies**

```powershell
python -m venv venv
.\venv\Scripts\activate
python -m pip install -r requirements.txt
```

**Step 3: Environment variables**

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here

LEGAL_RAG_LLM_MODEL=llama-3.1-8b-instant
LEGAL_RAG_EMBEDDING_MODEL=keepitreal/vietnamese-sbert
LEGAL_RAG_RERANKER_MODEL=BAAI/bge-reranker-v2-m3

LEGAL_RAG_GRAPH_ENABLED=true
LEGAL_RAG_NEO4J_URI=bolt://localhost:7687
LEGAL_RAG_NEO4J_USER=neo4j
LEGAL_RAG_NEO4J_PASSWORD=your_neo4j_password
LEGAL_RAG_GRAPH_EXTRACTION_MODE=hybrid
```

If Neo4j is unavailable, the runtime chain disables graph retrieval for that chain instance and continues with hybrid text retrieval.

### Usage

**Step 4: Crawl VBPL documents**

```powershell
.\venv\Scripts\python.exe -m scripts.crawl_vbpl
```

By default, the crawler:

- searches central legal documents with the keyword `lao động`,
- filters document types to `Bộ luật`, `Luật`, `Nghị định`, and `Thông tư`,
- writes Markdown files to `data/raw/vbpl/markdown/`,
- stores crawl metadata in `data/raw/vbpl/metadata.sqlite`.

Useful options:

```powershell
# Crawl only a small sample
.\venv\Scripts\python.exe -m scripts.crawl_vbpl --max-docs 10

# Search more broadly across VBPL quick-search results
.\venv\Scripts\python.exe -m scripts.crawl_vbpl --keyword-scope all

# Also export metadata JSON files
.\venv\Scripts\python.exe -m scripts.crawl_vbpl --write-json
```

**Step 5: Build retrieval indexes**

Build Chroma and BM25 from the crawled Markdown files:

```powershell
.\venv\Scripts\python.exe -m scripts.ingest `
  --data-dir data/raw/vbpl/markdown `
  --metadata-db data/raw/vbpl/metadata.sqlite
```

Build Chroma, BM25, and Neo4j graph with rule-only extraction:

```powershell
.\venv\Scripts\python.exe -m scripts.ingest `
  --data-dir data/raw/vbpl/markdown `
  --metadata-db data/raw/vbpl/metadata.sqlite `
  --build-graph `
  --graph-extraction-mode rule `
  --graph-audit-output data/indexes/graph_extraction_audit.json
```

Reset Neo4j legal graph nodes before rebuilding graph facts:

```powershell
.\venv\Scripts\python.exe -m scripts.ingest `
  --data-dir data/raw/vbpl/markdown `
  --metadata-db data/raw/vbpl/metadata.sqlite `
  --build-graph `
  --reset-graph `
  --graph-extraction-mode rule `
  --graph-audit-output data/indexes/graph_extraction_audit.json
```

By default, indexes are written to:

- Chroma DB: `data/indexes/chroma_db/`
- BM25 index: `data/indexes/bm25_retriever.pkl`

You can tune chunking and write batches:

```powershell
.\venv\Scripts\python.exe -m scripts.ingest `
  --data-dir data/raw/vbpl/markdown `
  --metadata-db data/raw/vbpl/metadata.sqlite `
  --chunk-size 1000 `
  --chunk-overlap 200 `
  --embedding-batch-size 64 `
  --chroma-batch-size 512
```

**Step 6: Run the Chatbot UI**

```powershell
.\venv\Scripts\python.exe -m streamlit run .\app\streamlit_app.py
```

Access the application at `http://localhost:8501`.

**Step 7: Run evaluation or regression checks**

Evaluate the pipeline with Ragas:

```powershell
.\venv\Scripts\python.exe -m scripts.evaluate
.\venv\Scripts\python.exe -m scripts.evaluate --verbose
.\venv\Scripts\python.exe -m scripts.evaluate --output results.csv
```

Focused regression checks:

```powershell
.\venv\Scripts\python.exe -m pytest tests/test_answer_verifier.py tests/test_legal_retrieval.py tests/test_legal_chunking.py tests/graph -q
.\venv\Scripts\python.exe -m compileall src scripts tests
```
