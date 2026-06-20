# Current Retrieval Context Contract

This document describes what the runtime chain sends to the prompt, trace, and verifier.

## 1. Why There Are Two Contexts

The chain formats retrieved evidence into two sections:

- `Dq`: direct textual evidence used to answer legal questions.
- `Gq`: graph relation evidence used to add legal status or relationship context.

The prompt is intentionally strict: legal conclusions must be grounded in `Dq`. `Gq` can support relationship statements such as amendment, repeal, replacement, guidance, detailed regulation, or related-document context.

## 2. Document Split Rule

Implemented in `src/chains/rag_chain.py`.

After rerank and optional graph expansion:

- Documents with no `graph_source` go to `Dq`.
- Documents whose `graph_source` starts with `query_anchor` go to `Dq`.
- Other graph documents go to `Gq`.

This means explicit query anchors such as "Điều 7 Nghị định 145/2020/NĐ-CP" are treated as direct answer evidence, not only graph context.

## 3. Dq Format

Each Dq item is formatted like:

```text
--- Dq TAI LIEU SO 1 ---
[Nguon]: ...
[Loai nguon]: ...
[Truy van neo]: ...
[Van ban]: ...
[Dieu]: ...
[Khoan]: ...
[Trich dan]: ...
[Noi dung]: ...
```

Not every line appears for every document. The formatter prefers explicit metadata and falls back to filename parsing when older chunks lack document-number metadata.

Important metadata:

- `source`
- `document_id`
- `doc_number`
- `document_type`
- `title`
- `article_no`
- `clause_no`
- `Dieu`
- `Khoan`
- `query_anchor_raw`

## 4. Gq Format

Each Gq item is formatted like:

```text
--- Gq QUAN HE SO 1 ---
[Nguon]: ...
[Loai nguon]: graph_direct_incoming
[Van ban]: ...
[Dieu]: ...
[Khoan]: ...
[Trich dan]: ...
[Quan he]: huong=incoming, loai=AMENDS, anchor=...
[Noi dung]: ...
```

Gq content is compacted to avoid flooding the prompt. It should explain why a related legal unit was retrieved, not replace direct legal evidence.

Important graph metadata:

- `graph_source`
- `graph_direction`
- `graph_relation_type`
- `query_anchor_raw`
- `query_anchor_doc_number`
- `query_anchor_article_no`
- `query_anchor_clause_no`

## 5. Trace Contract

When `return_context_list=True`, the chain returns:

- `dq_context`
- `gq_context`
- `dq_context_list`
- `gq_context_list`
- `dq_context_trace`
- `gq_context_trace`

Trace rows include:

- `context_rank`
- `source`
- `document_id`
- `chunk_id`
- `chunk_index`
- `retrieval_rank`
- `fused_score`
- `retrieval_trace`
- `rerank_rank`
- `rerank_score`
- `graph_source`
- `query_anchor_raw`
- `query_anchor_doc_number`
- `query_anchor_article_no`
- `query_anchor_clause_no`

The Streamlit app displays both context text and trace JSON.

## 6. Answer Verification

Implemented in `src/chains/answer_verifier.py`.

The verifier:

- accepts the configured fallback answer,
- extracts citations from non-fallback answers,
- fails answers with no citation,
- fails answers whose citations do not appear in `Dq` or `Gq`,
- accepts document citations when the normalized document number is present in context.

Failure reasons:

- `missing_citation`
- `unsupported_citation`

If verification fails, `rag_chain.py` replaces the generated answer with `NO_CONTEXT_ANSWER`.

## 7. Practical Implications

- If a correct answer is rejected, inspect Dq/Gq metadata before weakening verifier logic.
- `Dq` should expose document, article, clause, and quote metadata clearly.
- `Gq` should stay relation-focused and compact.
- Missing `chunk_id`, relation type, or query-anchor fields usually indicates a trace/metadata issue, not necessarily a retrieval failure.

