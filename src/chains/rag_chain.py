"""Main Legal RAG chain factory."""

from __future__ import annotations

import re
from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from config.prompts import NO_CONTEXT_ANSWER, QA_SYSTEM_PROMPT
from config.settings import DEFAULT_CONFIG
from src.chains.answer_verifier import verify_answer
from src.chains.query_rewriter import build_query_rewriter
from src.embeddings.embedding_model import get_embedding_model
from src.generators.groq_generator import build_groq_llm
from src.graph.graph_retriever import GraphRetriever
from src.graph.neo4j_store import Neo4jLegalGraphStore
from src.rerankers.cross_encoder_reranker import CrossEncoderReranker
from src.retrievers.bm25_retriever import BM25RetrieverAdapter, load_bm25_retriever
from src.retrievers.hybrid_retriever import HybridRetriever
from src.retrievers.semantic_retriever import SemanticRetriever
from src.vectorstores.chroma_store import load_chroma_store


def get_hr_bot(
    *,
    llm_model: str = DEFAULT_CONFIG.models.llm_model,
    llm_temperature: float = 0.2,
    embedding_model: str = DEFAULT_CONFIG.models.embedding_model,
    reranker_model: str = DEFAULT_CONFIG.models.reranker_model,
    reranker_max_length: int = DEFAULT_CONFIG.models.reranker_max_length,
    retrieval_k: int = DEFAULT_CONFIG.retrieval.retrieval_k,
    rerank_top_k: int = DEFAULT_CONFIG.retrieval.rerank_top_k,
    rrf_k: int = DEFAULT_CONFIG.retrieval.rrf_k,
    semantic_weight: float = DEFAULT_CONFIG.retrieval.semantic_weight,
    bm25_weight: float = DEFAULT_CONFIG.retrieval.bm25_weight,
    rerank_min_score: float | None = DEFAULT_CONFIG.retrieval.rerank_min_score,
    chroma_path: str | None = None,
    bm25_path: str | None = None,
    return_context_list: bool = False,
    graph_enabled: bool | None = None,
):
    """Build the LCEL RAG chain used by the UI and evaluation."""

    chroma_dir = Path(chroma_path) if chroma_path else DEFAULT_CONFIG.paths.chroma_dir
    bm25_file = Path(bm25_path) if bm25_path else DEFAULT_CONFIG.paths.bm25_path

    print(f"  [bot] LLM: {llm_model} (temp={llm_temperature})")
    llm = build_groq_llm(model=llm_model, temperature=llm_temperature)

    print(f"  [bot] Embedding: {embedding_model}")
    embedding = get_embedding_model(embedding_model)

    print(f"  [bot] Chroma DB: {chroma_dir}")
    vectorstore = load_chroma_store(chroma_dir, embedding)
    graph_retriever = _build_graph_retriever(
        vectorstore,
        enabled=DEFAULT_CONFIG.graph.enabled if graph_enabled is None else graph_enabled,
    )

    print(f"  [bot] Retriever: k={retrieval_k}")
    semantic = SemanticRetriever(vectorstore, search_k=retrieval_k)
    keyword = BM25RetrieverAdapter(load_bm25_retriever(bm25_file))
    retriever = HybridRetriever(
        semantic=semantic,
        keyword=keyword,
        rrf_k=rrf_k,
        semantic_weight=semantic_weight,
        bm25_weight=bm25_weight,
    )

    print(f"  [bot] Reranker: {reranker_model}")
    if rerank_min_score is not None:
        print(f"  [bot] Rerank threshold: {rerank_min_score}")
    reranker = CrossEncoderReranker(reranker_model, max_length=reranker_max_length)

    rephrase_step = RunnablePassthrough.assign(
        _rephrased=build_query_rewriter(llm)
    )
    retrieval_step = RunnableLambda(
        lambda state: state | _retrieve_context(
            state["_rephrased"],
            retriever=retriever,
            reranker=reranker,
            retrieval_k=retrieval_k,
            rerank_top_k=rerank_top_k,
            rerank_min_score=rerank_min_score,
            graph_retriever=graph_retriever,
            return_context_list=return_context_list,
        )
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", QA_SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    return rephrase_step | retrieval_step | _build_verified_answer_step(qa_prompt, llm)


def _retrieve_context(
    query: str,
    *,
    retriever,
    reranker,
    retrieval_k: int,
    rerank_top_k: int,
    rerank_min_score: float | None = None,
    graph_retriever=None,
    return_context_list: bool,
) -> dict:
    docs = retriever.retrieve(query, retrieval_k)
    top_docs = _rerank_documents(
        reranker,
        query,
        docs,
        rerank_top_k,
        min_score=rerank_min_score,
    )
    if graph_retriever:
        top_docs = _pack_graph_context(query, top_docs, graph_retriever, rerank_top_k)
    dq_docs, gq_docs = _split_dq_gq_documents(top_docs)
    if not dq_docs and not gq_docs:
        result = {"dq_context": "", "gq_context": ""}
        if return_context_list:
            result["dq_context_list"] = []
            result["gq_context_list"] = []
            result["dq_context_trace"] = []
            result["gq_context_trace"] = []
        return result

    dq_context_list = [_format_dq_document(doc, i) for i, doc in enumerate(dq_docs, start=1)]
    gq_context_list = [_format_gq_document(doc, i) for i, doc in enumerate(gq_docs, start=1)]
    result = {
        "dq_context": "\n\n".join(dq_context_list),
        "gq_context": "\n\n".join(gq_context_list),
    }
    if return_context_list:
        result["dq_context_list"] = dq_context_list
        result["gq_context_list"] = gq_context_list
        result["dq_context_trace"] = [_document_trace(doc, i) for i, doc in enumerate(dq_docs, start=1)]
        result["gq_context_trace"] = [_document_trace(doc, i) for i, doc in enumerate(gq_docs, start=1)]
    return result


def _rerank_documents(reranker, query: str, docs: list, top_k: int, *, min_score: float | None) -> list:
    if hasattr(reranker, "rerank_with_scores"):
        return reranker.rerank_with_scores(query, docs, top_k, min_score=min_score)

    reranked = reranker.rerank(query, docs, top_k)
    if min_score is None:
        return reranked
    return [
        doc
        for doc in reranked
        if doc.metadata.get("rerank_score") is not None
        and float(doc.metadata["rerank_score"]) >= min_score
    ]


def _document_trace(doc, index: int) -> dict:
    metadata = doc.metadata or {}
    return {
        "context_rank": index,
        "source": metadata.get("source", "Tai lieu noi bo"),
        "document_id": metadata.get("document_id", ""),
        "chunk_id": metadata.get("chunk_id") or metadata.get("id", ""),
        "chunk_index": metadata.get("chunk_index", ""),
        "retrieval_rank": metadata.get("retrieval_rank"),
        "fused_score": metadata.get("fused_score"),
        "retrieval_trace": metadata.get("retrieval_trace", []),
        "rerank_rank": metadata.get("rerank_rank"),
        "rerank_score": metadata.get("rerank_score"),
        "graph_source": metadata.get("graph_source", ""),
        "query_anchor_raw": metadata.get("query_anchor_raw", ""),
        "query_anchor_doc_number": metadata.get("query_anchor_doc_number", ""),
        "query_anchor_article_no": metadata.get("query_anchor_article_no", ""),
        "query_anchor_clause_no": metadata.get("query_anchor_clause_no", ""),
    }


def _split_dq_gq_documents(documents: list) -> tuple[list, list]:
    dq_docs = []
    gq_docs = []
    for doc in documents:
        graph_source = str(doc.metadata.get("graph_source") or "")
        if graph_source.startswith("query_anchor"):
            dq_docs.append(doc)
        elif graph_source:
            gq_docs.append(doc)
        else:
            dq_docs.append(doc)
    return dq_docs, gq_docs


def _format_dq_document(doc, index: int) -> str:
    metadata = doc.metadata or {}
    source = _source_display(metadata.get("source", "Tai lieu noi bo"))
    citation_lines = _legal_metadata_lines(metadata, source)
    graph_source = str(metadata.get("graph_source") or "")
    source_type = f"[Loai nguon]: {graph_source}\n" if graph_source else ""
    query_anchor = _query_anchor_line(metadata)
    citation_text = citation_lines if citation_lines else ""
    return (
        f"--- Dq TAI LIEU SO {index} ---\n"
        f"[Nguon]: {source}\n"
        f"{source_type}"
        f"{query_anchor}"
        f"{citation_text}"
        f"[Noi dung]: {doc.page_content}"
    )


def _legal_metadata_lines(metadata: dict, source: str) -> str:
    document = _document_label(metadata, source)
    article_heading, article_no = _article_label(metadata)
    clause_heading, clause_no = _clause_label(metadata)
    canonical_unit = _canonical_unit_citation(article_no, clause_no)

    lines = []
    if document:
        lines.append(f"[Van ban]: {document}")
    if article_heading:
        lines.append(f"[Dieu]: {article_heading}")
    if clause_heading:
        lines.append(f"[Khoan]: {clause_heading}")
    if canonical_unit:
        lines.append(f"[Trich dan]: {canonical_unit}")
    return "\n".join(lines) + ("\n" if lines else "")


def _document_label(metadata: dict, source: str) -> str:
    doc_number = str(
        metadata.get("doc_number")
        or metadata.get("so_ky_hieu")
        or metadata.get("document_number")
        or metadata.get("query_anchor_doc_number")
        or _doc_number_from_source(source)
        or ""
    ).strip()
    document_type = str(metadata.get("document_type") or "").strip()
    title = str(metadata.get("title") or metadata.get("document_title") or "").strip()

    parts = []
    if document_type:
        parts.append(document_type)
    if doc_number:
        parts.append(doc_number)
    label = " ".join(parts).strip()
    if title and title != label:
        label = f"{label} - {title}" if label else title
    return label


def _article_label(metadata: dict) -> tuple[str, str]:
    heading = str(metadata.get("Dieu") or metadata.get("article_title") or "").strip()
    article_no = str(
        metadata.get("article_no")
        or metadata.get("query_anchor_article_no")
        or _number_from_heading(heading)
        or ""
    ).strip()
    if heading:
        return heading, article_no
    if article_no:
        return f"\u0110i\u1ec1u {article_no}", article_no
    return "", ""


def _clause_label(metadata: dict) -> tuple[str, str]:
    heading = str(metadata.get("Khoan") or metadata.get("clause_title") or "").strip()
    clause_no = str(
        metadata.get("clause_no")
        or metadata.get("query_anchor_clause_no")
        or _number_from_heading(heading)
        or ""
    ).strip()
    if clause_no == "0":
        clause_no = ""
    if heading:
        return heading, clause_no
    if clause_no:
        return f"Kho\u1ea3n {clause_no}", clause_no
    return "", ""


def _canonical_unit_citation(article_no: str, clause_no: str) -> str:
    if article_no and clause_no:
        return f"Kho\u1ea3n {clause_no} \u0110i\u1ec1u {article_no}"
    if article_no:
        return f"\u0110i\u1ec1u {article_no}"
    return ""


def _number_from_heading(value: str) -> str:
    match = re.search(r"\b(?:\u0110i\u1ec1u|Dieu|Kho\u1ea3n|Khoan)\s+(\d+[a-z]?)\b", value or "", re.I)
    return match.group(1) if match else ""


def _doc_number_from_source(source: str) -> str:
    stem = Path(_source_display(source)).stem.lower()
    match = re.match(r"(?P<number>\d{1,4})-(?P<year>\d{4})-(?P<rest>[a-z0-9-]+)", stem)
    if not match:
        return ""

    stop_words = {
        "bo",
        "luat",
        "nghi",
        "dinh",
        "thong",
        "tu",
        "quyet",
        "huong",
        "dan",
        "quy",
        "chi",
        "tiet",
    }
    code_parts = []
    for token in match.group("rest").split("-"):
        if code_parts and token in stop_words:
            break
        code_parts.append(token)
        if len(code_parts) >= 4:
            break
    if not code_parts:
        return ""
    return f"{match.group('number')}/{match.group('year')}/{'-'.join(code_parts).upper()}"


def _format_gq_document(doc, index: int) -> str:
    metadata = doc.metadata or {}
    source = _source_display(metadata.get("source", "Neo4j graph"))
    citation_lines = _legal_metadata_lines(metadata, source)
    details = _graph_context_details(metadata)
    detail_text = f"\n[Quan he]: {details}" if details else ""
    content = _compact_context(doc.page_content)
    return (
        f"--- Gq QUAN HE SO {index} ---\n"
        f"[Nguon]: {source}\n"
        f"[Loai nguon]: {metadata.get('graph_source', 'graph')}\n"
        f"{citation_lines.rstrip()}"
        f"{detail_text}\n"
        f"[Noi dung]: {content}"
    )


def _graph_context_details(metadata: dict) -> str:
    parts = []
    if metadata.get("graph_direction"):
        parts.append(f"huong={metadata['graph_direction']}")
    if metadata.get("graph_relation_type"):
        parts.append(f"loai={metadata['graph_relation_type']}")
    if metadata.get("query_anchor_raw"):
        parts.append(f"anchor={metadata['query_anchor_raw']}")
    return ", ".join(parts)


def _query_anchor_line(metadata: dict) -> str:
    anchor = str(metadata.get("query_anchor_raw") or "").strip()
    return f"[Truy van neo]: {anchor}\n" if anchor else ""


def _source_display(source: str) -> str:
    value = str(source or "").strip()
    if not value:
        return "Tai lieu noi bo"
    normalized = value.replace("\\", "/")
    return normalized.rsplit("/", 1)[-1] or value


def _compact_context(content: str, max_chars: int = 1200) -> str:
    text = re.sub(r"\s+", " ", content or "").strip()
    if len(text) <= max_chars:
        return text
    cut_at = text.rfind(" ", 0, max_chars)
    if cut_at < max_chars // 2:
        cut_at = max_chars
    return text[:cut_at].rstrip() + "..."


def _build_graph_retriever(vectorstore, *, enabled: bool):
    if not enabled:
        return None
    try:
        store = Neo4jLegalGraphStore(
            DEFAULT_CONFIG.graph.neo4j_uri,
            DEFAULT_CONFIG.graph.neo4j_user,
            DEFAULT_CONFIG.graph.neo4j_password,
            database=DEFAULT_CONFIG.graph.neo4j_database,
        )
        print(
            "  [bot] Graph retrieval: enabled "
            f"(internal={DEFAULT_CONFIG.retrieval.graph_internal_ref_k}, "
            f"external={DEFAULT_CONFIG.retrieval.graph_external_scope_k})"
        )
        return GraphRetriever(
            store,
            vectorstore=vectorstore,
            internal_ref_k=DEFAULT_CONFIG.retrieval.graph_internal_ref_k,
            external_scope_k=DEFAULT_CONFIG.retrieval.graph_external_scope_k,
        )
    except Exception as exc:
        print(f"  [bot] Graph retrieval disabled: {exc}")
        return None


def _pack_graph_context(query: str, seed_docs: list, graph_retriever, rerank_top_k: int) -> list:
    main_limit = min(len(seed_docs), min(rerank_top_k, 6))
    main_docs = seed_docs[:main_limit]
    graph_docs = graph_retriever.retrieve(query, main_docs)
    packed = _dedupe_documents(main_docs + graph_docs)
    max_total = (
        rerank_top_k
        + graph_retriever.internal_ref_k
        + graph_retriever.external_scope_k
        + getattr(graph_retriever, "query_anchor_article_k", 0)
    )
    return packed[:max_total]


def _dedupe_documents(documents: list) -> list:
    seen: set[tuple[str, str]] = set()
    unique = []
    for doc in documents:
        marker = (
            str(doc.metadata.get("document_id", "")),
            doc.page_content,
        )
        if marker not in seen:
            seen.add(marker)
            unique.append(doc)
    return unique


def _build_answer_step(prompt, llm):
    def _invoke(state: dict) -> str:
        dq_context = (state.get("dq_context") or "").strip()
        if not dq_context:
            return NO_CONTEXT_ANSWER
        return (prompt | llm | StrOutputParser()).invoke(state)

    return RunnableLambda(_invoke)


def _build_verified_answer_step(prompt, llm):
    raw_answer_step = _build_answer_step(prompt, llm)

    def _invoke(state: dict) -> dict:
        raw_answer = raw_answer_step.invoke(state)
        result = verify_answer(
            raw_answer,
            state.get("dq_context", ""),
            state.get("gq_context", ""),
        )
        answer = raw_answer if result.passed else NO_CONTEXT_ANSWER
        return state | {
            "answer": answer,
            "verification_result": result.to_dict(),
        }

    return RunnableLambda(_invoke)
