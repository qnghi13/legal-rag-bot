"""Rule-based extraction for Vietnamese legal document graphs."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from config.settings import DEFAULT_CONFIG
from src.graph.llm_extractor import (
    GroqLegalGraphLLMExtractor,
    LLMAmendmentFact,
    LLMDocumentRelationFact,
    LLMExtractionResult,
    LLMGraphPayload,
    LLMReferenceFact,
)
from src.graph.schema import (
    AmendmentOperation,
    Article,
    Clause,
    DocumentRelation,
    ExtractedLegalGraph,
    LegalDocument,
    ReferenceMention,
)
from src.ingestion.text_extractor import normalize_legal_headings


ARTICLE_HEADING_RE = re.compile(r"^###\s+Điều\s+([\w]+)\.?\s*(.*)$", re.I)
CLAUSE_HEADING_RE = re.compile(r"^####\s+Khoản\s+([\w]+)\.?\s*(.*)$", re.I)
DOC_NUMBER_RE = re.compile(
    r"\b(?:Bộ luật|Luật|Nghị định|Thông tư|Quyết định)?\s*"
    r"(?:số\s*)?([0-9]{1,4}/[0-9]{4}/[A-ZĐ][A-Z0-9Đ/-]*)",
    re.I,
)
NAMED_LAW_RE = re.compile(
    r"\b((?:Bộ luật|Luật)\s+[A-ZÀ-Ỹ][^.;,\n()]{2,120}?)(?=(?:\s+ngày|\s+năm|[.;,\n)]|$))",
    re.I,
)
BASED_ON_RE = re.compile(r"^\s*Căn cứ\s+(.+)", re.I)
ARTICLE_REF_RE = re.compile(
    r"(?:(?:điểm)\s+([a-zđ])\s+)?"
    r"(?:(?:khoản)\s+(\d+[a-z]?)\s+)?"
    r"Điều\s+(\d+[a-z]?)",
    re.I,
)
CURRENT_ARTICLE_REF_RE = re.compile(
    r"(?:(?:điểm)\s+([a-zđ])\s+)?"
    r"(?:(?:khoản)\s+(\d+[a-z]?)\s+)?"
    r"Điều\s+này",
    re.I,
)
POINT_CLAUSE_ARTICLE_RE = re.compile(
    r"điểm\s+([a-zđ])\s+khoản\s+(\d+[a-z]?)\s+Điều\s+(\d+[a-z]?)",
    re.I,
)
AMENDMENT_RE = re.compile(
    r"(?P<op>Sửa đổi,\s*bổ sung|Sửa đổi|Bổ sung|Bãi bỏ|Thay thế|Hết hiệu lực)"
    r"\s+(?P<target>[^:\n.]{0,180}?Điều\s+\d+[a-z]?|[^:\n.]{0,180}?khoản\s+\d+[a-z]?\s+Điều\s+\d+[a-z]?|[^:\n.]{0,180}?điểm\s+[a-zđ]\s+khoản\s+\d+[a-z]?\s+Điều\s+\d+[a-z]?|Điều\s+\d+[a-z]?)",
    re.I,
)
SUPPLEMENT_CLAUSE_INTO_ARTICLE_RE = re.compile(
    r"(?P<op>Bổ sung)\s+khoản\s+(?P<clause>\d+[a-z]?)\s+vào\s+Điều\s+(?P<article>\d+[a-z]?)",
    re.I,
)
GUIDE_HINT_RE = re.compile(r"\b(hướng dẫn|quy định chi tiết|thi hành)\b", re.I)
AMENDMENT_HINT_RE = re.compile(r"\b(sửa đổi|bổ sung|bãi bỏ|thay thế|hết hiệu lực)\b", re.I)

RELATION_TYPES = {
    "BASED_ON",
    "AMENDS",
    "SUPPLEMENTS",
    "REPEALS",
    "REPLACES",
    "GUIDES",
    "DETAILS",
    "CITES",
}

GENERIC_NAMED_LAW_REFS = {
    "phap luat",
    "phap luat viet nam",
    "luat viet nam",
    "luat quoc te",
    "luat nuoc so tai",
    "luat cua nuoc so tai",
}


@dataclass
class GraphExtractionAudit:
    documents_seen: int = 0
    documents_llm_called: int = 0
    documents_llm_skipped_private: int = 0
    llm_parse_failures: int = 0
    llm_facts_accepted: int = 0
    llm_facts_rejected: int = 0
    empty_graph_documents: int = 0
    documents_without_clauses: int = 0
    documents_with_duplicate_clause_keys: int = 0
    low_confidence_relations: int = 0
    primary_units: int = 0
    quoted_replacement_blocks: int = 0
    relations_rejected_missing_evidence: int = 0
    ambiguous_references: int = 0
    same_document_refs_blocked_by_max_article: int = 0
    incoming_outgoing_edges_materialized: int = 0
    relations_by_type: dict[str, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "documents_seen": self.documents_seen,
            "documents_llm_called": self.documents_llm_called,
            "documents_llm_skipped_private": self.documents_llm_skipped_private,
            "llm_parse_failures": self.llm_parse_failures,
            "llm_facts_accepted": self.llm_facts_accepted,
            "llm_facts_rejected": self.llm_facts_rejected,
            "empty_graph_documents": self.empty_graph_documents,
            "documents_without_clauses": self.documents_without_clauses,
            "documents_with_duplicate_clause_keys": self.documents_with_duplicate_clause_keys,
            "low_confidence_relations": self.low_confidence_relations,
            "primary_units": self.primary_units,
            "quoted_replacement_blocks": self.quoted_replacement_blocks,
            "relations_rejected_missing_evidence": self.relations_rejected_missing_evidence,
            "ambiguous_references": self.ambiguous_references,
            "same_document_refs_blocked_by_max_article": self.same_document_refs_blocked_by_max_article,
            "incoming_outgoing_edges_materialized": self.incoming_outgoing_edges_materialized,
            "relations_by_type": self.relations_by_type or {},
        }


def extract_graph_from_markdown(
    markdown: str,
    metadata: dict[str, Any],
    *,
    extraction_mode: str | None = None,
    llm_extractor: Any | None = None,
    allow_private_llm_extraction: bool = False,
    audit: GraphExtractionAudit | None = None,
    min_confidence: float | None = None,
) -> ExtractedLegalGraph:
    """Extract legal graph facts from a single normalized Markdown document."""

    audit = audit or GraphExtractionAudit()
    audit.documents_seen += 1
    extraction_mode = _normalize_extraction_mode(extraction_mode)
    min_confidence = (
        DEFAULT_CONFIG.graph.llm_min_confidence
        if min_confidence is None
        else min_confidence
    )
    document = _document_from_metadata(metadata)
    normalized = normalize_legal_headings(markdown)
    articles, clauses = _extract_structure(normalized, document)
    if not articles and not clauses:
        articles, clauses = _extract_legacy_structure(normalized, document)
    graph = ExtractedLegalGraph(document=document, articles=articles, clauses=clauses)

    for clause in clauses:
        graph.references.extend(
            _extract_clause_references(
                clause,
                document,
                clause.text,
                max_article_no=_max_numeric_article_no(articles),
            )
        )
        graph.amendments.extend(_extract_amendments(clause, document, clause.text))

    source_text = _strip_markdown_noise(normalized)
    graph.document_relations.extend(_extract_document_relations(document, source_text))

    # Amendment clauses often imply a document-level relationship even when the
    # title split has hidden the document number from the local clause text.
    if graph.amendments:
        for relation in _extract_document_relations(document, document.title):
            if relation.relation_type in {"AMENDS", "SUPPLEMENTS", "REPEALS", "REPLACES"}:
                graph.document_relations.append(relation)

    if extraction_mode in {"hybrid", "llm_shadow"} and _should_call_llm(
        graph,
        metadata,
        normalized,
    ):
        if _is_public_vbpl_metadata(metadata) or allow_private_llm_extraction or not DEFAULT_CONFIG.graph.llm_public_only:
            llm_result = _run_llm_extraction(
                llm_extractor=llm_extractor,
                metadata=metadata,
                normalized=normalized,
                graph=graph,
            )
            audit.documents_llm_called += int(llm_result.called)
            audit.llm_parse_failures += int(llm_result.parse_failed)
            if extraction_mode == "hybrid" and not llm_result.parse_failed:
                accepted, rejected = _merge_llm_payload(
                    graph,
                    llm_result.payload,
                    normalized,
                    min_confidence=min_confidence,
                )
                audit.llm_facts_accepted += accepted
                audit.llm_facts_rejected += rejected
                audit.relations_rejected_missing_evidence += rejected
        else:
            audit.documents_llm_skipped_private += 1

    _finalize_graph(graph, normalized, min_confidence=min_confidence, audit=audit)
    return graph


def extract_graph_from_vbpl_record(
    record: dict[str, Any],
    *,
    extraction_mode: str | None = None,
    llm_extractor: Any | None = None,
    allow_private_llm_extraction: bool = False,
    audit: GraphExtractionAudit | None = None,
) -> ExtractedLegalGraph | None:
    """Build graph facts from one row of the VBPL SQLite metadata registry."""

    markdown_path = Path(record.get("markdown_path") or "")
    if not markdown_path.exists():
        return None

    metadata = _metadata_from_record(record)
    markdown = markdown_path.read_text(encoding="utf-8")
    graph = extract_graph_from_markdown(
        markdown,
        metadata,
        extraction_mode=extraction_mode,
        llm_extractor=llm_extractor,
        allow_private_llm_extraction=allow_private_llm_extraction,
        audit=audit,
    )

    source_text = _source_text_from_json(record.get("json_path"))
    if source_text:
        graph.document_relations.extend(
            _dedupe_relations(_extract_document_relations(graph.document, source_text))
        )
        graph.document_relations[:] = _dedupe_relations(graph.document_relations)
    return graph


def load_vbpl_records(metadata_db: str | Path) -> list[dict[str, Any]]:
    """Load crawled VBPL records from the crawler metadata registry."""

    with sqlite3.connect(metadata_db) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT
                document_id, source_url, title, document_type, status, doc_number,
                issuing_agency, issue_date, effective_date, expiry_date, crawled_at,
                updated_at, json_path, markdown_path, content_sha256, metadata_json
            FROM vbpl_documents
            WHERE crawl_status = 'crawled'
            ORDER BY issue_date, document_id
            """
        ).fetchall()
    return [dict(row) for row in rows]


def extract_graphs_from_vbpl_metadata(
    metadata_db: str | Path,
    *,
    extraction_mode: str | None = None,
    llm_extractor: Any | None = None,
    allow_private_llm_extraction: bool = False,
    audit: GraphExtractionAudit | None = None,
) -> list[ExtractedLegalGraph]:
    graphs: list[ExtractedLegalGraph] = []
    for record in load_vbpl_records(metadata_db):
        graph = extract_graph_from_vbpl_record(
            record,
            extraction_mode=extraction_mode,
            llm_extractor=llm_extractor,
            allow_private_llm_extraction=allow_private_llm_extraction,
            audit=audit,
        )
        if graph:
            graphs.append(graph)
    return graphs


def normalize_doc_number(value: str) -> str:
    return re.sub(r"\s+", "", value or "").upper()


def normalize_ref_key(value: str) -> str:
    text = unicodedata.normalize("NFKC", value or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def clause_id(document_id: str, article_no: str, clause_no: str, occurrence: int = 1) -> str:
    suffix = "" if occurrence == 1 else f":o{occurrence}"
    return f"{document_id}:a{article_no}:c{clause_no}{suffix}"


def article_id(document_id: str, article_no: str) -> str:
    return f"{document_id}:a{article_no}"


def _document_from_metadata(metadata: dict[str, Any]) -> LegalDocument:
    document_id = str(
        metadata.get("document_id")
        or metadata.get("source_url")
        or metadata.get("markdown_path")
        or _digest(metadata.get("title", "document"))
    )
    return LegalDocument(
        document_id=document_id,
        doc_number=normalize_doc_number(str(metadata.get("doc_number") or metadata.get("so_ky_hieu") or "")),
        title=str(metadata.get("title") or ""),
        document_type=str(metadata.get("document_type") or ""),
        status=str(metadata.get("status") or ""),
        issuing_agency=str(metadata.get("issuing_agency") or metadata.get("co_quan_ban_hanh") or ""),
        issue_date=str(metadata.get("issue_date") or metadata.get("ngay_ban_hanh") or ""),
        effective_date=str(metadata.get("effective_date") or metadata.get("ngay_hieu_luc") or ""),
        source_url=str(metadata.get("source_url") or ""),
        markdown_path=str(metadata.get("markdown_path") or ""),
    )


def _metadata_from_record(record: dict[str, Any]) -> dict[str, Any]:
    metadata = {key: value for key, value in record.items() if value not in (None, "")}
    raw_nested = record.get("metadata_json")
    if raw_nested:
        try:
            nested = json.loads(raw_nested).get("metadata") or {}
            metadata.update({key: value for key, value in nested.items() if value not in (None, "")})
        except json.JSONDecodeError:
            pass
    return metadata


def _extract_structure(markdown: str, document: LegalDocument) -> tuple[list[Article], list[Clause]]:
    articles: list[Article] = []
    clauses: list[Clause] = []
    current_article_no = ""
    current_article_title = ""
    current_article_start_line = 0
    current_clause_no = ""
    current_clause_heading = ""
    current_clause_start_line = 0
    current_clause_lines: list[str] = []
    current_clause_contains_quote = False
    current_article_target_doc_number = ""
    current_article_target_title_hint = ""
    seen_clauses: dict[tuple[str, str], int] = {}
    in_quoted_block = False

    def flush_clause(end_line: int) -> None:
        nonlocal current_clause_no, current_clause_heading, current_clause_start_line
        nonlocal current_clause_lines, current_clause_contains_quote
        if not current_article_no or not current_clause_no:
            current_clause_lines = []
            current_clause_contains_quote = False
            return
        text = "\n".join(line.rstrip() for line in current_clause_lines).strip()
        if not text:
            current_clause_lines = []
            current_clause_contains_quote = False
            return
        key = (current_article_no, current_clause_no)
        occurrence = seen_clauses.get(key, 0) + 1
        seen_clauses[key] = occurrence
        cid = clause_id(document.document_id, current_article_no, current_clause_no, occurrence)
        clauses.append(
            Clause(
                id=cid,
                document_id=document.document_id,
                article_no=current_article_no,
                clause_no=current_clause_no,
                title=current_clause_heading,
                text=text,
                chunk_id=_digest(cid),
                markdown_path=document.markdown_path,
                occurrence=occurrence,
                unit_type="clause" if current_clause_no != "0" else "article_intro",
                structural_path=f"article:{current_article_no}/clause:{current_clause_no}",
                source_start_line=current_clause_start_line,
                source_end_line=end_line,
                quote_state="inside_quote" if current_clause_contains_quote else "outside_quote",
                contains_quoted_replacement=current_clause_contains_quote,
                target_document_number_hint=current_article_target_doc_number,
                target_title_hint=current_article_target_title_hint,
            )
        )
        current_clause_lines = []
        current_clause_contains_quote = False

    for line_no, raw_line in enumerate(markdown.splitlines(), start=1):
        line = raw_line.strip()
        quote_transition = _quote_transition(line)
        if quote_transition in {"open", "single"} and current_clause_no:
            current_clause_contains_quote = True
        if quote_transition == "open":
            in_quoted_block = True
        article_match = ARTICLE_HEADING_RE.match(line)
        if article_match and not in_quoted_block:
            flush_clause(line_no - 1)
            current_article_no = article_match.group(1)
            current_article_title = article_match.group(2).strip()
            current_article_start_line = line_no
            current_article_target_doc_number, current_article_target_title_hint = _target_hint_from_text(
                current_article_title,
                document.doc_number,
            )
            current_clause_no = ""
            current_clause_heading = ""
            articles.append(
                Article(
                    id=article_id(document.document_id, current_article_no),
                    document_id=document.document_id,
                    article_no=current_article_no,
                    title=current_article_title,
                    structural_path=f"article:{current_article_no}",
                    source_start_line=line_no,
                    source_end_line=line_no,
                )
            )
            if AMENDMENT_HINT_RE.search(current_article_title):
                current_clause_no = "0"
                current_clause_heading = "Article heading"
                current_clause_start_line = line_no
                current_clause_lines = [current_article_title]
                current_clause_contains_quote = False
            continue

        clause_match = CLAUSE_HEADING_RE.match(line)
        if clause_match and current_article_no and not in_quoted_block:
            flush_clause(line_no - 1)
            current_clause_no = clause_match.group(1)
            current_clause_heading = clause_match.group(2).strip()
            current_clause_start_line = line_no
            current_clause_contains_quote = False
            continue

        if current_article_no and not current_clause_no and line:
            # Some articles have article-level text before the first numbered
            # clause. Store it as pseudo clause 0 so graph retrieval can cite it.
            current_clause_no = "0"
            current_clause_heading = "Article intro"
            current_clause_start_line = line_no
            current_clause_contains_quote = False

        if current_clause_no:
            current_clause_lines.append(raw_line)
        if quote_transition == "close":
            in_quoted_block = False

    flush_clause(len(markdown.splitlines()))
    return _dedupe_articles(articles), clauses


def _extract_legacy_structure(markdown: str, document: LegalDocument) -> tuple[list[Article], list[Clause]]:
    lines = [line.rstrip() for line in markdown.splitlines()]
    content_lines = [line for line in lines if line.strip()]
    if not content_lines:
        return [], []

    article_no = "legacy-1"
    article = Article(
        id=article_id(document.document_id, article_no),
        document_id=document.document_id,
        article_no=article_no,
        title=document.title or "Legacy legal units",
        unit_type="legacy_document",
        structural_path="legacy:1",
        source_start_line=1,
        source_end_line=len(lines),
    )
    clauses: list[Clause] = []
    current_no = "1"
    current_start = 1
    current_lines: list[str] = []
    seen = 0

    def flush(end_line: int) -> None:
        nonlocal seen, current_lines, current_no, current_start
        text = "\n".join(line for line in current_lines if line.strip()).strip()
        if not text:
            current_lines = []
            return
        seen += 1
        clause_no = current_no or str(seen)
        cid = clause_id(document.document_id, article_no, clause_no, seen)
        clauses.append(
            Clause(
                id=cid,
                document_id=document.document_id,
                article_no=article_no,
                clause_no=clause_no,
                title=f"Legacy unit {clause_no}",
                text=text,
                chunk_id=_digest(cid),
                markdown_path=document.markdown_path,
                occurrence=seen,
                unit_type="legacy_clause",
                structural_path=f"legacy:1/unit:{clause_no}",
                source_start_line=current_start,
                source_end_line=end_line,
                confidence=0.7,
            )
        )
        current_lines = []

    for line_no, raw_line in enumerate(lines, start=1):
        stripped = raw_line.strip()
        legacy_heading = re.match(r"^(?P<num>(?:[IVXLCDM]+|\d+|[a-zA-Z]))[\.)]\s+.+", stripped)
        if legacy_heading and current_lines:
            flush(line_no - 1)
            current_no = legacy_heading.group("num")
            current_start = line_no
        elif not current_lines and stripped:
            current_start = line_no
        current_lines.append(raw_line)
    flush(len(lines))
    return [article], clauses


def _extract_clause_references(
    source_clause: Clause,
    document: LegalDocument,
    text: str,
    *,
    max_article_no: int = 0,
) -> list[ReferenceMention]:
    references: list[ReferenceMention] = []
    for match in ARTICLE_REF_RE.finditer(text):
        if _mention_is_inside_amendment_operation(text, match.start(), match.end()):
            continue
        point, clause_no, article_no = match.groups()
        raw_text = match.group(0)
        ref_type = "clause" if clause_no else "article"
        if point:
            ref_type = "point"
        target_clause_no = clause_no or ""
        resolution = _resolve_article_reference(
            document=document,
            text=text,
            start=match.start(),
            end=match.end(),
            article_no=article_no,
            max_article_no=max_article_no,
            source_clause=source_clause,
        )
        references.append(
            ReferenceMention(
                id=_mention_id(source_clause.id, raw_text, match.start()),
                source_clause_id=source_clause.id,
                raw_text=raw_text,
                ref_type=ref_type,
                scope=resolution["scope"],
                target_document_number=resolution["target_document_number"],
                target_article_no=article_no,
                target_clause_no=target_clause_no,
                target_point=(point or "").lower(),
                confidence=resolution["confidence"],
                evidence_text=resolution["evidence_text"],
                resolution_reason=resolution["resolution_reason"],
            )
        )

    for match in CURRENT_ARTICLE_REF_RE.finditer(text):
        if _mention_is_inside_amendment_operation(text, match.start(), match.end()):
            continue
        point, clause_no = match.groups()
        raw_text = match.group(0)
        ref_type = "clause" if clause_no else "article"
        if point:
            ref_type = "point"
        references.append(
            ReferenceMention(
                id=_mention_id(source_clause.id, raw_text, match.start()),
                source_clause_id=source_clause.id,
                raw_text=raw_text,
                ref_type=ref_type,
                scope="same_document",
                target_document_number=document.doc_number,
                target_article_no=source_clause.article_no,
                target_clause_no=clause_no or "",
                target_point=(point or "").lower(),
                confidence=0.95,
                evidence_text=_evidence_window(text, match.start(), match.end()),
                resolution_reason="The reference uses 'Điều này', so it resolves to the current article.",
            )
        )

    for law in _named_law_refs(text):
        references.append(
            ReferenceMention(
                id=_mention_id(source_clause.id, law, 0),
                source_clause_id=source_clause.id,
                raw_text=law,
                ref_type="external_document",
                scope="external_document",
                target_title_hint=law,
                confidence=0.75,
                evidence_text=law,
                resolution_reason="Named legal document mention detected in the source clause.",
            )
        )
    return references


def _extract_amendments(
    source_clause: Clause,
    document: LegalDocument,
    text: str,
) -> list[AmendmentOperation]:
    amendments: list[AmendmentOperation] = []
    target_doc_number = (
        source_clause.target_document_number_hint
        or _first_external_doc_number(document.title, document.doc_number)
        or _first_external_doc_number(text, document.doc_number)
    )
    for line in text.splitlines():
        candidate = re.sub(r"^[\s\"“”'.,;:-]+", "", line.strip())
        supplement_into = SUPPLEMENT_CLAUSE_INTO_ARTICLE_RE.match(candidate)
        if supplement_into:
            raw_text = supplement_into.group(0)
            amendments.append(
                AmendmentOperation(
                    id=_mention_id(source_clause.id, raw_text, supplement_into.start()),
                    source_clause_id=source_clause.id,
                    operation_type="supplement",
                    raw_text=raw_text,
                    target_document_number=target_doc_number,
                    target_article_no=supplement_into.group("article"),
                    target_clause_no=supplement_into.group("clause"),
                    new_text=_quoted_replacement_text(text),
                    evidence_text=raw_text,
                    resolution_reason="Target document is inherited from the article/document amendment context.",
                )
            )
            continue
        match = AMENDMENT_RE.match(candidate)
        if not match:
            continue
        raw_text = match.group(0)
        op_text = match.group("op").lower()
        target_text = match.group("target")
        operation_type = _operation_type(op_text)
        point, clause_no, article_no = _target_parts(target_text)
        amendments.append(
            AmendmentOperation(
                id=_mention_id(source_clause.id, raw_text, match.start()),
                source_clause_id=source_clause.id,
                operation_type=operation_type,
                raw_text=raw_text,
                target_document_number=target_doc_number,
                target_article_no=article_no,
                target_clause_no=clause_no,
                target_point=point,
                new_text=_quoted_replacement_text(text),
                evidence_text=raw_text,
                resolution_reason=(
                    "Target document is inherited from the article/document amendment context."
                    if source_clause.target_document_number_hint
                    else "Target document is resolved from the nearest external document number."
                ),
            )
        )
    return amendments


def _extract_document_relations(
    document: LegalDocument,
    source_text: str,
) -> list[DocumentRelation]:
    relations: list[DocumentRelation] = []
    for raw_line in _iter_relevant_lines(source_text):
        based_on = BASED_ON_RE.match(raw_line)
        if based_on:
            for ref in _document_refs_from_line(raw_line):
                relations.append(_doc_relation(document, "BASED_ON", raw_line, ref))
            continue

        if AMENDMENT_HINT_RE.search(raw_line):
            relation_type = _document_relation_type(raw_line)
            for ref in _document_refs_from_line(raw_line):
                if ref.get("doc_number") != document.doc_number:
                    relations.append(_doc_relation(document, relation_type, raw_line, ref))

        if GUIDE_HINT_RE.search(raw_line):
            relation_type = "DETAILS" if re.search(r"quy định chi tiết", raw_line, re.I) else "GUIDES"
            for ref in _document_refs_from_line(raw_line):
                if ref.get("doc_number") != document.doc_number:
                    relations.append(_doc_relation(document, relation_type, raw_line, ref))
    return relations


def _doc_relation(
    document: LegalDocument,
    relation_type: str,
    raw_text: str,
    ref: dict[str, str],
) -> DocumentRelation:
    return DocumentRelation(
        source_document_id=document.document_id,
        relation_type=relation_type if relation_type in RELATION_TYPES else "CITES",
        raw_text=raw_text[:1000],
        target_document_number=ref.get("doc_number", ""),
        target_title_hint=ref.get("title_hint", ""),
        confidence=0.9 if ref.get("doc_number") else 0.7,
        evidence_text=raw_text[:500],
    )


def _normalize_extraction_mode(value: str | None) -> str:
    mode = (value or DEFAULT_CONFIG.graph.extraction_mode or "hybrid").strip().lower()
    if mode not in {"rule", "hybrid", "llm_shadow"}:
        raise ValueError(f"Unsupported graph extraction mode: {mode}")
    return mode


def _should_call_llm(
    graph: ExtractedLegalGraph,
    metadata: dict[str, Any],
    normalized_text: str,
) -> bool:
    text = "\n".join(
        [
            str(metadata.get("title") or ""),
            normalized_text[:4000],
        ]
    )
    return (
        not graph.articles
        or not graph.clauses
        or bool(AMENDMENT_HINT_RE.search(text))
        or _has_ambiguous_named_law(graph)
        or _has_suspicious_same_document_refs(graph)
    )


def _is_public_vbpl_metadata(metadata: dict[str, Any]) -> bool:
    source_url = str(metadata.get("source_url") or "").lower()
    markdown_path = str(metadata.get("markdown_path") or "").replace("\\", "/").lower()
    return "vbpl.vn" in source_url or "/vbpl/" in markdown_path or "data/raw/vbpl/" in markdown_path


def _run_llm_extraction(
    *,
    llm_extractor: Any | None,
    metadata: dict[str, Any],
    normalized: str,
    graph: ExtractedLegalGraph,
) -> LLMExtractionResult:
    extractor = llm_extractor or GroqLegalGraphLLMExtractor(model=DEFAULT_CONFIG.graph.llm_model)
    if not hasattr(extractor, "extract"):
        raise TypeError("llm_extractor must provide an extract(...) method")
    return extractor.extract(
        document_metadata=metadata,
        normalized_text=normalized,
        units=_graph_units_for_llm(graph),
    )


def _graph_units_for_llm(graph: ExtractedLegalGraph) -> list[dict[str, Any]]:
    units = []
    for clause in graph.clauses:
        units.append(
            {
                "source_clause_id": clause.id,
                "article_no": clause.article_no,
                "clause_no": clause.clause_no,
                "title": clause.title,
                "unit_type": clause.unit_type,
                "structural_path": clause.structural_path,
                "contains_quoted_replacement": clause.contains_quoted_replacement,
                "target_document_number_hint": clause.target_document_number_hint,
                "target_title_hint": clause.target_title_hint,
                "unit_source": clause.unit_source,
                "text": clause.text[:1000],
            }
        )
    return units


def _merge_llm_payload(
    graph: ExtractedLegalGraph,
    payload: LLMGraphPayload,
    source_text: str,
    *,
    min_confidence: float,
) -> tuple[int, int]:
    accepted = 0
    rejected = 0
    for fact in payload.references:
        reference = _reference_from_llm_fact(graph, fact, source_text, min_confidence)
        if reference:
            graph.references.append(reference)
            accepted += 1
        else:
            rejected += 1
    for fact in payload.amendments:
        amendment = _amendment_from_llm_fact(graph, fact, source_text, min_confidence)
        if amendment:
            graph.amendments.append(amendment)
            accepted += 1
        else:
            rejected += 1
    for fact in payload.document_relations:
        relation = _relation_from_llm_fact(graph, fact, source_text, min_confidence)
        if relation:
            graph.document_relations.append(relation)
            accepted += 1
        else:
            rejected += 1
    return accepted, rejected


def _reference_from_llm_fact(
    graph: ExtractedLegalGraph,
    fact: LLMReferenceFact,
    source_text: str,
    min_confidence: float,
) -> ReferenceMention | None:
    if not _llm_fact_is_supported(fact.confidence, fact.evidence_text, source_text, min_confidence):
        return None
    if _is_generic_named_law(fact.target_title_hint or fact.raw_text):
        return None
    source_clause = _find_source_clause(graph, fact.source_article_no, fact.source_clause_no)
    if not source_clause:
        return None
    return ReferenceMention(
        id=_mention_id(source_clause.id, fact.raw_text, 0),
        source_clause_id=source_clause.id,
        raw_text=fact.raw_text,
        ref_type=fact.ref_type,
        scope=fact.scope,
        target_document_number=normalize_doc_number(fact.target_document_number),
        target_article_no=fact.target_article_no,
        target_clause_no=fact.target_clause_no,
        target_point=fact.target_point.lower(),
        target_title_hint=fact.target_title_hint,
        confidence=fact.confidence,
        evidence_text=fact.evidence_text,
        resolution_reason=fact.resolution_reason,
        extraction_method="llm",
        source_span=_source_span_for_evidence(source_text, fact.evidence_text),
        effective_date=fact.effective_date,
    )


def _amendment_from_llm_fact(
    graph: ExtractedLegalGraph,
    fact: LLMAmendmentFact,
    source_text: str,
    min_confidence: float,
) -> AmendmentOperation | None:
    if not _llm_fact_is_supported(fact.confidence, fact.evidence_text, source_text, min_confidence):
        return None
    source_clause = _find_source_clause(graph, fact.source_article_no, fact.source_clause_no)
    if not source_clause:
        return None
    return AmendmentOperation(
        id=_mention_id(source_clause.id, fact.raw_text, 0),
        source_clause_id=source_clause.id,
        operation_type=fact.operation_type,
        raw_text=fact.raw_text,
        target_document_number=normalize_doc_number(fact.target_document_number),
        target_article_no=fact.target_article_no,
        target_clause_no=fact.target_clause_no,
        target_point=fact.target_point.lower(),
        new_text=fact.new_text,
        confidence=fact.confidence,
        evidence_text=fact.evidence_text,
        resolution_reason=fact.resolution_reason,
        extraction_method="llm",
        source_span=_source_span_for_evidence(source_text, fact.evidence_text),
        effective_date=fact.effective_date,
    )


def _relation_from_llm_fact(
    graph: ExtractedLegalGraph,
    fact: LLMDocumentRelationFact,
    source_text: str,
    min_confidence: float,
) -> DocumentRelation | None:
    if not _llm_fact_is_supported(fact.confidence, fact.evidence_text, source_text, min_confidence):
        return None
    if _is_generic_named_law(fact.target_title_hint or fact.raw_text):
        return None
    return DocumentRelation(
        source_document_id=graph.document.document_id,
        relation_type=fact.relation_type,
        raw_text=fact.raw_text[:1000],
        target_document_number=normalize_doc_number(fact.target_document_number),
        target_title_hint=fact.target_title_hint,
        confidence=fact.confidence,
        evidence_text=fact.evidence_text,
        resolution_reason=fact.resolution_reason,
        extraction_method="llm",
        source_span=_source_span_for_evidence(source_text, fact.evidence_text),
        effective_date=fact.effective_date,
    )


def _find_source_clause(
    graph: ExtractedLegalGraph,
    article_no: str,
    clause_no: str,
) -> Clause | None:
    article_no = str(article_no or "")
    clause_no = str(clause_no or "")
    for clause in graph.clauses:
        if article_no and clause.article_no != article_no:
            continue
        if clause_no and clause.clause_no != clause_no:
            continue
        return clause
    return graph.clauses[0] if graph.clauses else None


def _llm_fact_is_supported(
    confidence: float,
    evidence_text: str,
    source_text: str,
    min_confidence: float,
) -> bool:
    if confidence < min_confidence:
        return False
    evidence = re.sub(r"\s+", " ", evidence_text or "").strip()
    if not evidence:
        return False
    source = re.sub(r"\s+", " ", source_text or "")
    return evidence in source


def _finalize_graph(
    graph: ExtractedLegalGraph,
    source_text: str,
    *,
    min_confidence: float,
    audit: GraphExtractionAudit,
) -> None:
    filtered_references, blocked_refs = _filter_references(
        graph.references,
        graph,
        min_confidence=min_confidence,
    )
    graph.references[:] = _dedupe_by_id(filtered_references)
    graph.amendments[:] = _dedupe_by_id(graph.amendments)
    graph.document_relations[:] = _dedupe_relations(
        _filter_relations(graph.document_relations, min_confidence=min_confidence)
    )
    audit.primary_units += len(graph.clauses)
    audit.quoted_replacement_blocks += sum(1 for clause in graph.clauses if clause.contains_quoted_replacement)
    audit.ambiguous_references += sum(1 for reference in graph.references if "ambiguous" in reference.scope)
    audit.same_document_refs_blocked_by_max_article += blocked_refs
    audit.incoming_outgoing_edges_materialized += sum(
        1 for reference in graph.references if "ambiguous" not in reference.scope
    ) + len(graph.amendments)
    relation_counts: dict[str, int] = {}
    relation_counts["REFERENCE"] = len(graph.references)
    for amendment in graph.amendments:
        relation_type = _operation_relation_type(amendment.operation_type)
        relation_counts[relation_type] = relation_counts.get(relation_type, 0) + 1
    for relation in graph.document_relations:
        relation_counts[relation.relation_type] = relation_counts.get(relation.relation_type, 0) + 1
    if audit.relations_by_type is None:
        audit.relations_by_type = {}
    for key, value in relation_counts.items():
        audit.relations_by_type[key] = audit.relations_by_type.get(key, 0) + value
    if not graph.articles and not graph.clauses and source_text.strip():
        audit.empty_graph_documents += 1
    if not graph.clauses and source_text.strip():
        audit.documents_without_clauses += 1
    if _has_duplicate_clause_keys(graph.clauses):
        audit.documents_with_duplicate_clause_keys += 1
    audit.low_confidence_relations += sum(
        1 for relation in graph.document_relations if relation.confidence < min_confidence
    )


def _filter_references(
    references: list[ReferenceMention],
    graph: ExtractedLegalGraph,
    *,
    min_confidence: float,
) -> tuple[list[ReferenceMention], int]:
    max_article = _max_numeric_article_no(graph.articles)
    filtered = []
    blocked_by_max_article = 0
    for reference in references:
        if _is_generic_named_law(reference.target_title_hint or reference.raw_text):
            continue
        if (
            reference.scope == "same_document"
            and _article_no_exceeds(reference.target_article_no, max_article)
        ):
            blocked_by_max_article += 1
            continue
        if reference.extraction_method == "llm" and reference.confidence < min_confidence:
            continue
        filtered.append(reference)
    return filtered, blocked_by_max_article


def _filter_relations(
    relations: list[DocumentRelation],
    *,
    min_confidence: float,
) -> list[DocumentRelation]:
    filtered = []
    for relation in relations:
        if _is_generic_named_law(relation.target_title_hint or relation.raw_text):
            continue
        if relation.extraction_method == "llm" and relation.confidence < min_confidence:
            continue
        filtered.append(relation)
    return filtered


def _has_ambiguous_named_law(graph: ExtractedLegalGraph) -> bool:
    values = [ref.target_title_hint or ref.raw_text for ref in graph.references]
    values.extend(rel.target_title_hint or rel.raw_text for rel in graph.document_relations)
    return any(_is_generic_named_law(value) for value in values)


def _has_suspicious_same_document_refs(graph: ExtractedLegalGraph) -> bool:
    max_article = _max_numeric_article_no(graph.articles)
    return any(
        ref.scope == "same_document" and _article_no_exceeds(ref.target_article_no, max_article)
        for ref in graph.references
    )


def _is_generic_named_law(value: str) -> bool:
    key = _ascii_key(value)
    return any(generic in key for generic in GENERIC_NAMED_LAW_REFS)


def _ascii_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value or "")
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_text = re.sub(r"\s+", " ", ascii_text).strip().lower()
    return ascii_text


def _max_numeric_article_no(articles: list[Article]) -> int:
    values = [_article_no_int(article.article_no) for article in articles]
    return max(values or [0])


def _article_no_exceeds(article_no: str, max_article: int) -> bool:
    value = _article_no_int(article_no)
    return bool(value and max_article and value > max_article)


def _article_no_int(article_no: str) -> int:
    match = re.match(r"(\d+)", str(article_no or ""))
    return int(match.group(1)) if match else 0


def _has_duplicate_clause_keys(clauses: list[Clause]) -> bool:
    seen: set[tuple[str, str]] = set()
    for clause in clauses:
        key = (clause.article_no, clause.clause_no)
        if key in seen:
            return True
        seen.add(key)
    return False


def _source_span_for_evidence(source_text: str, evidence_text: str) -> str:
    if not evidence_text:
        return ""
    index = source_text.find(evidence_text)
    if index < 0:
        return ""
    start_line = source_text[:index].count("\n") + 1
    end_line = start_line + evidence_text.count("\n")
    return f"L{start_line}-L{end_line}"


def _document_refs_from_line(line: str) -> list[dict[str, str]]:
    refs: list[dict[str, str]] = [
        {"doc_number": normalize_doc_number(match.group(1)), "title_hint": ""}
        for match in DOC_NUMBER_RE.finditer(line)
    ]
    refs.extend({"doc_number": "", "title_hint": law} for law in _named_law_refs(line))
    return _dedupe_ref_dicts(refs)


def _line_toggles_quote(line: str) -> bool:
    stripped = line.strip()
    quote_chars = ('"', "“", "”", "‘", "’")
    if len(stripped) >= 2 and stripped.startswith(quote_chars) and stripped.endswith(quote_chars):
        return False
    return stripped.startswith(quote_chars) or stripped.endswith(quote_chars)


def _quote_transition(line: str) -> str:
    stripped = line.strip()
    if not stripped:
        return "none"
    opening_chars = ('"', "“", "‘")
    closing_chars = ('"', "”", "’")
    opens = stripped.startswith(opening_chars)
    closes = stripped.endswith(closing_chars)
    if opens and closes and len(stripped) > 1:
        return "single"
    if opens:
        return "open"
    if closes:
        return "close"
    return "none"


def _target_hint_from_text(text: str, current_doc_number: str) -> tuple[str, str]:
    doc_number = _first_external_doc_number(text, current_doc_number)
    if doc_number:
        return doc_number, ""
    laws = _named_law_refs(text)
    return "", laws[0] if laws else ""


def _resolve_article_reference(
    *,
    document: LegalDocument,
    text: str,
    start: int,
    end: int,
    article_no: str,
    max_article_no: int,
    source_clause: Clause,
) -> dict[str, str | float]:
    evidence = _evidence_window(text, start, end)
    doc_number = _doc_number_near_mention(text, start, end)
    if doc_number:
        return {
            "scope": "external_document",
            "target_document_number": doc_number,
            "confidence": 0.92,
            "evidence_text": evidence,
            "resolution_reason": "A document number appears in the same local phrase as the article/clause reference.",
        }
    if source_clause.target_document_number_hint and _looks_like_amendment_target_phrase(text, start):
        return {
            "scope": "external_document",
            "target_document_number": source_clause.target_document_number_hint,
            "confidence": 0.88,
            "evidence_text": evidence,
            "resolution_reason": "The reference is in an amendment target phrase and inherits the target document from the parent context.",
        }
    context = _context_window(text, start, end)
    if _has_current_document_marker(context):
        return {
            "scope": "same_document",
            "target_document_number": document.doc_number,
            "confidence": 0.9,
            "evidence_text": evidence,
            "resolution_reason": "The local phrase uses a current-document marker such as 'Thông tư này' or 'Nghị định này'.",
        }
    if _article_no_exceeds(article_no, max_article_no):
        return {
            "scope": "ambiguous_external",
            "target_document_number": "",
            "confidence": 0.45,
            "evidence_text": evidence,
            "resolution_reason": "The referenced article number exceeds the source document's max article number, so it is not resolved to the current document.",
        }
    return {
        "scope": "same_document",
        "target_document_number": document.doc_number,
        "confidence": 0.8,
        "evidence_text": evidence,
        "resolution_reason": "No nearby external document marker was found and the article number fits the source document.",
    }


def _doc_number_near_mention(text: str, start: int, end: int) -> str:
    after = text[end : min(len(text), end + 180)]
    before = text[max(0, start - 120) : start]
    for candidate in (after, before):
        match = DOC_NUMBER_RE.search(candidate)
        if match:
            return normalize_doc_number(match.group(1))
    return ""


def _context_window(text: str, start: int, end: int, *, before: int = 120, after: int = 180) -> str:
    return text[max(0, start - before) : min(len(text), end + after)]


def _evidence_window(text: str, start: int, end: int) -> str:
    window = _context_window(text, start, end, before=80, after=120)
    return re.sub(r"\s+", " ", window).strip()[:500]


def _has_current_document_marker(text: str) -> bool:
    return bool(
        re.search(
            r"\b(văn bản này|thông tư này|nghị định này|luật này|bộ luật này|điều này|khoản này)\b",
            text,
            re.I,
        )
    )


def _looks_like_amendment_target_phrase(text: str, start: int) -> bool:
    before = text[max(0, start - 80) : start]
    return bool(re.search(r"\b(sửa đổi|bổ sung|thay thế|bãi bỏ|vào)\b", before, re.I))


def _mention_is_inside_amendment_operation(text: str, start: int, end: int) -> bool:
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", end)
    if line_end < 0:
        line_end = len(text)
    line = re.sub(r"^[\s\"“”'.,;:-]+", "", text[line_start:line_end].strip())
    if SUPPLEMENT_CLAUSE_INTO_ARTICLE_RE.match(line):
        return True
    return bool(AMENDMENT_RE.match(line))


def _named_law_refs(text: str) -> list[str]:
    names = []
    for match in NAMED_LAW_RE.finditer(text):
        name = re.sub(r"\s+", " ", match.group(1)).strip(" ,.;")
        if DOC_NUMBER_RE.search(name):
            continue
        names.append(name)
    return list(dict.fromkeys(names))


def _target_parts(target_text: str) -> tuple[str, str, str]:
    point_match = POINT_CLAUSE_ARTICLE_RE.search(target_text)
    if point_match:
        return (
            point_match.group(1).lower(),
            point_match.group(2),
            point_match.group(3),
        )
    article_match = ARTICLE_REF_RE.search(target_text)
    if not article_match:
        return "", "", ""
    point, clause_no, article_no = article_match.groups()
    return (point or "").lower(), clause_no or "", article_no or ""


def _operation_type(op_text: str) -> str:
    if "bãi bỏ" in op_text or "hết hiệu lực" in op_text:
        return "repeal"
    if "thay thế" in op_text:
        return "replace"
    if "bổ sung" in op_text and "sửa đổi" not in op_text:
        return "supplement"
    return "amend"


def _document_relation_type(line: str) -> str:
    lowered = line.lower()
    if "bãi bỏ" in lowered or "hết hiệu lực" in lowered:
        return "REPEALS"
    if "thay thế" in lowered:
        return "REPLACES"
    if "bổ sung" in lowered and "sửa đổi" not in lowered:
        return "SUPPLEMENTS"
    return "AMENDS"


def _operation_relation_type(operation_type: str) -> str:
    normalized = (operation_type or "").lower()
    if normalized == "supplement":
        return "SUPPLEMENTS"
    if normalized == "repeal":
        return "REPEALS"
    if normalized == "replace":
        return "REPLACES"
    return "AMENDS"


def _first_doc_number(text: str) -> str:
    match = DOC_NUMBER_RE.search(text or "")
    return normalize_doc_number(match.group(1)) if match else ""


def _first_external_doc_number(text: str, current_doc_number: str) -> str:
    current = normalize_doc_number(current_doc_number)
    for match in DOC_NUMBER_RE.finditer(text or ""):
        doc_number = normalize_doc_number(match.group(1))
        if doc_number and doc_number != current:
            return doc_number
    return ""


def _quoted_replacement_text(text: str) -> str:
    match = re.search(r"[\"“](.+)[\"”]", text, re.S)
    if not match:
        return ""
    return match.group(1).strip()[:5000]


def _iter_relevant_lines(text: str) -> Iterable[str]:
    for line in text.splitlines():
        stripped = re.sub(r"\s+", " ", line).strip()
        if len(stripped) >= 8:
            yield stripped


def _strip_markdown_noise(markdown: str) -> str:
    text = re.sub(r"<!--.*?-->", "", markdown, flags=re.S)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.M)
    return text


def _source_text_from_json(json_path: str | None) -> str:
    if not json_path:
        return ""
    path = Path(json_path)
    if not path.exists():
        return ""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    tree = payload.get("content_tree") or []
    texts = [str(block.get("text", "")) for block in tree if isinstance(block, dict)]
    return "\n".join(text for text in texts if text)


def _dedupe_articles(articles: list[Article]) -> list[Article]:
    seen: set[str] = set()
    unique: list[Article] = []
    for article in articles:
        if article.id not in seen:
            seen.add(article.id)
            unique.append(article)
    return unique


def _dedupe_by_id(items):
    best_by_id: dict[str, Any] = {}
    for item in items:
        item_id = getattr(item, "id", "")
        existing = best_by_id.get(item_id)
        if existing is None or getattr(item, "confidence", 1.0) > getattr(existing, "confidence", 1.0):
            best_by_id[item_id] = item
    return list(best_by_id.values())


def _dedupe_relations(relations: list[DocumentRelation]) -> list[DocumentRelation]:
    best: dict[tuple[str, str, str, str], DocumentRelation] = {}
    for relation in relations:
        key = (
            relation.source_document_id,
            relation.relation_type,
            relation.target_document_number,
            normalize_ref_key(relation.target_title_hint),
        )
        existing = best.get(key)
        if existing is None or relation.confidence > existing.confidence:
            best[key] = relation
    return list(best.values())


def _dedupe_ref_dicts(refs: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str]] = set()
    unique: list[dict[str, str]] = []
    for ref in refs:
        key = (ref.get("doc_number", ""), normalize_ref_key(ref.get("title_hint", "")))
        if key != ("", "") and key not in seen:
            seen.add(key)
            unique.append(ref)
    return unique


def _mention_id(source_id: str, raw_text: str, offset: int) -> str:
    return _digest(f"{source_id}:{offset}:{raw_text}")


def _digest(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()
