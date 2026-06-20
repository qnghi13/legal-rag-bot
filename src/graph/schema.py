"""Typed graph objects used by the Legal GraphRAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LegalDocument:
    document_id: str
    doc_number: str = ""
    title: str = ""
    document_type: str = ""
    status: str = ""
    issuing_agency: str = ""
    issue_date: str = ""
    effective_date: str = ""
    source_url: str = ""
    markdown_path: str = ""


@dataclass(frozen=True)
class Article:
    id: str
    document_id: str
    article_no: str
    title: str = ""
    unit_type: str = "article"
    structural_path: str = ""
    source_start_line: int = 0
    source_end_line: int = 0
    extraction_method: str = "rule"


@dataclass(frozen=True)
class Clause:
    id: str
    document_id: str
    article_no: str
    clause_no: str
    text: str
    title: str = ""
    chunk_id: str = ""
    markdown_path: str = ""
    occurrence: int = 1
    unit_type: str = "clause"
    structural_path: str = ""
    source_start_line: int = 0
    source_end_line: int = 0
    extraction_method: str = "rule"
    confidence: float = 1.0
    quote_state: str = "outside_quote"
    contains_quoted_replacement: bool = False
    target_document_number_hint: str = ""
    target_title_hint: str = ""
    unit_source: str = "main_text"


@dataclass(frozen=True)
class LegalDocumentRef:
    ref_key: str
    doc_number: str = ""
    title_hint: str = ""
    resolved: bool = False


@dataclass(frozen=True)
class ReferenceMention:
    id: str
    source_clause_id: str
    raw_text: str
    ref_type: str
    scope: str = "unknown"
    target_document_number: str = ""
    target_article_no: str = ""
    target_clause_no: str = ""
    target_point: str = ""
    target_title_hint: str = ""
    confidence: float = 1.0
    evidence_text: str = ""
    resolution_reason: str = ""
    extraction_method: str = "rule"
    source_span: str = ""
    effective_date: str = ""


@dataclass(frozen=True)
class AmendmentOperation:
    id: str
    source_clause_id: str
    operation_type: str
    raw_text: str
    target_document_number: str = ""
    target_article_no: str = ""
    target_clause_no: str = ""
    target_point: str = ""
    new_text: str = ""
    confidence: float = 1.0
    evidence_text: str = ""
    resolution_reason: str = ""
    extraction_method: str = "rule"
    source_span: str = ""
    effective_date: str = ""


@dataclass(frozen=True)
class DocumentRelation:
    source_document_id: str
    relation_type: str
    raw_text: str
    target_document_number: str = ""
    target_title_hint: str = ""
    confidence: float = 1.0
    evidence_text: str = ""
    resolution_reason: str = ""
    extraction_method: str = "rule"
    source_span: str = ""
    effective_date: str = ""


@dataclass
class ExtractedLegalGraph:
    document: LegalDocument
    articles: list[Article] = field(default_factory=list)
    clauses: list[Clause] = field(default_factory=list)
    references: list[ReferenceMention] = field(default_factory=list)
    amendments: list[AmendmentOperation] = field(default_factory=list)
    document_relations: list[DocumentRelation] = field(default_factory=list)
