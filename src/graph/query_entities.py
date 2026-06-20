"""Regex-based query entity anchors for legal graph retrieval."""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.graph.extractor import normalize_doc_number


DOC_NUMBER_PATTERN = (
    r"(?:Bộ luật|Luật|Nghị định|Thông tư|Quyết định)?\s*"
    r"(?:số\s*)?(?P<doc_number>[0-9]{1,4}/[0-9]{4}/[A-ZĐ][A-Z0-9Đ/-]*)"
)
DOC_NUMBER_RE = re.compile(DOC_NUMBER_PATTERN, re.I)
UNIT_BEFORE_DOC_RE = re.compile(
    r"(?:(?:khoản)\s+(?P<clause>\d+[a-z]?)\s+)?"
    r"Điều\s+(?P<article>\d+[a-z]?)"
    r"(?P<between>[^.;\n]{0,160}?)"
    + DOC_NUMBER_PATTERN,
    re.I,
)
DOC_BEFORE_UNIT_RE = re.compile(
    DOC_NUMBER_PATTERN
    + r"(?P<between>[^.;\n]{0,160}?)"
    r"(?:(?:khoản)\s+(?P<clause>\d+[a-z]?)\s+)?"
    r"Điều\s+(?P<article>\d+[a-z]?)",
    re.I,
)


@dataclass(frozen=True)
class QueryAnchor:
    raw_text: str
    doc_number: str
    article_no: str = ""
    clause_no: str = ""

    def to_seed(self) -> dict[str, str]:
        return {
            "doc_number": self.doc_number,
            "article_no": self.article_no,
            "clause_no": self.clause_no,
            "query_anchor_raw": self.raw_text,
        }

    def trace_metadata(self) -> dict[str, str]:
        return {
            "query_anchor_raw": self.raw_text,
            "query_anchor_doc_number": self.doc_number,
            "query_anchor_article_no": self.article_no,
            "query_anchor_clause_no": self.clause_no,
        }


def parse_query_anchors(query: str) -> list[QueryAnchor]:
    """Extract document-backed legal anchors from a user query."""

    anchors: list[tuple[int, int, QueryAnchor]] = []
    for pattern in (UNIT_BEFORE_DOC_RE, DOC_BEFORE_UNIT_RE):
        for match in pattern.finditer(query or ""):
            doc_number = normalize_doc_number(match.group("doc_number"))
            if not doc_number:
                continue
            anchors.append(
                (
                    match.start(),
                    match.end(),
                    QueryAnchor(
                        raw_text=_clean_raw(match.group(0)),
                        doc_number=doc_number,
                        article_no=(match.group("article") or "").lower(),
                        clause_no=(match.group("clause") or "").lower(),
                    ),
                )
            )

    unit_spans = [(start, end) for start, end, anchor in anchors if anchor.article_no]
    for match in DOC_NUMBER_RE.finditer(query or ""):
        if any(start <= match.start() and match.end() <= end for start, end in unit_spans):
            continue
        doc_number = normalize_doc_number(match.group("doc_number"))
        if doc_number:
            anchors.append(
                (
                    match.start(),
                    match.end(),
                    QueryAnchor(
                        raw_text=_clean_raw(match.group(0)),
                        doc_number=doc_number,
                    ),
                )
            )

    ordered: list[QueryAnchor] = []
    seen: set[tuple[str, str, str]] = set()
    for _, _, anchor in sorted(anchors, key=lambda item: item[0]):
        key = (anchor.doc_number, anchor.article_no, anchor.clause_no)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(anchor)
    return ordered


def _clean_raw(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()
