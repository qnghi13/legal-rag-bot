"""Rule-based post-generation answer verification."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass

from config.prompts import NO_CONTEXT_ANSWER
from src.graph.extractor import normalize_doc_number


CLAUSE_ARTICLE_RE = re.compile(r"\bKhoản\s+\d+[a-z]?\s+Điều\s+\d+[a-z]?\b", re.I)
ARTICLE_RE = re.compile(r"\bĐiều\s+\d+[a-z]?\b", re.I)
DOCUMENT_RE = re.compile(
    r"\b(?:Bộ luật|Luật|Nghị định|Thông tư|Quyết định)\s+"
    r"(?:số\s*)?[0-9]{1,4}/[0-9]{4}/[A-ZĐ][A-Z0-9Đ/-]*",
    re.I,
)


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    citations_found: list[str]
    unsupported_citations: list[str]
    fallback_reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def verify_answer(answer: str, dq_context: str, gq_context: str) -> VerificationResult:
    """Verify that a non-fallback answer cites evidence present in Dq or Gq."""

    normalized_answer = _normalize_text(answer)
    if not normalized_answer or normalized_answer == _normalize_text(NO_CONTEXT_ANSWER):
        return VerificationResult(
            passed=True,
            citations_found=[],
            unsupported_citations=[],
        )

    citations = extract_citations(answer)
    if not citations:
        return VerificationResult(
            passed=False,
            citations_found=[],
            unsupported_citations=[],
            fallback_reason="missing_citation",
        )

    normalized_context = _normalize_text(f"{dq_context}\n{gq_context}")
    unsupported = [
        citation
        for citation in citations
        if not _citation_supported(citation, normalized_context)
    ]
    if unsupported:
        return VerificationResult(
            passed=False,
            citations_found=citations,
            unsupported_citations=unsupported,
            fallback_reason="unsupported_citation",
        )
    return VerificationResult(
        passed=True,
        citations_found=citations,
        unsupported_citations=[],
    )


def extract_citations(text: str) -> list[str]:
    citations: list[tuple[int, int, str]] = []
    occupied: list[tuple[int, int]] = []
    for match in CLAUSE_ARTICLE_RE.finditer(text or ""):
        citations.append((match.start(), match.end(), _clean_citation(match.group(0))))
        occupied.append((match.start(), match.end()))

    for match in DOCUMENT_RE.finditer(text or ""):
        citations.append((match.start(), match.end(), _clean_citation(match.group(0))))
        occupied.append((match.start(), match.end()))

    for match in ARTICLE_RE.finditer(text or ""):
        if any(start <= match.start() and match.end() <= end for start, end in occupied):
            continue
        citations.append((match.start(), match.end(), _clean_citation(match.group(0))))

    unique: list[str] = []
    seen: set[str] = set()
    for _, _, citation in sorted(citations, key=lambda item: item[0]):
        key = _normalize_text(citation)
        if key not in seen:
            seen.add(key)
            unique.append(citation)
    return unique


def _citation_supported(citation: str, normalized_context: str) -> bool:
    normalized_citation = _normalize_text(citation)
    if normalized_citation and normalized_citation in normalized_context:
        return True
    doc_number = _doc_number_from_citation(citation)
    return bool(doc_number and _normalize_text(doc_number) in normalized_context)


def _doc_number_from_citation(citation: str) -> str:
    match = re.search(r"[0-9]{1,4}/[0-9]{4}/[A-ZĐ][A-Z0-9Đ/-]*", citation or "", re.I)
    return normalize_doc_number(match.group(0)) if match else ""


def _clean_citation(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFC", value or "")
    normalized = re.sub(r"\s+", " ", normalized).strip().lower()
    return normalized
