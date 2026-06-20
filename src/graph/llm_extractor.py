"""Structured LLM extraction for Vietnamese legal graph facts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError


RELATION_TYPES = {
    "AMENDS",
    "SUPPLEMENTS",
    "REPEALS",
    "REPLACES",
    "GUIDES",
    "DETAILS",
    "BASED_ON",
    "CITES",
}


class LLMReferenceFact(BaseModel):
    source_article_no: str = ""
    source_clause_no: str = ""
    raw_text: str
    ref_type: str = "external_document"
    scope: str = "unknown"
    target_document_number: str = ""
    target_article_no: str = ""
    target_clause_no: str = ""
    target_point: str = ""
    target_title_hint: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_text: str
    resolution_reason: str = ""
    effective_date: str = ""


class LLMAmendmentFact(BaseModel):
    source_article_no: str = ""
    source_clause_no: str = ""
    operation_type: Literal["amend", "supplement", "repeal", "replace"]
    raw_text: str
    target_document_number: str = ""
    target_article_no: str = ""
    target_clause_no: str = ""
    target_point: str = ""
    new_text: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_text: str
    resolution_reason: str = ""
    effective_date: str = ""


class LLMDocumentRelationFact(BaseModel):
    relation_type: Literal[
        "AMENDS",
        "SUPPLEMENTS",
        "REPEALS",
        "REPLACES",
        "GUIDES",
        "DETAILS",
        "BASED_ON",
        "CITES",
    ]
    raw_text: str
    target_document_number: str = ""
    target_title_hint: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_text: str
    resolution_reason: str = ""
    effective_date: str = ""


class LLMGraphPayload(BaseModel):
    references: list[LLMReferenceFact] = Field(default_factory=list)
    amendments: list[LLMAmendmentFact] = Field(default_factory=list)
    document_relations: list[LLMDocumentRelationFact] = Field(default_factory=list)


@dataclass
class LLMExtractionResult:
    payload: LLMGraphPayload = field(default_factory=LLMGraphPayload)
    called: bool = False
    parse_failed: bool = False
    raw_response: str = ""
    error: str = ""


class GroqLegalGraphLLMExtractor:
    """Small adapter around Groq/LangChain that returns validated graph facts."""

    def __init__(
        self,
        *,
        model: str,
        temperature: float = 0.0,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self._client = client

    def extract(
        self,
        *,
        document_metadata: dict[str, Any],
        normalized_text: str,
        units: list[dict[str, Any]],
    ) -> LLMExtractionResult:
        messages = _build_messages(
            document_metadata=document_metadata,
            normalized_text=normalized_text,
            units=units,
        )
        result = LLMExtractionResult(called=True)
        try:
            response = self._get_client().invoke(messages)
            result.raw_response = _message_content(response)
            result.payload = _parse_payload(result.raw_response)
        except Exception as exc:  # pragma: no cover - network/provider dependent
            result.parse_failed = True
            result.error = str(exc)
        return result

    def _get_client(self):
        if self._client is not None:
            return self._client
        from langchain_groq import ChatGroq

        self._client = ChatGroq(model=self.model, temperature=self.temperature)
        return self._client


def _build_messages(
    *,
    document_metadata: dict[str, Any],
    normalized_text: str,
    units: list[dict[str, Any]],
) -> list[tuple[str, str]]:
    compact_text = normalized_text[:12000]
    compact_units = units[:80]
    schema_hint = {
        "references": [
            {
                "source_article_no": "1",
                "source_clause_no": "2",
                "raw_text": "khoản 2 Điều 3 của Thông tư số ...",
                "ref_type": "clause",
                "scope": "external_document",
                "target_document_number": "01/2025/TT-BNV",
                "target_article_no": "3",
                "target_clause_no": "2",
                "target_point": "",
                "target_title_hint": "",
                "confidence": 0.9,
                "evidence_text": "exact short source span",
                "resolution_reason": "document number is near the article reference",
                "effective_date": "",
            }
        ],
        "amendments": [],
        "document_relations": [],
    }
    system = (
        "You extract Vietnamese legal knowledge graph facts. Return JSON only. "
        "Prefer precision over recall. Include only facts supported by exact evidence_text. "
        "Extract facts at Article/Clause level only; if a point/diem is mentioned, put it in target_point "
        "and do not invent a separate point node. Do not output containment relationships such as "
        "HAS_ARTICLE, HAS_CLAUSE, or CONTAINS. Do not treat headings inside quoted replacement text "
        "as source units of the current document. If the target is ambiguous, use scope='ambiguous' "
        "or leave unresolved fields blank instead of assuming the current document. "
        "Allowed document relation types: AMENDS, SUPPLEMENTS, REPEALS, REPLACES, "
        "GUIDES, DETAILS, BASED_ON, CITES. Allowed amendment operation_type values: "
        "amend, supplement, repeal, replace."
    )
    human = (
        "Document metadata:\n"
        f"{json.dumps(document_metadata, ensure_ascii=False)}\n\n"
        "Known legal units:\n"
        f"{json.dumps(compact_units, ensure_ascii=False)}\n\n"
        "Each unit may include target_document_number_hint inherited from a parent amendment heading. "
        "Use that hint only when the unit text is an amendment operation targeting that parent document.\n\n"
        "Return JSON shaped like this example, with arrays possibly empty:\n"
        f"{json.dumps(schema_hint, ensure_ascii=False)}\n\n"
        "Source text:\n"
        f"{compact_text}"
    )
    return [("system", system), ("human", human)]


def _parse_payload(raw_response: str) -> LLMGraphPayload:
    raw_response = _strip_code_fence(raw_response)
    try:
        return LLMGraphPayload.model_validate(json.loads(raw_response))
    except (json.JSONDecodeError, ValidationError):
        try:
            from json_repair import repair_json

            repaired = repair_json(raw_response)
            return LLMGraphPayload.model_validate(json.loads(repaired))
        except Exception as exc:
            raise ValueError(f"Cannot parse LLM graph JSON: {exc}") from exc


def _strip_code_fence(text: str) -> str:
    stripped = (text or "").strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped


def _message_content(response: Any) -> str:
    if isinstance(response, str):
        return response
    content = getattr(response, "content", None)
    if isinstance(content, str):
        return content
    return str(response)
