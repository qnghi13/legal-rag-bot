"""PDF loading helpers."""

from __future__ import annotations


def pdf_to_markdown(file_path: str) -> str:
    """Extract a PDF into markdown with PyMuPDF4LLM."""

    import pymupdf4llm

    return pymupdf4llm.to_markdown(file_path)
