"""Generic Markdown chunking utilities."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter


def split_markdown_by_headers(
    markdown_text: str,
    *,
    chunk_size: int,
    chunk_overlap: int,
    headers_to_split_on: list[tuple[str, str]] | None = None,
) -> list[Document]:
    """Split markdown by headers first, then by character length."""

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
        or [
            ("#", "Header1"),
            ("##", "Header2"),
            ("###", "Header3"),
            ("####", "Header4"),
        ]
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    header_docs = markdown_splitter.split_text(markdown_text)
    return text_splitter.split_documents(header_docs)
