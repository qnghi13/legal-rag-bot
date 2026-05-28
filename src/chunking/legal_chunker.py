"""Legal document chunking by Markdown hierarchy."""

from __future__ import annotations

import os

from langchain_core.documents import Document
from src.chunking.markdown_chunker import split_markdown_by_headers
from src.ingestion.text_extractor import process_file_to_markdown


def load_and_chunk_folder(
    folder_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    print(f"[ingest] Scanning folder: {folder_path}")
    print(f"[ingest] Chunk size={chunk_size}, overlap={chunk_overlap}")

    final_chunks: list[Document] = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if not os.path.isfile(file_path):
            continue

        try:
            print(f"  -> Processing: {filename}")
            raw_markdown = process_file_to_markdown(file_path)
            if not raw_markdown:
                continue

            chunks = split_markdown_by_headers(
                raw_markdown,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                headers_to_split_on=[
                    ("#", "Chuong"),
                    ("##", "Muc"),
                    ("###", "Dieu"),
                    ("####", "Khoan"),
                ],
            )
            for chunk in chunks:
                chunk.metadata["source"] = filename
                _prepend_header_tag(chunk)

            final_chunks.extend(chunks)
        except Exception as exc:
            print(f"[ingest] Failed to read {filename}: {exc}")

    print(f"[ingest] Created {len(final_chunks)} semantic chunks.")
    if final_chunks:
        print("[ingest] Sample chunk metadata:", final_chunks[0].metadata)
        print("[ingest] Sample chunk content:", final_chunks[0].page_content[:1000])

    return final_chunks


def _prepend_header_tag(chunk: Document) -> None:
    header_parts = [
        chunk.metadata.get("Chuong", ""),
        chunk.metadata.get("Muc", ""),
        chunk.metadata.get("Dieu", ""),
        chunk.metadata.get("Khoan", ""),
    ]
    header_tag = "[" + " | ".join(part for part in header_parts if part) + "]"
    chunk.page_content = f"{header_tag}\n{chunk.page_content}"
