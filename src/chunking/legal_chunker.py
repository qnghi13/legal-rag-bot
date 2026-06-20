"""Legal document chunking by Markdown hierarchy plus crawl metadata."""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.chunking.markdown_chunker import split_markdown_by_headers
from src.ingestion.text_extractor import normalize_legal_headings


LEGAL_HEADER_RE = re.compile(
    r"^#{1,6}\s+(?:Chương|Mục|Điều|Khoản)\b",
    re.IGNORECASE | re.MULTILINE,
)
HTML_COMMENT_RE = re.compile(r"^\s*<!--.*?-->\s*$", re.MULTILINE)
HEADER_METADATA_KEYS = ("Chuong", "Muc", "Dieu", "Khoan")
DOCUMENT_METADATA_KEYS = (
    "document_id",
    "source_url",
    "title",
    "document_type",
    "status",
    "doc_number",
    "issuing_agency",
    "issue_date",
    "effective_date",
    "expiry_date",
    "crawled_at",
    "updated_at",
    "markdown_path",
    "content_sha256",
)


def load_and_chunk_folder(
    folder_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    metadata_db: str | None = None,
    strip_preamble: bool = True,
) -> list[Document]:
    folder = Path(folder_path)
    metadata_db_path = _resolve_metadata_db(folder, metadata_db)
    metadata_by_path = _load_metadata_registry(metadata_db_path)

    print(f"[ingest] Scanning Markdown folder: {folder}")
    if metadata_db_path:
        print(f"[ingest] Loading metadata DB: {metadata_db_path}")
    else:
        print("[ingest] Metadata DB not found; using file-level metadata only.")
    print(f"[ingest] Chunk size={chunk_size}, overlap={chunk_overlap}")

    final_chunks: list[Document] = []
    markdown_files = sorted(folder.rglob("*.md"))
    for file_path in markdown_files:
        if not file_path.is_file():
            continue

        try:
            print(f"  -> Processing: {file_path.name}")
            raw_markdown = file_path.read_text(encoding="utf-8")
            if not raw_markdown:
                continue

            markdown = normalize_legal_headings(raw_markdown)
            markdown = _remove_html_comments(markdown)
            if strip_preamble:
                markdown = _strip_before_first_legal_header(markdown)

            doc_metadata = _metadata_for_file(
                file_path,
                folder,
                metadata_by_path,
            )
            chunks = _split_legal_markdown(
                markdown,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            for index, chunk in enumerate(chunks):
                chunk.metadata.update(doc_metadata)
                chunk.metadata["chunk_index"] = index
                _prepend_header_tag(chunk)

            final_chunks.extend(chunks)
        except Exception as exc:
            print(f"[ingest] Failed to read {file_path.name}: {exc}")

    print(f"[ingest] Created {len(final_chunks)} semantic chunks.")
    if final_chunks:
        print("[ingest] Sample chunk metadata:", final_chunks[0].metadata)
        print("[ingest] Sample chunk content:", final_chunks[0].page_content[:1000])

    return final_chunks


def _split_legal_markdown(
    markdown: str,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    if LEGAL_HEADER_RE.search(markdown):
        return split_markdown_by_headers(
            markdown,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            headers_to_split_on=[
                ("#", "Chuong"),
                ("##", "Muc"),
                ("###", "Dieu"),
                ("####", "Khoan"),
            ],
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    return splitter.create_documents([markdown])


def _resolve_metadata_db(folder: Path, metadata_db: str | None) -> Path | None:
    if metadata_db:
        path = Path(metadata_db)
        if not path.exists():
            raise FileNotFoundError(f"Metadata DB not found: {path}")
        return path

    candidates = [
        folder / "metadata.sqlite",
        folder.parent / "metadata.sqlite",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_metadata_registry(db_path: Path | None) -> dict[str, dict[str, Any]]:
    if not db_path:
        return {}

    registry: dict[str, dict[str, Any]] = {}
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT
                document_id, source_url, title, document_type, status, doc_number,
                issuing_agency, issue_date, effective_date, expiry_date, crawled_at,
                updated_at, markdown_path, content_sha256, metadata_json
            FROM vbpl_documents
            WHERE crawl_status = 'crawled'
            """
        )
        for row in rows:
            metadata = _row_to_metadata(dict(row))
            markdown_path = metadata.get("markdown_path", "")
            if not markdown_path:
                continue
            path = Path(markdown_path)
            registry[_path_key(path)] = metadata
            registry[path.name] = metadata
    return registry


def _row_to_metadata(row: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        key: row.get(key, "")
        for key in DOCUMENT_METADATA_KEYS
        if row.get(key) not in (None, "")
    }
    metadata_json = row.get("metadata_json")
    if metadata_json:
        try:
            payload = json.loads(metadata_json)
            nested = payload.get("metadata") or {}
            for key, value in nested.items():
                metadata.setdefault(key, value)
        except json.JSONDecodeError:
            pass
    return _clean_metadata(metadata)


def _metadata_for_file(
    file_path: Path,
    root_folder: Path,
    registry: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    metadata = (
        registry.get(_path_key(file_path))
        or registry.get(_path_key(file_path.relative_to(root_folder)))
        or registry.get(file_path.name)
        or {}
    )
    fallback = {
        "source": file_path.name,
        "markdown_path": str(file_path),
    }
    return _clean_metadata(fallback | metadata)


def _remove_html_comments(markdown: str) -> str:
    return HTML_COMMENT_RE.sub("", markdown).strip()


def _strip_before_first_legal_header(markdown: str) -> str:
    match = LEGAL_HEADER_RE.search(markdown)
    if not match:
        return markdown.strip()
    return markdown[match.start() :].strip()


def _clean_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in metadata.items():
        if value in (None, ""):
            continue
        if isinstance(value, (str, int, float, bool)):
            cleaned[key] = value
        else:
            cleaned[key] = json.dumps(value, ensure_ascii=False)
    return cleaned


def _path_key(path: Path) -> str:
    return str(path).replace("\\", "/").lower()


def _prepend_header_tag(chunk: Document) -> None:
    header_parts = [chunk.metadata.get(key, "") for key in HEADER_METADATA_KEYS]
    header_tag = "[" + " | ".join(part for part in header_parts if part) + "]"
    if header_tag != "[]":
        chunk.page_content = f"{header_tag}\n{chunk.page_content}"
