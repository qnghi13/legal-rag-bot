"""Convert supported source documents into markdown text."""

from __future__ import annotations

import os
import re

from src.ingestion.pdf_loader import pdf_to_markdown


_CHAPTER_RE = re.compile(r"^Chương\s+[IVXLCDM]+\b.*", re.IGNORECASE)
_SECTION_RE = re.compile(r"^Mục\s+\d+[a-zA-Z]?\b.*", re.IGNORECASE)
_ARTICLE_RE = re.compile(r"^Điều\s+\d+[a-zA-Z]?\.\s+.*", re.IGNORECASE)
_CLAUSE_RE = re.compile(r"^(\d{1,3})\.(?:\s+(.+)|\s*)$")
_CLAUSE_HEADING_RE = re.compile(r"^Khoản\s+(\d{1,3})(?:\.\s*(.*)|\s*)$", re.IGNORECASE)
_SEPARATOR_RE = re.compile(r"^-{3,}$")


def normalize_legal_headings(text: str) -> str:
    """Normalize Vietnamese legal headings into Markdown headers."""

    lines = [_strip_markdown_heading_markup(line) for line in text.splitlines()]
    normalized: list[str] = []
    in_article = False
    index = 0

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        if not stripped:
            normalized.append(line)
            index += 1
            continue

        if _CHAPTER_RE.match(stripped):
            chapter_title, next_index = _consume_heading_with_following_title(
                lines,
                index,
            )
            normalized.append(f"# {chapter_title}")
            in_article = False
            index = next_index
            continue

        if _SECTION_RE.match(stripped):
            normalized.append(f"## {stripped}")
            in_article = False
            index += 1
            continue

        if _ARTICLE_RE.match(stripped):
            normalized.append(f"### {stripped}")
            in_article = True
            index += 1
            continue

        clause_heading_match = _CLAUSE_HEADING_RE.match(stripped)
        if in_article and clause_heading_match:
            clause_number, clause_text = clause_heading_match.groups()
            normalized.append(f"#### Khoản {clause_number}")
            if clause_text:
                normalized.append(clause_text)
            index += 1
            continue

        clause_match = _CLAUSE_RE.match(stripped)
        if in_article and clause_match:
            clause_number, clause_text = clause_match.groups()
            normalized.append(f"#### Khoản {clause_number}")
            if clause_text:
                normalized.append(clause_text)
            index += 1
            continue

        normalized.append(line)
        index += 1

    return "\n".join(normalized)


def _strip_markdown_heading_markup(line: str) -> str:
    """Remove noisy PDF-to-markdown heading markup while keeping normal text."""

    line = re.sub(r"^\s*#+\s*", "", line)
    line = line.replace("**", "")
    return line.rstrip()


def _consume_heading_with_following_title(lines: list[str], index: int) -> tuple[str, int]:
    chapter = lines[index].strip()
    next_index = index + 1
    title_index = _find_next_nonempty_line(lines, next_index)

    if title_index is None:
        return chapter, next_index

    title = lines[title_index].strip()
    if _is_legal_heading(title) or _SEPARATOR_RE.match(title):
        return chapter, next_index

    return f"{chapter} - {title}", title_index + 1


def _find_next_nonempty_line(lines: list[str], start_index: int) -> int | None:
    for index in range(start_index, len(lines)):
        if lines[index].strip():
            return index
    return None


def _is_legal_heading(line: str) -> bool:
    return bool(
        _CHAPTER_RE.match(line)
        or _SECTION_RE.match(line)
        or _ARTICLE_RE.match(line)
        or _CLAUSE_RE.match(line)
        or _CLAUSE_HEADING_RE.match(line)
    )


def process_file_to_markdown(file_path: str) -> str:
    filename = os.path.basename(file_path)
    lowered = filename.lower()

    if lowered.endswith(".pdf"):
        md_text = pdf_to_markdown(file_path)
        return normalize_legal_headings(md_text)

    if lowered.endswith((".txt", ".md")):
        with open(file_path, "r", encoding="utf-8") as f:
            return normalize_legal_headings(f.read())

    return ""
