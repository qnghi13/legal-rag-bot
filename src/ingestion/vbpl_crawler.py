"""Crawler for Vietnamese legal documents from vbpl.vn."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urljoin, urlparse

import requests
from bs4 import BeautifulSoup, NavigableString, Tag

from src.ingestion.text_extractor import normalize_legal_headings


LEGACY_SEARCH_URL = "https://vbpl.vn/TW/Pages/vbpq-timkiem.aspx"
VBPL_API_BASE_URL = "https://vbpl-bientap-gateway.moj.gov.vn/api"
VBPL_DOCUMENT_SEARCH_PATH = "/qtdc/public/doc/all"
VBPL_DOCUMENT_DETAIL_PATH = "/qtdc/public/doc"
VBPL_COMBOBOX_PATH = "/qtdc/public/doc/combobox"
ALLOWED_DOCUMENT_TYPES = ("Bộ luật", "Luật", "Nghị định", "Thông tư")
ALLOWED_STATUSES = ("Còn hiệu lực", "Hết hiệu lực một phần")
DOCUMENT_TYPE_IDS = {
    "Bộ luật": "404b68a7-8e71-4ee5-a6c0-07e59f35f824",
    "Luật": "11025e19-2dd6-4165-85ad-ab6241186a1a",
    "Nghị định": "0d08b84c-7de7-4800-8760-2a68265e7890",
    "Thông tư": "178c63a9-73ff-4fd4-9d91-18d690520090",
}
STATUS_IDS = {
    "Còn hiệu lực": "1419f6be-4a15-44a7-97ac-ea042770a514",
    "Hết hiệu lực một phần": "9c20e89d-e048-4f3a-b6c2-df87fe0b1ada",
}
RESULT_LINK_RE = re.compile(r"vbpq-(?:toanvan|thuoctinh)\.aspx\?ItemID=\d+", re.I)


@dataclass(frozen=True)
class VBPLSearchResult:
    document_id: str
    source_url: str
    title: str
    document_type: str
    status: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class VBPLDocument:
    source_url: str
    title: str
    document_type: str
    status: str
    crawled_at: str
    file_path: str
    metadata: dict[str, str]
    content_html: str
    content_tree: list[dict[str, Any]]
    content_text: str

    def to_json(self) -> dict[str, Any]:
        return {
            "source_url": self.source_url,
            "title": self.title,
            "document_type": self.document_type,
            "status": self.status,
            "crawled_at": self.crawled_at,
            "file_path": self.file_path,
            "metadata": self.metadata,
            "content_html": self.content_html,
            "content_tree": self.content_tree,
            "content_text": self.content_text,
        }


class VBPLCrawlerError(RuntimeError):
    """Raised when the crawler cannot collect enough data safely."""


class VBPLCrawler:
    """Crawl VBPL documents and write JSON plus Markdown outputs."""

    def __init__(
        self,
        *,
        keyword: str = "lao động",
        document_types: tuple[str, ...] = ALLOWED_DOCUMENT_TYPES,
        statuses: tuple[str, ...] = ALLOWED_STATUSES,
        output_dir: str | Path = "data/raw/vbpl",
        keyword_scope: str = "title",
        metadata_db: str | Path | None = None,
        write_json: bool = False,
        include_content_json: bool = False,
        delay_seconds: float = 0.5,
        timeout_seconds: int = 30,
    ) -> None:
        if keyword_scope not in {"title", "all"}:
            raise ValueError("keyword_scope must be one of: title, all")

        self.keyword = keyword
        self.document_types = tuple(document_types)
        self.statuses = tuple(statuses)
        self.output_dir = Path(output_dir)
        self.keyword_scope = keyword_scope
        self.metadata_db = Path(metadata_db) if metadata_db else self.output_dir / "metadata.sqlite"
        self.write_json = write_json
        self.include_content_json = include_content_json
        self.delay_seconds = delay_seconds
        self.timeout_seconds = timeout_seconds
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0 Safari/537.36"
                ),
                "Accept-Language": "vi,en-US;q=0.9,en;q=0.8",
            }
        )

    def crawl(self, *, max_docs: int | None = None) -> list[Path]:
        return self._crawl_with_requests(max_docs=max_docs)

    def _crawl_with_requests(self, *, max_docs: int | None) -> list[Path]:
        results = self._collect_search_results_with_api(max_docs=max_docs)
        if not results:
            raise VBPLCrawlerError(
                "No matching documents found from the VBPL public API."
            )
        return self._write_results(results, max_docs=max_docs)

    def _collect_search_results_with_api(
        self,
        *,
        max_docs: int | None,
    ) -> list[VBPLSearchResult]:
        collected: list[VBPLSearchResult] = []
        seen: set[str] = set()
        page = 1
        page_size = 100
        type_ids = self._resolve_category_ids(
            group_code="LoaiVanBan",
            names=self.document_types,
            fallback=DOCUMENT_TYPE_IDS,
        )
        status_ids = self._resolve_category_ids(
            group_code="TrangThaiHieuLuc",
            names=self.statuses,
            fallback=STATUS_IDS,
        )

        while True:
            payload = {
                "pageNumber": page,
                "pageSize": page_size,
                "sortBy": "issueDate",
                "sortDirection": "desc",
                "groupVbpl": True,
                "score": False,
                "agencyLevel": "TRUNG_UONG",
                "docType": type_ids,
                "effStatus": status_ids,
            }
            if self.keyword_scope == "title":
                payload.update(
                    {
                        "keyword": self.keyword,
                        "documentName": self.keyword,
                        "optionDoc": "title",
                        "matchMode": "all_words",
                    }
                )
            else:
                payload["keywordQuickSearch"] = self.keyword
            payload = {key: value for key, value in payload.items() if value}
            data = self._api_post(
                VBPL_DOCUMENT_SEARCH_PATH,
                payload,
                timeout=self.timeout_seconds,
            )
            items = data.get("items") or []
            page_results = [
                search_result_from_api_item(item, document_types=self.document_types)
                for item in items
            ]
            new_results = [
                result
                for result in page_results
                if result
                and result.document_id not in seen
                and (
                    self.keyword_scope != "title"
                    or text_contains_keyword(result.title, self.keyword)
                )
            ]
            for result in new_results:
                seen.add(result.document_id)
                collected.append(result)
                if max_docs and len(collected) >= max_docs:
                    return collected

            total = int(data.get("total") or 0)
            if not items or page * page_size >= total:
                return collected

            page += 1
            time.sleep(self.delay_seconds)

    def _resolve_category_ids(
        self,
        *,
        group_code: str,
        names: tuple[str, ...],
        fallback: dict[str, str],
    ) -> list[str]:
        try:
            data = self._api_get(
                VBPL_COMBOBOX_PATH,
                {"groupCode": group_code, "includeInactive": True},
                timeout=self.timeout_seconds,
            )
            items = data.get("items") or []
            ids_by_name = {normalize_space(item.get("name", "")): item.get("id") for item in items}
            ids = [ids_by_name[name] for name in names if ids_by_name.get(name)]
            if ids:
                return ids
        except Exception as exc:
            print(f"[vbpl] could not resolve {group_code} ids from API: {exc}")
        return [fallback[name] for name in names if name in fallback]

    def _api_get(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        *,
        timeout: int,
    ) -> dict[str, Any]:
        response = self.session.get(
            f"{VBPL_API_BASE_URL}{path}",
            params=params,
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
        return payload.get("data") if isinstance(payload, dict) and "data" in payload else payload

    def _api_post(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        timeout: int,
    ) -> dict[str, Any]:
        response = self.session.post(
            f"{VBPL_API_BASE_URL}{path}",
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        body = response.json()
        return body.get("data") if isinstance(body, dict) and "data" in body else body

    def _write_results(
        self,
        results: list[VBPLSearchResult],
        *,
        max_docs: int | None,
        detail_html_by_url: dict[str, str] | None = None,
    ) -> list[Path]:
        markdown_dir = self.output_dir / "markdown"
        json_dir = self.output_dir / "json"
        if self.write_json:
            json_dir.mkdir(parents=True, exist_ok=True)
        markdown_dir.mkdir(parents=True, exist_ok=True)
        init_metadata_db(self.metadata_db)

        written: list[Path] = []
        seen_docs: set[str] = set()
        for result in results:
            dedupe_key = _dedupe_key(result)
            if dedupe_key in seen_docs:
                continue
            seen_docs.add(dedupe_key)

            if detail_html_by_url and result.source_url in detail_html_by_url:
                try:
                    document = self.document_from_html(
                        result,
                        detail_html_by_url[result.source_url],
                        markdown_dir=markdown_dir,
                    )
                except Exception as exc:
                    print(f"[vbpl] skipped {result.source_url}: {exc}")
                    upsert_crawl_error(self.metadata_db, result, str(exc))
                    continue
            else:
                try:
                    document = self.fetch_document(result, markdown_dir=markdown_dir)
                except Exception as exc:
                    print(f"[vbpl] skipped {result.source_url}: {exc}")
                    upsert_crawl_error(self.metadata_db, result, str(exc))
                    continue
            slug = make_slug(document.title, document.metadata.get("so_ky_hieu", ""))
            json_path = json_dir / f"{slug}.json" if self.write_json else None
            markdown_path = markdown_dir / f"{slug}.md"

            document = VBPLDocument(
                source_url=document.source_url,
                title=document.title,
                document_type=document.document_type,
                status=document.status,
                crawled_at=document.crawled_at,
                file_path=str(markdown_path),
                metadata=document.metadata,
                content_html=document.content_html,
                content_tree=document.content_tree,
                content_text=document.content_text,
            )

            markdown_text = document_to_markdown(document)
            markdown_path.write_text(markdown_text, encoding="utf-8")
            metadata_payload = document_to_metadata_json(
                document,
                json_path=json_path,
                markdown_path=markdown_path,
                markdown_text=markdown_text,
                include_content=self.include_content_json,
            )
            if json_path is not None:
                json_path.write_text(
                    json.dumps(metadata_payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            upsert_crawled_document(self.metadata_db, metadata_payload)
            written.append(markdown_path)
            print(f"[vbpl] saved {markdown_path}")

            if max_docs and len(written) >= max_docs:
                break

            time.sleep(self.delay_seconds)

        return written

    def fetch_document(
        self,
        result: VBPLSearchResult,
        *,
        markdown_dir: Path,
    ) -> VBPLDocument:
        if result.document_id:
            data = self._api_get(
                f"{VBPL_DOCUMENT_DETAIL_PATH}/{result.document_id}",
                timeout=self.timeout_seconds,
            )
            return self.document_from_api_detail(result, data, markdown_dir=markdown_dir)

        response = self.session.get(result.source_url, timeout=self.timeout_seconds)
        response.raise_for_status()
        return self.document_from_html(result, response.text, markdown_dir=markdown_dir)

    def document_from_api_detail(
        self,
        result: VBPLSearchResult,
        data: dict[str, Any],
        *,
        markdown_dir: Path,
    ) -> VBPLDocument:
        content_html = ((data.get("documentContent") or {}).get("content") or "").strip()
        if not content_html:
            raise VBPLCrawlerError(f"Document has no HTML content: {result.source_url}")

        soup = BeautifulSoup(content_html, "html.parser")
        content_node = soup.body or soup.find()
        if content_node is None:
            raise VBPLCrawlerError(f"Cannot parse document content: {result.source_url}")

        clean_node = clean_content_node(content_node)
        content_tree = html_to_content_tree(clean_node)
        metadata = result.metadata | metadata_from_api_detail(data)
        title = normalize_space(data.get("title") or result.title)
        document_type = get_nested_name(data.get("docType")) or result.document_type
        status = get_nested_name(data.get("effStatus")) or result.status
        source_url = make_public_detail_url(data.get("id") or result.document_id, title, category="trung-uong")
        slug = make_slug(title, metadata.get("so_ky_hieu", ""))

        return VBPLDocument(
            source_url=source_url,
            title=title,
            document_type=document_type,
            status=status,
            crawled_at=datetime.now(timezone.utc).isoformat(),
            file_path=str(markdown_dir / f"{slug}.md"),
            metadata=metadata,
            content_html=str(clean_node),
            content_tree=content_tree,
            content_text=clean_node.get_text("\n", strip=True),
        )

    def document_from_html(
        self,
        result: VBPLSearchResult,
        html: str,
        *,
        markdown_dir: Path,
    ) -> VBPLDocument:
        soup = BeautifulSoup(html, "html.parser")
        content_node = find_document_content_node(soup)
        if content_node is None:
            raise VBPLCrawlerError(f"Cannot find document content: {result.source_url}")

        clean_node = clean_content_node(content_node)
        content_tree = html_to_content_tree(clean_node)
        metadata = result.metadata | parse_detail_metadata(soup)
        title = result.title or metadata.get("title", "")
        slug = make_slug(title, metadata.get("so_ky_hieu", ""))

        return VBPLDocument(
            source_url=result.source_url,
            title=title,
            document_type=result.document_type or metadata.get("document_type", ""),
            status=result.status or metadata.get("status", ""),
            crawled_at=datetime.now(timezone.utc).isoformat(),
            file_path=str(markdown_dir / f"{slug}.md"),
            metadata=metadata,
            content_html=str(clean_node),
            content_tree=content_tree,
            content_text=clean_node.get_text("\n", strip=True),
        )


def parse_search_results(
    soup: BeautifulSoup,
    *,
    base_url: str,
    document_types: tuple[str, ...] = ALLOWED_DOCUMENT_TYPES,
    statuses: tuple[str, ...] = ALLOWED_STATUSES,
) -> list[VBPLSearchResult]:
    results: list[VBPLSearchResult] = []
    seen: set[str] = set()

    for link in soup.find_all("a", href=True):
        href = str(link["href"])
        if not (RESULT_LINK_RE.search(href) or "/van-ban/" in href):
            continue

        source_url = normalize_detail_url(urljoin(base_url, href))
        if source_url in seen:
            continue

        card = find_result_container(link)
        title = normalize_space(link.get_text(" ", strip=True))
        if card is not None:
            heading = card.find(["h1", "h2", "h3", "h4"])
            if heading:
                title = normalize_space(heading.get_text(" ", strip=True))
            elif not title:
                title = infer_title_from_text(card.get_text(" ", strip=True), document_types)

        combined_text = normalize_space((card or link).get_text(" ", strip=True))
        status = infer_status(combined_text, statuses)
        document_type = infer_document_type(title or combined_text, document_types)
        if not title or not status or not document_type:
            continue

        metadata = parse_metadata_from_text(combined_text)
        if not is_allowed_result(document_type, status, document_types, statuses):
            continue

        seen.add(source_url)
        results.append(
            VBPLSearchResult(
                document_id=extract_document_id_from_url(source_url),
                source_url=source_url,
                title=title,
                document_type=document_type,
                status=status,
                metadata=metadata,
            )
        )

    return results


def search_result_from_api_item(
    item: dict[str, Any],
    *,
    document_types: tuple[str, ...] = ALLOWED_DOCUMENT_TYPES,
) -> VBPLSearchResult | None:
    document_id = str(item.get("id") or "").strip()
    title = normalize_space(item.get("title") or "")
    document_type = get_nested_name(item.get("docType"))
    status = get_nested_name(item.get("effStatus"))
    if not document_id or not title or document_type not in document_types:
        return None

    source_url = make_public_detail_url(document_id, title, category="trung-uong")
    return VBPLSearchResult(
        document_id=document_id,
        source_url=source_url,
        title=title,
        document_type=document_type,
        status=status,
        metadata=metadata_from_api_summary(item),
    )


def is_allowed_result(
    document_type: str,
    status: str,
    document_types: tuple[str, ...] = ALLOWED_DOCUMENT_TYPES,
    statuses: tuple[str, ...] = ALLOWED_STATUSES,
) -> bool:
    return document_type in document_types and status in statuses


def find_result_container(link: Tag) -> Tag | None:
    node: Tag | None = link
    for _ in range(5):
        if node is None:
            return None
        text = node.get_text(" ", strip=True)
        if "Trạng thái" in text or any(status in text for status in ALLOWED_STATUSES):
            return node
        node = node.parent if isinstance(node.parent, Tag) else None
    return link.parent if isinstance(link.parent, Tag) else None


def has_next_page(soup: BeautifulSoup, current_page: int) -> bool:
    next_text = soup.find(string=re.compile(r"\bSau\b", re.I))
    if next_text:
        parent = next_text.parent if isinstance(next_text.parent, Tag) else None
        return parent is None or "disabled" not in " ".join(parent.get("class", []))
    return bool(soup.find("a", string=re.compile(rf"^\s*{current_page + 1}\s*$")))


def find_document_content_node(soup: BeautifulSoup) -> Tag | None:
    selectors = [
        "#toanvan",
        "#divContentDoc",
        ".content1",
        ".fulltext",
        ".doc-content",
        "[class*=document-content]",
        "[class*=content]",
        "article",
        "main",
    ]
    for selector in selectors:
        node = soup.select_one(selector)
        if node and _looks_like_legal_text(node):
            return node

    candidates = [
        node
        for node in soup.find_all(["div", "section", "article", "main"])
        if _looks_like_legal_text(node)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda item: len(item.get_text(" ", strip=True)))


def clean_content_node(node: Tag) -> Tag:
    soup = BeautifulSoup(str(node), "html.parser")
    root = soup.find()
    if root is None:
        raise VBPLCrawlerError("Empty document content node.")
    if root.name == "html" and root.body:
        root = root.body

    for noisy in root.select("script, style, noscript, iframe, button, svg"):
        noisy.decompose()
    for tag in root.find_all(True):
        attrs = {
            name: value
            for name, value in tag.attrs.items()
            if name in {"href", "src", "colspan", "rowspan"}
        }
        tag.attrs = attrs
    return root


def html_to_content_tree(node: Tag) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for child in node.children:
        blocks.extend(_node_to_blocks(child))
    return [block for block in blocks if block.get("text") or block.get("rows")]


def _node_to_blocks(node: Tag | NavigableString) -> list[dict[str, Any]]:
    if isinstance(node, NavigableString):
        text = normalize_space(str(node))
        return [{"type": "paragraph", "text": text}] if text else []
    if not isinstance(node, Tag):
        return []

    name = node.name.lower()
    if name in {"script", "style", "noscript"}:
        return []
    if name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
        return [{"type": "heading", "level": int(name[1]), "text": node.get_text(" ", strip=True)}]
    if name == "table":
        rows = []
        for tr in node.find_all("tr"):
            row = [normalize_space(cell.get_text(" ", strip=True)) for cell in tr.find_all(["th", "td"])]
            if row:
                rows.append(row)
        return [{"type": "table", "rows": rows}]
    if name in {"ul", "ol"}:
        items = [
            normalize_space(li.get_text(" ", strip=True))
            for li in node.find_all("li", recursive=False)
        ]
        return [{"type": "list", "ordered": name == "ol", "items": [item for item in items if item]}]
    if name in {"p", "div", "section", "article", "main", "center"}:
        direct_text = normalize_space(
            " ".join(str(child).strip() for child in node.children if isinstance(child, NavigableString))
        )
        child_blocks: list[dict[str, Any]] = []
        for child in node.children:
            if isinstance(child, Tag):
                child_blocks.extend(_node_to_blocks(child))
        if child_blocks:
            return ([{"type": "paragraph", "text": direct_text}] if direct_text else []) + child_blocks
        text = normalize_space(node.get_text(" ", strip=True))
        return [{"type": "paragraph", "text": text}] if text else []

    text = normalize_space(node.get_text(" ", strip=True))
    return [{"type": "paragraph", "text": text}] if text else []


def document_to_markdown(document: VBPLDocument) -> str:
    lines = [
        f"<!-- source_url: {document.source_url} -->",
        f"<!-- document_type: {document.document_type} -->",
        f"<!-- status: {document.status} -->",
        f"# {document.title}",
        "",
    ]
    lines.extend(content_tree_to_markdown(document.content_tree))
    markdown = "\n".join(lines)
    return normalize_legal_headings(markdown)


def document_to_metadata_json(
    document: VBPLDocument,
    *,
    json_path: Path | None,
    markdown_path: Path,
    markdown_text: str,
    include_content: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "document_id": document.metadata.get("document_id", ""),
        "source_url": document.source_url,
        "title": document.title,
        "document_type": document.document_type,
        "status": document.status,
        "crawled_at": document.crawled_at,
        "file_path": str(markdown_path),
        "json_path": str(json_path) if json_path else "",
        "markdown_path": str(markdown_path),
        "content_sha256": hashlib.sha256(markdown_text.encode("utf-8")).hexdigest(),
        "content_bytes": len(markdown_text.encode("utf-8")),
        "metadata": document.metadata,
    }
    if include_content:
        payload.update(
            {
                "content_html": document.content_html,
                "content_tree": document.content_tree,
                "content_text": document.content_text,
            }
        )
    return payload


def init_metadata_db(db_path: str | Path) -> None:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS vbpl_documents (
                document_id TEXT PRIMARY KEY,
                source_url TEXT NOT NULL,
                title TEXT NOT NULL,
                document_type TEXT,
                status TEXT,
                doc_number TEXT,
                issuing_agency TEXT,
                issue_date TEXT,
                effective_date TEXT,
                expiry_date TEXT,
                crawled_at TEXT,
                updated_at TEXT,
                json_path TEXT,
                markdown_path TEXT,
                content_sha256 TEXT,
                content_bytes INTEGER,
                crawl_status TEXT NOT NULL DEFAULT 'crawled',
                error TEXT,
                metadata_json TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_vbpl_documents_type_status
            ON vbpl_documents(document_type, status)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_vbpl_documents_issue_date
            ON vbpl_documents(issue_date)
            """
        )


def upsert_crawled_document(db_path: str | Path, payload: dict[str, Any]) -> None:
    init_metadata_db(db_path)
    metadata = payload.get("metadata") or {}
    registry_payload = {
        key: value
        for key, value in payload.items()
        if key not in {"content_html", "content_tree", "content_text"}
    }
    document_id = payload.get("document_id") or metadata.get("document_id") or extract_document_id_from_url(
        payload.get("source_url", "")
    )
    if not document_id:
        document_id = hashlib.sha256(payload.get("source_url", "").encode("utf-8")).hexdigest()

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO vbpl_documents (
                document_id, source_url, title, document_type, status, doc_number,
                issuing_agency, issue_date, effective_date, expiry_date, crawled_at,
                updated_at, json_path, markdown_path, content_sha256, content_bytes,
                crawl_status, error, metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'crawled', NULL, ?)
            ON CONFLICT(document_id) DO UPDATE SET
                source_url=excluded.source_url,
                title=excluded.title,
                document_type=excluded.document_type,
                status=excluded.status,
                doc_number=excluded.doc_number,
                issuing_agency=excluded.issuing_agency,
                issue_date=excluded.issue_date,
                effective_date=excluded.effective_date,
                expiry_date=excluded.expiry_date,
                crawled_at=excluded.crawled_at,
                updated_at=excluded.updated_at,
                json_path=excluded.json_path,
                markdown_path=excluded.markdown_path,
                content_sha256=excluded.content_sha256,
                content_bytes=excluded.content_bytes,
                crawl_status='crawled',
                error=NULL,
                metadata_json=excluded.metadata_json
            """,
            (
                document_id,
                payload.get("source_url", ""),
                payload.get("title", ""),
                payload.get("document_type", ""),
                payload.get("status", ""),
                metadata.get("so_ky_hieu", ""),
                metadata.get("co_quan_ban_hanh", ""),
                metadata.get("ngay_ban_hanh", ""),
                metadata.get("ngay_hieu_luc", ""),
                metadata.get("ngay_het_hieu_luc", ""),
                payload.get("crawled_at", ""),
                metadata.get("updated_at", ""),
                payload.get("json_path", ""),
                payload.get("markdown_path") or payload.get("file_path", ""),
                payload.get("content_sha256", ""),
                int(payload.get("content_bytes") or 0),
                json.dumps(registry_payload, ensure_ascii=False),
            ),
        )


def upsert_crawl_error(db_path: str | Path, result: VBPLSearchResult, error: str) -> None:
    init_metadata_db(db_path)
    document_id = result.document_id or extract_document_id_from_url(result.source_url)
    if not document_id:
        document_id = hashlib.sha256(result.source_url.encode("utf-8")).hexdigest()
    now = datetime.now(timezone.utc).isoformat()
    metadata = result.metadata or {}
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO vbpl_documents (
                document_id, source_url, title, document_type, status, doc_number,
                issuing_agency, issue_date, effective_date, expiry_date, crawled_at,
                updated_at, json_path, markdown_path, content_sha256, content_bytes,
                crawl_status, error, metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '', '', '', '', 0, 'error', ?, ?)
            ON CONFLICT(document_id) DO UPDATE SET
                source_url=excluded.source_url,
                title=excluded.title,
                document_type=excluded.document_type,
                status=excluded.status,
                doc_number=excluded.doc_number,
                issuing_agency=excluded.issuing_agency,
                issue_date=excluded.issue_date,
                effective_date=excluded.effective_date,
                expiry_date=excluded.expiry_date,
                crawled_at=excluded.crawled_at,
                crawl_status='error',
                error=excluded.error,
                metadata_json=excluded.metadata_json
            """,
            (
                document_id,
                result.source_url,
                result.title,
                result.document_type,
                result.status,
                metadata.get("so_ky_hieu", ""),
                metadata.get("co_quan_ban_hanh", ""),
                metadata.get("ngay_ban_hanh", ""),
                metadata.get("ngay_hieu_luc", ""),
                metadata.get("ngay_het_hieu_luc", ""),
                now,
                error,
                json.dumps(
                    {
                        "document_id": document_id,
                        "source_url": result.source_url,
                        "title": result.title,
                        "document_type": result.document_type,
                        "status": result.status,
                        "metadata": metadata,
                        "error": error,
                    },
                    ensure_ascii=False,
                ),
            ),
        )


def content_tree_to_markdown(blocks: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for block in blocks:
        block_type = block.get("type")
        if block_type == "heading":
            level = min(max(int(block.get("level", 2)), 1), 6)
            lines.extend([f"{'#' * level} {block.get('text', '')}", ""])
        elif block_type == "table":
            rows = block.get("rows", [])
            if rows:
                width = max(len(row) for row in rows)
                normalized_rows = [row + [""] * (width - len(row)) for row in rows]
                lines.append("| " + " | ".join(normalized_rows[0]) + " |")
                lines.append("| " + " | ".join("---" for _ in range(width)) + " |")
                for row in normalized_rows[1:]:
                    lines.append("| " + " | ".join(row) + " |")
                lines.append("")
        elif block_type == "list":
            for index, item in enumerate(block.get("items", []), start=1):
                prefix = f"{index}. " if block.get("ordered") else "- "
                lines.append(prefix + str(item))
            lines.append("")
        else:
            text = str(block.get("text", "")).strip()
            if text:
                lines.extend([text, ""])
    return lines


def parse_detail_metadata(soup: BeautifulSoup) -> dict[str, str]:
    text = normalize_space(soup.get_text(" ", strip=True))
    metadata = parse_metadata_from_text(text)
    label_map = {
        "Số hiệu": "so_ky_hieu",
        "Số ký hiệu": "so_ky_hieu",
        "Loại văn bản": "document_type",
        "Cơ quan ban hành": "co_quan_ban_hanh",
        "Người ký": "nguoi_ky",
        "Ngày ban hành": "ngay_ban_hanh",
        "Ngày có hiệu lực": "ngay_hieu_luc",
        "Ngày hiệu lực": "ngay_hieu_luc",
        "Ngày hết hiệu lực": "ngay_het_hieu_luc",
        "Tình trạng hiệu lực": "status",
    }
    for label, key in label_map.items():
        match = re.search(rf"{re.escape(label)}\s*:?\s*([^|]+?)(?=\s+[A-ZĐÂĂÊÔƠƯ][^:]+:|$)", text)
        if match:
            metadata[key] = normalize_space(match.group(1))
    return metadata


def metadata_from_api_summary(item: dict[str, Any]) -> dict[str, str]:
    metadata = {
        "so_ky_hieu": normalize_space(item.get("docNum") or ""),
        "co_quan_ban_hanh": normalize_space(item.get("agencyName") or ""),
        "ngay_ban_hanh": format_api_date(item.get("issueDate")),
        "ngay_hieu_luc": format_api_date(item.get("effFrom")),
        "ngay_het_hieu_luc": format_api_date(item.get("effTo")),
    }
    return {key: value for key, value in metadata.items() if value}


def metadata_from_api_detail(data: dict[str, Any]) -> dict[str, str]:
    metadata = metadata_from_api_summary(data)
    metadata.update(
        {
            "nguoi_ky": normalize_space(data.get("signerName") or data.get("signer") or ""),
            "document_id": normalize_space(data.get("id") or ""),
            "updated_at": format_api_date(data.get("updatedDate")),
        }
    )
    doc_type = get_nested_name(data.get("docType"))
    status = get_nested_name(data.get("effStatus"))
    if doc_type:
        metadata["document_type"] = doc_type
    if status:
        metadata["status"] = status
    return {key: value for key, value in metadata.items() if value}


def get_nested_name(value: Any) -> str:
    if isinstance(value, dict):
        return normalize_space(value.get("name") or "")
    return normalize_space(value or "")


def format_api_date(value: Any) -> str:
    if not value:
        return ""
    text = str(value)
    match = re.match(r"(\d{4})-(\d{2})-(\d{2})", text)
    if not match:
        return normalize_space(text)
    year, month, day = match.groups()
    return f"{day}/{month}/{year}"


def parse_metadata_from_text(text: str) -> dict[str, str]:
    metadata: dict[str, str] = {}
    patterns = {
        "ngay_ban_hanh": r"Ngày ban hành\s*:?\s*(\d{1,2}/\d{1,2}/\d{4})|Ban hành\s*:?\s*(\d{1,2}/\d{1,2}/\d{4})",
        "ngay_hieu_luc": r"Ngày hiệu lực\s*:?\s*(\d{1,2}/\d{1,2}/\d{4})|Hiệu lực\s*:?\s*(\d{1,2}/\d{1,2}/\d{4})",
        "ngay_het_hieu_luc": r"Ngày hết hiệu lực\s*:?\s*(\d{1,2}/\d{1,2}/\d{4})",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.I)
        if match:
            metadata[key] = next(group for group in match.groups() if group)

    number_match = re.search(
        r"\b(?:Bộ luật|Luật|Nghị định|Thông tư)\s+(?:số\s+)?([^,\s]+/\d{4}/[A-ZĐ0-9-]+|Không số)",
        text,
        re.I,
    )
    if number_match:
        metadata["so_ky_hieu"] = number_match.group(1)
    return metadata


def infer_status(text: str, statuses: tuple[str, ...] = ALLOWED_STATUSES) -> str:
    lowered = text.lower()
    for status in statuses:
        if status.lower() in lowered:
            return status
    return ""


def infer_document_type(
    text: str,
    document_types: tuple[str, ...] = ALLOWED_DOCUMENT_TYPES,
) -> str:
    normalized = normalize_space(text)
    for document_type in sorted(document_types, key=len, reverse=True):
        if re.search(rf"(^|\s){re.escape(document_type)}(\s|$)", normalized, re.I):
            return document_type
    return ""


def infer_title_from_text(text: str, document_types: tuple[str, ...]) -> str:
    for document_type in document_types:
        match = re.search(rf"{re.escape(document_type)}\s+.+?(?=\s+Trạng thái|\s+Ngày ban hành|$)", text)
        if match:
            return normalize_space(match.group(0))
    return ""


def normalize_detail_url(url: str) -> str:
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    if "ItemID" not in query:
        return url
    path = parsed.path.replace("vbpq-thuoctinh.aspx", "vbpq-toanvan.aspx")
    return parsed._replace(path=path, query=f"ItemID={query['ItemID'][0]}").geturl()


def extract_document_id_from_url(url: str) -> str:
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    if "ItemID" in query:
        return query["ItemID"][0]
    last_part = parsed.path.rstrip("/").split("/")[-1]
    if "--" in last_part:
        return last_part.rsplit("--", 1)[-1]
    if re.fullmatch(r"[0-9a-fA-F-]{16,}|\d+", last_part):
        return last_part
    return ""


def make_public_detail_url(document_id: str, title: str, *, category: str) -> str:
    slug = make_slug(title)
    if slug:
        return f"https://vbpl.vn/van-ban/{category}/{slug}--{document_id}"
    return f"https://vbpl.vn/van-ban/chi-tiet/{document_id}"


def make_slug(title: str, doc_number: str = "") -> str:
    raw = f"{doc_number}-{title}" if doc_number else title
    raw = raw.replace("đ", "d").replace("Đ", "D")
    ascii_text = unicodedata.normalize("NFKD", raw).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", ascii_text.lower()).strip("-")
    return slug[:140] or "vbpl-document"


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def text_contains_keyword(text: str, keyword: str) -> bool:
    normalized_text = normalize_search_text(text)
    normalized_keyword = normalize_search_text(keyword)
    return bool(normalized_keyword and normalized_keyword in normalized_text)


def normalize_search_text(text: str) -> str:
    text = (text or "").replace("đ", "d").replace("Đ", "D")
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return normalize_space(text.lower())


def _dedupe_key(result: VBPLSearchResult) -> str:
    number = result.metadata.get("so_ky_hieu", "")
    return normalize_space(f"{number}|{result.title}").lower() or result.source_url


def _looks_like_legal_text(node: Tag) -> bool:
    text = normalize_space(node.get_text(" ", strip=True))
    legal_markers = ("Điều", "Chương", "Căn cứ", "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM")
    return len(text) >= 50 and any(marker in text for marker in legal_markers)


def parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Crawl labor-related legal documents from vbpl.vn")
    parser.add_argument("--keyword", default="lao động")
    parser.add_argument("--document-types", default=",".join(ALLOWED_DOCUMENT_TYPES))
    parser.add_argument("--statuses", default=",".join(ALLOWED_STATUSES))
    parser.add_argument("--output-dir", default="data/raw/vbpl")
    parser.add_argument(
        "--metadata-db",
        default=None,
        help="SQLite crawl registry path. Defaults to <output-dir>/metadata.sqlite.",
    )
    parser.add_argument("--engine", choices=("requests",), default="requests", help=argparse.SUPPRESS)
    parser.add_argument(
        "--keyword-scope",
        choices=("title", "all"),
        default="title",
        help="Search in document titles by default; use 'all' for the broader VBPL quick search.",
    )
    parser.add_argument(
        "--write-json",
        action="store_true",
        help="Also write metadata JSON files. By default metadata is stored only in SQLite.",
    )
    parser.add_argument(
        "--include-content-json",
        action="store_true",
        help="When --write-json is enabled, include content_html/content_tree/content_text in JSON.",
    )
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--delay-seconds", type=float, default=0.5)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    crawler = VBPLCrawler(
        keyword=args.keyword,
        document_types=parse_csv(args.document_types),
        statuses=parse_csv(args.statuses),
        output_dir=args.output_dir,
        keyword_scope=args.keyword_scope,
        metadata_db=args.metadata_db,
        write_json=args.write_json,
        include_content_json=args.include_content_json,
        delay_seconds=args.delay_seconds,
    )
    written = crawler.crawl(max_docs=args.max_docs)
    print(f"[vbpl] wrote {len(written)} Markdown files")
    return 0
