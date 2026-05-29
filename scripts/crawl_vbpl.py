"""CLI entry point for crawling labor-related legal documents from vbpl.vn."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.vbpl_crawler import main


if __name__ == "__main__":
    raise SystemExit(main())
