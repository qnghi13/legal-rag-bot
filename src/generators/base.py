"""Generator interfaces."""

from __future__ import annotations

from typing import Protocol

from langchain_core.documents import Document


class BaseGenerator(Protocol):
    def generate_answer(self, query: str, context: list[Document]) -> str:
        """Generate an answer from a query and retrieved documents."""

