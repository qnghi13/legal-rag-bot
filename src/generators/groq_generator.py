"""Groq LLM factory."""

from __future__ import annotations

from langchain_groq import ChatGroq


def build_groq_llm(model: str, temperature: float = 0.2) -> ChatGroq:
    return ChatGroq(model=model, temperature=temperature)

