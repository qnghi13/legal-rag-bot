"""Query rewriting chain."""

from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from config.prompts import REPHRASE_SYSTEM_PROMPT


def build_query_rewriter(llm):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", REPHRASE_SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    return prompt | llm | StrOutputParser()

