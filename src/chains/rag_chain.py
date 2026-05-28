"""Main Legal RAG chain factory."""

from __future__ import annotations

from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from config.prompts import NO_CONTEXT_ANSWER, QA_SYSTEM_PROMPT
from config.settings import DEFAULT_CONFIG
from src.chains.query_rewriter import build_query_rewriter
from src.embeddings.embedding_model import get_embedding_model
from src.generators.groq_generator import build_groq_llm
from src.rerankers.cross_encoder_reranker import CrossEncoderReranker
from src.retrievers.bm25_retriever import BM25RetrieverAdapter, load_bm25_retriever
from src.retrievers.hybrid_retriever import HybridRetriever
from src.retrievers.semantic_retriever import SemanticRetriever
from src.vectorstores.chroma_store import load_chroma_store


def get_hr_bot(
    *,
    llm_model: str = DEFAULT_CONFIG.models.llm_model,
    llm_temperature: float = 0.2,
    embedding_model: str = DEFAULT_CONFIG.models.embedding_model,
    reranker_model: str = DEFAULT_CONFIG.models.reranker_model,
    reranker_max_length: int = DEFAULT_CONFIG.models.reranker_max_length,
    retrieval_k: int = DEFAULT_CONFIG.retrieval.retrieval_k,
    rerank_top_k: int = DEFAULT_CONFIG.retrieval.rerank_top_k,
    chroma_path: str | None = None,
    bm25_path: str | None = None,
    return_context_list: bool = False,
):
    """Build the LCEL RAG chain used by the UI and evaluation."""

    chroma_dir = Path(chroma_path) if chroma_path else DEFAULT_CONFIG.paths.chroma_dir
    bm25_file = Path(bm25_path) if bm25_path else DEFAULT_CONFIG.paths.bm25_path

    print(f"  [bot] LLM: {llm_model} (temp={llm_temperature})")
    llm = build_groq_llm(model=llm_model, temperature=llm_temperature)

    print(f"  [bot] Embedding: {embedding_model}")
    embedding = get_embedding_model(embedding_model)

    print(f"  [bot] Chroma DB: {chroma_dir}")
    vectorstore = load_chroma_store(chroma_dir, embedding)

    print(f"  [bot] Retriever: k={retrieval_k}")
    semantic = SemanticRetriever(vectorstore, search_k=retrieval_k)
    keyword = BM25RetrieverAdapter(load_bm25_retriever(bm25_file))
    retriever = HybridRetriever(semantic=semantic, keyword=keyword)

    print(f"  [bot] Reranker: {reranker_model}")
    reranker = CrossEncoderReranker(reranker_model, max_length=reranker_max_length)

    rephrase_step = RunnablePassthrough.assign(
        _rephrased=build_query_rewriter(llm)
    )
    retrieval_step = RunnableLambda(
        lambda state: state | _retrieve_context(
            state["_rephrased"],
            retriever=retriever,
            reranker=reranker,
            retrieval_k=retrieval_k,
            rerank_top_k=rerank_top_k,
            return_context_list=return_context_list,
        )
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", QA_SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    return rephrase_step | retrieval_step | RunnablePassthrough.assign(
        answer=_build_answer_step(qa_prompt, llm)
    )


def _retrieve_context(
    query: str,
    *,
    retriever,
    reranker,
    retrieval_k: int,
    rerank_top_k: int,
    return_context_list: bool,
) -> dict:
    docs = retriever.retrieve(query, retrieval_k)
    top_docs = reranker.rerank(query, docs, rerank_top_k)
    if not top_docs:
        result = {"context": ""}
        if return_context_list:
            result["context_list"] = []
        return result

    formatted_docs = [_format_document(doc, i) for i, doc in enumerate(top_docs, start=1)]
    result = {"context": "\n\n".join(formatted_docs)}
    if return_context_list:
        result["context_list"] = formatted_docs
    return result


def _format_document(doc, index: int) -> str:
    source = doc.metadata.get("source", "Tai lieu noi bo")
    return (
        f"--- TAI LIEU SO {index} ---\n"
        f"[Nguon]: {source}\n"
        f"[Noi dung]: {doc.page_content}"
    )


def _build_answer_step(prompt, llm):
    def _invoke(state: dict) -> str:
        context = (state.get("context") or "").strip()
        if not context:
            return NO_CONTEXT_ANSWER
        return (prompt | llm | StrOutputParser()).invoke(state)

    return RunnableLambda(_invoke)

