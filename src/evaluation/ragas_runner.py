"""Ragas evaluation runner for the Legal RAG pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Sequence

from datasets import Dataset
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.run_config import RunConfig

from config.settings import DEFAULT_CONFIG
from src.chains.rag_chain import get_hr_bot
from src.evaluation.datasets import DEFAULT_GROUND_TRUTHS, DEFAULT_QUESTIONS
from src.evaluation.metrics import DEFAULT_RAGAS_METRICS


@dataclass
class EvaluationRunner:
    llm_model: str = DEFAULT_CONFIG.models.llm_model
    llm_temperature: float = 0.2
    embedding_model: str = DEFAULT_CONFIG.models.embedding_model
    reranker_model: str = DEFAULT_CONFIG.models.reranker_model
    reranker_max_length: int = DEFAULT_CONFIG.models.reranker_max_length
    retrieval_k: int = DEFAULT_CONFIG.retrieval.retrieval_k
    rerank_top_k: int = DEFAULT_CONFIG.retrieval.rerank_top_k
    rrf_k: int = DEFAULT_CONFIG.retrieval.rrf_k
    semantic_weight: float = DEFAULT_CONFIG.retrieval.semantic_weight
    bm25_weight: float = DEFAULT_CONFIG.retrieval.bm25_weight
    rerank_min_score: float | None = DEFAULT_CONFIG.retrieval.rerank_min_score
    chroma_path: str | None = None
    bm25_path: str | None = None
    judge_llm: str = DEFAULT_CONFIG.models.judge_llm_model
    judge_embedding: str = DEFAULT_CONFIG.models.embedding_model
    judge_temperature: float = 0
    timeout: int = 120
    max_workers: int = 2
    max_retries: int = 10

    def run(
        self,
        *,
        questions: Sequence[str] | str | None = None,
        ground_truths: Sequence[str] | str | None = None,
        output: str | None = None,
        verbose: bool = False,
    ):
        questions_list = _coerce_list(questions) or DEFAULT_QUESTIONS
        ground_truths_list = _coerce_list(ground_truths) or DEFAULT_GROUND_TRUTHS

        if len(questions_list) != len(ground_truths_list):
            raise ValueError(
                "questions and ground_truths must have the same length "
                f"({len(questions_list)} != {len(ground_truths_list)})"
            )

        print(f"[eval] Questions: {len(questions_list)}")
        if verbose:
            for index, question in enumerate(questions_list, start=1):
                print(f"  {index}. {question[:100]}")

        rag_chain = get_hr_bot(
            llm_model=self.llm_model,
            llm_temperature=self.llm_temperature,
            embedding_model=self.embedding_model,
            reranker_model=self.reranker_model,
            reranker_max_length=self.reranker_max_length,
            retrieval_k=self.retrieval_k,
            rerank_top_k=self.rerank_top_k,
            rrf_k=self.rrf_k,
            semantic_weight=self.semantic_weight,
            bm25_weight=self.bm25_weight,
            rerank_min_score=self.rerank_min_score,
            chroma_path=self.chroma_path,
            bm25_path=self.bm25_path,
            return_context_list=True,
        )

        answers: list[str] = []
        contexts: list[list[str]] = []
        for index, question in enumerate(questions_list, start=1):
            print(f"[eval] Running {index}/{len(questions_list)}")
            response = rag_chain.invoke({"input": question, "chat_history": []})
            answers.append(response["answer"])
            contexts.append(response.get("dq_context_list", []))

        eval_data = {
            "question": questions_list,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths_list,
        }
        valid_indices = [
            index
            for index, truth in enumerate(ground_truths_list)
            if truth and not truth.startswith("Không trả lời được")
        ]
        eval_data = {
            key: [values[index] for index in valid_indices]
            for key, values in eval_data.items()
        }
        if not eval_data["question"]:
            raise ValueError("No valid evaluation questions after filtering ground truths.")

        judge_llm = ChatGroq(model=self.judge_llm, temperature=self.judge_temperature)
        judge_embedding = HuggingFaceEmbeddings(model_name=self.judge_embedding)
        result = evaluate(
            Dataset.from_dict(eval_data),
            metrics=DEFAULT_RAGAS_METRICS,
            llm=judge_llm,
            embeddings=judge_embedding,
            run_config=RunConfig(
                timeout=self.timeout,
                max_workers=self.max_workers,
                max_retries=self.max_retries,
            ),
            raise_exceptions=False,
        )

        df_result = result.to_pandas()
        _print_summary(df_result)
        if output:
            df_result.to_csv(output, index=False, encoding="utf-8-sig")
            print(f"[eval] Saved results to: {output}")
        return df_result


def run_evaluation(**kwargs):
    runner_kwargs = {
        key: kwargs.pop(key)
        for key in list(kwargs.keys())
        if key in EvaluationRunner.__dataclass_fields__
    }
    return EvaluationRunner(**runner_kwargs).run(**kwargs)


def _coerce_list(value: Sequence[str] | str | None) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return json.loads(value)
    return list(value)


def _print_summary(df_result) -> None:
    print("\n[eval] Ragas metrics")
    labels = {
        "faithfulness": "Faithfulness",
        "answer_relevancy": "Answer relevancy",
        "context_precision": "Context precision",
        "context_recall": "Context recall",
    }
    for column, label in labels.items():
        if column in df_result.columns:
            print(f"  - {label}: {df_result[column].mean():.4f}")
    print(f"  - Evaluated questions: {len(df_result)}")
