"""Run Ragas evaluation from the scripts entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import DEFAULT_CONFIG
from src.evaluation.ragas_runner import run_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the Legal RAG pipeline with Ragas",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--llm-model", default=DEFAULT_CONFIG.models.llm_model)
    parser.add_argument("--llm-temperature", type=float, default=0.2)
    parser.add_argument("--embedding-model", default=DEFAULT_CONFIG.models.embedding_model)
    parser.add_argument("--reranker-model", default=DEFAULT_CONFIG.models.reranker_model)
    parser.add_argument("--reranker-max-length", type=int, default=DEFAULT_CONFIG.models.reranker_max_length)
    parser.add_argument("--retrieval-k", type=int, default=DEFAULT_CONFIG.retrieval.retrieval_k)
    parser.add_argument("--rerank-top-k", type=int, default=DEFAULT_CONFIG.retrieval.rerank_top_k)
    parser.add_argument("--rrf-k", type=int, default=DEFAULT_CONFIG.retrieval.rrf_k)
    parser.add_argument("--semantic-weight", type=float, default=DEFAULT_CONFIG.retrieval.semantic_weight)
    parser.add_argument("--bm25-weight", type=float, default=DEFAULT_CONFIG.retrieval.bm25_weight)
    parser.add_argument("--rerank-min-score", type=float, default=DEFAULT_CONFIG.retrieval.rerank_min_score)
    parser.add_argument("--chroma-path", default=None)
    parser.add_argument("--bm25-path", default=None)
    parser.add_argument("--judge-llm", default=DEFAULT_CONFIG.models.judge_llm_model)
    parser.add_argument("--judge-embedding", default=DEFAULT_CONFIG.models.embedding_model)
    parser.add_argument("--judge-temperature", type=float, default=0)
    parser.add_argument("--questions", default=None)
    parser.add_argument("--ground-truths", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_evaluation(
        llm_model=args.llm_model,
        llm_temperature=args.llm_temperature,
        embedding_model=args.embedding_model,
        reranker_model=args.reranker_model,
        reranker_max_length=args.reranker_max_length,
        retrieval_k=args.retrieval_k,
        rerank_top_k=args.rerank_top_k,
        rrf_k=args.rrf_k,
        semantic_weight=args.semantic_weight,
        bm25_weight=args.bm25_weight,
        rerank_min_score=args.rerank_min_score,
        chroma_path=args.chroma_path,
        bm25_path=args.bm25_path,
        judge_llm=args.judge_llm,
        judge_embedding=args.judge_embedding,
        judge_temperature=args.judge_temperature,
        questions=args.questions,
        ground_truths=args.ground_truths,
        output=args.output,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
