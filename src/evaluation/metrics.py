"""Ragas metric selection."""

from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

DEFAULT_RAGAS_METRICS = [
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
]

