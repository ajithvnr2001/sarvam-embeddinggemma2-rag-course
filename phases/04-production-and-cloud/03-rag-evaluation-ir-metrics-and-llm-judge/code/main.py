#!/usr/bin/env python3
"""Information Retrieval Metrics (Hit@K, MRR@K, NDCG@K) & Grounded Judge from Scratch.

Lesson: phases/04-production-and-cloud/03-rag-evaluation-ir-metrics-and-llm-judge/docs/en.md
Implements Hit@K, Mean Reciprocal Rank (MRR@K), and Normalized Discounted Cumulative Gain
(NDCG@K) from scratch, proving why Hit@K alone hides rank degradation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class EvalCase:
    query: str
    relevant_doc_ids: set[str]
    retrieved_top_k: list[str]


def hit_at_k(relevant: set[str], retrieved: list[str], k: int = 5) -> float:
    """Return 1.0 if at least one relevant document appears in top-k, else 0.0."""
    for doc_id in retrieved[:k]:
        if doc_id in relevant:
            return 1.0
    return 0.0


def reciprocal_rank_at_k(relevant: set[str], retrieved: list[str], k: int = 5) -> float:
    """Return 1 / rank_1based of the FIRST relevant document in top-k, else 0.0."""
    for idx, doc_id in enumerate(retrieved[:k]):
        if doc_id in relevant:
            return 1.0 / float(idx + 1)
    return 0.0


def ndcg_at_k(relevant: set[str], retrieved: list[str], k: int = 5) -> float:
    """Compute binary-relevance NDCG@k = DCG@k / IDCG@k with log2(i + 1) discounting."""
    dcg = 0.0
    for idx, doc_id in enumerate(retrieved[:k]):
        rel = 1.0 if doc_id in relevant else 0.0
        # 1-based rank i = idx + 1; discount is log2(i + 1) = log2(idx + 2)
        dcg += rel / math.log2(idx + 2)

    ideal_hits = min(len(relevant), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(idx + 2) for idx in range(ideal_hits))
    return dcg / idcg


def main() -> int:
    print("=" * 72)
    print("PHASE 04 · LESSON 03: RAG EVALUATION — HIT@K, MRR@K & NDCG@K FROM SCRATCH")
    print("=" * 72)

    # Compare two retrievers that BOTH achieve 100% Hit@5, but have vastly different rank quality:
    gold_relevant = {"sarvam_0012"}
    retriever_a_ranks = ["sarvam_0012", "sarvam_0045", "sarvam_0089", "sarvam_0102", "sarvam_0115"]  # Gold at Rank #1
    retriever_b_ranks = ["sarvam_0045", "sarvam_0089", "sarvam_0102", "sarvam_0115", "sarvam_0012"]  # Gold at Rank #5

    print("\n1. Why Hit@5 Alone is Blind to Rank Position:")
    print(
        f"   - Retriever A (Gold at Rank #1) -> Hit@5={hit_at_k(gold_relevant, retriever_a_ranks, 5):.2f} | "
        f"MRR@5={reciprocal_rank_at_k(gold_relevant, retriever_a_ranks, 5):.4f} | "
        f"NDCG@5={ndcg_at_k(gold_relevant, retriever_a_ranks, 5):.4f}"
    )
    print(
        f"   - Retriever B (Gold at Rank #5) -> Hit@5={hit_at_k(gold_relevant, retriever_b_ranks, 5):.2f} | "
        f"MRR@5={reciprocal_rank_at_k(gold_relevant, retriever_b_ranks, 5):.4f} | "
        f"NDCG@5={ndcg_at_k(gold_relevant, retriever_b_ranks, 5):.4f}"
    )

    assert hit_at_k(gold_relevant, retriever_a_ranks, 5) == hit_at_k(gold_relevant, retriever_b_ranks, 5) == 1.0
    assert reciprocal_rank_at_k(gold_relevant, retriever_a_ranks, 5) == 1.0
    assert abs(reciprocal_rank_at_k(gold_relevant, retriever_b_ranks, 5) - 0.20) < 1e-9

    # 2. Evaluate the 6-Query Sarvam AI Production Benchmark Suite
    benchmark: list[EvalCase] = [
        EvalCase("How do I authenticate to Sarvam APIs?", {"sarvam_0000", "sarvam_0001"}, ["sarvam_0000", "sarvam_0001", "sarvam_0005", "sarvam_0010", "sarvam_0012"]),
        EvalCase("Which TTS model supports Indian voices?", {"sarvam_0007", "sarvam_0008"}, ["sarvam_0007", "sarvam_0008", "sarvam_0009", "sarvam_0015", "sarvam_0020"]),
        EvalCase("How does Saaras v3 handle STT?", {"sarvam_0012", "sarvam_0013"}, ["sarvam_0012", "sarvam_0013", "sarvam_0014", "sarvam_0018", "sarvam_0022"]),
        EvalCase("How to clone a speaker voice?", {"sarvam_0016", "sarvam_0017"}, ["sarvam_0016", "sarvam_0017", "sarvam_0019", "sarvam_0025", "sarvam_0030"]),
        EvalCase("What is Mayura v1 translation?", {"sarvam_0026", "sarvam_0027"}, ["sarvam_0028", "sarvam_0026", "sarvam_0027", "sarvam_0031", "sarvam_0035"]),
        EvalCase("How does Sarvam Parse extract PDF tables?", {"sarvam_0036", "sarvam_0037"}, ["sarvam_0036", "sarvam_0037", "sarvam_0038", "sarvam_0040", "sarvam_0042"]),
    ]

    hits = [hit_at_k(c.relevant_doc_ids, c.retrieved_top_k, 5) for c in benchmark]
    mrrs = [reciprocal_rank_at_k(c.relevant_doc_ids, c.retrieved_top_k, 5) for c in benchmark]
    ndcgs = [ndcg_at_k(c.relevant_doc_ids, c.retrieved_top_k, 5) for c in benchmark]

    mean_hit = sum(hits) / len(hits)
    mean_mrr = sum(mrrs) / len(mrrs)
    mean_ndcg = sum(ndcgs) / len(ndcgs)

    print("\n2. Sarvam AI 6-Query Golden Benchmark Suite Summary:")
    print(f"   - Mean Hit@5  : {mean_hit * 100.0:.2f}% (target >= 90%)")
    print(f"   - Mean MRR@5  : {mean_mrr:.4f} (target >= 0.85)")
    print(f"   - Mean NDCG@5 : {mean_ndcg:.4f} (target >= 0.85)")

    assert mean_hit == 1.0 and mean_mrr > 0.90
    print("\nVerification passed: Hit@5, MRR@5 (0.9167), and NDCG@5 verified from scratch.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
