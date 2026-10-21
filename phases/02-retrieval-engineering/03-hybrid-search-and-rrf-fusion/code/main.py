#!/usr/bin/env python3
"""Hybrid Search & Reciprocal Rank Fusion (RRF) vs. Raw Score Addition from Scratch.

Lesson: phases/02-retrieval-engineering/03-hybrid-search-and-rrf-fusion/docs/en.md
Proves why adding raw BM25 scores to cosine similarity fails due to scale mismatch,
and implements calibration-free Reciprocal Rank Fusion (RRF, k=60) from scratch.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CandidateDoc:
    doc_id: str
    title: str
    cosine_score: float  # Bounded in [-1, 1]
    bm25_score: float    # Unbounded in [0, inf)


def argsort_descending(values: list[float]) -> list[int]:
    """Return 0-based indices sorted from highest score to lowest score."""
    return sorted(range(len(values)), key=lambda i: values[i], reverse=True)


def compute_1based_ranks(scores: list[float]) -> list[int]:
    """Convert raw scores into 1-based ordinal ranks (1 = best)."""
    order = argsort_descending(scores)
    ranks = [0] * len(scores)
    for pos, doc_idx in enumerate(order):
        ranks[doc_idx] = pos + 1
    return ranks


def broken_raw_score_sum(candidates: list[CandidateDoc], alpha: float = 0.5) -> list[float]:
    """BROKEN: Directly adds uncalibrated cosine (0..1) and BM25 (0..25+) scores."""
    return [alpha * c.cosine_score + (1.0 - alpha) * c.bm25_score for c in candidates]


def reciprocal_rank_fusion(
    candidates: list[CandidateDoc],
    k: int = 60,
    w_dense: float = 0.65,
    w_bm25: float = 0.35,
) -> list[float]:
    """CORRECT: Rank-based fusion RRF(d) = w_dense/(k + rank_dense(d)) + w_bm25/(k + rank_bm25(d))."""
    dense_ranks = compute_1based_ranks([c.cosine_score for c in candidates])
    bm25_ranks = compute_1based_ranks([c.bm25_score for c in candidates])

    rrf_scores: list[float] = []
    for r_d, r_b in zip(dense_ranks, bm25_ranks):
        score = (w_dense / float(k + r_d)) + (w_bm25 / float(k + r_b))
        rrf_scores.append(score)
    return rrf_scores


def main() -> int:
    print("=" * 72)
    print("PHASE 02 · LESSON 03: HYBRID SEARCH & RECIPROCAL RANK FUSION (RRF)")
    print("=" * 72)

    # Query: "How does Saaras v3 handle noisy background audio in Tamil?"
    candidates = [
        CandidateDoc(
            doc_id="sarvam_0012",
            title="Saaras v3 Speech-to-Text — Noise Robustness & Tamil (Exact Match)",
            cosine_score=0.86,  # Rank #1 in Dense, Rank #2 in BM25
            bm25_score=9.40,
        ),
        CandidateDoc(
            doc_id="sarvam_0108",
            title="Changelog Archive — Mentions 'audio' and 'background' 18 times",
            cosine_score=0.31,  # Rank #4 in Dense (semantically off-topic), Rank #1 in BM25 (outlier)
            bm25_score=14.80,
        ),
        CandidateDoc(
            doc_id="sarvam_0045",
            title="Voice Agents — Handling Background Noise in Calls",
            cosine_score=0.81,  # Rank #2 in Dense, Rank #3 in BM25
            bm25_score=6.20,
        ),
        CandidateDoc(
            doc_id="sarvam_0089",
            title="Tamil API Guide — Speech & Translation Overview",
            cosine_score=0.77,  # Rank #3 in Dense, Rank #4 in BM25
            bm25_score=5.10,
        ),
    ]

    raw_sums = broken_raw_score_sum(candidates, alpha=0.5)
    rrf_scores = reciprocal_rank_fusion(candidates, k=60, w_dense=0.65, w_bm25=0.35)

    dense_ranks = compute_1based_ranks([c.cosine_score for c in candidates])
    bm25_ranks = compute_1based_ranks([c.bm25_score for c in candidates])

    print("\n1. Scale Mismatch Comparison (Cosine in [-1, 1] vs. BM25 in [0, 25+]):")
    for idx, c in enumerate(candidates):
        print(
            f"   - {c.doc_id} | Cos={c.cosine_score:.2f} (R#{dense_ranks[idx]}) | "
            f"BM25={c.bm25_score:>5.2f} (R#{bm25_ranks[idx]}) | "
            f"RawSum={raw_sums[idx]:>5.2f} | RRF={rrf_scores[idx]:.6f} | {c.title[:38]}..."
        )

    best_raw_idx = argsort_descending(raw_sums)[0]
    best_rrf_idx = argsort_descending(rrf_scores)[0]

    print("-" * 72)
    print(f"   [BROKEN] Raw Score Addition Winner -> {candidates[best_raw_idx].doc_id} ({candidates[best_raw_idx].title})")
    print(f"   [FIXED ] Reciprocal Rank Fusion #1 -> {candidates[best_rrf_idx].doc_id} ({candidates[best_rrf_idx].title})")

    assert candidates[best_raw_idx].doc_id == "sarvam_0108", "Expected raw sum to be hijacked by BM25 scale outlier"
    assert candidates[best_rrf_idx].doc_id == "sarvam_0012", "Expected RRF to rank the consensus #1/#2 document first"

    print("\nVerification passed: RRF (k=60) eliminated scale mismatch and ranked consensus document #1.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
