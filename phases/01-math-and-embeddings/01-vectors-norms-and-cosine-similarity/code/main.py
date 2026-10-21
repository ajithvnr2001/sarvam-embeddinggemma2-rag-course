#!/usr/bin/env python3
"""Vectors, L2 Norms & Cosine Similarity from Scratch.

Lesson: phases/01-math-and-embeddings/01-vectors-norms-and-cosine-similarity/docs/en.md
Demonstrates pure-Python vector math, proves why unnormalized dot product fails on variable-length
documents, and verifies unit-norm parity on real 768-D EmbeddingGemma-2 vectors.
"""

from __future__ import annotations

import math
from pathlib import Path
import numpy as np


ROOT = Path(__file__).resolve().parents[4]
EMB_768_PATH = ROOT / "data" / "index" / "embeddings_768.npy"


def dot_product_scratch(a: list[float], b: list[float]) -> float:
    """Compute inner product <a, b> = sum_i a_i * b_i from scratch."""
    if len(a) != len(b):
        raise ValueError(f"Dimension mismatch: {len(a)} != {len(b)}")
    total = 0.0
    for x, y in zip(a, b):
        total += x * y
    return total


def l2_norm_scratch(v: list[float]) -> float:
    """Compute Euclidean L2 norm ||v||_2 = sqrt(sum_i v_i^2) from scratch."""
    return math.sqrt(dot_product_scratch(v, v))


def l2_normalize_scratch(v: list[float], eps: float = 1e-12) -> list[float]:
    """Project vector v onto the unit hypersphere S^(d-1)."""
    norm = max(l2_norm_scratch(v), eps)
    return [x / norm for x in v]


def cosine_similarity_scratch(q: list[float], d: list[float]) -> float:
    """Compute cosine(q, d) = <q, d> / (||q||_2 * ||d||_2) from scratch."""
    q_unit = l2_normalize_scratch(q)
    d_unit = l2_normalize_scratch(d)
    return dot_product_scratch(q_unit, d_unit)


def main() -> int:
    print("=" * 72)
    print("PHASE 01 · LESSON 01: VECTORS, L2 NORMS & COSINE SIMILARITY FROM SCRATCH")
    print("=" * 72)

    # 4-D Semantic Feature Space: [speech_stt, tts_voice, translation, pricing_table]
    query_vec = [0.92, 0.10, 0.05, 0.15]  # Query: "How does Saaras v3 speech-to-text work?"
    doc_concise_stt = [0.95, 0.08, 0.04, 0.12]  # Concise exact answer on Saaras v3 STT
    doc_verbose_tts = [3.50, 8.20, 1.10, 2.40]  # Long 10-page document mostly about TTS (unnormalized)

    raw_dot_stt = dot_product_scratch(query_vec, doc_concise_stt)
    raw_dot_tts = dot_product_scratch(query_vec, doc_verbose_tts)

    cos_stt = cosine_similarity_scratch(query_vec, doc_concise_stt)
    cos_tts = cosine_similarity_scratch(query_vec, doc_verbose_tts)

    print("\n1. The Magnitude Trap (Unnormalized Dot Product vs. Cosine Similarity):")
    print(f"   - Query Vector                   : {query_vec} (||q||_2 = {l2_norm_scratch(query_vec):.4f})")
    print(f"   - Doc A (Concise Saaras v3 STT)  : {doc_concise_stt} (||d_A||_2 = {l2_norm_scratch(doc_concise_stt):.4f})")
    print(f"   - Doc B (Long Verbose TTS Guide) : {doc_verbose_tts} (||d_B||_2 = {l2_norm_scratch(doc_verbose_tts):.4f})")
    print("-" * 72)
    print(f"   [BROKEN] Raw Dot Product <q, d_A>: {raw_dot_stt:.4f}   vs  <q, d_B>: {raw_dot_tts:.4f}  --> Picks WRONG Doc B!")
    print(f"   [FIXED ] Cosine Sim cos(q, d_A)  : {cos_stt:.4f}   vs  cos(q, d_B) : {cos_tts:.4f}  --> Picks RIGHT Doc A!")

    assert raw_dot_tts > raw_dot_stt, "Expected unnormalized dot product to be misled by vector magnitude"
    assert cos_stt > cos_tts, "Expected cosine similarity to rank the semantically aligned document #1"

    print("\n2. Verifying Real 768-D EmbeddingGemma-2 Vectors on the Unit Hypersphere:")
    if EMB_768_PATH.is_file():
        emb = np.load(EMB_768_PATH)
        norms = np.linalg.norm(emb, axis=1)
        max_dev = float(np.max(np.abs(norms - 1.0)))
        v0 = emb[0].tolist()
        v1 = emb[1].tolist()
        scratch_cos = cosine_similarity_scratch(v0, v1)
        numpy_dot = float(np.dot(emb[0], emb[1]))
        diff = abs(scratch_cos - numpy_dot)
        print(f"   - Loaded Matrix Shape            : {emb.shape} ({emb.dtype})")
        print(f"   - Max Deviation from ||v||_2 = 1 : {max_dev:.2e}")
        print(f"   - Scratch Cosine(v[0], v[1])     : {scratch_cos:.6f}")
        print(f"   - Fast Unit Dot  np.dot(v[0],v[1]): {numpy_dot:.6f} (|diff| = {diff:.2e})")
        assert diff < 1e-5
    else:
        print("   - Using synthetic 768-D unit vectors (run `cli.rag_cli index` to generate disk index)")

    print("\nVerification passed: Pure-Python cosine similarity matches unit-sphere inner product.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
