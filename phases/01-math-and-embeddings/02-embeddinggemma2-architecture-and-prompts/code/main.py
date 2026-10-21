#!/usr/bin/env python3
"""EmbeddingGemma-2 Architecture, Masked Mean Pooling & Asymmetric Task Prompts from Scratch.

Lesson: phases/01-math-and-embeddings/02-embeddinggemma2-architecture-and-prompts/docs/en.md
Implements attention-mask-aware mean pooling from scratch, demonstrates why averaging over
[PAD] tokens corrupts short query embeddings, and verifies asymmetric prompt formatting.
"""

from __future__ import annotations

import numpy as np


def format_query_prompt(query: str) -> str:
    """Official google/embeddinggemma-2 asymmetric query prompt."""
    return f"task: search result | query: {query.strip()}"


def format_document_prompt(title: str, content: str) -> str:
    """Official google/embeddinggemma-2 asymmetric document prompt."""
    clean_title = title.strip() or "Sarvam AI Documentation"
    return f"title: {clean_title} | text: {content.strip()}"


def naive_unmasked_mean_pool(hidden_states: np.ndarray) -> np.ndarray:
    """BROKEN: Averages across all sequence positions including [PAD] tokens."""
    pooled = hidden_states.mean(axis=1)
    norms = np.linalg.norm(pooled, axis=1, keepdims=True)
    return pooled / np.maximum(norms, 1e-12)


def masked_mean_pool(hidden_states: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """CORRECT: Averages only over real tokens (m_t = 1) and L2-normalizes."""
    # hidden_states: (B, T, D), attention_mask: (B, T)
    mask_expanded = attention_mask[:, :, None].astype(np.float64)  # (B, T, 1)
    summed = np.sum(hidden_states * mask_expanded, axis=1)  # (B, D)
    token_counts = np.maximum(np.sum(mask_expanded, axis=1), 1e-9)  # (B, 1)
    pooled = summed / token_counts
    norms = np.linalg.norm(pooled, axis=1, keepdims=True)
    return pooled / np.maximum(norms, 1e-12)


def main() -> int:
    print("=" * 72)
    print("PHASE 01 · LESSON 02: EMBEDDINGGEMMA-2 POOLING & ASYMMETRIC PROMPTS")
    print("=" * 72)

    q_raw = "What is the speaker parameter for Bulbul v3 TTS?"
    d_title = "Text to Speech — Bulbul v3 REST API"
    d_body = "Pass `speaker: 'shubh'` and `model: 'bulbul:v3'` to synthesize natural Indian voices."

    q_formatted = format_query_prompt(q_raw)
    d_formatted = format_document_prompt(d_title, d_body)

    print("\n1. Asymmetric Instruction Prompt Templates (`google/embeddinggemma-2`):")
    print(f"   - Query Template   : {q_formatted}")
    print(f"   - Document Template: {d_formatted}")

    assert q_formatted.startswith("task: search result | query: ")
    assert d_formatted.startswith("title: ") and " | text: " in d_formatted

    # 2. Prove why Masked Mean Pooling is essential when batching short queries with long docs
    rng = np.random.default_rng(42)
    dim = 16
    # Suppose real tokens of a 4-token query have clean semantic signal,
    # while 12 [PAD] positions carry a non-zero residual bias vector.
    true_token_states = rng.normal(loc=0.5, scale=0.2, size=(4, dim))
    pad_bias_state = np.full((12, dim), -0.45)

    # Unpadded batch (length = 4, all mask = 1)
    unpadded_batch = true_token_states[None, :, :]  # (1, 4, D)
    unpadded_mask = np.ones((1, 4), dtype=np.int64)

    # Padded batch (length = 16: 4 real tokens + 12 PAD tokens)
    padded_batch = np.concatenate([true_token_states, pad_bias_state], axis=0)[None, :, :]  # (1, 16, D)
    padded_mask = np.array([[1, 1, 1, 1] + [0] * 12], dtype=np.int64)

    gold_vec = masked_mean_pool(unpadded_batch, unpadded_mask)[0]
    naive_padded_vec = naive_unmasked_mean_pool(padded_batch)[0]
    masked_padded_vec = masked_mean_pool(padded_batch, padded_mask)[0]

    sim_naive = float(np.dot(gold_vec, naive_padded_vec))
    sim_masked = float(np.dot(gold_vec, masked_padded_vec))

    print("\n2. Padding Contamination Experiment (4 Real Tokens Padded to 16):")
    print(f"   - [BROKEN] Unmasked Mean Pool Cosine vs. Unpadded Gold : {sim_naive:.6f} (Corrupted by PAD tokens!)")
    print(f"   - [FIXED ] Masked Mean Pool Cosine   vs. Unpadded Gold : {sim_masked:.6f} (Exact 1.000000 match!)")

    assert abs(sim_masked - 1.0) < 1e-12, "Masked mean pooling must be 100% invariant to right-padding"
    assert sim_naive < 0.0, "Expected unmasked pooling to be severely distorted by pad bias"

    print("\nVerification passed: Asymmetric prompts and masked mean pooling verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
