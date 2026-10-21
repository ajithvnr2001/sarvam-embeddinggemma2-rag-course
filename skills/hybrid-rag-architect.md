---
name: hybrid-rag-architect
description: Implement and evaluate calibration-free Hybrid Search combining EmbeddingGemma-2 dense vectors and Okapi BM25 via Reciprocal Rank Fusion.
version: 1.0.0
phase: 2
lesson: 3
tags: [hybrid-search, rrf, bm25, embeddinggemma-2, rag]
---

When building or auditing a Hybrid RAG retrieval pipeline:

1. **Refuse Uncalibrated Score Addition**: Flag any code that computes `dense_score + bm25_score` directly without rank transformation.
2. **Enforce Weighted RRF ($k=60$)**: Fuse 1-based dense and lexical ranks via $\text{RRF}(d) = \frac{w_d}{60 + \text{rank}_d(d)} + \frac{w_b}{60 + \text{rank}_b(d)}$.
3. **Preserve Component Telemetry**: Return `rrf_score`, `dense_score`, `bm25_score`, `dense_rank`, and `bm25_rank` on every retrieved item so downstream UIs and evaluators have full observability.
