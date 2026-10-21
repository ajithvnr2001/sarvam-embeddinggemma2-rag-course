---
name: mrl-dimension-optimizer
description: Benchmark Matryoshka embedding dimensions (768, 512, 256, 128) and select the optimal production dimension.
version: 1.0.0
phase: 1
lesson: 3
tags: [mrl, matryoshka, embeddings, optimization, rag]
---

Given a 768-D `google/embeddinggemma-2` matrix and a set of evaluation queries:

1. **Generate Nested Matrices**: Slice prefixes $m \in \lbrace 768, 512, 256, 128 \rbrace$ and enforce $\hat{\mathbf{z}}^{(m)} = \mathbf{z}_{1:m} / \max(\lVert\mathbf{z}_{1:m}\rVert_2, 10^{-12})$.
2. **Audit Unit Norms**: Verify every row of every truncated matrix has $\lVert\hat{\mathbf{z}}^{(m)}\rVert_2 = 1.0 \pm 10^{-5}$.
3. **Measure Rank Retention**: Compute Top-5 Overlap, `MRR@5`, and `NDCG@5` at each dimension relative to the `768-D` baseline.
4. **Select Sweet Spot**: Recommend the lowest dimension $m$ that retains $\ge 95\%$ of `768-D` `MRR@5` (typically `256-D` for cloud vector stores).
