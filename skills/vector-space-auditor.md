---
name: vector-space-auditor
description: Audit an embedding matrix (.npy or JSONL) for unit-norm compliance, numerical stability, and anisotropy.
version: 1.0.0
phase: 1
lesson: 1
tags: [embeddings, cosine-similarity, l2-norm, audit]
---

Given a path to a saved embedding matrix (`.npy` or `.jsonl`):

1. **Shape & Dtype Check**: Report $(N, d)$ and verify `float32` or `float64` precision.
2. **Unit Hypersphere Check**: Compute $\lVert\mathbf{v}_i\rVert_2$ for all $i \in \{1..N\}$ and assert $\max_i |\lVert\mathbf{v}_i\rVert_2 - 1.0| < 10^{-5}$.
3. **NaN / Inf / Zero Guard**: Fail immediately if any coordinate is `NaN`, `Inf`, or if any row norm is $< 10^{-8}$.
4. **Isotropy & Near-Duplicate Report**: Compute the mean off-diagonal cosine similarity and flag any pair $(i, j)$ with $\cos(\mathbf{v}_i, \mathbf{v}_j) > 0.995$ as a near-duplicate chunk.
