---
name: embeddinggemma2-integrator
description: Audit and implement Hugging Face google/embeddinggemma-2 encoding with masked mean pooling and asymmetric prompts.
version: 1.0.0
phase: 1
lesson: 2
tags: [embeddinggemma-2, transformers, pooling, rag]
---

When integrating or reviewing a `google/embeddinggemma-2` embedding module:

1. **Asymmetric Prefix Check**: Ensure queries use `task: search result | query: ` and documents use `title: {title} | text: `.
2. **Batch Invariance Test**: Encode a short 5-word query alone (`batch_size=1`) and alongside a 400-word document (`batch_size=2`). Assert that the query vector is identical ($\cos(\mathbf{q}_{\text{solo}}, \mathbf{q}_{\text{batched}}) > 0.999999$).
3. **Attention Mask Pooling**: Refuse any implementation that uses `outputs.last_hidden_state[:, 0]` (CLS pooling) or unmasked `.mean(dim=1)` on `google/embeddinggemma-2`.
4. **Device & Dtype Safety**: Cast pooled vectors to `float32` before L2 normalization to avoid `bfloat16` underflow.
