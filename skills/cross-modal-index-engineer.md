---
name: cross-modal-index-engineer
description: Build and validate unified Text + Image + 16kHz Audio vector indices on S^767.
version: 1.0.0
phase: 3
lesson: 1
tags: [multimodal, audio, vision, embeddinggemma-2, rag]
---

When building or auditing a multimodal RAG index (`multimodal_items.json` + `multimodal_embeddings_768.npy`):

1. **Audio Asset Check**: Verify every `.wav` file is 16,000 Hz, mono (`nchannels == 1`), 16-bit PCM, with non-zero RMS amplitude and no hard clipping.
2. **Image Asset Check**: Verify every `.png` diagram is readable, valid RGB, and served cleanly via `/media/<filename>`.
3. **Shared Hypersphere Verification**: Assert that `multimodal_embeddings_768.npy` has shape `(N, 768)` and every row across all three modalities satisfies $|\lVert\mathbf{v}_i\rVert_2 - 1.0| < 10^{-5}$.
4. **Cross-Modal Recall Test**: Run a text query and verify that semantically matching `text`, `image`, and `audio` items all achieve positive cosine alignment.
