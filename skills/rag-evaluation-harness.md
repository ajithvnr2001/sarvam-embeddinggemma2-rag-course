---
name: rag-evaluation-harness
description: Execute retrieval (Hit@K, MRR@K, NDCG@K) and generation faithfulness regression gates on a RAG pipeline.
version: 1.0.0
phase: 4
lesson: 3
tags: [evaluation, mrr, ndcg, ci-cd, rag]
---

When evaluating a RAG index or pull request:

1. **Run Retrieval Benchmark**: Compute `Hit@5`, `MRR@5`, and `NDCG@5` across the golden query suite.
2. **Run MRL Retention Audit**: Verify that `256-D` MRL vectors retain $\ge 95\%$ of `768-D` top-5 overlap.
3. **Run Citation Guardrail Check**: Verify zero phantom `[Doc i]` references across all generated answers.
4. **Enforce Regression Gate**: Fail the check if `Hit@5 < 0.90` or `MRR@5 < 0.85`.
