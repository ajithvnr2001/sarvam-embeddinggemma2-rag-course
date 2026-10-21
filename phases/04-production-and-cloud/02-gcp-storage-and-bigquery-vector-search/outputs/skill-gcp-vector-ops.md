---
name: gcp-vector-ops
description: Validate BigQuery vector JSONL exports, verify REPEATED FLOAT64 array integrity, and audit COSINE_DISTANCE SQL queries.
version: 1.0.0
phase: 4
lesson: 2
tags: [gcp, bigquery, gcs, vector-search, rag]
---

When preparing or auditing a Google Cloud BigQuery vector deployment:

1. **JSONL Vector Integrity**: Verify that every row in `bq_vectors_256d.jsonl` has a non-empty `embedding` list of exact length $d$ (`256`), with zero `NaN` or `null` entries, and unit norm $\lVert\mathbf{v}\rVert_2 = 1.0 \pm 10^{-4}$.
2. **Sort Order Guard**: Inspect all SQL queries using `COSINE_DISTANCE` or `EUCLIDEAN_DISTANCE` and refuse any query that sorts `DESC` instead of `ASC`.
3. **Teardown Readiness**: Ensure every provisioned GCS bucket and BigQuery dataset is registered in `.env` so `python -m cli.rag_cli destroy --yes` can remove 100% of cloud resources cleanly.
