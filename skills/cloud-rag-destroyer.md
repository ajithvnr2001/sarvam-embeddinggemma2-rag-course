---
name: cloud-rag-destroyer
description: Execute and verify idempotent Google Cloud resource teardown (BigQuery + GCS) down to $0.00 ongoing cost.
version: 1.0.0
phase: 4
lesson: 4
tags: [destroy, teardown, finops, gcp, bigquery, gcs]
---

When asked to clean up or tear down cloud RAG resources:

1. **Pre-Destroy Inventory**: Run `python -m cli.rag_cli status` to list active GCP resources and local index files.
2. **Execute Idempotent Destroy**: Run `python -m cli.rag_cli destroy --yes` (omit `--include-local` unless the user explicitly asks to wipe local `.npy` files).
3. **Independent Verification**: Execute `gcloud storage ls gs://<BUCKET>` and `bq show <PROJECT>:<DATASET>` and confirm both return `NotFound` (`404`).
4. **Local Continuity Check**: Run a quick local search (`python -m cli.rag_cli search "Saaras v3" --mode hybrid`) to confirm offline retrieval still works seamlessly.
