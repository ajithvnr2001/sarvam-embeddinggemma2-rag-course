---
name: rag-preflight-auditor
description: Audit a local or cloud Python workspace for EmbeddingGemma-2, Sarvam corpus, and GCP readiness before running RAG workloads.
version: 1.0.0
phase: 0
lesson: 1
tags: [preflight, environment, rag, gcp, verification]
---

Given a target learning route (`beginner`, `rag-foundations`, `multimodal`, or `gcp-production`) and terminal access, execute the following verification protocol:

1. **Runtime Check**: Verify Python >= 3.10 and ensure `.venv` is isolated from system packages.
2. **Corpus Integrity**: Confirm `data/sarvam_docs/` exists and contains all 64 Markdown files across `api-reference/` and `documentation/`.
3. **Dependency Audit**: Check `numpy`, `torch`, `transformers`, `rank_bm25`, `fastapi`, `google-cloud-bigquery`, and `google-cloud-storage`.
4. **Secret Hygiene**: Verify `.gitignore` excludes `.env` and `*.json` service account keys so credentials can never be committed.
5. **Remediation Output**: Output a structured PASS/FAIL table with exact shell commands for any failed probe.
