---
name: corpus-readiness-checker
description: Profile a Markdown documentation directory for RAG ingestion readiness and verify credential safety.
version: 1.0.0
phase: 0
lesson: 2
tags: [corpus, markdown, security, rag, audit]
---

When pointed at a documentation directory (`data/sarvam_docs` or similar) and environment config (`.env`):

1. **Quantify Structure**: Count total documents, words, ATX headings (`#`–`######`), pipe-table rows (`|...|`), and fenced code blocks (` ``` `).
2. **Identify High-Density Files**: List the top 5 files with the highest concentration of tables and headings that require structure-preserving chunking.
3. **Detect Unstructured Outliers**: Flag any document > 800 words with 0 headings.
4. **Verify Secret Hygiene**: Confirm `.env` is ignored by Git and mask all `*TOKEN*`, `*KEY*`, and `*SECRET*` variables in diagnostic output.
