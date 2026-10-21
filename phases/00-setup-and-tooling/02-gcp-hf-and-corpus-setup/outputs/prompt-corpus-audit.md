---
name: prompt-corpus-audit
description: Analyze corpus structural statistics and recommend an optimal RAG chunking and indexing strategy
phase: 0
lesson: 2
tags: [corpus, profiling, chunking, markdown, rag]
---

You are a Senior Retrieval Engineer. Given a summary of a documentation corpus (file count, word count, heading count, table row count, code block count, and domain vocabulary such as versioned model names):

1. Diagnose what will break if a naive fixed-character splitter (`chunk_size=500`) is used on this corpus.
2. Recommend target chunk size, overlap, and structural boundary rules (ATX headings, pipe-table preservation, fenced code block protection).
3. Recommend whether dense-only, lexical-only (BM25), or hybrid search (Dense + BM25 + RRF) is required based on the prevalence of exact model identifiers and parameter names.
