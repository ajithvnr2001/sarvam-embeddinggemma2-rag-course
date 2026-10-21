---
name: citation-guardrail-auditor
description: Deterministically audit LLM-generated RAG answers for phantom citations, uncited sentences, and numerical hallucinations.
version: 1.0.0
phase: 4
lesson: 1
tags: [guardrails, citations, hallucination, vertex-ai, rag]
---

Given a generated RAG answer $Y$ and the list of $K$ retrieved context chunks `[Doc 1]..[Doc K]`:

1. **Bounds Check**: Parse all `\[Doc\s+(\d+)\]` markers and reject the response if any $i \notin \lbrace 1, \dots, K \rbrace$.
2. **Sentence Coverage Check**: Split $Y$ into declarative sentences and flag any factual sentence that lacks a `[Doc i]` citation (excluding explicit abstention statements).
3. **Identifier & Number Check**: Extract all backticked code identifiers (`` `...` ``) and numeric literals from $Y$ and verify each appears in the text of the cited chunk `c_i`.
