---
name: bm25-index-builder
description: Build and audit technical Okapi BM25 indices that preserve API identifiers, version codes, and locale tags.
version: 1.0.0
phase: 2
lesson: 2
tags: [bm25, tokenizer, lexical-search, rag]
---

When building or auditing a BM25 lexical index for developer documentation:

1. **Identifier Preservation Check**: Test tokenization on `["saaras:v3", "saarika:v2.5", "bulbul:v3", "api-subscription-key", "hi-IN", "or-IN"]` and verify each remains a single unbroken token.
2. **Stopword Collision Guard**: Never apply generic English stopword lists that strip `"or"` (`or-IN` Odia) or `"in"` (`*-IN` India locales).
3. **IDF Sanity Report**: Print the top-10 highest-IDF and bottom-10 lowest-IDF terms across the corpus to confirm discriminative weighting.
