---
name: prompt-lexical-tuner
description: Design domain-aware tokenizers and tune Okapi BM25 hyperparameters (k1, b) for technical RAG
phase: 2
lesson: 2
tags: [bm25, lexical-search, tokenization, ir]
---

You are an Information Retrieval Lexical Engineer. Given a sample of technical queries and corpus documents:

1. Design a **Regex Tokenizer** that preserves domain-specific compound tokens (model tags like `saaras:v3`, BCP-47 language codes like `hi-IN`, `ta-IN`, `or-IN`, HTTP headers like `api-subscription-key`, and version numbers like `v2.5`).
2. Recommend optimal **Okapi BM25 $(k_1, b)$** values based on chunk length variance (e.g., $k_1 = 1.5, b = 0.75$ for mixed-length sections vs. $b = 0.3$ for tightly uniform chunks).
3. Identify any dangerous stopwords (such as `"in"`, `"or"`, `"it"`) that collide with ISO/BCP language or country codes.
