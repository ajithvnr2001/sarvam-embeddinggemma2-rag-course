#!/usr/bin/env python3
"""Okapi BM25 & Technical Tokenizer from Scratch.

Lesson: phases/02-retrieval-engineering/02-okapi-bm25-from-scratch/docs/en.md
Implements a technical-identifier-preserving tokenizer, Robertson-Sparck Jones IDF,
and Okapi BM25 (k1=1.5, b=0.75) from scratch, proving why BM25 beats raw TF on
Sarvam AI model codes (`saaras:v3`, `bulbul:v3`, `api-subscription-key`).
"""

from __future__ import annotations

import math
import re
from collections import Counter


def tokenize_technical(text: str) -> list[str]:
    """Tokenize while keeping colons, hyphens, and dots inside model codes (`saaras:v3`, `hi-in`)."""
    return re.findall(r"[a-z0-9_]+(?:[:.-][a-z0-9_]+)*", text.lower())


class OkapiBM25Scratch:
    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.tokenized_docs = [tokenize_technical(doc) for doc in documents]
        self.n_docs = len(self.tokenized_docs)
        self.doc_lens = [len(tokens) for tokens in self.tokenized_docs]
        self.avgdl = sum(self.doc_lens) / max(self.n_docs, 1)
        self.tf_maps = [Counter(tokens) for tokens in self.tokenized_docs]

        # Document frequency df(t): number of documents containing term t
        df_counts: Counter[str] = Counter()
        for tf_map in self.tf_maps:
            for term in tf_map.keys():
                df_counts[term] += 1

        # Robertson-Sparck Jones IDF: log(1 + (N - df + 0.5) / (df + 0.5))
        self.idf: dict[str, float] = {}
        for term, df in df_counts.items():
            self.idf[term] = math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))

    def score_query(self, query: str) -> list[float]:
        q_terms = tokenize_technical(query)
        scores = [0.0] * self.n_docs
        for i in range(self.n_docs):
            dl = self.doc_lens[i]
            tf_map = self.tf_maps[i]
            total = 0.0
            for term in q_terms:
                f = tf_map.get(term, 0)
                if f == 0:
                    continue
                idf_val = self.idf.get(term, 0.0)
                numerator = f * (self.k1 + 1.0)
                denominator = f + self.k1 * (1.0 - self.b + self.b * (dl / max(self.avgdl, 1e-9)))
                total += idf_val * (numerator / denominator)
            scores[i] = total
        return scores


def naive_raw_tf_score(documents: list[str], query: str) -> list[int]:
    """BROKEN: Counts raw term occurrences with no IDF weighting or saturation."""
    q_terms = tokenize_technical(query)
    out: list[int] = []
    for doc in documents:
        tf_map = Counter(tokenize_technical(doc))
        out.append(sum(tf_map.get(t, 0) for t in q_terms))
    return out


def main() -> int:
    print("=" * 72)
    print("PHASE 02 · LESSON 02: OKAPI BM25 & TECHNICAL TOKENIZATION FROM SCRATCH")
    print("=" * 72)

    corpus = [
        # Doc 0: Exact concise match for saaras:v3
        "Use model saaras:v3 with header api-subscription-key for streaming speech-to-text transcription.",
        # Doc 1: Verbose general overview repeating common words 'use', 'model', 'with', 'for' many times
        (
            "When you use a model in our platform, use the model with your application for best results. "
            "You can use a model with Python or use a model with REST for text generation and chat."
        ),
        # Doc 2: Bulbul v3 TTS doc
        "Use model bulbul:v3 with speaker shubh for natural Indian language text-to-speech synthesis.",
        # Doc 3: Mayura v1 Translation doc
        "Use model mayura:v1 with source_language_code hi-IN for formal translation across 22 languages.",
    ]

    query = "How to use model saaras:v3 with api-subscription-key"
    bm25 = OkapiBM25Scratch(corpus, k1=1.5, b=0.75)

    print("\n1. Inverse Document Frequency (IDF) — Rare Technical Codes vs. Common Words:")
    for term in ("saaras:v3", "api-subscription-key", "bulbul:v3", "model", "use"):
        print(f"   - IDF({term:<22}) = {bm25.idf.get(term, 0.0):.4f}")

    assert bm25.idf["saaras:v3"] > bm25.idf["model"] * 3.0, "Rare model code must have much higher IDF than common word"

    raw_tf_scores = naive_raw_tf_score(corpus, query)
    bm25_scores = bm25.score_query(query)

    best_tf_idx = int( max(range(len(corpus)), key=lambda i: raw_tf_scores[i]) )
    best_bm25_idx = int( max(range(len(corpus)), key=lambda i: bm25_scores[i]) )

    print(f"\n2. Query Ranking Comparison: {query!r}")
    for idx in range(len(corpus)):
        print(
            f"   - Doc {idx} | Raw TF Score = {raw_tf_scores[idx]:>2} | Okapi BM25 Score = {bm25_scores[idx]:>6.4f} | {corpus[idx][:56]}..."
        )
    print("-" * 72)
    print(f"   [BROKEN] Raw TF Winner    -> Doc {best_tf_idx} (Fooled by repeated common words 'use', 'model', 'with'!)")
    print(f"   [FIXED ] Okapi BM25 Winner-> Doc {best_bm25_idx} (Correctly ranks exact `saaras:v3` & `api-subscription-key` #1!)")

    assert best_tf_idx == 1, "Expected naive TF to be fooled by Doc 1's repeated common words"
    assert best_bm25_idx == 0, "Expected Okapi BM25 to rank Doc 0 #1"

    print("\nVerification passed: Technical tokenizer, RSJ IDF, and Okapi BM25 verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
