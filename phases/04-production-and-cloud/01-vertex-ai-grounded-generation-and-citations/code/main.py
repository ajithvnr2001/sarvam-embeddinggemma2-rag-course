#!/usr/bin/env python3
"""Vertex AI Grounded Generation Prompt Builder & Deterministic Citation Verifier from Scratch.

Lesson: phases/04-production-and-cloud/01-vertex-ai-grounded-generation-and-citations/docs/en.md
Implements numbered [Doc i] context formatting and a deterministic citation guardrail that
detects phantom out-of-bounds citations ([Doc 99]) and uncited factual assertions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ContextChunk:
    doc_num: int
    chunk_id: str
    title: str
    url: str
    content: str


@dataclass(frozen=True)
class CitationAudit:
    is_valid: bool
    cited_doc_nums: list[int]
    phantom_doc_nums: list[int]
    uncited_sentences: list[str]


def build_grounded_prompt(query: str, chunks: list[ContextChunk]) -> str:
    """Format retrieved chunks into numbered [Doc i] evidence blocks with strict grounding rules."""
    blocks: list[str] = []
    for c in chunks:
        blocks.append(
            f"[Doc {c.doc_num}] (id={c.chunk_id} | title={c.title} | url={c.url})\n{c.content}"
        )
    context_str = "\n\n".join(blocks)
    return (
        "SYSTEM INSTRUCTION: Answer ONLY using the numbered [Doc i] evidence blocks below.\n"
        "Every factual claim MUST include an inline citation like [Doc 1] or [Doc 2].\n"
        "If the context does not contain the answer, reply: 'I cannot verify this from the provided Sarvam documentation.'\n\n"
        f"CONTEXT:\n{context_str}\n\n"
        f"USER QUESTION: {query}\n"
        "GROUNDED ANSWER:"
    )


def verify_citations(answer: str, max_doc_num: int) -> CitationAudit:
    """Deterministically verify that every [Doc i] citation is in {1..max_doc_num} and claims are cited."""
    raw_nums = [int(m) for m in re.findall(r"\[Doc\s+(\d+)\]", answer)]
    cited_unique = sorted(set(raw_nums))
    phantom = [n for n in cited_unique if n < 1 or n > max_doc_num]

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer.strip()) if s.strip()]
    uncited: list[str] = []
    for sent in sentences:
        if "cannot verify this" in sent.lower():
            continue
        if not re.search(r"\[Doc\s+\d+\]", sent):
            uncited.append(sent)

    is_valid = (len(cited_unique) > 0) and (len(phantom) == 0) and (len(uncited) == 0)
    return CitationAudit(
        is_valid=is_valid,
        cited_doc_nums=cited_unique,
        phantom_doc_nums=phantom,
        uncited_sentences=uncited,
    )


def main() -> int:
    print("=" * 72)
    print("PHASE 04 · LESSON 01: GROUNDED GENERATION & CITATION GUARDRAILS")
    print("=" * 72)

    chunks = [
        ContextChunk(
            doc_num=1,
            chunk_id="sarvam_0001",
            title="Authentication — API Subscription Key",
            url="https://docs.sarvam.ai/api-reference/authentication",
            content="All Sarvam AI REST API requests require the `api-subscription-key` HTTP header.",
        ),
        ContextChunk(
            doc_num=2,
            chunk_id="sarvam_0014",
            title="Speech to Text — Saaras v3",
            url="https://docs.sarvam.ai/documentation/speech-to-text",
            content="Model `saaras:v3` supports automatic code-mixing and 22 Indian languages at 16kHz.",
        ),
    ]

    prompt = build_grounded_prompt("How do I authenticate to Saaras v3?", chunks)
    print("\n1. Constructed Grounded Prompt Preview:")
    for line in prompt.splitlines()[:7]:
        print(f"   {line}")

    hallucinated_response = (
        "You must pass the `api-subscription-key` header [Doc 1]. "
        "Model `saaras:v3` also supports 128kHz satellite audio [Doc 9]. "
        "It is completely free for unlimited commercial usage."
    )
    grounded_response = (
        "Authenticate all Sarvam AI REST requests by passing the `api-subscription-key` HTTP header [Doc 1]. "
        "Once authenticated, set `model` to `saaras:v3` to transcribe 16kHz audio across 22 Indian languages [Doc 2]."
    )

    bad_audit = verify_citations(hallucinated_response, max_doc_num=len(chunks))
    good_audit = verify_citations(grounded_response, max_doc_num=len(chunks))

    print("\n2. Deterministic Citation Guardrail Audit:")
    print(
        f"   - [REJECTED] Hallucinated Output -> valid={bad_audit.is_valid} | "
        f"phantom_docs={bad_audit.phantom_doc_nums} | uncited_sentences={len(bad_audit.uncited_sentences)}"
    )
    print(
        f"   - [ACCEPTED] Grounded Output     -> valid={good_audit.is_valid} | "
        f"cited_docs={good_audit.cited_doc_nums} | phantom_docs={bad_audit.phantom_doc_nums[:0]}"
    )

    assert not bad_audit.is_valid and bad_audit.phantom_doc_nums == [9] and len(bad_audit.uncited_sentences) == 1
    assert good_audit.is_valid and good_audit.cited_doc_nums == [1, 2]

    print("\nVerification passed: Citation verifier caught phantom [Doc 9] and uncited claims.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
