#!/usr/bin/env python3
"""Structure-Aware Markdown Chunking vs. Naive Fixed-Window Slicing from Scratch.

Lesson: phases/02-retrieval-engineering/01-structure-aware-markdown-chunking/docs/en.md
Demonstrates why fixed-character chunking corrupts Markdown tables and code blocks containing
`#` comments, and implements a structure-aware ATX heading + code-fence state machine chunker.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


SAMPLE_SARVAM_MD = """# Speech to Text REST API (Saaras v3)

Transcribe Indian language audio files with high accuracy and automatic code-mixing support.

## Request Parameters

| Parameter | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `model` | `string` | Yes | Set to `saaras:v3` for state-of-the-art transcription |
| `language_code` | `string` | Optional | BCP-47 code such as `hi-IN`, `ta-IN`, or `unknown` |
| `with_timestamps` | `boolean` | Optional | Return word-level start and end timestamps |

## Code Example

```bash
# Send audio file to Sarvam Saaras v3 endpoint
curl -X POST https://api.sarvam.ai/speech-to-text \\
  -H "api-subscription-key: $SARVAM_API_KEY" \\
  -F "file=@sample_tamil.wav" \\
  -F "model=saaras:v3"
```
"""


@dataclass(frozen=True)
class Chunk:
    heading: str
    content: str
    formatted_for_embedding: str


def naive_fixed_window_chunker(text: str, window_chars: int = 220, overlap_chars: int = 30) -> list[str]:
    """BROKEN: Blindly slices every `window_chars` characters regardless of tables or code blocks."""
    chunks: list[str] = []
    step = max(1, window_chars - overlap_chars)
    for start in range(0, len(text), step):
        piece = text[start : start + window_chars].strip()
        if piece:
            chunks.append(piece)
        if start + window_chars >= len(text):
            break
    return chunks


def structure_aware_markdown_chunker(doc_title: str, md_text: str) -> list[Chunk]:
    """CORRECT: Splits on ATX headings outside fenced code blocks and injects hierarchical breadcrumbs."""
    lines = md_text.strip().splitlines()
    sections: list[tuple[str, list[str]]] = []
    current_heading = "Overview"
    current_lines: list[str] = []
    in_code_block = False

    for line in lines:
        if line.strip().startswith("```"):
            in_code_block = not in_code_block

        # Only treat `#` as a Markdown heading if we are NOT inside a ``` code block!
        heading_match = not in_code_block and re.match(r"^(#{1,3})\s+(.+)$", line.strip())
        if heading_match:
            if current_lines:
                body = "\n".join(current_lines).strip()
                if body:
                    sections.append((current_heading, current_lines))
                current_lines = []
            current_heading = heading_match.group(2).strip()
            current_lines.append(line)
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_heading, current_lines))

    result: list[Chunk] = []
    for heading, sec_lines in sections:
        body = "\n".join(sec_lines).strip()
        full_title = f"{doc_title} — {heading}" if heading != doc_title else doc_title
        formatted = f"title: {full_title} | text: {body}"
        result.append(Chunk(heading=heading, content=body, formatted_for_embedding=formatted))
    return result


def has_broken_code_fence(chunk_text: str) -> bool:
    """Return True if a chunk opens a ``` fence without closing it (or vice versa)."""
    fence_count = len(re.findall(r"^```", chunk_text, flags=re.MULTILINE))
    return (fence_count % 2) != 0


def main() -> int:
    print("=" * 72)
    print("PHASE 02 · LESSON 01: STRUCTURE-AWARE MARKDOWN CHUNKING FROM SCRATCH")
    print("=" * 72)

    naive_chunks = naive_fixed_window_chunker(SAMPLE_SARVAM_MD, window_chars=220, overlap_chars=30)
    struct_chunks = structure_aware_markdown_chunker("Speech to Text API", SAMPLE_SARVAM_MD)

    naive_broken_fences = sum(1 for c in naive_chunks if has_broken_code_fence(c))
    struct_broken_fences = sum(1 for c in struct_chunks if has_broken_code_fence(c.content))

    print(f"\n1. Naive Fixed-Window Chunker (220 chars) -> Produced {len(naive_chunks)} chunks:")
    print(f"   - Chunks with severed ``` code fences : {naive_broken_fences}")
    print(f"   - Sample severed chunk (#2):\n     {naive_chunks[1]!r}")

    print(f"\n2. Structure-Aware Markdown Chunker       -> Produced {len(struct_chunks)} semantic sections:")
    print(f"   - Chunks with severed ``` code fences : {struct_broken_fences}")
    for idx, ch in enumerate(struct_chunks, start=1):
        print(f"   - Section {idx} [{ch.heading:<18}] -> {len(ch.content):>3} chars | Prefix: {ch.formatted_for_embedding[:58]}...")

    assert naive_broken_fences > 0, "Expected naive fixed-window chunking to sever the bash code block"
    assert struct_broken_fences == 0, "Structure-aware chunker must never sever fenced code blocks"
    assert len(struct_chunks) == 3, "Expected exactly 3 semantic sections (# Overview, ## Request Parameters, ## Code Example)"

    print("\nVerification passed: Structure-aware chunker preserved tables, bash comments, and breadcrumbs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
