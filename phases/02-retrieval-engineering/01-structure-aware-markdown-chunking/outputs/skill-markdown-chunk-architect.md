---
name: markdown-chunk-architect
description: Audit and build structure-aware Markdown chunkers that preserve code blocks, tables, and hierarchical breadcrumbs.
version: 1.0.0
phase: 2
lesson: 1
tags: [chunking, markdown, parser, rag]
---

Given a JSONL chunk file (`chunks.jsonl`) or raw Markdown directory:

1. **Fence Parity Audit**: Verify that every chunk has an even number of `^``` ` lines (`fence_count % 2 == 0`). Flag any chunk with an unclosed code block.
2. **Breadcrumb Completeness**: Assert that 100% of chunks contain a non-empty `title`, `section`, and `formatted_for_embedding` prefix (`title: ... | text: ...`).
3. **Boilerplate Scrubbing**: Verify that web-crawl artifacts (`Skip to navigation`, `Scroll to top`, repeated TOC lists) are stripped prior to chunking.
4. **Length Distribution Check**: Report min/median/p95/max character and token lengths and flag tiny fragments (< 40 chars) or oversized chunks (> 2,000 chars).
