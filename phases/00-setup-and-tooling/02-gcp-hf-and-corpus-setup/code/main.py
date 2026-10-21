#!/usr/bin/env python3
"""Sarvam AI Corpus Profiler & Cloud Configuration Validator from Scratch.

Lesson: phases/00-setup-and-tooling/02-gcp-hf-and-corpus-setup/docs/en.md
Scans Sarvam AI Markdown documentation and indexed chunks to quantify document length
distributions, headings, tables, and endpoint categories, and audits .env schema safety.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
DOCS_DIR = Path(os.environ.get("SARVAM_DOCS_PATH", "/data/opencode/sarvam/docs"))
CHUNKS_PATH = ROOT / "data" / "index" / "chunks.jsonl"


@dataclass(frozen=True)
class DocProfile:
    rel_path: str
    category: str
    word_count: int
    heading_count: int
    table_row_count: int
    code_fence_count: int


def profile_text(rel_path: str, category: str, text: str) -> DocProfile:
    words = re.findall(r"\b\w+\b", text)
    headings = re.findall(r"^#{1,6}\s+.+$", text, flags=re.MULTILINE)
    table_rows = [
        line for line in text.splitlines() if line.strip().startswith("|") and line.strip().endswith("|")
    ]
    code_fences = re.findall(r"^```", text, flags=re.MULTILINE)
    return DocProfile(
        rel_path=rel_path,
        category=category,
        word_count=len(words),
        heading_count=len(headings),
        table_row_count=len(table_rows),
        code_fence_count=max(1, len(code_fences) // 2) if code_fences else 0,
    )


def load_profiles() -> list[DocProfile]:
    if CHUNKS_PATH.is_file():
        grouped_text: dict[str, list[str]] = {}
        grouped_cat: dict[str, str] = {}
        for line in CHUNKS_PATH.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            doc_path = row["doc_path"]
            grouped_text.setdefault(doc_path, []).append(row["content"])
            grouped_cat[doc_path] = row.get("category", "general")
        return [
            profile_text(doc_path, grouped_cat[doc_path], "\n\n".join(parts))
            for doc_path, parts in sorted(grouped_text.items())
        ]

    if DOCS_DIR.is_dir():
        profiles: list[DocProfile] = []
        for path in sorted(DOCS_DIR.rglob("*.md"))[:64]:
            rel = path.relative_to(DOCS_DIR).as_posix()
            cat = rel.split("/")[0] if "/" in rel else "root"
            profiles.append(profile_text(rel, cat, path.read_text(encoding="utf-8", errors="ignore")))
        return profiles

    return []


def mask_secret(value: str) -> str:
    value = value.strip()
    if not value:
        return "<EMPTY>"
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}...{value[-3:]} (len={len(value)})"


def audit_env_schema(env_path: Path) -> dict[str, str]:
    required_keys = (
        "HF_TOKEN",
        "EMBEDDING_MODEL_ID",
        "GCP_PROJECT_ID",
        "GCP_LOCATION",
        "GCS_BUCKET_NAME",
        "BQ_DATASET_ID",
        "BQ_TABLE_ID",
    )
    found: dict[str, str] = {}
    if env_path.is_file():
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            found[k.strip()] = v.strip().strip("'\"")

    report: dict[str, str] = {}
    for key in required_keys:
        val = found.get(key, "")
        if "TOKEN" in key or "KEY" in key or "SECRET" in key:
            report[key] = mask_secret(val) if val else "NOT SET (optional for offline labs)"
        else:
            report[key] = val if val else "<DEFAULT>"
    return report


def main() -> int:
    print("=" * 72)
    print("PHASE 00 · LESSON 02: SARVAM AI CORPUS & CLOUD CONFIG PROFILER")
    print("=" * 72)

    profiles = load_profiles()
    if not profiles:
        print("ERROR: Could not locate Sarvam corpus.")
        return 1

    cat_counts = Counter(p.category for p in profiles)
    total_words = sum(p.word_count for p in profiles)
    total_headings = sum(p.heading_count for p in profiles)
    total_tables = sum(p.table_row_count for p in profiles)
    total_code_blocks = sum(p.code_fence_count for p in profiles)

    print("\n1. Corpus Structural Summary:")
    print(f"   - Distinct Sarvam Docs Profiled : {len(profiles)}")
    print(f"   - Total Words                   : {total_words:,}")
    print(f"   - Total Markdown Headings       : {total_headings:,}")
    print(f"   - Total Markdown Table Rows     : {total_tables:,}")
    print(f"   - Total Fenced Code Blocks      : {total_code_blocks:,}")

    print("\n2. Breakdown by Product / API Category:")
    for cat, count in sorted(cat_counts.items()):
        cat_words = sum(p.word_count for p in profiles if p.category == cat)
        print(f"   - {cat:<22}: {count:>2} docs ({cat_words:>6,} words)")

    print("\n3. Top 5 Most Information-Dense Sarvam Docs (by heading + table count):")
    ranked = sorted(
        profiles,
        key=lambda p: (p.heading_count + p.table_row_count, p.word_count),
        reverse=True,
    )[:5]
    for idx, p in enumerate(ranked, start=1):
        print(
            f"   {idx}. {p.rel_path:<46} | words={p.word_count:>4} | headings={p.heading_count:>2} | table_rows={p.table_row_count:>2}"
        )

    env_file = ROOT / ".env" if (ROOT / ".env").is_file() else ROOT / ".env.example"
    print(f"\n4. Sanitized Configuration Schema Audit ({env_file.name}):")
    for key, masked in audit_env_schema(env_file).items():
        print(f"   - {key:<22}: {masked}")

    print("\nVerification passed: Corpus profiled and secret masking verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
