#!/usr/bin/env python3
"""Route-aware environment preflight for Sarvam AI + EmbeddingGemma-2 RAG from Scratch.

Lesson: phases/00-setup-and-tooling/01-dev-environment-and-preflight/docs/en.md
Run this file from the repository root before starting a learning route.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[4]


@dataclass(frozen=True)
class Result:
    ok: bool
    detail: str


@dataclass(frozen=True)
class Probe:
    label: str
    run: Callable[[], Result]
    fix: str


@dataclass(frozen=True)
class Route:
    label: str
    required: tuple[str, ...]
    optional: tuple[str, ...]
    next_command: str


def python_result() -> Result:
    version = platform.python_version()
    executable = sys.executable
    if sys.version_info < (3, 10):
        return Result(False, f"found Python {version} at {executable}; need Python 3.10+")
    return Result(True, f"Python {version} at {executable}")


def command_result(command: str) -> Result:
    path = shutil.which(command)
    if path is None:
        return Result(False, f"{command!r} was not found on PATH")
    try:
        process = subprocess.run(
            [path, "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return Result(False, f"could not run {path}: {exc}")

    output = (process.stdout or process.stderr).strip().splitlines()
    detail = output[0] if output else f"exit code {process.returncode}"
    if process.returncode != 0:
        return Result(False, detail)
    return Result(True, f"{detail} at {path}")


def module_result(module: str) -> Result:
    if importlib.util.find_spec(module) is None:
        return Result(False, f"{module!r} is not importable by {sys.executable}")
    return Result(True, f"importable by {sys.executable}")


def corpus_result() -> Result:
    docs_dir = Path(os.environ.get("SARVAM_DOCS_PATH", "/data/opencode/sarvam/docs"))
    chunks_file = ROOT / "data" / "index" / "chunks.jsonl"
    if docs_dir.is_dir():
        md_files = list(docs_dir.rglob("*.md"))
        if md_files:
            return Result(True, f"{len(md_files)} Sarvam Markdown files ready at {docs_dir}")
    if chunks_file.is_file():
        lines = [ln for ln in chunks_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
        return Result(True, f"{len(lines)} indexed Sarvam chunks ready in data/index/chunks.jsonl")
    return Result(False, f"missing Sarvam corpus at {docs_dir} or {chunks_file}")


def accelerator_result() -> Result:
    if importlib.util.find_spec("torch") is None:
        return Result(True, "CPU mode active (PyTorch optional for from-scratch labs)")
    try:
        import torch

        if torch.cuda.is_available():
            return Result(True, f"CUDA GPU detected: {torch.cuda.get_device_name(0)}")
    except Exception as exc:
        return Result(True, f"CPU fallback active ({type(exc).__name__})")
    return Result(True, "CPU mode active (100% supported for EmbeddingGemma-2 & from-scratch labs)")


PROBES: dict[str, Probe] = {
    "python": Probe(
        "Python 3.10+",
        python_result,
        "Create a virtual environment with Python 3.10+ (`python3 -m venv .venv && source .venv/bin/activate`).",
    ),
    "git": Probe(
        "Git CLI",
        lambda: command_result("git"),
        "Install git via `sudo apt-get update && sudo apt-get install -y git`.",
    ),
    "corpus": Probe(
        "Sarvam AI Corpus",
        corpus_result,
        "Ensure `/data/opencode/sarvam/docs` or `data/index/chunks.jsonl` is present.",
    ),
    "numpy": Probe(
        "NumPy",
        lambda: module_result("numpy"),
        "Run `pip install -r requirements.txt` inside `.venv`.",
    ),
    "torch": Probe(
        "PyTorch",
        lambda: module_result("torch"),
        "Run `pip install torch` inside `.venv`.",
    ),
    "transformers": Probe(
        "Hugging Face Transformers",
        lambda: module_result("transformers"),
        "Run `pip install transformers huggingface-hub` inside `.venv`.",
    ),
    "fastapi": Probe(
        "FastAPI Web Studio",
        lambda: module_result("fastapi"),
        "Run `pip install fastapi uvicorn` inside `.venv`.",
    ),
    "gcloud": Probe(
        "Google Cloud SDK (gcloud)",
        lambda: command_result("gcloud"),
        "Install Google Cloud SDK and run `gcloud auth application-default login`.",
    ),
    "accelerator": Probe(
        "Compute Backend",
        accelerator_result,
        "CPU works out of the box; CUDA GPU accelerates large batch indexing.",
    ),
}


ROUTES: dict[str, Route] = {
    "beginner": Route(
        "Beginner From-Scratch Route (Phases 0-2)",
        ("python", "git", "corpus"),
        ("numpy", "accelerator"),
        "python3 phases/01-math-and-embeddings/01-vectors-norms-and-cosine-similarity/code/main.py",
    ),
    "rag-foundations": Route(
        "EmbeddingGemma-2 & Hybrid Retrieval Route (Phases 0-2)",
        ("python", "git", "corpus"),
        ("numpy", "torch", "transformers", "accelerator"),
        "python3 phases/01-math-and-embeddings/02-embeddinggemma2-architecture-and-prompts/code/main.py",
    ),
    "multimodal": Route(
        "Multimodal Audio & Vision RAG Route (Phase 3)",
        ("python", "git", "corpus"),
        ("numpy", "torch", "transformers", "fastapi"),
        "python3 phases/03-multimodal-rag/01-cross-modal-audio-and-image-rag/code/main.py",
    ),
    "gcp-production": Route(
        "Full Stack GCP Vertex AI, BigQuery & FinOps Destroy Route (Phases 0-4)",
        ("python", "git", "corpus"),
        ("numpy", "torch", "transformers", "fastapi", "gcloud"),
        "python3 phases/04-production-and-cloud/01-vertex-ai-grounded-generation-and-citations/code/main.py",
    ),
}


def run_preflight(route_name: str = "beginner") -> int:
    route = ROUTES[route_name]
    print("=" * 72)
    print(f"SARVAM AI x EMBEDDINGGEMMA-2 RAG PREFLIGHT — Route: {route.label}")
    print("=" * 72)

    failures: list[Probe] = []
    print("\n[Required Checks]")
    for key in route.required:
        probe = PROBES[key]
        res = probe.run()
        status = "PASS" if res.ok else "FAIL"
        print(f"  [{status}] {probe.label:<28} -> {res.detail}")
        if not res.ok:
            failures.append(probe)

    print("\n[Optional / Production Checks]")
    for key in route.optional:
        probe = PROBES[key]
        res = probe.run()
        status = "READY" if res.ok else "INFO "
        print(f"  [{status}] {probe.label:<28} -> {res.detail}")

    print("-" * 72)
    if failures:
        print(f"Result: {len(failures)} required check(s) failed:")
        for probe in failures:
            print(f"  - {probe.label}: {probe.fix}")
        return 1

    print("Result: All required checks passed! Your environment is ready.")
    print(f"Next step:\n  {route.next_command}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--route",
        choices=tuple(ROUTES.keys()),
        default="beginner",
        help="Learning route to validate (default: beginner)",
    )
    parser.add_argument(
        "--list-routes",
        action="store_true",
        help="List all available learning routes and exit",
    )
    args = parser.parse_args()

    if args.list_routes:
        for key, route in ROUTES.items():
            print(f"{key:<18} {route.label}")
        return 0

    return run_preflight(args.route)


if __name__ == "__main__":
    raise SystemExit(main())
