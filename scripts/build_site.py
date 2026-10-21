#!/usr/bin/env python3
"""Build the interactive static website (`site/`) for Cloudflare Pages deployment.

Reads all 5 phases and 13 lessons (`docs/en.md`, `code/main.py`, `quiz.json`,
`outputs/prompt-*.md`, `outputs/skill-*.md`), `cheatsheets/sarvam-rag.md`,
`ROADMAP.md`, and `course/*.md`, and generates a full interactive learning portal
in `site/` with KaTeX math, Mermaid diagrams, interactive quizzes, code tabs,
skill/prompt copy buttons, search, dark/light mode, and localStorage progress tracking.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE_DIR = ROOT / "site"

PHASE_META = {
    "00-setup-and-tooling": {
        "number": "Phase 00",
        "title": "Setup & Tooling",
        "subtitle": "Verify your Python/PyTorch/GCP workstation, authenticate with Hugging Face & Google Cloud, and audit the Sarvam AI corpus.",
        "color": "#3b82f6",
        "est": "~1.25 hrs",
    },
    "01-math-and-embeddings": {
        "number": "Phase 01",
        "title": "Math & Embeddings",
        "subtitle": "Vectors, L2 normalization, cosine similarity, asymmetric prompts, and Matryoshka truncation from first principles.",
        "color": "#8b5cf6",
        "est": "~2.25 hrs",
    },
    "02-retrieval-engineering": {
        "number": "Phase 02",
        "title": "Retrieval Engineering",
        "subtitle": "Structure-aware Markdown chunking, Okapi BM25 from scratch, and Hybrid Reciprocal Rank Fusion (RRF).",
        "color": "#10b981",
        "est": "~2.5 hrs",
    },
    "03-multimodal-rag": {
        "number": "Phase 03",
        "title": "Multimodal RAG",
        "subtitle": "Cross-modal retrieval across text, PNG architecture cards, and WAV/MP3 audio samples in one unified 768-D vector space.",
        "color": "#f59e0b",
        "est": "~1.0 hr",
    },
    "04-production-and-cloud": {
        "number": "Phase 04",
        "title": "Production & Cloud",
        "subtitle": "Vertex AI Gemini grounded generation, BigQuery vector search, IR evaluation metrics, and Cloud FinOps teardown to $0.00.",
        "color": "#ec4899",
        "est": "~3.0 hrs",
    },
}

LEARNING_PATHS = [
    {
        "id": "all",
        "name": "All Phases (Complete Curriculum)",
        "badge": "5 Phases · 13 Lessons · ~10 hrs",
        "desc": "Complete end-to-end journey from 3D vector math to production GCP BigQuery Vector Search and $0.00 FinOps teardown.",
        "lessons": None,
    },
    {
        "id": "beginner",
        "name": "Beginner Track (Zero-to-Production)",
        "badge": "All 13 Lessons · Step-by-Step",
        "desc": "Start at environment verification and build every RAG primitive in pure Python before touching cloud SDKs.",
        "lessons": None,
    },
    {
        "id": "rag-foundations",
        "name": "RAG Foundations (Math + Hybrid Search)",
        "badge": "6 Lessons · ~4.5 hrs",
        "desc": "Cosine similarity, Matryoshka (MRL) truncation, Markdown chunking, Okapi BM25, and Hybrid RRF fusion.",
        "lessons": [
            "01-math-and-embeddings/01-vectors-norms-and-cosine-similarity",
            "01-math-and-embeddings/02-embeddinggemma2-architecture-and-prompts",
            "01-math-and-embeddings/03-matryoshka-representation-learning",
            "02-retrieval-engineering/01-structure-aware-markdown-chunking",
            "02-retrieval-engineering/02-okapi-bm25-from-scratch",
            "02-retrieval-engineering/03-hybrid-search-and-rrf-fusion",
        ],
    },
    {
        "id": "multimodal",
        "name": "Multimodal Audio & Image RAG",
        "badge": "3 Lessons · ~2.5 hrs",
        "desc": "Encode text queries, PNG architecture cards, and WAV/MP3 speech audio into one shared 768-D vector space.",
        "lessons": [
            "01-math-and-embeddings/02-embeddinggemma2-architecture-and-prompts",
            "01-math-and-embeddings/03-matryoshka-representation-learning",
            "03-multimodal-rag/01-cross-modal-audio-and-image-rag",
        ],
    },
    {
        "id": "gcp-production",
        "name": "GCP Production & FinOps Destroy",
        "badge": "6 Lessons · ~4.0 hrs",
        "desc": "Deploy grounded RAG with Vertex AI Gemini, BigQuery COSINE_DISTANCE, IR eval gates, and verified $0.00 teardown.",
        "lessons": [
            "00-setup-and-tooling/01-dev-environment-and-preflight",
            "00-setup-and-tooling/02-gcp-hf-and-corpus-setup",
            "04-production-and-cloud/01-vertex-ai-grounded-generation-and-citations",
            "04-production-and-cloud/02-gcp-storage-and-bigquery-vector-search",
            "04-production-and-cloud/03-rag-evaluation-ir-metrics-and-llm-judge",
            "04-production-and-cloud/04-cli-ui-operations-and-resource-destroy",
        ],
    },
]


def extract_title_and_tagline(md_text: str) -> tuple[str, str, str]:
    title = "Lesson"
    tagline = ""
    est_time = "~45 minutes"
    for line in md_text.splitlines():
        if line.startswith("# ") and title == "Lesson":
            title = line[2:].strip()
        elif line.startswith("> ") and not tagline:
            tagline = line[2:].strip()
        elif line.startswith("**Time:**"):
            est_time = line.replace("**Time:**", "").strip()
    return title, tagline, est_time


def collect_curriculum() -> list[dict]:
    phases_dir = ROOT / "phases"
    phases_data = []

    for phase_path in sorted(phases_dir.iterdir()):
        if not phase_path.is_dir():
            continue
        phase_slug = phase_path.name
        meta = PHASE_META.get(
            phase_slug,
            {
                "number": phase_slug[:2],
                "title": phase_slug,
                "subtitle": "",
                "color": "#3b82f6",
                "est": "~2 hrs",
            },
        )
        lessons = []
        for lesson_path in sorted(phase_path.iterdir()):
            if not lesson_path.is_dir():
                continue
            doc_file = lesson_path / "docs" / "en.md"
            if not doc_file.exists():
                continue
            md_text = doc_file.read_text(encoding="utf-8")
            title, tagline, est_time = extract_title_and_tagline(md_text)

            code_file = lesson_path / "code" / "main.py"
            code_text = code_file.read_text(encoding="utf-8") if code_file.exists() else ""

            quiz_file = lesson_path / "quiz.json"
            quiz_data = json.loads(quiz_file.read_text(encoding="utf-8")) if quiz_file.exists() else {"questions": []}

            outputs_dir = lesson_path / "outputs"
            prompt_name, prompt_text = "", ""
            skill_name, skill_text = "", ""
            if outputs_dir.exists():
                for out_file in sorted(outputs_dir.iterdir()):
                    if out_file.name.startswith("prompt-"):
                        prompt_name = out_file.name
                        prompt_text = out_file.read_text(encoding="utf-8")
                    elif out_file.name.startswith("skill-"):
                        skill_name = out_file.name
                        skill_text = out_file.read_text(encoding="utf-8")

            lesson_id = f"{phase_slug}/{lesson_path.name}"
            lessons.append(
                {
                    "id": lesson_id,
                    "phase_slug": phase_slug,
                    "phase_number": meta["number"],
                    "phase_title": meta["title"],
                    "lesson_slug": lesson_path.name,
                    "lesson_num": lesson_path.name.split("-")[0],
                    "title": title,
                    "tagline": tagline,
                    "time": est_time,
                    "markdown": md_text,
                    "code_path": f"phases/{lesson_id}/code/main.py",
                    "code": code_text,
                    "quiz": quiz_data,
                    "prompt_name": prompt_name,
                    "prompt_text": prompt_text,
                    "skill_name": skill_name,
                    "skill_text": skill_text,
                }
            )

        phases_data.append(
            {
                "slug": phase_slug,
                "number": meta["number"],
                "title": meta["title"],
                "subtitle": meta["subtitle"],
                "color": meta["color"],
                "est": meta["est"],
                "lessons": lessons,
            }
        )
    return phases_data


def collect_deep_dive_modules() -> list[dict]:
    course_dir = ROOT / "course"
    modules = []
    if not course_dir.exists():
        return modules
    for md_file in sorted(course_dir.glob("*.md")):
        text = md_file.read_text(encoding="utf-8")
        title = md_file.stem
        for line in text.splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
                break
        modules.append(
            {
                "slug": md_file.stem,
                "filename": f"course/{md_file.name}",
                "title": title,
                "markdown": text,
            }
        )
    return modules


def build() -> None:
    SITE_DIR.mkdir(parents=True, exist_ok=True)
    assets_dir = SITE_DIR / "assets" / "screenshots"
    assets_dir.mkdir(parents=True, exist_ok=True)

    # Copy UI screenshots
    src_screenshots = ROOT / "data" / "artifacts" / "screenshots"
    if src_screenshots.exists():
        for png in src_screenshots.glob("*.png"):
            shutil.copy2(png, assets_dir / png.name)

    phases = collect_curriculum()
    cheatsheet_md = (ROOT / "cheatsheets" / "sarvam-rag.md").read_text(encoding="utf-8")
    roadmap_md = (ROOT / "ROADMAP.md").read_text(encoding="utf-8")
    teardown_txt = ""
    teardown_file = ROOT / "data" / "artifacts" / "teardown_verification.txt"
    if teardown_file.exists():
        teardown_txt = teardown_file.read_text(encoding="utf-8")

    deep_dive_modules = collect_deep_dive_modules()

    bundle = {
        "phases": phases,
        "learning_paths": LEARNING_PATHS,
        "cheatsheet_md": cheatsheet_md,
        "roadmap_md": roadmap_md,
        "teardown_txt": teardown_txt,
        "deep_dive_modules": deep_dive_modules,
    }

    (SITE_DIR / "course-data.js").write_text(
        "window.COURSE_DATA = " + json.dumps(bundle, ensure_ascii=False) + ";\n",
        encoding="utf-8",
    )

    # Write _headers for Cloudflare Pages caching & security
    (SITE_DIR / "_headers").write_text(
        "/*\n  X-Content-Type-Options: nosniff\n  X-Frame-Options: DENY\n  Referrer-Policy: strict-origin-when-cross-origin\n",
        encoding="utf-8",
    )

    print(f"Built site/course-data.js with {len(phases)} phases and {sum(len(p['lessons']) for p in phases)} lessons.")


if __name__ == "__main__":
    build()
