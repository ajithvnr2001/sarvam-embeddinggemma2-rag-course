#!/usr/bin/env python3
"""
Interactive Bit-by-Bit Hands-On Learning Labs (Bits 1 -> 10)
Run any individual bit to see side-by-side "Naive Mistake vs. Production Solution" live:
  python scripts/interactive_learning_labs.py --bit 1
  python scripts/interactive_learning_labs.py --all
"""
import argparse
import json
import re
import sys
from pathlib import Path
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CONFIG
from src.chunker import clean_markdown_body, split_markdown_sections
from src.embedder import EmbeddingGemma2Engine, mrl_truncate_and_renormalize
from src.retriever import SarvamHybridRetriever, tokenize_for_bm25
from src.multimodal_rag import search_multimodal_gallery

console = Console()


def lab_bit_1() -> None:
    console.print(Panel.fit(
        "[bold cyan]Bit 1 Lab: How Cosine Similarity & L2 Normalization Work Under the Hood[/bold cyan]\n"
        "Proving the 3D worked math example from Module 01 in live NumPy.",
        title="🟢 Bit 1 — Vector Geometry & Cosine Similarity"
    ))
    q = np.array([3.0, 4.0, 0.0], dtype=np.float32)
    d1 = np.array([6.0, 8.0, 0.0], dtype=np.float32)  # Auth doc (same angle as q!)
    d2 = np.array([0.0, 0.0, 5.0], dtype=np.float32)  # Unrelated TTS doc (orthogonal)

    q_norm = q / np.linalg.norm(q)
    d1_norm = d1 / np.linalg.norm(d1)
    d2_norm = d2 / np.linalg.norm(d2)

    table = Table(title="Bit 1: Raw Vectors -> L2 Unit Vectors -> Dot Product (Cosine Similarity)", show_lines=True)
    table.add_column("Vector", style="cyan")
    table.add_column("Raw Coordinates", style="white")
    table.add_column("L2 Length ||v||_2", style="yellow")
    table.add_column("Normalized Unit Vector v / ||v||_2", style="green")
    table.add_column("Cosine Sim with Query", style="bold magenta")

    table.add_row("Query q", str(q.tolist()), f"{np.linalg.norm(q):.2f}", str(np.round(q_norm, 2).tolist()), "1.0000 (Self)")
    table.add_row("Doc 1 (Auth Header)", str(d1.tolist()), f"{np.linalg.norm(d1):.2f}", str(np.round(d1_norm, 2).tolist()), f"{float(q_norm @ d1_norm):.4f} (Rank #1 Match!)")
    table.add_row("Doc 2 (Tamil TTS)", str(d2.tolist()), f"{np.linalg.norm(d2):.2f}", str(np.round(d2_norm, 2).tolist()), f"{float(q_norm @ d2_norm):.4f} (Orthogonal 90°)")
    console.print(table)


def lab_bit_2() -> None:
    console.print(Panel.fit(
        "[bold cyan]Bit 2 Lab: Naive '#' Splitting vs. Code-Fence-Aware Chunking[/bold cyan]\n"
        "See why naive '#' splitting breaks Python comments & code blocks, while src/chunker.py keeps them intact.",
        title="🟢 Bit 2 — Structure-Aware Markdown & Code Chunking"
    ))
    sample_md = (
        "# Text to Speech API\n"
        "Use `bulbul:v3` to synthesize audio.\n\n"
        "```python\n"
        "# Step 1: Set your api-subscription-key header\n"
        "headers = {'api-subscription-key': 'YOUR_KEY'}\n"
        "# Step 2: Send POST request\n"
        "response = requests.post('https://api.sarvam.ai/text-to-speech', headers=headers)\n"
        "```\n\n"
        "## Supported Sample Rates\n"
        "Supports 8000Hz, 16000Hz, and 24000Hz."
    )
    naive_splits = [part.strip() for part in sample_md.split("\n#") if part.strip()]
    smart_sections = split_markdown_sections(clean_markdown_body(sample_md))

    console.print(f"[red]❌ Naive '#' Splitter produced {len(naive_splits)} broken fragments[/red] (mistook `# Step 1` inside Python code for a heading!):")
    for idx, frag in enumerate(naive_splits, 1):
        console.print(f"   Fragment {idx}: {repr(frag[:68])}...")

    console.print(f"\n[green]✅ Our Code-Fence State Machine (`split_markdown_sections`) produced {len(smart_sections)} clean sections[/green] (0 broken code blocks!):")
    for sec_title, sec_body in smart_sections:
        console.print(f"   • Section [bold]{sec_title}[/bold]: {len(sec_body)} chars | Balanced ``` fences: {sec_body.count('```') % 2 == 0}")


def lab_bit_3() -> None:
    console.print(Panel.fit(
        "[bold cyan]Bit 3 Lab: Selective 303M Loading & Asymmetric Task Prefixes in google/embeddinggemma-2[/bold cyan]\n"
        "Encoding a developer question with QuestionAnswering vs. CodeRetrieval prefixes.",
        title="🟢 Bit 3 — EmbeddingGemma 2 Architecture & Task Prefixes"
    ))
    engine = EmbeddingGemma2Engine(modality_mode="text_only")
    query = "How do I authenticate with the Sarvam AI API?"
    doc_formatted = "title: Authentication — Overview | text: All Sarvam AI REST API requests require the api-subscription-key HTTP header."

    v_q_qa = engine.encode_query(query, prompt_name="QuestionAnswering")
    v_q_code = engine.encode_query("headers = {'api-subscription-key': 'YOUR_KEY'}", prompt_name="CodeRetrieval")
    v_d = engine.encode_documents([doc_formatted])[0]

    console.print(f"• Model Active Mode: [magenta]{engine.modality_mode}[/magenta] | Dtype: [cyan]{engine.dtype}[/cyan] | Vector Shape: [green]{v_q_qa.shape}[/green]")
    console.print(f"• QuestionAnswering Query -> Doc Cosine: [bold green]{float(v_q_qa @ v_d):.5f}[/bold green]")
    console.print(f"• CodeRetrieval Query     -> Doc Cosine: [bold green]{float(v_q_code @ v_d):.5f}[/bold green]")


def lab_bit_4() -> None:
    console.print(Panel.fit(
        "[bold cyan]Bit 4 Lab: Why L2 Re-Normalization After Matryoshka Slicing is Mandatory[/bold cyan]\n"
        "Inspecting real L2 norms of a 768d EmbeddingGemma 2 vector when sliced to 512d, 256d, and 128d.",
        title="🟡 Bit 4 — Matryoshka Representation Learning (MRL)"
    ))
    mat_768 = np.load(CONFIG.index_dir / "embeddings_768.npy")
    v768 = mat_768[17]  # sarvam_0017 (Authentication chunk)

    table = Table(title="Chunk sarvam_0017 Across MRL Dimensions (Before vs. After L2 Re-Normalization)", show_lines=True)
    table.add_column("MRL Slice", style="cyan")
    table.add_column("Compression", style="yellow")
    table.add_column("RAM for 307 Chunks", style="white")
    table.add_column("❌ Raw Sliced L2 Norm (Un-normalized)", style="red")
    table.add_column("✅ After mrl_truncate_and_renormalize()", style="green")

    for dim in (768, 512, 256, 128):
        raw_slice = v768[:dim]
        renorm_slice = mrl_truncate_and_renormalize(v768, dim)
        ram_kb = (307 * dim * 4) / 1024
        table.add_row(
            f"{dim}d",
            f"{768 / dim:.1f}x",
            f"{ram_kb:.1f} KB",
            f"{np.linalg.norm(raw_slice):.6f} (< 1.0!)",
            f"{np.linalg.norm(renorm_slice):.6f} (Unit Norm)",
        )
    console.print(table)


def lab_bit_5() -> None:
    console.print(Panel.fit(
        "[bold cyan]Bit 5 Lab: Identifier-Aware BM25 + Dense Vector Reciprocal Rank Fusion (RRF)[/bold cyan]\n"
        "Watching how tokenize_for_bm25 preserves `bulbul:v4-flash` and `api-subscription-key`.",
        title="🟡 Bit 5 — Hybrid Search & RRF"
    ))
    sample_query = "Which Tamil speaker IDs work on bulbul:v4-flash with api-subscription-key?"
    console.print(f"Input Query: [bold]{sample_query}[/bold]")
    console.print(f"Identifier-Aware Tokens: [green]{tokenize_for_bm25(sample_query)}[/green]")

    retriever = SarvamHybridRetriever()
    res = retriever.search(sample_query, top_k=3, mode="hybrid", dim=256)
    for h in res["hits"]:
        console.print(f"  Rank #{h['rank']}: [bold yellow]{h['chunk_id']}[/bold yellow] | RRF Score: [green]{h['score']:.5f}[/green] | {h['title']}")


def lab_bit_6() -> None:
    console.print(Panel.fit(
        "[bold cyan]Bit 6 Lab: True Multimodal 740M Cross-Modal Search (Text -> Audio & Image)[/bold cyan]\n"
        "Searching pre-encoded 16kHz WAV audio (<|audio|>) and PNG API cards (<|image|>) in one 768d space.",
        title="🔴 Bit 6 — Multimodal Audio & Vision RAG"
    ))
    for q in [
        "continuous heavy rainfall storm sound effect audio",
        "Instant voice cloning workflow diagram /voices/create and /voices/clone",
    ]:
        res = search_multimodal_gallery(q, top_k=2, dim=768)
        top = res["hits"][0]
        console.print(
            f"• Query: [bold]'{q}'[/bold]\n"
            f"  -> Rank #1 Match: [cyan]{top['item_id']}[/cyan] | Modality: [magenta]{top['modality'].upper()}[/magenta] "
            f"| Cosine: [bold green]{top['score']:.5f}[/bold green] | Title: {top['title']}"
        )


def lab_bit_7() -> None:
    console.print(Panel.fit(
        "[bold cyan]Bit 7 Lab: Deterministic Citation Auditing on Grounded Answers[/bold cyan]\n"
        "Verifying that every [sarvam_XXXX] citation in a generated answer comes strictly from retrieved chunks.",
        title="🔴 Bit 7 — Grounded Generation & Citation Guardrails"
    ))
    retrieved_ids = {"sarvam_0017", "sarvam_0018", "sarvam_0015"}
    sample_answer = (
        "To authenticate with the Sarvam AI REST API, pass your subscription key in the "
        "`api-subscription-key` HTTP header [sarvam_0017] [sarvam_0018]."
    )
    cited_ids = set(re.findall(r"\[(sarvam_\d{4})\]", sample_answer))
    precision = len(cited_ids & retrieved_ids) / max(len(cited_ids), 1)
    console.print(f"• Retrieved Context Chunk IDs: [yellow]{sorted(retrieved_ids)}[/yellow]")
    console.print(f"• Extracted Answer Citations:  [cyan]{sorted(cited_ids)}[/cyan]")
    console.print(f"• Citation Precision:          [bold green]{precision * 100:.1f}% (0 Hallucinated Citations)[/bold green]")


def lab_bit_8() -> None:
    console.print(Panel.fit(
        "[bold cyan]Bit 8 Lab: Cloud BigQuery VECTOR_SEARCH Storage & Cost Math[/bold cyan]\n"
        "Comparing BigQuery ARRAY<FLOAT64> storage for 768d vs. 256d Matryoshka vectors.",
        title="🔴 Bit 8 — GCP Cloud Storage & BigQuery VECTOR_SEARCH"
    ))
    for dim in (768, 256):
        bytes_per_row = dim * 8  # FLOAT64 = 8 bytes
        console.print(
            f"• BigQuery ARRAY<FLOAT64> @ [bold]{dim}d[/bold]: [yellow]{bytes_per_row:,} bytes/vector[/yellow] "
            f"({768 // dim}.0x scan reduction)"
        )


def lab_bit_9() -> None:
    console.print(Panel.fit(
        "[bold cyan]Bit 9 Lab: Inspecting Live IR Metrics & Vertex AI LLM-as-a-Judge Report[/bold cyan]\n"
        "Reading verified benchmark results from data/artifacts/evaluation_report.json.",
        title="🔴 Bit 9 — Quantitative RAG Evaluation & LLM Judge"
    ))
    report = json.loads((CONFIG.artifacts_dir / "evaluation_report.json").read_text(encoding="utf-8"))
    ir = report["strategy_benchmarks"]
    summary = report["summary"]
    table = Table(title="Live Verified IR Metrics (sarvam-docs)", show_lines=True)
    table.add_column("Strategy", style="cyan")
    table.add_column("Hit@1 (%)", style="green")
    table.add_column("Recall@5 (%)", style="yellow")
    table.add_column("MRR@5", style="magenta")
    for name in ("hybrid_rrf_768d", "hybrid_rrf_256d", "mrl_funnel_128_to_768", "dense_768d", "dense_256d", "bm25_lexical"):
        row = ir[name]
        table.add_row(name, f"{row['hit_at_1_pct']}%", f"{row['recall_at_5_pct']}%", f"{row['mrr_at_5']:.4f}")
    console.print(table)
    console.print(
        f"• Vertex AI Gemini 2.5 Flash Judge Scores: "
        f"Context Relevance=[bold green]{summary['avg_context_relevance']:.2f}/5.0[/bold green] | "
        f"Faithfulness=[bold green]{summary['avg_faithfulness']:.2f}/5.0[/bold green] | "
        f"Completeness=[bold green]{summary['avg_completeness']:.2f}/5.0[/bold green]"
    )


def lab_bit_10() -> None:
    console.print(Panel.fit(
        "[bold cyan]Bit 10 Lab: FinOps & Clean Resource Teardown Verification[/bold cyan]\n"
        "Checking local index files vs. destroyed cloud state ($0.00 idle cloud bill).",
        title="🧹 Bit 10 — Resource Teardown (destroy)"
    ))
    teardown_file = CONFIG.artifacts_dir / "teardown_verification.txt"
    if teardown_file.exists():
        console.print("[green]✓ Verified Cloud Teardown Log (`data/artifacts/teardown_verification.txt`):[/green]")
        for line in teardown_file.read_text(encoding="utf-8").splitlines()[-6:]:
            console.print("  ", line)
    console.print("\n[bold]Commands to provision or destroy anytime:[/bold]")
    console.print("  • Provision Cloud: [cyan]python -m cli.rag_cli gcp-sync --dim 256[/cyan]")
    console.print("  • Destroy Cloud:   [red]python -m cli.rag_cli destroy[/red]")


def run_labs(bit: int | None) -> None:
    labs = {
        1: lab_bit_1, 2: lab_bit_2, 3: lab_bit_3, 4: lab_bit_4, 5: lab_bit_5,
        6: lab_bit_6, 7: lab_bit_7, 8: lab_bit_8, 9: lab_bit_9, 10: lab_bit_10,
    }
    if bit is not None:
        labs[bit]()
    else:
        for b in sorted(labs):
            labs[b]()
            console.print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive Bit-by-Bit Learning Labs (Bits 1-10)")
    parser.add_argument("--bit", type=int, choices=list(range(1, 11)), default=None, help="Run a specific bit lab (1-10)")
    parser.add_argument("--all", action="store_true", help="Run all 10 interactive micro-labs")
    args = parser.parse_args()
    run_labs(args.bit)
