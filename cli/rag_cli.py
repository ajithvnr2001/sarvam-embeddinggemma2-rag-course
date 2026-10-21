#!/usr/bin/env python3
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

from src.config import CONFIG
from src.chunker import build_sarvam_corpus, save_chunks_jsonl, load_chunks_jsonl
from src.embedder import EmbeddingGemma2Engine, mrl_truncate_and_renormalize
from src.retriever import SarvamHybridRetriever
from src.generator import VertexAIGeminiGenerator
from src.gcp_integration import GCPVectorPlatform
from src.multimodal_rag import build_multimodal_gallery, search_multimodal_gallery

console = Console()


def cmd_ingest(args: argparse.Namespace) -> None:
    console.print(
        Panel.fit(
            f"[bold cyan]Ingesting Sarvam Docs Corpus[/bold cyan]\n"
            f"Source: [yellow]{CONFIG.sarvam_docs_path}[/yellow]\n"
            f"Model:  [green]{CONFIG.embedding_model_id}[/green] (Selective Mode: [magenta]text_only 270M[/magenta])",
            title="Phase 1 — Corpus Chunking & EmbeddingGemma 2 Indexing",
        )
    )
    chunks = build_sarvam_corpus(CONFIG.sarvam_docs_path, max_files=args.max_files)
    if args.max_chunks and len(chunks) > args.max_chunks:
        chunks = chunks[: args.max_chunks]

    chunks_path = CONFIG.index_dir / "chunks.jsonl"
    save_chunks_jsonl(chunks, chunks_path)
    console.print(f"[green]✓[/green] Created [bold]{len(chunks)}[/bold] semantic markdown/code chunks -> {chunks_path}")

    embedder = EmbeddingGemma2Engine(modality_mode="text_only")
    console.print(f"[green]✓[/green] Loaded {CONFIG.embedding_model_id} in {embedder.load_time_sec}s (dtype={embedder.dtype})")

    formatted_texts = [c.formatted_for_embedding for c in chunks]
    t0 = time.time()
    emb_768 = embedder.encode_documents(formatted_texts, batch_size=args.batch_size, truncate_dim=768)
    elapsed = round(time.time() - t0, 2)

    np.save(CONFIG.index_dir / "embeddings_768.npy", emb_768)
    for d in (512, 256, 128):
        np.save(CONFIG.index_dir / f"embeddings_{d}.npy", mrl_truncate_and_renormalize(emb_768, d))

    stats = {
        "total_chunks": len(chunks),
        "embedding_model": CONFIG.embedding_model_id,
        "modality_mode": "text_only (270M active parameters)",
        "native_dim": 768,
        "mrl_dims_saved": [768, 512, 256, 128],
        "encode_time_sec": elapsed,
        "chunks_per_sec": round(len(chunks) / max(elapsed, 0.01), 2),
        "categories": {},
    }
    for c in chunks:
        stats["categories"][c.category] = stats["categories"].get(c.category, 0) + 1

    (CONFIG.index_dir / "index_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    console.print(
        f"[bold green]✓ Indexing Complete![/bold green] Encoded {len(chunks)} chunks in {elapsed}s "
        f"({stats['chunks_per_sec']} chunks/s). Saved 768d, 512d, 256d, and 128d MRL matrices."
    )


def cmd_search(args: argparse.Namespace) -> None:
    retriever = SarvamHybridRetriever()
    res = retriever.search(
        query=args.query,
        top_k=args.top_k,
        mode=args.mode,
        dim=args.dim,
        prompt_name=args.prompt_name,
        category_filter=args.category,
    )

    table = Table(
        title=(
            f"Retrieval Results | Query: '{args.query}' | Mode: {args.mode.upper()} | "
            f"Dim: {args.dim}d | Prefix: {args.prompt_name} | Total: {res['total_ms']} ms"
        ),
        show_lines=True,
    )
    table.add_column("Rank", justify="center", style="bold cyan", width=5)
    table.add_column("Chunk ID", style="yellow", width=12)
    table.add_column("Category", style="magenta", width=15)
    table.add_column("Title / Section", style="bold white", width=36)
    table.add_column("Score", justify="right", style="green", width=10)
    table.add_column("Snippet", style="dim white", width=55)

    for h in res["hits"]:
        snippet = h["content"].replace("\n", " ")[:180] + "..."
        table.add_row(
            str(h["rank"]),
            h["chunk_id"],
            h["category"],
            h["title"],
            f"{h['score']:.4f}",
            snippet,
        )
    console.print(table)


def cmd_ask(args: argparse.Namespace) -> None:
    retriever = SarvamHybridRetriever()
    ret_res = retriever.search(
        query=args.query,
        top_k=args.top_k,
        mode=args.mode,
        dim=args.dim,
        prompt_name=args.prompt_name,
    )
    generator = VertexAIGeminiGenerator()
    gen_res = generator.generate_grounded_answer(query=args.query, retrieved_hits=ret_res["hits"])

    console.print(
        Panel(
            Markdown(gen_res["answer"]),
            title=(
                f"[bold green]Grounded RAG Answer[/bold green] | "
                f"Model: [cyan]{gen_res['model']}[/cyan] ({gen_res['project_id']}) | "
                f"Retrieval: {args.mode.upper()} ({args.dim}d, {ret_res['total_ms']}ms) | "
                f"LLM: {gen_res['latency_ms']}ms"
            ),
            border_style="green",
        )
    )

    src_table = Table(title="Cited Sarvam Documentation Chunks", show_lines=False)
    src_table.add_column("Chunk ID", style="yellow")
    src_table.add_column("Title", style="white")
    src_table.add_column("Score", style="green")
    src_table.add_column("Source URL", style="cyan")
    for s in gen_res["sources"]:
        src_table.add_row(s["chunk_id"], s["title"], f"{s['score']:.4f}", s["url"])
    console.print(src_table)

    if args.judge:
        judge_scores = generator.evaluate_rag_response(
            query=args.query,
            retrieved_hits=ret_res["hits"],
            generated_answer=gen_res["answer"],
        )
        console.print(
            Panel(
                json.dumps(judge_scores, indent=2),
                title="[bold yellow]Vertex AI Gemini LLM-as-a-Judge Evaluation[/bold yellow]",
                border_style="yellow",
            )
        )


def cmd_mrl_benchmark(args: argparse.Namespace) -> None:
    retriever = SarvamHybridRetriever()
    comp = retriever.compare_mrl_dimensions(query=args.query, top_k=args.top_k)

    table = Table(
        title=f"Matryoshka Representation Learning (MRL) Live Comparison\nQuery: '{args.query}'",
        show_lines=True,
    )
    table.add_column("Dimension", style="bold cyan", justify="center")
    table.add_column("Compression", style="magenta", justify="center")
    table.add_column("Bytes/Vec (FP32)", justify="right")
    table.add_column("Search Time (ms)", justify="right", style="green")
    table.add_column("Top-1 Chunk ID", style="yellow")
    table.add_column("Top-1 Cosine", justify="right", style="bold green")
    table.add_column(f"Top-{args.top_k} Overlap vs 768d", justify="right", style="bold white")

    for d_str in ("768", "512", "256", "128"):
        row = comp["dimensions"][d_str]
        table.add_row(
            f"{row['dimension']}d",
            row["compression_ratio"],
            f"{row['bytes_per_vector_fp32']} B",
            f"{row['search_ms']:.4f}",
            f"{row['top1_chunk_id']} ({row['top1_title'][:28]}...)",
            f"{row['top1_cosine']:.5f}",
            f"{row['topk_overlap_with_768d']}%",
        )
    console.print(table)


def cmd_multimodal(args: argparse.Namespace) -> None:
    res = search_multimodal_gallery(query=args.query, top_k=args.top_k, dim=args.dim)
    table = Table(
        title=f"Unified 768d Multimodal Search (Text -> Image / Audio / Interleaved)\nQuery: '{args.query}' ({res['dimension']}d)",
        show_lines=True,
    )
    table.add_column("Rank", justify="center", style="cyan")
    table.add_column("ID", style="yellow")
    table.add_column("Modality", style="magenta")
    table.add_column("Title", style="bold white")
    table.add_column("Cosine Sim", justify="right", style="green")
    table.add_column("Asset Path", style="dim cyan")

    for h in res["hits"]:
        table.add_row(
            str(h["rank"]),
            h["id"],
            h["modality"],
            h["title"],
            f"{h['cosine_similarity']:.5f}",
            h["asset_path"],
        )
    console.print(table)


def cmd_gcp_sync(args: argparse.Namespace) -> None:
    chunks = load_chunks_jsonl(CONFIG.index_dir / "chunks.jsonl")
    emb_768 = np.load(CONFIG.index_dir / "embeddings_768.npy")
    gcp = GCPVectorPlatform()
    console.print(
        f"[cyan]Syncing {len(chunks)} chunks ({args.dim}d MRL vectors) to GCP Project "
        f"[bold]{gcp.project_id}[/bold] (GCS + BigQuery)...[/cyan]"
    )
    res = gcp.sync_to_gcs_and_bigquery(chunks, emb_768, dim=args.dim)
    console.print(Panel(json.dumps(res, indent=2), title="[bold green]GCP Cloud Storage & BigQuery Sync Complete[/bold green]"))


def cmd_bq_search(args: argparse.Namespace) -> None:
    embedder = EmbeddingGemma2Engine(modality_mode="text_only")
    q_vec = embedder.encode_query(args.query, prompt_name="QuestionAnswering", truncate_dim=args.dim)
    gcp = GCPVectorPlatform()
    res = gcp.bigquery_vector_search(q_vec, top_k=args.top_k, dim=args.dim)

    table = Table(
        title=f"GCP BigQuery VECTOR_SEARCH Results | Table: {res['table']} | Latency: {res['latency_ms']} ms",
        show_lines=True,
    )
    table.add_column("Chunk ID", style="yellow")
    table.add_column("Category", style="magenta")
    table.add_column("Title", style="bold white")
    table.add_column("Cosine Sim", justify="right", style="green")
    table.add_column("Preview", style="dim white")

    for r in res["results"]:
        table.add_row(
            str(r.get("chunk_id", "")),
            str(r.get("category", "")),
            str(r.get("title", "")),
            str(r.get("cosine_similarity", "")),
            str(r.get("content_preview", "")).replace("\n", " ")[:140] + "...",
        )
    console.print(table)


def cmd_chat(args: argparse.Namespace) -> None:
    retriever = SarvamHybridRetriever()
    generator = VertexAIGeminiGenerator()
    console.print(
        Panel.fit(
            "[bold green]Sarvam AI + EmbeddingGemma 2 Interactive RAG CLI[/bold green]\n"
            "Type your question and press Enter. Type [yellow]exit[/yellow] or [yellow]quit[/yellow] to leave.",
            border_style="cyan",
        )
    )
    while True:
        try:
            q = console.input("\n[bold cyan]Sarvam-RAG > [/bold cyan]").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q.lower() in ("exit", "quit"):
            break
        ret = retriever.search(q, top_k=args.top_k, mode=args.mode, dim=args.dim)
        ans = generator.generate_grounded_answer(q, ret["hits"])
        console.print(Panel(Markdown(ans["answer"]), title=f"Answer ({ans['latency_ms']} ms)", border_style="green"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag_cli",
        description="End-to-End CLI for Google EmbeddingGemma 2 + Sarvam Docs Multimodal RAG on GCP",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ing = sub.add_parser("ingest", help="Chunk sarvam-docs and build EmbeddingGemma 2 MRL & BM25 indices")
    p_ing.add_argument("--max-files", type=int, default=85, help="Max markdown files to index")
    p_ing.add_argument("--max-chunks", type=int, default=180, help="Max chunks to encode")
    p_ing.add_argument("--batch-size", type=int, default=16, help="Encoding batch size")
    p_ing.set_defaults(func=cmd_ingest)

    def add_query_arg(p: argparse.ArgumentParser, help_text: str) -> None:
        p.add_argument("pos_query", nargs="?", default=None, help=help_text)
        p.add_argument("-q", "--query", dest="opt_query", type=str, default=None, help=help_text)

    p_srch = sub.add_parser("search", help="Search the Sarvam Docs index (dense, bm25, hybrid, mrl_funnel)")
    add_query_arg(p_srch, "Search query")
    p_srch.add_argument("--mode", choices=["dense", "bm25", "hybrid", "mrl_funnel"], default="hybrid")
    p_srch.add_argument("--dim", type=int, choices=[768, 512, 256, 128], default=768)
    p_srch.add_argument("--top-k", type=int, default=5)
    p_srch.add_argument("--prompt-name", type=str, default="QuestionAnswering")
    p_srch.add_argument("--category", type=str, default="all")
    p_srch.set_defaults(func=cmd_search)

    p_ask = sub.add_parser("ask", help="Retrieve + synthesize a grounded answer via GCP Vertex AI Gemini")
    add_query_arg(p_ask, "Developer question")
    p_ask.add_argument("--mode", choices=["dense", "bm25", "hybrid", "mrl_funnel"], default="hybrid")
    p_ask.add_argument("--dim", type=int, choices=[768, 512, 256, 128], default=768)
    p_ask.add_argument("--top-k", type=int, default=5)
    p_ask.add_argument("--prompt-name", type=str, default="QuestionAnswering")
    p_ask.add_argument("--judge", action="store_true", help="Run Vertex AI LLM-as-a-Judge evaluation")
    p_ask.set_defaults(func=cmd_ask)

    p_mrl = sub.add_parser("mrl-benchmark", help="Compare Matryoshka dimensions (768d, 512d, 256d, 128d)")
    add_query_arg(p_mrl, "Query to benchmark across MRL dimensions")
    p_mrl.add_argument("--top-k", type=int, default=5)
    p_mrl.set_defaults(func=cmd_mrl_benchmark)

    p_mm = sub.add_parser("multimodal", help="Cross-modal search across Text, Images (<|image|>), and Audio (<|audio|>)")
    add_query_arg(p_mm, "Natural language query")
    p_mm.add_argument("--dim", type=int, choices=[768, 512, 256, 128], default=768)
    p_mm.add_argument("--top-k", type=int, default=5)
    p_mm.set_defaults(func=cmd_multimodal)

    p_gcp = sub.add_parser("gcp-sync", help="Upload vectors to GCP Cloud Storage & BigQuery Vector Search table")
    p_gcp.add_argument("--dim", type=int, choices=[768, 512, 256, 128], default=256)
    p_gcp.set_defaults(func=cmd_gcp_sync)

    p_bq = sub.add_parser("bq-search", help="Run cloud-native VECTOR_SEARCH SQL query on GCP BigQuery")
    add_query_arg(p_bq, "Search query")
    p_bq.add_argument("--dim", type=int, choices=[768, 512, 256, 128], default=256)
    p_bq.add_argument("--top-k", type=int, default=5)
    p_bq.set_defaults(func=cmd_bq_search)

    p_chat = sub.add_parser("chat", help="Interactive terminal RAG chat session")
    p_chat.add_argument("--mode", choices=["dense", "bm25", "hybrid", "mrl_funnel"], default="hybrid")
    p_chat.add_argument("--dim", type=int, choices=[768, 512, 256, 128], default=768)
    p_chat.add_argument("--top-k", type=int, default=5)
    p_chat.set_defaults(func=cmd_chat)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if hasattr(args, "opt_query") or hasattr(args, "pos_query"):
        args.query = getattr(args, "opt_query", None) or getattr(args, "pos_query", None)
        if not args.query:
            parser.error("A query string is required (pass it positionally or via --query / -q).")
    args.func(args)


if __name__ == "__main__":
    main()
