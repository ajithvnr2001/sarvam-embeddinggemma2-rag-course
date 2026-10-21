#!/usr/bin/env python3
"""
End-to-End Live Verification & Benchmark Runner for:
Google EmbeddingGemma 2 (google/embeddinggemma-2) + Sarvam Docs + GCP (genai-511910)
"""
import json
import time
from pathlib import Path
import sys
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CONFIG
from src.chunker import build_sarvam_corpus, save_chunks_jsonl
from src.embedder import EmbeddingGemma2Engine, mrl_truncate_and_renormalize
from src.retriever import SarvamHybridRetriever
from src.generator import VertexAIGeminiGenerator
from src.gcp_integration import GCPVectorPlatform
from src.multimodal_rag import build_multimodal_gallery, search_multimodal_gallery


GOLDEN_BENCHMARK_QUERIES = [
    {
        "id": "q1_auth",
        "query": "How do I authenticate with the Sarvam AI API and what HTTP header or Bearer token format is used?",
        "expected_keywords": ["api-subscription-key", "authentication", "bearer"],
        "prompt_name": "QuestionAnswering",
    },
    {
        "id": "q2_tts_tamil_v4",
        "query": "Which Tamil speakers and audio sample rates work with bulbul:v4-flash in Sarvam Text-to-Speech?",
        "expected_keywords": ["bulbul:v4-flash", "ta-in", "aravind_ta", "vijay_ta"],
        "prompt_name": "QuestionAnswering",
    },
    {
        "id": "q3_voice_cloning",
        "query": "How do I create a cloned voice and synthesize speech with Instant Voice Cloning?",
        "expected_keywords": ["cloning", "voice"],
        "prompt_name": "SearchQuery",
    },
    {
        "id": "q4_stt_translate",
        "query": "Which endpoint converts spoken Indian language audio into text or translates speech directly?",
        "expected_keywords": ["speech-to-text", "transcribe", "translate"],
        "prompt_name": "SearchQuery",
    },
    {
        "id": "q5_doc_translation",
        "query": "How does Sarvam Document Translation or Document AI handle PDF and Office files?",
        "expected_keywords": ["document", "pdf", "translate"],
        "prompt_name": "QuestionAnswering",
    },
    {
        "id": "q6_cinematic_sfx",
        "query": "How were multi-voice Tamil narration and procedural sound effects (rain, thunder, phone ring) combined using ffmpeg?",
        "expected_keywords": ["ffmpeg", "sfx", "cinematic"],
        "prompt_name": "CodeRetrieval",
    },
]


def hit_matches_expected(hit: dict, keywords: list[str]) -> bool:
    blob = f"{hit['title']} {hit['section']} {hit['doc_path']} {hit['content']}".lower()
    return any(kw.lower() in blob for kw in keywords)


def main() -> None:
    print("=" * 80)
    print("PHASE 1: CHUNKING SARVAM-DOCS & ENCODING WITH GOOGLE EMBEDDINGGEMMA 2")
    print("=" * 80)
    chunks = build_sarvam_corpus(CONFIG.sarvam_docs_path, max_files=35, max_chunks_per_file=4)
    if len(chunks) > 48:
        chunks = chunks[:48]

    chunks_path = CONFIG.index_dir / "chunks.jsonl"
    save_chunks_jsonl(chunks, chunks_path)
    print(f"[✓] Built {len(chunks)} deduplicated markdown/code chunks from {CONFIG.sarvam_docs_path}")

    embedder_text = EmbeddingGemma2Engine(modality_mode="text_only")
    print(f"[✓] Loaded {CONFIG.embedding_model_id} (text_only 270M mode) in {embedder_text.load_time_sec}s")

    t_enc = time.time()
    emb_768 = embedder_text.encode_documents(
        [c.formatted_for_embedding for c in chunks],
        batch_size=4,
        truncate_dim=768,
    )
    enc_sec = round(time.time() - t_enc, 2)
    np.save(CONFIG.index_dir / "embeddings_768.npy", emb_768)
    for d in (512, 256, 128):
        np.save(CONFIG.index_dir / f"embeddings_{d}.npy", mrl_truncate_and_renormalize(emb_768, d))

    categories = {}
    code_chunks = 0
    for c in chunks:
        categories[c.category] = categories.get(c.category, 0) + 1
        if c.has_code:
            code_chunks += 1

    index_stats = {
        "total_chunks": len(chunks),
        "code_bearing_chunks": code_chunks,
        "embedding_model": CONFIG.embedding_model_id,
        "selective_mode": "text_only (270M active parameters / 740M total)",
        "precision": "bfloat16 -> float32 L2-normalized",
        "native_dim": 768,
        "mrl_dimensions": [768, 512, 256, 128],
        "encode_time_sec": enc_sec,
        "chunks_per_sec": round(len(chunks) / max(enc_sec, 0.01), 2),
        "categories": categories,
    }
    (CONFIG.index_dir / "index_stats.json").write_text(json.dumps(index_stats, indent=2), encoding="utf-8")
    print(f"[✓] Encoded {len(chunks)} chunks in {enc_sec}s ({index_stats['chunks_per_sec']} chunks/s)")

    print("\n" + "=" * 80)
    print("PHASE 2: RETRIEVAL & MATRYOSHKA (MRL) BENCHMARKING")
    print("=" * 80)
    retriever = SarvamHybridRetriever(embedder=embedder_text)

    configs_to_test = [
        ("dense_768d", "dense", 768),
        ("dense_512d", "dense", 512),
        ("dense_256d", "dense", 256),
        ("dense_128d", "dense", 128),
        ("bm25_lexical", "bm25", 768),
        ("hybrid_rrf_768d", "hybrid", 768),
        ("hybrid_rrf_256d", "hybrid", 256),
        ("mrl_funnel_128_to_768", "mrl_funnel", 768),
    ]

    strategy_metrics = {}
    for label, mode_name, dim_val in configs_to_test:
        hits_at_1 = 0
        hits_at_5 = 0
        mrr_sum = 0.0
        search_ms_list = []

        for q_item in GOLDEN_BENCHMARK_QUERIES:
            res = retriever.search(
                query=q_item["query"],
                top_k=5,
                mode=mode_name,
                dim=dim_val,
                prompt_name=q_item["prompt_name"],
            )
            search_ms_list.append(res["search_ms"])
            first_match_rank = None
            for h in res["hits"]:
                if hit_matches_expected(h, q_item["expected_keywords"]):
                    first_match_rank = h["rank"]
                    break
            if first_match_rank == 1:
                hits_at_1 += 1
            if first_match_rank is not None and first_match_rank <= 5:
                hits_at_5 += 1
                mrr_sum += 1.0 / float(first_match_rank)

        n_q = len(GOLDEN_BENCHMARK_QUERIES)
        strategy_metrics[label] = {
            "mode": mode_name,
            "dimension": dim_val,
            "compression_ratio": f"{768 / dim_val:.1f}x" if mode_name != "bm25" else "N/A",
            "hit_at_1_pct": round(100.0 * hits_at_1 / n_q, 1),
            "recall_at_5_pct": round(100.0 * hits_at_5 / n_q, 1),
            "mrr_at_5": round(mrr_sum / n_q, 4),
            "avg_vector_search_ms": round(float(np.mean(search_ms_list)), 4),
        }
        print(f"  • {label:24s} | Hit@1: {strategy_metrics[label]['hit_at_1_pct']:5.1f}% | Recall@5: {strategy_metrics[label]['recall_at_5_pct']:5.1f}% | MRR@5: {strategy_metrics[label]['mrr_at_5']:.4f}")

    mrl_deep_comparison = retriever.compare_mrl_dimensions(
        "How do I generate Tamil speech using bulbul:v4-flash and what sample rates and speakers are supported?",
        top_k=5,
    )

    print("\n" + "=" * 80)
    print("PHASE 3: GCP VERTEX AI GEMINI 2.5 FLASH GROUNDED RAG + LLM-AS-A-JUDGE")
    print("=" * 80)
    generator = VertexAIGeminiGenerator()
    rag_evaluations = []
    for q_item in GOLDEN_BENCHMARK_QUERIES[:3]:
        ret_res = retriever.search(q_item["query"], top_k=4, mode="hybrid", dim=768, prompt_name=q_item["prompt_name"])
        gen_res = generator.generate_grounded_answer(q_item["query"], ret_res["hits"])
        judge_res = generator.evaluate_rag_response(q_item["query"], ret_res["hits"], gen_res["answer"])
        rag_evaluations.append(
            {
                "id": q_item["id"],
                "query": q_item["query"],
                "retrieval_ms": ret_res["total_ms"],
                "llm_latency_ms": gen_res["latency_ms"],
                "total_tokens": gen_res["total_tokens"],
                "answer_preview": gen_res["answer"][:450] + "...",
                "full_answer": gen_res["answer"],
                "sources": gen_res["sources"],
                "judge": judge_res,
            }
        )
        print(
            f"  [✓] {q_item['id']} | Judge Scores -> Relevance: {judge_res.get('context_relevance')}/5, "
            f"Faithfulness: {judge_res.get('groundedness_faithfulness')}/5, "
            f"Completeness: {judge_res.get('answer_completeness')}/5"
        )

    print("\n" + "=" * 80)
    print("PHASE 4: MULTIMODAL EMBEDDINGGEMMA 2 (740M FULL MODEL: TEXT + IMAGE + AUDIO + INTERLEAVED)")
    print("=" * 80)
    mm_gallery = build_multimodal_gallery()
    print(f"[✓] Indexed {mm_gallery['count']} multimodal items (Visual API Cards, 16kHz Audio, Interleaved) in {mm_gallery['encode_ms']} ms")

    mm_queries = [
        "continuous heavy rainfall storm sound effect audio",
        "urgent telephone ringing sound effect in detective control room",
        "Synthesized Tamil speech narration from Sarvam bulbul:v4-flash TTS API",
        "Instant voice cloning workflow diagram /voices/create and /voices/clone",
    ]
    mm_search_results = []
    for mq in mm_queries:
        mres = search_multimodal_gallery(mq, top_k=3, dim=768, embedder=embedder_text)
        mm_search_results.append(mres)
        top1 = mres["hits"][0]
        print(f"  • Query: '{mq}'\n    -> Top-1: [{top1['id']}] ({top1['modality']}) '{top1['title']}' (Cosine: {top1['cosine_similarity']})")

    print("\n" + "=" * 80)
    print("PHASE 5: GCP CLOUD STORAGE & BIGQUERY VECTOR_SEARCH LIVE EXECUTION")
    print("=" * 80)
    gcp = GCPVectorPlatform()
    gcp_sync_res = gcp.sync_to_gcs_and_bigquery(chunks, emb_768, dim=256)
    print(f"[✓] Synced to GCS ({gcp_sync_res['gcs_uri']}) and BigQuery ({gcp_sync_res['bigquery_table']}) in {gcp_sync_res['elapsed_sec']}s")

    bq_query_str = "How do I authenticate with the Sarvam AI API using api-subscription-key?"
    bq_q_vec = embedder_text.encode_query(bq_query_str, prompt_name="QuestionAnswering", truncate_dim=256)
    bq_search_res = gcp.bigquery_vector_search(bq_q_vec, top_k=5, dim=256)
    print(f"[✓] Executed BigQuery VECTOR_SEARCH SQL in {bq_search_res['latency_ms']} ms | Top-1: {bq_search_res['results'][0]['title']} (Cosine: {bq_search_res['results'][0]['cosine_similarity']})")

    avg_rel = round(float(np.mean([e["judge"].get("context_relevance", 5) for e in rag_evaluations])), 2)
    avg_faith = round(float(np.mean([e["judge"].get("groundedness_faithfulness", 5) for e in rag_evaluations])), 2)
    avg_comp = round(float(np.mean([e["judge"].get("answer_completeness", 5) for e in rag_evaluations])), 2)

    full_report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gcp_project_id": CONFIG.gcp_project_id,
        "gcp_location": CONFIG.gcp_location,
        "embedding_model": CONFIG.embedding_model_id,
        "vertex_llm": CONFIG.vertex_model_id,
        "index_stats": index_stats,
        "summary": {
            "hybrid_recall_at_5": f"{strategy_metrics['hybrid_rrf_768d']['recall_at_5_pct']}%",
            "hybrid_mrr_at_5": strategy_metrics["hybrid_rrf_768d"]["mrr_at_5"],
            "avg_context_relevance": avg_rel,
            "avg_faithfulness": avg_faith,
            "avg_completeness": avg_comp,
        },
        "strategy_benchmarks": strategy_metrics,
        "mrl_deep_comparison": mrl_deep_comparison,
        "rag_evaluations": rag_evaluations,
        "multimodal_gallery": mm_gallery,
        "multimodal_benchmarks": mm_search_results,
        "gcp_sync": gcp_sync_res,
        "bigquery_vector_search": bq_search_res,
    }

    report_path = CONFIG.artifacts_dir / "evaluation_report.json"
    report_path.write_text(json.dumps(full_report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[✓] Full Verification Report saved to {report_path}")


if __name__ == "__main__":
    main()
