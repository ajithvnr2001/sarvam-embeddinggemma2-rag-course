import json
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from rank_bm25 import BM25Okapi

from src.chunker import DocChunk, load_chunks_jsonl
from src.embedder import EmbeddingGemma2Engine, mrl_truncate_and_renormalize, l2_normalize
from src.config import CONFIG


def tokenize_for_bm25(text: str) -> List[str]:
    """
    Tokenizer preserving technical identifiers (`bulbul:v4-flash`, `api-subscription-key`, `ta-IN`, `/text-to-speech`).
    """
    return re.findall(r"[a-zA-Z0-9_:\-/\.]{2,}", text.lower())


@dataclass
class RetrievalHit:
    rank: int
    chunk_id: str
    title: str
    section: str
    category: str
    url: str
    doc_path: str
    has_code: bool
    score: float
    dense_score: Optional[float]
    bm25_score: Optional[float]
    content: str
    retrieval_mode: str
    dimension: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SarvamHybridRetriever:
    """
    Multi-strategy RAG Retriever over Sarvam AI documentation:
    - Dense Vector Search across MRL dimensions (768d, 512d, 256d, 128d)
    - BM25 Lexical Search for exact API endpoints, headers, speaker IDs, and error codes
    - Hybrid Search via Reciprocal Rank Fusion (RRF)
    - Two-Stage Matryoshka Funnel Retrieval (128d/256d fast candidate scan -> 768d exact rescoring)
    """

    def __init__(
        self,
        index_dir: Path = CONFIG.index_dir,
        embedder: Optional[EmbeddingGemma2Engine] = None,
    ):
        self.index_dir = index_dir
        self.chunks_path = index_dir / "chunks.jsonl"
        self.embeddings_768_path = index_dir / "embeddings_768.npy"

        if not self.chunks_path.exists() or not self.embeddings_768_path.exists():
            raise FileNotFoundError(
                f"Index files not found in {index_dir}. Run `python -m cli.rag_cli ingest` first."
            )

        self.chunks: List[DocChunk] = load_chunks_jsonl(self.chunks_path)
        self.emb_768 = l2_normalize(np.load(self.embeddings_768_path).astype(np.float32))

        # Precompute Matryoshka truncated + L2 re-normalized matrices in memory
        self.emb_by_dim: Dict[int, np.ndarray] = {
            768: self.emb_768,
            512: mrl_truncate_and_renormalize(self.emb_768, 512),
            256: mrl_truncate_and_renormalize(self.emb_768, 256),
            128: mrl_truncate_and_renormalize(self.emb_768, 128),
        }

        # Build BM25 index over title + section + content
        tokenized_corpus = [
            tokenize_for_bm25(f"{c.title} {c.section} {c.category} {c.content}")
            for c in self.chunks
        ]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.embedder = embedder

    def _ensure_embedder(self) -> EmbeddingGemma2Engine:
        if self.embedder is None:
            self.embedder = EmbeddingGemma2Engine(modality_mode="text_only")
        return self.embedder

    def search(
        self,
        query: str,
        top_k: int = 5,
        mode: str = "hybrid",
        dim: int = 768,
        prompt_name: str = "QuestionAnswering",
        category_filter: Optional[str] = None,
        funnel_shortlist_k: int = 25,
        funnel_stage1_dim: int = 128,
        rrf_k: int = 60,
    ) -> Dict[str, Any]:
        """
        Execute retrieval with detailed latency & scoring telemetry.
        Modes: `dense`, `bm25`, `hybrid`, `mrl_funnel`.
        """
        t0 = time.perf_counter()
        if dim not in self.emb_by_dim:
            raise ValueError(f"Unsupported MRL dimension {dim}. Choose from 768, 512, 256, 128.")

        valid_mask = np.ones(len(self.chunks), dtype=bool)
        if category_filter and category_filter != "all":
            valid_mask = np.array([c.category == category_filter for c in self.chunks], dtype=bool)
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) == 0:
            return {"hits": [], "latency_ms": 0.0, "mode": mode, "dim": dim}

        dense_scores_full: Optional[np.ndarray] = None
        bm25_scores_full: Optional[np.ndarray] = None
        encode_ms = 0.0

        if mode in ("dense", "hybrid", "mrl_funnel"):
            t_enc = time.perf_counter()
            q_emb_768 = self._ensure_embedder().encode_query(query, prompt_name=prompt_name, truncate_dim=768)
            encode_ms = (time.perf_counter() - t_enc) * 1000.0

        t_search = time.perf_counter()

        if mode == "dense":
            q_vec = mrl_truncate_and_renormalize(q_emb_768, dim) if dim < 768 else q_emb_768
            dense_scores_full = self.emb_by_dim[dim] @ q_vec
            masked_scores = dense_scores_full[valid_indices]
            top_rel = np.argsort(-masked_scores)[:top_k]
            chosen_indices = valid_indices[top_rel]
            final_scores = [float(dense_scores_full[i]) for i in chosen_indices]

        elif mode == "bm25":
            q_tokens = tokenize_for_bm25(query)
            bm25_scores_full = np.array(self.bm25.get_scores(q_tokens), dtype=np.float32)
            masked_scores = bm25_scores_full[valid_indices]
            top_rel = np.argsort(-masked_scores)[:top_k]
            chosen_indices = valid_indices[top_rel]
            final_scores = [float(bm25_scores_full[i]) for i in chosen_indices]

        elif mode == "hybrid":
            # Dense at requested MRL dimension + BM25 fused via Reciprocal Rank Fusion (RRF)
            q_vec = mrl_truncate_and_renormalize(q_emb_768, dim) if dim < 768 else q_emb_768
            dense_scores_full = self.emb_by_dim[dim] @ q_vec

            q_tokens = tokenize_for_bm25(query)
            bm25_scores_full = np.array(self.bm25.get_scores(q_tokens), dtype=np.float32)

            dense_order = valid_indices[np.argsort(-dense_scores_full[valid_indices])]
            bm25_order = valid_indices[np.argsort(-bm25_scores_full[valid_indices])]

            dense_rank = {int(idx): r + 1 for r, idx in enumerate(dense_order)}
            bm25_rank = {int(idx): r + 1 for r, idx in enumerate(bm25_order)}

            rrf_scores: Dict[int, float] = {}
            for idx in valid_indices:
                i = int(idx)
                # Weight dense 0.65 and lexical BM25 0.35
                rrf_scores[i] = (0.65 / (rrf_k + dense_rank[i])) + (0.35 / (rrf_k + bm25_rank[i]))

            chosen_indices = sorted(valid_indices, key=lambda i: -rrf_scores[int(i)])[:top_k]
            final_scores = [float(rrf_scores[int(i)]) for i in chosen_indices]

        elif mode == "mrl_funnel":
            # Stage 1: Ultra-fast shortlist at funnel_stage1_dim (e.g. 128d)
            q_short = mrl_truncate_and_renormalize(q_emb_768, funnel_stage1_dim)
            stage1_scores = self.emb_by_dim[funnel_stage1_dim][valid_indices] @ q_short
            shortlist_rel = np.argsort(-stage1_scores)[: max(top_k, funnel_shortlist_k)]
            shortlist_indices = valid_indices[shortlist_rel]

            # Stage 2: Exact rerank using full 768d vectors only on shortlisted candidates
            dense_scores_full = np.zeros(len(self.chunks), dtype=np.float32)
            stage2_scores = self.emb_768[shortlist_indices] @ q_emb_768
            for idx_val, sc in zip(shortlist_indices, stage2_scores):
                dense_scores_full[int(idx_val)] = float(sc)

            reranked_order = np.argsort(-stage2_scores)[:top_k]
            chosen_indices = shortlist_indices[reranked_order]
            final_scores = [float(dense_scores_full[int(i)]) for i in chosen_indices]
        else:
            raise ValueError(f"Unknown retrieval mode: {mode}")

        search_ms = (time.perf_counter() - t_search) * 1000.0
        total_ms = (time.perf_counter() - t0) * 1000.0

        hits: List[RetrievalHit] = []
        for rank_idx, (doc_idx, score_val) in enumerate(zip(chosen_indices, final_scores), start=1):
            c = self.chunks[int(doc_idx)]
            hits.append(
                RetrievalHit(
                    rank=rank_idx,
                    chunk_id=c.chunk_id,
                    title=c.title,
                    section=c.section,
                    category=c.category,
                    url=c.url,
                    doc_path=c.doc_path,
                    has_code=c.has_code,
                    score=round(score_val, 5),
                    dense_score=round(float(dense_scores_full[int(doc_idx)]), 5)
                    if dense_scores_full is not None
                    else None,
                    bm25_score=round(float(bm25_scores_full[int(doc_idx)]), 4)
                    if bm25_scores_full is not None
                    else None,
                    content=c.content,
                    retrieval_mode=mode,
                    dimension=dim if mode != "mrl_funnel" else 768,
                )
            )

        return {
            "query": query,
            "mode": mode,
            "dimension": dim,
            "prompt_name": prompt_name,
            "encode_ms": round(encode_ms, 2),
            "search_ms": round(search_ms, 3),
            "total_ms": round(total_ms, 2),
            "hits": [h.to_dict() for h in hits],
        }

    def compare_mrl_dimensions(self, query: str, top_k: int = 5, prompt_name: str = "QuestionAnswering") -> Dict[str, Any]:
        """Compare 768d, 512d, 256d, and 128d retrieval side-by-side on the same query."""
        q_emb_768 = self._ensure_embedder().encode_query(query, prompt_name=prompt_name, truncate_dim=768)
        baseline_hits = None
        results_by_dim = {}

        for d in (768, 512, 256, 128):
            t0 = time.perf_counter()
            q_vec = mrl_truncate_and_renormalize(q_emb_768, d) if d < 768 else q_emb_768
            scores = self.emb_by_dim[d] @ q_vec
            top_idx = np.argsort(-scores)[:top_k]
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            top_ids = [self.chunks[int(i)].chunk_id for i in top_idx]
            if d == 768:
                baseline_hits = set(top_ids)
            overlap = len(set(top_ids) & baseline_hits) / float(top_k) if baseline_hits else 1.0

            results_by_dim[str(d)] = {
                "dimension": d,
                "compression_ratio": f"{768 / d:.1f}x",
                "bytes_per_vector_fp32": d * 4,
                "search_ms": round(elapsed_ms, 4),
                "top1_chunk_id": top_ids[0],
                "top1_title": self.chunks[int(top_idx[0])].title,
                "top1_cosine": round(float(scores[int(top_idx[0])]), 5),
                "topk_overlap_with_768d": round(overlap * 100.0, 1),
                "top_hits": [
                    {
                        "rank": r + 1,
                        "chunk_id": self.chunks[int(i)].chunk_id,
                        "title": self.chunks[int(i)].title,
                        "cosine": round(float(scores[int(i)]), 5),
                    }
                    for r, i in enumerate(top_idx)
                ],
            }

        return {"query": query, "top_k": top_k, "dimensions": results_by_dim}
