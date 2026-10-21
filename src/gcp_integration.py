import json
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

from src.config import CONFIG
from src.chunker import DocChunk
from src.embedder import mrl_truncate_and_renormalize


def run_cmd(cmd: List[str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return proc.stdout.strip()


class GCPVectorPlatform:
    """
    Integrates the local EmbeddingGemma 2 + Sarvam Docs index with Google Cloud Platform:
    1. Google Cloud Storage (`gs://genai-511910-sarvam-rag`) for corpus & vector JSONL storage
    2. Google BigQuery (`genai-511910.sarvam_rag.docs_embeddings`) with native `ARRAY<FLOAT64>` vector column
    3. Cloud-native `VECTOR_SEARCH` SQL execution using COSINE distance on BigQuery
    """

    def __init__(
        self,
        project_id: str = CONFIG.gcp_project_id,
        location: str = CONFIG.gcp_location,
        bucket_name: str = CONFIG.gcs_bucket_name,
        dataset_id: str = CONFIG.bq_dataset_id,
        table_id: str = CONFIG.bq_table_id,
    ):
        self.project_id = project_id
        self.location = location
        self.bucket_name = bucket_name
        self.dataset_id = dataset_id
        self.table_id = table_id

    def export_bigquery_jsonl(
        self,
        chunks: List[DocChunk],
        embeddings_768: np.ndarray,
        out_jsonl: Path,
        dim: int = 256,
    ) -> Path:
        """
        Export chunks + Matryoshka-truncated (or full 768d) normalized embeddings as newline-delimited JSON
        ready for BigQuery `ARRAY<FLOAT64>` ingestion.
        """
        vecs = mrl_truncate_and_renormalize(embeddings_768, dim) if dim < 768 else embeddings_768
        out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with out_jsonl.open("w", encoding="utf-8") as f:
            for c, v in zip(chunks, vecs):
                row = {
                    "chunk_id": c.chunk_id,
                    "doc_path": c.doc_path,
                    "url": c.url,
                    "title": c.title,
                    "section": c.section,
                    "category": c.category,
                    "has_code": c.has_code,
                    "char_count": c.char_count,
                    "content": c.content,
                    "embedding_dim": dim,
                    "embedding": [round(float(x), 6) for x in v.tolist()],
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return out_jsonl

    def sync_to_gcs_and_bigquery(
        self,
        chunks: List[DocChunk],
        embeddings_768: np.ndarray,
        dim: int = 256,
    ) -> Dict[str, Any]:
        """
        1. Ensure GCS bucket `gs://<bucket_name>` exists
        2. Upload `bq_vectors.jsonl` to GCS
        3. Ensure BigQuery dataset `<dataset_id>` exists
        4. Load `bq_vectors.jsonl` into `<project_id>:<dataset_id>.<table_id>` with explicit schema
        """
        t0 = time.time()
        jsonl_path = CONFIG.index_dir / f"bq_vectors_{dim}d.jsonl"
        self.export_bigquery_jsonl(chunks, embeddings_768, jsonl_path, dim=dim)

        # 1. Create GCS bucket if not exists
        bucket_uri = f"gs://{self.bucket_name}"
        check_bucket = subprocess.run(
            ["gcloud", "storage", "buckets", "describe", bucket_uri, f"--project={self.project_id}"],
            capture_output=True,
            text=True,
        )
        if check_bucket.returncode != 0:
            run_cmd(
                [
                    "gcloud",
                    "storage",
                    "buckets",
                    "create",
                    bucket_uri,
                    f"--project={self.project_id}",
                    f"--location={self.location}",
                ]
            )

        # 2. Upload JSONL to GCS
        gcs_object_uri = f"{bucket_uri}/sarvam_rag/bq_vectors_{dim}d.jsonl"
        run_cmd(["gcloud", "storage", "cp", str(jsonl_path), gcs_object_uri, f"--project={self.project_id}"])

        # 3. Create BigQuery dataset if not exists
        subprocess.run(
            [
                "bq",
                f"--location={self.location}",
                "mk",
                "--dataset",
                f"--project_id={self.project_id}",
                f"{self.project_id}:{self.dataset_id}",
            ],
            capture_output=True,
            text=True,
        )

        # 4. Write schema JSON and load into BigQuery
        schema_path = CONFIG.index_dir / "bq_schema.json"
        schema = [
            {"name": "chunk_id", "type": "STRING", "mode": "REQUIRED"},
            {"name": "doc_path", "type": "STRING", "mode": "NULLABLE"},
            {"name": "url", "type": "STRING", "mode": "NULLABLE"},
            {"name": "title", "type": "STRING", "mode": "NULLABLE"},
            {"name": "section", "type": "STRING", "mode": "NULLABLE"},
            {"name": "category", "type": "STRING", "mode": "NULLABLE"},
            {"name": "has_code", "type": "BOOLEAN", "mode": "NULLABLE"},
            {"name": "char_count", "type": "INTEGER", "mode": "NULLABLE"},
            {"name": "content", "type": "STRING", "mode": "NULLABLE"},
            {"name": "embedding_dim", "type": "INTEGER", "mode": "NULLABLE"},
            {"name": "embedding", "type": "FLOAT64", "mode": "REPEATED"},
        ]
        schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")

        table_ref = f"{self.project_id}:{self.dataset_id}.{self.table_id}"
        run_cmd(
            [
                "bq",
                f"--location={self.location}",
                "load",
                "--replace",
                "--source_format=NEWLINE_DELIMITED_JSON",
                f"--project_id={self.project_id}",
                table_ref,
                gcs_object_uri,
                str(schema_path),
            ]
        )

        elapsed_sec = round(time.time() - t0, 2)
        return {
            "status": "SUCCESS",
            "project_id": self.project_id,
            "location": self.location,
            "gcs_uri": gcs_object_uri,
            "bigquery_table": f"{self.project_id}.{self.dataset_id}.{self.table_id}",
            "rows_uploaded": len(chunks),
            "embedding_dim": dim,
            "elapsed_sec": elapsed_sec,
        }

    def bigquery_vector_search(
        self,
        query_vec: np.ndarray,
        top_k: int = 5,
        dim: int = 256,
    ) -> Dict[str, Any]:
        """
        Execute a live cloud-native `VECTOR_SEARCH` SQL query on BigQuery!
        """
        t0 = time.perf_counter()
        q_trunc = mrl_truncate_and_renormalize(query_vec, dim) if len(query_vec) > dim else query_vec
        vec_literal = "[" + ", ".join(f"{float(x):.6f}" for x in q_trunc.tolist()) + "]"
        fq_table = f"`{self.project_id}.{self.dataset_id}.{self.table_id}`"

        sql = f"""
SELECT
  base.chunk_id AS chunk_id,
  base.title AS title,
  base.category AS category,
  base.url AS url,
  base.doc_path AS doc_path,
  ROUND(1.0 - distance, 5) AS cosine_similarity,
  ROUND(distance, 5) AS cosine_distance,
  SUBSTR(base.content, 1, 500) AS content_preview
FROM
  VECTOR_SEARCH(
    TABLE {fq_table},
    'embedding',
    (SELECT {vec_literal} AS query_embedding),
    'query_embedding',
    top_k => {int(top_k)},
    distance_type => 'COSINE'
  )
ORDER BY distance ASC
"""
        out = run_cmd(
            [
                "bq",
                f"--location={self.location}",
                "query",
                f"--project_id={self.project_id}",
                "--use_legacy_sql=false",
                "--format=json",
                sql,
            ]
        )
        rows = json.loads(out) if out else []
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return {
            "engine": "GCP BigQuery VECTOR_SEARCH (COSINE)",
            "table": f"{self.project_id}.{self.dataset_id}.{self.table_id}",
            "dimension": dim,
            "top_k": top_k,
            "latency_ms": elapsed_ms,
            "results": rows,
        }
