import os
from pathlib import Path
from dataclasses import dataclass

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_dotenv_file(env_path: Path = PROJECT_ROOT / ".env") -> None:
    """Lightweight .env loader without external dependencies."""
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip("'").strip('"')
        if k and k not in os.environ:
            os.environ[k] = v


load_dotenv_file()


@dataclass
class RAGConfig:
    hf_token: str = os.environ.get("HF_TOKEN", "")
    embedding_model_id: str = os.environ.get("EMBEDDING_MODEL_ID", "google/embeddinggemma-2")
    gcp_project_id: str = os.environ.get("GCP_PROJECT_ID", "genai-511910")
    gcp_location: str = os.environ.get("GCP_LOCATION", "us-central1")
    gcs_bucket_name: str = os.environ.get("GCS_BUCKET_NAME", "genai-511910-sarvam-rag")
    bq_dataset_id: str = os.environ.get("BQ_DATASET_ID", "sarvam_rag")
    bq_table_id: str = os.environ.get("BQ_TABLE_ID", "docs_embeddings")
    vertex_model_id: str = os.environ.get("VERTEX_MODEL_ID", "gemini-2.5-flash")
    sarvam_docs_path: Path = Path(os.environ.get("SARVAM_DOCS_PATH", "/data/opencode/sarvam/docs"))
    sarvam_repo_root: Path = Path(os.environ.get("SARVAM_REPO_ROOT", "/data/opencode/sarvam"))
    index_dir: Path = PROJECT_ROOT / "data" / "index"
    multimodal_dir: Path = PROJECT_ROOT / "data" / "multimodal_samples"
    artifacts_dir: Path = PROJECT_ROOT / "data" / "artifacts"
    mrl_dimensions: tuple = (768, 512, 256, 128)


CONFIG = RAGConfig()
CONFIG.index_dir.mkdir(parents=True, exist_ok=True)
CONFIG.multimodal_dir.mkdir(parents=True, exist_ok=True)
CONFIG.artifacts_dir.mkdir(parents=True, exist_ok=True)
