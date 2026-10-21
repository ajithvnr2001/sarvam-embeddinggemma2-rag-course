import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from src.config import CONFIG

TASK_PREFIX_MAP = {
    "SearchQuery": "task: search result | query: ",
    "QuestionAnswering": "task: question answering | query: ",
    "FactChecking": "task: fact checking | query: ",
    "CodeRetrieval": "task: code retrieval | query: ",
    "Classification": "task: classification | query: ",
    "Clustering": "task: clustering | query: ",
    "SentenceSimilarity": "task: sentence similarity | query: ",
}

MODALITY_CONFIG_KWARGS = {
    "text_only": {"vision_config": None, "audio_config": None},      # ~270M active params
    "text_image": {"audio_config": None},                            # ~440M active params
    "text_audio": {"vision_config": None},                           # ~570M active params
    "multimodal": {},                                                # ~740M active params
}


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize 1D or 2D numpy arrays safely."""
    if vectors.ndim == 1:
        norm = np.linalg.norm(vectors)
        return vectors / max(float(norm), 1e-12)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return vectors / norms


def mrl_truncate_and_renormalize(vectors: np.ndarray, dim: int) -> np.ndarray:
    """
    Matryoshka Representation Learning (MRL) truncation + mandatory L2 re-normalization.
    Supported dimensions for EmbeddingGemma 2: 768, 512, 256, 128.
    """
    if dim not in (768, 512, 256, 128):
        raise ValueError(f"Unsupported MRL dimension {dim}. Choose from 768, 512, 256, 128.")
    sliced = vectors[..., :dim].astype(np.float32)
    return l2_normalize(sliced)


class EmbeddingGemma2Engine:
    """
    Production wrapper around google/embeddinggemma-2 supporting:
    1. Selective modality encoder loading (text_only 270M, text_image 440M, text_audio 570M, multimodal 740M)
    2. Safe numerical precision (bfloat16 / float32 — never float16)
    3. Official asymmetric & symmetric task instruction prefixes
    4. Matryoshka Representation Learning (MRL) truncation & L2 re-normalization (768d, 512d, 256d, 128d)
    5. Multimodal & interleaved inputs (<|image|>, <|audio|>, <|video|>)
    """

    def __init__(
        self,
        model_id: str = CONFIG.embedding_model_id,
        modality_mode: str = "text_only",
        dtype: Optional[torch.dtype] = None,
    ):
        if modality_mode not in MODALITY_CONFIG_KWARGS:
            raise ValueError(f"Unknown modality_mode={modality_mode}. Options: {list(MODALITY_CONFIG_KWARGS)}")
        self.model_id = model_id
        self.modality_mode = modality_mode
        # Never use float16 on EmbeddingGemma 2 (causes activation overflow / NaNs).
        # Use bfloat16 on GPUs with native bf16 support, and float32 on CPUs (hardware AVX-512 acceleration).
        if dtype is None:
            dtype = (
                torch.bfloat16
                if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
                else torch.float32
            )
        self.dtype = dtype

        if CONFIG.hf_token:
            os.environ["HF_TOKEN"] = CONFIG.hf_token
            os.environ["HUGGING_FACE_HUB_TOKEN"] = CONFIG.hf_token

        # Prefer fast SSD cache (/root/.cache/huggingface) if available, fallback to /data/cache/huggingface
        ssd_cache = Path.home() / ".cache/huggingface/hub/models--google--embeddinggemma-2"
        data_cache = Path("/data/cache/huggingface/hub/models--google--embeddinggemma-2")
        if ssd_cache.exists():
            os.environ["HF_HOME"] = str(Path.home() / ".cache/huggingface")
            use_local_only = True
        elif data_cache.exists():
            os.environ["HF_HOME"] = "/data/cache/huggingface"
            use_local_only = True
        else:
            use_local_only = False

        t0 = time.time()
        self.model = SentenceTransformer(
            self.model_id,
            token=CONFIG.hf_token or None,
            local_files_only=use_local_only,
            model_kwargs={"torch_dtype": self.dtype},
            config_kwargs=MODALITY_CONFIG_KWARGS[self.modality_mode],
        )
        self.load_time_sec = round(time.time() - t0, 2)

    def encode_query(
        self,
        query: str,
        prompt_name: str = "QuestionAnswering",
        truncate_dim: int = 768,
    ) -> np.ndarray:
        """Encode a search/QA/code query using the official task instruction prefix."""
        vec = self.model.encode(
            query,
            prompt_name=prompt_name,
            truncate_dim=truncate_dim if truncate_dim < 768 else None,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return l2_normalize(vec.astype(np.float32))

    def encode_documents(
        self,
        formatted_docs: List[str],
        batch_size: int = 4,
        truncate_dim: int = 768,
        show_progress_bar: bool = True,
    ) -> np.ndarray:
        """
        Encode pre-formatted documents (`title: {title} | text: {content}`).
        Note: We do not pass prompt_name='Document' when documents already include a real title.
        """
        vecs = self.model.encode(
            formatted_docs,
            batch_size=batch_size,
            truncate_dim=truncate_dim if truncate_dim < 768 else None,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=show_progress_bar,
        )
        return l2_normalize(vecs.astype(np.float32))

    def encode_multimodal(
        self,
        inputs: Union[List[Any], Any],
        truncate_dim: int = 768,
        batch_size: int = 1,
    ) -> np.ndarray:
        """
        Encode multimodal items (PIL images, 16kHz mono audio arrays, or interleaved dicts
        with `<|image|>`, `<|audio|>`, `<|video|>` placeholders).
        """
        vecs = self.model.encode(
            inputs,
            batch_size=batch_size,
            truncate_dim=truncate_dim if truncate_dim < 768 else None,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return l2_normalize(vecs.astype(np.float32))
