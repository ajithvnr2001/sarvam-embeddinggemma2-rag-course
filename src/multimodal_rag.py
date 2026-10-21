import json
import shutil
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import librosa
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.config import CONFIG
from src.embedder import EmbeddingGemma2Engine, l2_normalize, mrl_truncate_and_renormalize


def create_visual_doc_card(
    out_path: Path,
    badge: str,
    title: str,
    endpoint: str,
    bullets: List[str],
    accent_rgb: tuple = (56, 189, 248),
) -> Path:
    """Create a clean visual API documentation diagram/card image for multimodal vision indexing."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (760, 420), color=(15, 23, 42))
    draw = ImageDraw.Draw(img)

    # Top accent bar & card border
    draw.rectangle([0, 0, 760, 10], fill=accent_rgb)
    draw.rectangle([16, 22, 744, 404], outline=(51, 65, 85), width=2)

    # Badge pill
    draw.rectangle([36, 40, 260, 72], fill=accent_rgb)
    draw.text((48, 49), badge, fill=(15, 23, 42))

    # Title & Endpoint box
    draw.text((36, 92), title, fill=(248, 250, 252))
    draw.rectangle([36, 132, 720, 174], fill=(30, 41, 59), outline=accent_rgb, width=1)
    draw.text((50, 146), endpoint, fill=accent_rgb)

    # Bullet points
    y = 198
    for b in bullets:
        draw.ellipse([40, y + 4, 50, y + 14], fill=accent_rgb)
        draw.text((64, y), b, fill=(203, 213, 225))
        y += 38

    draw.text((36, 372), "Sarvam AI Documentation  |  Google EmbeddingGemma 2 Multimodal RAG", fill=(100, 116, 139))
    img.save(out_path)
    return out_path


def build_multimodal_gallery(
    embedder: Optional[EmbeddingGemma2Engine] = None,
) -> Dict[str, Any]:
    """
    Build and index a real multimodal corpus combining:
    - Visual API Architecture Cards (Image modality `<|image|>`)
    - Acoustic Sound Effects & Sarvam Tamil TTS Speech Clips (Audio modality `<|audio|>`, 16kHz mono)
    - Interleaved Text + Image + Audio records (`<|image|>` + `<|audio|>`)
    """
    mm_dir = CONFIG.multimodal_dir
    mm_dir.mkdir(parents=True, exist_ok=True)

    # 1. Create Visual API Cards (Image Modality)
    card_specs = [
        {
            "id": "mm_img_tts",
            "modality": "image",
            "title": "Sarvam Bulbul v4 Flash Text-to-Speech Visual Architecture Card",
            "description": "Visual API specification diagram for POST /text-to-speech using bulbul:v4-flash with pitch, pace, loudness, and 44.1kHz audio.",
            "file": create_visual_doc_card(
                mm_dir / "sarvam_tts_bulbul_v4_card.png",
                badge="SARVAM TTS API",
                title="Text-to-Speech (bulbul:v4-flash & bulbul:v3)",
                endpoint="POST https://api.sarvam.ai/text-to-speech  |  Header: api-subscription-key",
                bullets=[
                    "Models: bulbul:v4-flash (low-latency expressive), bulbul:v3 (multi-speaker)",
                    "Controls: target_language_code (ta-IN, hi-IN), speaker, pace, pitch, loudness",
                    "Sample Rates: 8000, 16000, 22050, 24000, 44100, 48000 Hz (MP3 / WAV / PCM)",
                    "Max Input: 2500 characters per REST call with enable_preprocessing=true",
                ],
                accent_rgb=(56, 189, 248),
            ),
            "caption_text": "Sarvam AI Text-to-Speech API diagram for bulbul:v4-flash POST /text-to-speech endpoint with speaker, pace, pitch, loudness, and 44100 Hz sample rate <|image|>",
        },
        {
            "id": "mm_img_stt",
            "modality": "image",
            "title": "Sarvam Saaras & Saarika Speech-to-Text Visual Pipeline Card",
            "description": "Visual architecture card for Sarvam Speech-to-Text (/speech-to-text) and Speech-to-Text Translate (/speech-to-text-translate).",
            "file": create_visual_doc_card(
                mm_dir / "sarvam_stt_saaras_card.png",
                badge="SARVAM STT API",
                title="Speech-to-Text & Speech Translation (Saaras / Saarika)",
                endpoint="POST https://api.sarvam.ai/speech-to-text  &  /speech-to-text-translate",
                bullets=[
                    "Transcribes spoken Indian languages (Hindi, Tamil, Telugu, Kannada, Malayalam)",
                    "Supports real-time WebSocket streaming and batch REST file transcription",
                    "Speaker diarization, word-level timestamps, and automatic code-mixing support",
                    "Direct speech-to-English translation via Saaras model architecture",
                ],
                accent_rgb=(168, 85, 247),
            ),
            "caption_text": "Sarvam AI Speech-to-Text and Speech Translation architecture diagram for POST /speech-to-text with diarization and timestamps <|image|>",
        },
        {
            "id": "mm_img_cloning",
            "modality": "image",
            "title": "Sarvam Instant Voice Cloning & Dubbing Workflow Diagram",
            "description": "Visual workflow diagram showing /voices/create and /voices/clone for instant speaker voice cloning.",
            "file": create_visual_doc_card(
                mm_dir / "sarvam_voice_cloning_card.png",
                badge="VOICE CLONING",
                title="Instant Voice Cloning & Multilingual Studio Dubbing",
                endpoint="POST https://api.sarvam.ai/voices/create  ->  POST /voices/clone",
                bullets=[
                    "Step 1: Upload reference voice sample to /voices/create to obtain voice_id",
                    "Step 2: Synthesize speech in 10+ Indic languages with /voices/clone + voice_id",
                    "Best practices: Clean 16kHz-44.1kHz mono reference audio without background noise",
                    "Supports cross-lingual voice preservation and automated video dubbing",
                ],
                accent_rgb=(52, 211, 153),
            ),
            "caption_text": "Sarvam AI Instant Voice Cloning workflow diagram for /voices/create and /voices/clone with voice_id <|image|>",
        },
        {
            "id": "mm_img_mrl",
            "modality": "image",
            "title": "Google EmbeddingGemma 2 Multimodal & Matryoshka (MRL) Architecture Diagram",
            "description": "Visual architecture overview of EmbeddingGemma 2 (740M total params: 270M text + 170M vision + 300M audio) and MRL 768d/512d/256d/128d.",
            "file": create_visual_doc_card(
                mm_dir / "embeddinggemma2_mrl_card.png",
                badge="EMBEDDINGGEMMA 2",
                title="Unified 768d Multimodal Vector Space & Matryoshka Truncation",
                endpoint="google/embeddinggemma-2  |  740M Params (270M Text + 170M Vision + 300M Audio)",
                bullets=[
                    "Maps Text, Code, Images (<|image|>), Audio (<|audio|>), Video (<|video|>) to 768d",
                    "Matryoshka Representation Learning (MRL): 768d (1x), 512d (1.5x), 256d (3x), 128d (6x)",
                    "Selective Encoder Loading: Load text-only in 270M params to cut RAM by 63%",
                    "Precision rule: Run in bfloat16 or float32 (never float16) + L2 re-normalize after MRL",
                ],
                accent_rgb=(251, 191, 36),
            ),
            "caption_text": "Google EmbeddingGemma 2 architecture diagram showing 740M parameters, 270M text backbone, 170M vision encoder, 300M audio encoder, and Matryoshka MRL truncation <|image|>",
        },
    ]

    # 2. Prepare Audio Samples from sarvam-docs repo (16kHz mono for EmbeddingGemma 2's 300M audio encoder)
    sarvam_root = CONFIG.sarvam_repo_root
    audio_specs = [
        {
            "id": "mm_aud_rain",
            "modality": "audio",
            "title": "Cinematic Rain Ambience Bed (sfx/rain.wav)",
            "description": "Continuous rainfall acoustic sound effect bed used in the Sarvam Tamil thriller cinematic narration pipeline.",
            "src_path": sarvam_root / "sfx" / "rain.wav",
            "prompt_text": "Acoustic environmental sound effect from Sarvam cinematic audio pipeline: continuous heavy rain shower ambience <|audio|>",
        },
        {
            "id": "mm_aud_thunder",
            "modality": "audio",
            "title": "Thunder Crack Sound Effect (sfx/thunder.wav)",
            "description": "Loud thunderclap acoustic cue triggered by [SFX: thunder] in Sarvam multi-voice story narration.",
            "src_path": sarvam_root / "sfx" / "thunder.wav",
            "prompt_path_note": "sfx/thunder.wav",
            "prompt_text": "Acoustic sound effect cue from Sarvam cinematic story generator: loud thunder crack and storm rumble <|audio|>",
        },
        {
            "id": "mm_aud_phone",
            "modality": "audio",
            "title": "Telephone Ring Cue (sfx/phone_ring.wav)",
            "description": "Urgent midnight telephone ring sound effect triggered in the DSP Vivek Tamil thriller audio drama.",
            "src_path": sarvam_root / "sfx" / "phone_ring.wav",
            "prompt_text": "Acoustic sound effect cue: urgent telephone ringing alert in detective control room <|audio|>",
        },
        {
            "id": "mm_aud_tts_narrator",
            "modality": "audio",
            "title": "Sarvam Bulbul v4 Tamil Narrator Speech (line_01_narrator.mp3)",
            "description": "Synthesized Tamil speech generated by Sarvam `bulbul:v4-flash` (speaker `vijay_ta_narration`, pace 0.90): 'சென்னை மாநகரத்தின் அந்த நள்ளிரவு மழை...'",
            "src_path": sarvam_root / "cinematic_chunks" / "line_01_narrator.mp3",
            "prompt_text": "Synthesized Tamil speech audio clip from Sarvam bulbul:v4-flash TTS API (vijay_ta_narration speaker, ta-IN): dramatic midnight rain story narration <|audio|>",
        },
        {
            "id": "mm_aud_tts_vivek",
            "modality": "audio",
            "title": "Sarvam Bulbul v4 DSP Vivek Dialogue Speech (line_03_vivek.mp3)",
            "description": "Synthesized Tamil character dialogue generated by Sarvam `bulbul:v4-flash` (speaker `aravind_ta_suspense`, pace 0.98): 'விவேக் ஹியர்... சொல்லுங்க கோபி!'",
            "src_path": sarvam_root / "cinematic_chunks" / "line_03_vivek.mp3",
            "prompt_text": "Synthesized Tamil character dialogue speech from Sarvam bulbul:v4-flash TTS API (aravind_ta_suspense speaker, ta-IN): DSP Vivek answering phone call <|audio|>",
        },
    ]

    # 3. Interleaved Multimodal Item (Text + Image + Audio in one shared 8,192-token window!)
    if embedder is None or embedder.modality_mode != "multimodal":
        embedder = EmbeddingGemma2Engine(modality_mode="multimodal")

    encode_inputs = []
    metadata_records = []

    for c in card_specs:
        pil_img = Image.open(c["file"]).convert("RGB")
        encode_inputs.append({"text": c["caption_text"], "image": pil_img})
        metadata_records.append(
            {
                "id": c["id"],
                "modality": "image",
                "title": c["title"],
                "description": c["description"],
                "asset_path": str(c["file"].relative_to(CONFIG.index_dir.parent.parent)),
                "token_cost_estimate": 280 + len(c["caption_text"]) // 4,
            }
        )

    loaded_wavs = {}
    for a in audio_specs:
        if not a["src_path"].exists():
            continue
        dst_file = mm_dir / a["src_path"].name
        if not dst_file.exists():
            shutil.copy2(a["src_path"], dst_file)
        wav, _ = librosa.load(str(dst_file), sr=16000, mono=True)
        # Keep up to 6 seconds (150 audio tokens @ 25 tokens/sec)
        wav_clip = wav[: 16000 * 6].astype(np.float32)
        loaded_wavs[a["id"]] = wav_clip
        duration_sec = round(len(wav_clip) / 16000.0, 2)
        encode_inputs.append({"text": a["prompt_text"], "audio": wav_clip})
        metadata_records.append(
            {
                "id": a["id"],
                "modality": "audio",
                "title": a["title"],
                "description": a["description"],
                "asset_path": str(dst_file.relative_to(CONFIG.index_dir.parent.parent)),
                "duration_sec": duration_sec,
                "token_cost_estimate": int(duration_sec * 25) + len(a["prompt_text"]) // 4,
            }
        )

    # Add an interleaved Text + Image + Audio record
    if "mm_aud_tts_narrator" in loaded_wavs:
        tts_img = Image.open(card_specs[0]["file"]).convert("RGB")
        interleaved_text = (
            "Complete Sarvam Bulbul v4 Flash Text-to-Speech specification card <|image|> "
            "paired with live synthesized Tamil narration audio output <|audio|>."
        )
        encode_inputs.append(
            {
                "text": interleaved_text,
                "image": tts_img,
                "audio": loaded_wavs["mm_aud_tts_narrator"],
            }
        )
        metadata_records.append(
            {
                "id": "mm_interleaved_tts_bundle",
                "modality": "interleaved (text+image+audio)",
                "title": "Interleaved Sarvam TTS Spec Card + Live Tamil Speech Sample",
                "description": "Single unified 768d vector combining text description, visual API card (<|image|>), and synthesized 16kHz Tamil speech (<|audio|>).",
                "asset_path": "data/multimodal_samples/sarvam_tts_bulbul_v4_card.png + line_01_narrator.mp3",
                "token_cost_estimate": 280 + 150 + 30,
            }
        )

    t0 = time.perf_counter()
    vecs_768 = embedder.encode_multimodal(encode_inputs, truncate_dim=768)
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    np.save(CONFIG.index_dir / "multimodal_embeddings_768.npy", vecs_768)
    (CONFIG.index_dir / "multimodal_items.json").write_text(
        json.dumps(metadata_records, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return {
        "count": len(metadata_records),
        "encode_ms": elapsed_ms,
        "items": metadata_records,
    }


def search_multimodal_gallery(
    query: str,
    top_k: int = 5,
    dim: int = 768,
    embedder: Optional[EmbeddingGemma2Engine] = None,
) -> Dict[str, Any]:
    """Query the unified 768d multimodal space (Images, Audio, Interleaved) using natural language text."""
    items_path = CONFIG.index_dir / "multimodal_items.json"
    emb_path = CONFIG.index_dir / "multimodal_embeddings_768.npy"
    if not items_path.exists() or not emb_path.exists():
        build_multimodal_gallery(embedder=embedder)

    items = json.loads(items_path.read_text(encoding="utf-8"))
    emb_768 = l2_normalize(np.load(emb_path).astype(np.float32))
    gallery_vecs = mrl_truncate_and_renormalize(emb_768, dim) if dim < 768 else emb_768

    if embedder is None:
        embedder = EmbeddingGemma2Engine(modality_mode="text_only")

    t0 = time.perf_counter()
    q_vec = embedder.encode_query(query, prompt_name="SearchQuery", truncate_dim=dim)
    scores = gallery_vecs @ q_vec
    top_idx = np.argsort(-scores)[:top_k]
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    hits = []
    for rank, idx in enumerate(top_idx, start=1):
        item = dict(items[int(idx)])
        item["rank"] = rank
        item["cosine_similarity"] = round(float(scores[int(idx)]), 5)
        item["dimension"] = dim
        hits.append(item)

    return {
        "query": query,
        "dimension": dim,
        "latency_ms": elapsed_ms,
        "hits": hits,
    }
