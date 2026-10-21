# Module 06 (Advanced): Multimodal & Interleaved RAG — Text, Code, Visual API Diagrams (`<|image|>`) & 16kHz Audio (`<|audio|>`)

**Level:** Advanced  
**Estimated Time:** 50 minutes  
**Code Reference:** [`src/multimodal_rag.py`](../src/multimodal_rag.py)

---

## 1. Unified 768d Multimodal Vector Space in `google/embeddinggemma-2`

Unlike traditional text-only embedding models (`bge-large`, `e5-mistral`, or `EmbeddingGemma 1`), **`google/embeddinggemma-2`** natively embeds **4 modalities** into the exact same 768-dimensional vector space:
1. **Text & Code** (via the 270M Gemma 4 text backbone)
2. **Images** (via the 170M vision encoder using `<|image|>` placeholder tokens)
3. **Video** (sampled at 1 frame/sec via the vision encoder using `<|video|>`)
4. **Audio** (via the 300M `gemma4_audio` encoder at **16 kHz mono** using `<|audio|>`)

### Context Window & Token Budget Math (8,192 Shared Tokens)

All modalities draw from a single shared **8,192-token** context window:

| Modality | Placeholder Token | Token Consumption Rate | Maximum Standalone Input in 8K Window |
| :--- | :---: | :--- | :--- |
| **Text / Code** | N/A | `1 token` per subword | **8,192 tokens** |
| **Image** | `<|image|>` | `280 tokens` per image (default; configurable `70–1120` soft tokens) | **~29 images** (up to ~114 at 70 soft tokens) |
| **Video** | `<|video|>` | `140 tokens` per frame @ 1 FPS default | **~58 frames** (~1 minute of video) |
| **Audio** | `<|audio|>` | `25 tokens` per second of **16 kHz mono** audio | **~327 seconds** (~5.4 minutes of audio) |

---

## 2. How Interleaved Multimodal Encoding Works

In `sentence-transformers >= 6.1.0`, you can pass a dictionary combining text with `<|image|>`, `<|audio|>`, and `<|video|>` placeholders. Each placeholder token in `"text"` is replaced in-order by the embeddings from the vision/audio encoders before passing through the 24-layer Gemma 4 transformer and mean-pooling projection (`512 -> 768`):

```python
import librosa
from PIL import Image
import torch
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "google/embeddinggemma-2",
    model_kwargs={"torch_dtype": torch.bfloat16},
)

# 1. Load a Visual API Card (PIL Image)
img = Image.open("data/multimodal_samples/sarvam_tts_bulbul_v4_card.png").convert("RGB")

# 2. Load a Sarvam bulbul:v4-flash Tamil TTS audio clip resampled to 16kHz mono
wav, sr = librosa.load("data/multimodal_samples/line_01_narrator.mp3", sr=16000, mono=True)

# 3. Encode an interleaved Text + Image + Audio item into a single 768d vector!
emb = model.encode(
    [
        {
            "text": "Complete Sarvam Bulbul v4 Flash Text-to-Speech specification card <|image|> paired with live synthesized Tamil narration audio output <|audio|>.",
            "image": img,
            "audio": wav[:16000 * 6],
        }
    ],
    normalize_embeddings=True,
)
print("Interleaved embedding shape:", emb.shape)  # (1, 768)
```

---

## 3. Real Cross-Modal Retrieval Results on `sarvam-docs` Assets

In [`src/multimodal_rag.py`](../src/multimodal_rag.py), we index **10 real multimodal assets** from the project:
- **4 Visual API Architecture Cards (`<|image|>`)**:
  - `mm_img_tts`: Sarvam Bulbul v4 Flash Text-to-Speech Visual Architecture Card
  - `mm_img_stt`: Sarvam Saaras & Saarika Speech-to-Text Visual Pipeline Card
  - `mm_img_cloning`: Sarvam Instant Voice Cloning & Dubbing Workflow Diagram
  - `mm_img_mrl`: Google EmbeddingGemma 2 Multimodal & Matryoshka (MRL) Architecture Diagram
- **5 Audio Clips (`<|audio|>`, 16 kHz mono)**:
  - `mm_aud_rain`: Continuous rainfall ambience (`sfx/rain.wav`)
  - `mm_aud_thunder`: Thunder crack cue (`sfx/thunder.wav`)
  - `mm_aud_phone`: Midnight telephone ring cue (`sfx/phone_ring.wav`)
  - `mm_aud_tts_narrator`: Tamil narrator speech synthesized by `bulbul:v4-flash` (`line_01_narrator.mp3`)
  - `mm_aud_tts_vivek`: DSP Vivek Tamil dialogue synthesized by `bulbul:v4-flash` (`line_03_vivek.mp3`)
- **1 Interleaved Multi-Modal Bundle (`<|image|>` + `<|audio|>` + Text)**:
  - `mm_interleaved_tts_bundle`: Combined visual TTS specification card + synthesized Tamil speech audio

---

## 4. Hands-On CLI Lab: Cross-Modal Search

```bash
# Query 1: Retrieve audio sound effect using a text query
/data/opencode/sarvam-rag-course/.venv/bin/python -m cli.rag_cli multimodal \
  "continuous heavy rainfall storm sound effect audio"

# Query 2: Retrieve synthesized Tamil TTS audio & interleaved bundle
/data/opencode/sarvam-rag-course/.venv/bin/python -m cli.rag_cli multimodal \
  "Synthesized Tamil speech narration from Sarvam bulbul:v4-flash TTS API"

# Query 3: Retrieve visual workflow diagram
/data/opencode/sarvam-rag-course/.venv/bin/python -m cli.rag_cli multimodal \
  "Instant voice cloning workflow diagram /voices/create and /voices/clone"
```

---

## Next Module
Proceed to **[Module 07: Grounded Generation with GCP Vertex AI Gemini & Guardrails against Hallucination](./07_vertex_ai_grounded_generation.md)**.
