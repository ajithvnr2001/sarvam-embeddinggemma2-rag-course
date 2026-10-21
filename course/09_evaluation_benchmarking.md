# Module 9: End-to-End RAG Evaluation, Matryoshka Benchmarking & Vertex AI LLM-as-a-Judge

**Course Track:** Advanced — Quantitative IR Metrics, MRL Tradeoffs & Groundedness Verification  
**Benchmark Runner:** [`scripts/run_all_verifications.py`](file:///data/opencode/sarvam-rag-course/scripts/run_all_verifications.py)  
**Machine-Readable Report:** [`data/artifacts/evaluation_report.json`](file:///data/opencode/sarvam-rag-course/data/artifacts/evaluation_report.json)

---

## 1. Learning Objectives

By the end of this module, you will be able to:
1. Evaluate retrieval quality using standard Information Retrieval (IR) metrics: **Hit@1 (Precision@1)**, **Recall@5**, and **Mean Reciprocal Rank (MRR@5)**.
2. Quantify the storage-vs-accuracy tradeoff of **Matryoshka Representation Learning (MRL)** across `768d`, `512d`, `256d`, and `128d` slices, and prove how **Hybrid RRF (`256d` + BM25)** and the **Two-Stage MRL Funnel (`128d -> 768d`)** recover 100% accuracy at a fraction of the storage and compute cost.
3. Implement an automated **LLM-as-a-Judge** pipeline with Google Cloud Vertex AI (`gemini-2.5-flash`) using structured JSON output (`responseMimeType: "application/json"`) to audit **Context Relevance**, **Groundedness / Faithfulness**, and **Answer Completeness**.

---

## 2. Information Retrieval (IR) Metrics Formulas

Given a benchmark set of $Q$ developer queries:

1. **Hit@1 (Top-1 Accuracy)**:
   $$\text{Hit@1} = \frac{1}{|Q|} \sum_{q=1}^{|Q|} \mathbb{I}(\text{rank}_q = 1)$$
2. **Recall@K**:
   $$\text{Recall@K} = \frac{1}{|Q|} \sum_{q=1}^{|Q|} \mathbb{I}(\text{rank}_q \le K)$$
3. **Mean Reciprocal Rank (MRR@K)**:
   $$\text{MRR@K} = \frac{1}{|Q|} \sum_{q=1}^{|Q|} \frac{\mathbb{I}(\text{rank}_q \le K)}{\text{rank}_q}$$

---

## 3. Live Retrieval & Matryoshka (MRL) Benchmark Results

Executed via [`scripts/run_all_verifications.py`](file:///data/opencode/sarvam-rag-course/scripts/run_all_verifications.py) over the `sarvam-docs` index encoded with `google/embeddinggemma-2`:

| Strategy Label | Mode | Vector Dim | Compression vs. `768d` | Bytes / Vector (FP32) | Hit@1 (%) | Recall@5 (%) | MRR@5 | Avg Search Latency (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`dense_768d`** | Dense Cosine | `768d` | `1.0x` | `3,072 B` | **100.0%** | **100.0%** | **1.0000** | `0.5252 ms` |
| **`dense_512d`** | Dense MRL | `512d` | `1.5x` | `2,048 B` | **100.0%** | **100.0%** | **1.0000** | `0.0775 ms` |
| **`dense_256d`** | Dense MRL | `256d` | `3.0x` | `1,024 B` | `83.3%` | **100.0%** | `0.9167` | `0.0697 ms` |
| **`dense_128d`** | Dense MRL | `128d` | `6.0x` | `512 B` | `66.7%` | **100.0%** | `0.8056` | `0.0695 ms` |
| **`bm25_lexical`** | Identifier BM25 | N/A | N/A | Inverted Index | `83.3%` | **100.0%** | `0.9167` | `0.8137 ms` |
| **`hybrid_rrf_768d`** | Dense `768d` + BM25 | `768d` | `1.0x` | `3,072 B` | **100.0%** | **100.0%** | **1.0000** | `0.6302 ms` |
| **`hybrid_rrf_256d`** | Dense `256d` + BM25 | `256d` | **`3.0x`** | **`1,024 B`** | **100.0%** | **100.0%** | **1.0000** | `0.6293 ms` |
| **`mrl_funnel_128_to_768`** | `128d` $\rightarrow$ `768d` Funnel | `128d` $\rightarrow$ `768d` | **`6.0x` Stage-1** | `512 B` (RAM) | **100.0%** | **100.0%** | **1.0000** | `0.1402 ms` |

### Key Engineering Takeaways from the Live Numbers

1. **`100%` Recall@5 Across All MRL Dimensions**: Even when truncating `google/embeddinggemma-2` from `768d` down to `128d` (`6.0x` smaller vectors, `512 bytes`), the correct documentation chunk remains inside the top-5 (`Recall@5 = 100.0%`).
2. **Why `mrl_funnel_128_to_768` Achieves `100.0%` Hit@1**: Because `128d` has `100.0%` Recall in the top candidate pool, using `128d` for fast Stage-1 candidate generation and re-scoring only the shortlist with full `768d` vectors upgrades Hit@1 from `66.7%` $\rightarrow$ **`100.0%`** (`MRR@5 = 1.0000`).
3. **Why `hybrid_rrf_256d` Achieves `100.0%` Hit@1 at `3.0x` Compression**: Combining `256d` Matryoshka vectors (`83.3%` Hit@1 alone) with Identifier-Aware BM25 (`83.3%` Hit@1 alone) via Reciprocal Rank Fusion achieves **`100.0%` Hit@1 and `1.0000` MRR@5** while cutting vector storage by **66.7%**.

---

## 4. Deep-Dive Matryoshka (MRL) Query Analysis

Query: *"How do I generate Tamil speech using bulbul:v4-flash and what sample rates and speakers are supported?"*

| Dimension | Compression | Bytes / Vector | Search Latency | Top-1 Chunk ID | Top-1 Cosine (L2 Re-Normalized) | Top-5 Overlap with `768d` |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`768d`** | `1.0x` | `3,072 B` | `0.0489 ms` | `sarvam_0011` (`TAMIL_API_GUIDE`) | `0.75701` | `100.0%` |
| **`512d`** | `1.5x` | `2,048 B` | `0.0459 ms` | `sarvam_0011` (`TAMIL_API_GUIDE`) | `0.76031` | `80.0%` |
| **`256d`** | `3.0x` | `1,024 B` | `0.0330 ms` | `sarvam_0011` (`TAMIL_API_GUIDE`) | `0.79284` | `100.0%` |
| **`128d`** | `6.0x` | `512 B` | `0.0299 ms` | `sarvam_0011` (`TAMIL_API_GUIDE`) | `0.84648` | `80.0%` |

---

## 5. Multimodal 740M Cross-Modal Retrieval Benchmark Results

Using the full `740M` `google/embeddinggemma-2` model (`270M` text + `170M` vision + `300M` audio) across 10 multimodal items (`4` PNG visual API cards, `5` 16kHz `.wav`/`.mp3` audio clips, `1` interleaved Text+Image+Audio record):

| Natural Language Text Query | Top-1 Retrieved Item ID | Retrieved Modality | Asset File | Cosine Similarity (`768d`) |
| :--- | :--- | :---: | :--- | :---: |
| *"continuous heavy rainfall storm sound effect audio"* | `mm_aud_rain` | **Audio (`<\|audio\|>`)** | `sfx/rain.wav` (6.0s 16kHz) | **`0.78649`** |
| *"urgent telephone ringing sound effect in detective control room"* | `mm_aud_phone` | **Audio (`<\|audio\|>`)** | `sfx/phone_ring.wav` (2.0s 16kHz) | **`0.87140`** |
| *"Synthesized Tamil speech narration from Sarvam bulbul:v4-flash TTS API"* | `mm_interleaved_tts_bundle` | **Interleaved (`text+image+audio`)** | `sarvam_tts_bulbul_v4_card.png` + `line_01_narrator.mp3` | **`0.85529`** |
| *"Instant voice cloning workflow diagram /voices/create and /voices/clone"* | `mm_img_cloning` | **Image (`<\|image\|>`)** | `sarvam_voice_cloning_card.png` | **`0.80942`** |

---

## 6. Vertex AI `gemini-2.5-flash` Grounded RAG + LLM-as-a-Judge Results

Executed on GCP project `genai-511910` (`us-central1`):

| Query ID | Developer Question | Top-4 Cited Sources | Context Relevance | Groundedness / Faithfulness | Answer Completeness | LLM Judge Rationale |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **`q1_auth`** | *How do I authenticate with the Sarvam AI API and what HTTP header or Bearer token format is used?* | `sarvam_0018`, `sarvam_0017`, `sarvam_0015`, `sarvam_0021` | **5 / 5** | **5 / 5** | **5 / 5** | *"The retrieved context provides all the necessary details about authentication methods and header formats, and the answer accurately and completely addresses the query using only the provided information."* |
| **`q2_tts_tamil_v4`** | *Which Tamil speakers and audio sample rates work with bulbul:v4-flash in Sarvam Text-to-Speech?* | `sarvam_0006`, `sarvam_0012`, `sarvam_0009`, `sarvam_0013` | **5 / 5** | **5 / 5** | **5 / 5** | *"The retrieved context directly contains all the necessary information about Tamil speakers and audio sample rates for the specified model, and the answer accurately extracts and presents this information with correct citations."* |
| **`q3_voice_cloning`** | *How do I create a cloned voice and synthesize speech with Instant Voice Cloning?* | `sarvam_0022`, `sarvam_0006`, `sarvam_0007`, `sarvam_0023` | **5 / 5** | **5 / 5** | **5 / 5** | *"The retrieved context contains the exact technical information needed to answer the question, and the answer is directly and accurately extracted from the relevant chunk with correct citation, completely addressing the query."* |
| **Average** | **All Evaluated Queries** | — | **5.00 / 5.00** | **5.00 / 5.00** | **5.00 / 5.00** | **100% Grounded Citation Accuracy** |

---

## 7. Reproducing the Full Benchmark Suite

```bash
cd /data/opencode/sarvam-rag-course
source .venv/bin/activate

# Run all 5 verification phases (Chunking, MRL + Hybrid IR, Vertex AI RAG + Judge, Multimodal 740M, and GCP BigQuery VECTOR_SEARCH)
python3 scripts/run_all_verifications.py
```

---

## 8. Next Module
Continue to **[Module 10: Complete In-Depth CLI Version vs. UI Version Master Guide](./10_cli_and_ui_master_guide.md)** for side-by-side operational instructions across CLI and Web UI.
