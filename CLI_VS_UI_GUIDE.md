# Complete End-to-End Guide: CLI Version vs. Web UI Version

> **Side-by-Side Operational Reference** for building, running, evaluating, and deploying the **`google/embeddinggemma-2` + `sarvam-docs` + Google Cloud Platform (`genai-511910`)** RAG system using **both** the Command-Line Interface (CLI) and the Interactive Web UI.

---

## Table of Contents
1. [Architecture Comparison: CLI vs. Web UI](#1-architecture-comparison-cli-vs-web-ui)
2. [Google Cloud Platform (GCP): `gcloud` / `bq` CLI vs. Cloud Console Web UI](#2-google-cloud-platform-gcp-gcloud--bq-cli-vs-cloud-console-web-ui)
3. [Hugging Face & `google/embeddinggemma-2`: CLI vs. Hub Web UI](#3-hugging-face--googleembeddinggemma-2-cli-vs-hub-web-ui)
4. [GitHub & `sarvam-docs`: `git` / `gh` CLI vs. GitHub Web UI](#4-github--sarvam-docs-git--gh-cli-vs-github-web-ui)
5. [RAG Engine CLI Version (`cli/rag_cli.py`) — All 9 Subcommands In-Depth](#5-rag-engine-cli-version-clirag_clipy--all-9-subcommands-in-depth)
6. [RAG Studio Web UI Version (`ui/app.py`) — All 4 Interactive Tabs In-Depth](#6-rag-studio-web-ui-version-uiapppy--all-4-interactive-tabs-in-depth)
7. [Resource Teardown & Cost Cleanup (`destroy`) — CLI vs. Web UI](#7-resource-teardown--cost-cleanup-destroy--cli-vs-web-ui)

---

## 1. Architecture Comparison: CLI vs. Web UI

Both the **CLI (`cli/rag_cli.py`)** and the **Interactive Web Studio (`ui/app.py`)** call the exact same Python modules in [`src/`](./src):

| Feature / Capability | CLI Version ([`cli/rag_cli.py`](./cli/rag_cli.py)) | Web UI Version ([`ui/app.py`](./ui/app.py)) |
| :--- | :--- | :--- |
| **Primary Interface** | Rich ANSI terminal tables, panels & interactive REPL | FastAPI + 4-Tab Browser Studio (`http://localhost:8090`) |
| **Corpus Ingestion** | `python -m cli.rag_cli ingest` | Automatic pre-loaded index + `GET /api/status` |
| **Search Strategies** | `--mode {dense,bm25,hybrid,mrl_funnel}` | Tab 1 Dropdown (`Hybrid RRF`, `MRL Funnel`, `Dense`, `BM25`) |
| **Matryoshka Dimensions** | `--dim {768,512,256,128}` + `mrl-benchmark` subcommand | Tab 1 Selector + Tab 2 (`Matryoshka Live Explorer` 4-column grid) |
| **Grounded Vertex AI Answer** | `python -m cli.rag_cli ask "..." --judge` | Tab 1 (`Ask Vertex AI Gemini (RAG)` button + Judge score pills) |
| **Multimodal 740M Search** | `python -m cli.rag_cli multimodal "..."` | Tab 3 (`Multimodal Lab`: Visual Cards + 16kHz Audio + Interleaved) |
| **GCP GCS + BigQuery Sync** | `python -m cli.rag_cli gcp-sync --dim 256` | Tab 4 (`Sync Local Index -> GCS & BigQuery` button) |
| **BigQuery `VECTOR_SEARCH`** | `python -m cli.rag_cli bq-search "..."` | Tab 4 (`Run BigQuery VECTOR_SEARCH` button + live SQL viewer) |
| **Resource Teardown (`destroy`)** | `python -m cli.rag_cli destroy` | Tab 4 (`Destroy Cloud Resources (GCS + BQ)` button) |

---

## 2. Google Cloud Platform (GCP): `gcloud` / `bq` CLI vs. Cloud Console Web UI

### 2.1 Authentication & Project Configuration (`genai-511910`)

#### CLI Version
```bash
# Authenticate user account & Application Default Credentials
gcloud auth login
gcloud auth application-default login

# Set active project & default region
gcloud config set project genai-511910
gcloud config set compute/region us-central1

# Verify active account and token
gcloud auth list
gcloud auth print-access-token | head -c 24 && echo "..."
```

#### Web UI Version
1. Open [Google Cloud Console](https://console.cloud.google.com/) and sign in with `ajithkumar1962vnr@gmail.com`.
2. Click the **Project Selector** at the top left of the top bar and select **`genai-511910`**.
3. Open **IAM & Admin $\rightarrow$ Settings** to verify the project number and region settings.

---

### 2.2 Enabling Required GCP APIs

#### CLI Version
```bash
gcloud services enable \
  aiplatform.googleapis.com \
  bigquery.googleapis.com \
  storage.googleapis.com \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  --project=genai-511910
```

#### Web UI Version
1. Go to [APIs & Services $\rightarrow$ Enabled APIs & services](https://console.cloud.google.com/apis/dashboard?project=genai-511910).
2. Click **+ ENABLE APIS AND SERVICES**.
3. Search for and enable **Vertex AI API**, **BigQuery API**, **Cloud Storage API**, and **Cloud Run Admin API**.

---

### 2.3 Creating Cloud Storage Bucket & BigQuery Vector Table

#### CLI Version (`gcloud storage` & `bq`)
```bash
# 1. Create GCS bucket
gcloud storage buckets create gs://genai-511910-sarvam-rag \
  --project=genai-511910 \
  --location=us-central1 \
  --uniform-bucket-level-access

# 2. Upload JSONL embeddings
gcloud storage cp data/artifacts/docs_embeddings_256d.jsonl \
  gs://genai-511910-sarvam-rag/bq_load/docs_embeddings_256d.jsonl

# 3. Create BigQuery dataset & load JSONL with ARRAY<FLOAT64> embedding column
bq --location=us-central1 mk --dataset --if_not_exists genai-511910:sarvam_rag

bq --location=us-central1 load \
  --replace \
  --source_format=NEWLINE_DELIMITED_JSON \
  genai-511910:sarvam_rag.docs_embeddings \
  gs://genai-511910-sarvam-rag/bq_load/docs_embeddings_256d.jsonl
```

#### Web UI Version (Cloud Storage Browser & BigQuery Studio)
1. **Cloud Storage UI**:
   - Open [Cloud Storage Browser](https://console.cloud.google.com/storage/browser?project=genai-511910) $\rightarrow$ **+ CREATE**.
   - Name: `genai-511910-sarvam-rag`, Region: `us-central1`, Access control: `Uniform` $\rightarrow$ **Create**.
   - Click **UPLOAD FILES** inside the bucket to upload `docs_embeddings_256d.jsonl`.
2. **BigQuery Studio UI**:
   - Open [BigQuery Studio](https://console.cloud.google.com/bigquery?project=genai-511910).
   - Next to `genai-511910`, click **Create dataset** $\rightarrow$ Dataset ID: `sarvam_rag`, Location: `us-central1`.
   - Next to `sarvam_rag`, click **Create table** $\rightarrow$ Source: `Google Cloud Storage` (`gs://genai-511910-sarvam-rag/bq_load/docs_embeddings_256d.jsonl`), Format: `JSONL`, Table: `docs_embeddings`, Schema: `Auto-detect` $\rightarrow$ **Create table**.

---

## 3. Hugging Face & `google/embeddinggemma-2`: CLI vs. Hub Web UI

### CLI Version
```bash
source .venv/bin/activate
export HF_TOKEN="hf_YOUR_TOKEN_HERE"

# Verify Hugging Face account via CLI
hf auth whoami

# Download & verify google/embeddinggemma-2 (270M text-only selective load)
python3 -c "
from src.embedder import EmbeddingGemma2Engine
eng = EmbeddingGemma2Engine(modality_mode='text_only')
vec = eng.encode_query('How do I authenticate with Sarvam AI?', prompt_name='QuestionAnswering')
print('Shape:', vec.shape, 'L2 norm:', (vec @ vec) ** 0.5)
"
```

### Web UI Version
1. Open [huggingface.co/google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2).
2. Inspect the model card architecture (`740M` total parameters: `270M` Gemma 3 text backbone + `170M` vision encoder + `300M` audio encoder).
3. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) to create or rotate your `hf_...` token and place it in `.env`.

---

## 4. GitHub & `sarvam-docs`: `git` / `gh` CLI vs. GitHub Web UI

### CLI Version
```bash
# Audit working tree to ensure zero secrets are tracked
git status
git grep -E "ghp_|hf_" || echo "Clean: Zero secrets in tracked files"

# Commit and push
git add -A
git commit -m "Complete EmbeddingGemma 2 + Sarvam Docs + GCP RAG Course (CLI & UI)"
git push -u origin main
```

### Web UI Version
1. Visit [github.com/ajithvnr2001/sarvam-embeddinggemma2-rag-course](https://github.com/ajithvnr2001/sarvam-embeddinggemma2-rag-course) and [github.com/ajithvnr2001/sarvam-docs](https://github.com/ajithvnr2001/sarvam-docs).
2. Browse the interactive Markdown modules inside [`course/`](./course/) and view embedded UI screenshots and Mermaid diagrams directly in GitHub's Markdown renderer.

---

## 5. RAG Engine CLI Version (`cli/rag_cli.py`) — All 9 Subcommands In-Depth

```bash
cd /data/opencode/sarvam-rag-course
source .venv/bin/activate
```

### 5.1 `ingest` — Build the Chunks & Matryoshka Vector Index
```bash
python -m cli.rag_cli ingest
```
**What happens:**
- Crawls `/data/opencode/sarvam/docs` + root guides (`README_TTS.md`, `TAMIL_API_GUIDE.md`, `HOW_IT_WAS_DONE.md`).
- Strips navigation boilerplate, splits by Markdown headings (`#`, `##`, `###`) while keeping triple-backtick code blocks whole, and formats each chunk as `title: {title} | text: {content}`.
- Encodes `768d` vectors with `google/embeddinggemma-2` and writes `embeddings_768.npy`, `embeddings_512.npy`, `embeddings_256.npy`, and `embeddings_128.npy` (all L2 re-normalized).

### 5.2 `search` — Test Any Retrieval Strategy & MRL Dimension
```bash
python -m cli.rag_cli search \
  "Which Tamil speakers and audio sample rates work with bulbul:v4-flash?" \
  --mode hybrid \
  --dim 768 \
  --prompt-name QuestionAnswering \
  --top-k 5
```

### 5.3 `ask` — Grounded Synthesis with Vertex AI `gemini-2.5-flash` + LLM Judge
```bash
python -m cli.rag_cli ask \
  "How do I authenticate with the Sarvam AI API and what HTTP header is used?" \
  --mode hybrid \
  --dim 768 \
  --top-k 4 \
  --judge
```

### 5.4 `mrl-benchmark` — Compare `768d`, `512d`, `256d`, `128d` Side-by-Side
```bash
python -m cli.rag_cli mrl-benchmark \
  "How do I generate Tamil speech using bulbul:v4-flash and what sample rates and speakers are supported?" \
  --top-k 5
```

### 5.5 `multimodal` — Cross-Modal Text $\rightarrow$ Image + 16kHz Audio Search (740M Model)
```bash
python -m cli.rag_cli multimodal \
  "continuous heavy rainfall storm sound effect audio" \
  --dim 768 \
  --top-k 4
```

### 5.6 `gcp-sync` — Upload Chunks & Vectors to GCS & Load BigQuery Table
```bash
python -m cli.rag_cli gcp-sync --dim 256
```

### 5.7 `bq-search` — Execute Live BigQuery `VECTOR_SEARCH` SQL
```bash
python -m cli.rag_cli bq-search \
  "How do I authenticate with the Sarvam AI API using api-subscription-key?" \
  --dim 256 \
  --top-k 5
```

### 5.8 `chat` — Interactive Multi-Turn RAG REPL
```bash
python -m cli.rag_cli chat --mode hybrid --dim 768
```

### 5.9 `destroy` — Idempotent Teardown of Cloud & Local Runtime Resources
```bash
# Deletes BigQuery dataset (genai-511910:sarvam_rag), GCS bucket (gs://genai-511910-sarvam-rag), and stops Web UI server
python -m cli.rag_cli destroy

# Optional: Also purge local pre-computed .npy matrices
python -m cli.rag_cli destroy --purge-local-index
```

---

## 6. RAG Studio Web UI Version (`ui/app.py`) — All 4 Interactive Tabs In-Depth

Launch the server:
```bash
uvicorn ui.app:app --host 0.0.0.0 --port 8090
```

### Visual Walkthrough of the 4 Web Studio Tabs

| Tab | Name | Key Interactive Controls & Live Outputs |
| :--- | :--- | :--- |
| **Tab 1** | **Grounded RAG & Vertex AI Studio** | Preset question pills, Search Strategy selector (`hybrid`, `mrl_funnel`, `dense`, `bm25`), MRL dimension selector (`768d`, `512d`, `256d`, `128d`), Task Prompt Prefix selector, Vertex AI `gemini-2.5-flash` answer panel with `[sarvam_XXXX]` citations, LLM-as-a-Judge `/5` score badges, and Retrieved Evidence Cards. |
| **Tab 2** | **Matryoshka (MRL) Live Explorer** | Live 4-column comparison (`768d 1.0x`, `512d 1.5x`, `256d 3.0x`, `128d 6.0x`) showing bytes/vector savings, Top-1 match indicator, Top-K overlap %, and per-chunk cosine similarity. |
| **Tab 3** | **Multimodal Lab (740M Text + Image + Audio)** | Cross-modal search across 4 visual API architecture PNG cards (`<|image|>`), 5 real 16kHz mono `.wav`/`.mp3` audio clips (`<|audio|>`), and 1 interleaved Text+Image+Audio record in a shared `768d` vector space. |
| **Tab 4** | **GCP BigQuery Vector Search & Cloud Console** | One-click **`Sync Local Index -> GCS & BigQuery`**, live BigQuery `VECTOR_SEARCH` query runner, full GoogleSQL query inspector, and one-click **`Destroy Cloud Resources (GCS + BQ)`** teardown button. |

---

## 7. Resource Teardown & Cost Cleanup (`destroy`) — CLI vs. Web UI

To ensure **zero ongoing cloud storage or BigQuery costs** after finishing your hands-on labs:

| Resource | CLI Version (`cli/rag_cli.py` / `scripts/destroy_resources.sh`) | Web UI Version (RAG Studio UI / GCP Console UI) |
| :--- | :--- | :--- |
| **All Project Cloud & Runtime Resources (One-Click)** | `python -m cli.rag_cli destroy`<br/>*(or `bash scripts/destroy_resources.sh`)* | Open `http://localhost:8090` $\rightarrow$ **Tab 4 (`GCP BigQuery Vector Search`)** $\rightarrow$ Click red **`Destroy Cloud Resources (GCS + BQ)`** button |
| **BigQuery Dataset (`genai-511910:sarvam_rag`)** | `bq --location=us-central1 rm -r -f -d genai-511910:sarvam_rag` | [BigQuery Console](https://console.cloud.google.com/bigquery?project=genai-511910) $\rightarrow$ Expand `genai-511910` $\rightarrow$ Click `⋮` next to `sarvam_rag` $\rightarrow$ **Delete** |
| **GCS Bucket (`gs://genai-511910-sarvam-rag`)** | `gcloud storage rm --recursive gs://genai-511910-sarvam-rag --project=genai-511910` | [Cloud Storage Browser](https://console.cloud.google.com/storage/browser?project=genai-511910) $\rightarrow$ Select `genai-511910-sarvam-rag` $\rightarrow$ **Delete** |
| **Local Web Studio Server (Port `8090`)** | `pkill -f "uvicorn ui.app:app"` | Press `Ctrl+C` in the terminal running `uvicorn` |
| **Local Pre-Computed `.npy` Index Matrices** | `python -m cli.rag_cli destroy --purge-local-index` | Delete `data/index/*.npy` in file explorer |
