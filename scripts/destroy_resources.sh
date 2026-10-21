#!/usr/bin/env bash
# ==============================================================================
# End-to-End GCP & Local Resource Teardown Script (`destroy_resources.sh`)
# ==============================================================================
# Safely destroys the resources provisioned for the Sarvam + EmbeddingGemma 2
# RAG course so you incur ZERO ongoing Google Cloud Storage or BigQuery charges:
#   1. BigQuery Dataset & Vector Table: genai-511910:sarvam_rag.docs_embeddings
#   2. Cloud Storage Bucket & Objects:  gs://genai-511910-sarvam-rag
#   3. Local FastAPI Web Studio Server: uvicorn ui.app:app (Port 8090)
# ==============================================================================

set -u

PROJECT_ID="${GCP_PROJECT_ID:-genai-511910}"
LOCATION="${GCP_LOCATION:-us-central1}"
BUCKET_NAME="${GCS_BUCKET_NAME:-genai-511910-sarvam-rag}"
DATASET_ID="${BQ_DATASET_ID:-sarvam_rag}"

echo "=================================================================="
echo " 🧹 Starting Teardown for GCP Project: ${PROJECT_ID}"
echo "=================================================================="

# 1. Destroy BigQuery Dataset and all Vector Tables inside it
echo "[1/4] Deleting BigQuery Dataset & Vector Table (${PROJECT_ID}:${DATASET_ID})..."
if bq --location="${LOCATION}" rm -r -f -d "${PROJECT_ID}:${DATASET_ID}" 2>/dev/null; then
    echo "  ✓ Deleted BigQuery dataset: ${PROJECT_ID}:${DATASET_ID}"
else
    echo "  ℹ BigQuery dataset ${PROJECT_ID}:${DATASET_ID} already deleted or not found."
fi

# 2. Destroy Google Cloud Storage Bucket and all Vector JSONL objects
echo "[2/4] Deleting GCS Bucket & Objects (gs://${BUCKET_NAME})..."
if gcloud storage rm --recursive "gs://${BUCKET_NAME}" --project="${PROJECT_ID}" 2>/dev/null; then
    echo "  ✓ Deleted GCS bucket: gs://${BUCKET_NAME}"
else
    echo "  ℹ GCS bucket gs://${BUCKET_NAME} already deleted or not found."
fi

# 3. Stop local FastAPI Web UI server on Port 8090 if running
echo "[3/4] Stopping background Web UI Studio server (uvicorn ui.app:app)..."
if pkill -f "uvicorn ui.app:app" 2>/dev/null; then
    echo "  ✓ Stopped Web UI server on port 8090."
else
    echo "  ℹ Web UI server was not running."
fi

# 4. Verify Cloud Cleanup
echo "[4/4] Verifying zero remaining RAG cloud resources..."
if gcloud storage buckets describe "gs://${BUCKET_NAME}" --project="${PROJECT_ID}" >/dev/null 2>&1; then
    echo "  ✗ WARNING: GCS bucket gs://${BUCKET_NAME} still exists!"
else
    echo "  ✓ Verified: GCS bucket gs://${BUCKET_NAME} is completely removed."
fi

if bq show "${PROJECT_ID}:${DATASET_ID}" >/dev/null 2>&1; then
    echo "  ✗ WARNING: BigQuery dataset ${PROJECT_ID}:${DATASET_ID} still exists!"
else
    echo "  ✓ Verified: BigQuery dataset ${PROJECT_ID}:${DATASET_ID} is completely removed."
fi

echo "=================================================================="
echo " ✅ Teardown Complete! Zero ongoing GCS or BigQuery storage costs."
echo "    (To re-create cloud resources anytime, run: python -m cli.rag_cli gcp-sync)"
echo "=================================================================="
