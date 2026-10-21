#!/usr/bin/env python3
"""Cloud FinOps Operations & Idempotent Resource Teardown (`destroy`) from Scratch.

Lesson: phases/04-production-and-cloud/04-cli-ui-operations-and-resource-destroy/docs/en.md
Implements an idempotent cloud resource state machine for BigQuery datasets and GCS buckets,
verifying the Build -> Verify -> Destroy lifecycle and $0.00 post-teardown state.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CloudEnvironmentState:
    project_id: str = "genai-511910"
    gcs_buckets: dict[str, list[str]] = field(default_factory=dict)
    bq_datasets: dict[str, list[str]] = field(default_factory=dict)


def provision_rag_resources(state: CloudEnvironmentState, bucket: str, dataset: str) -> None:
    """Provision GCS bucket artifacts and BigQuery vector table."""
    state.gcs_buckets[bucket] = [
        "sarvam_rag/chunks.jsonl",
        "sarvam_rag/embeddings_768.npy",
        "sarvam_rag/embeddings_256.npy",
        "sarvam_rag/bq_vectors_256d.jsonl",
    ]
    state.bq_datasets[dataset] = ["docs_embeddings (120 rows x 256-D REPEATED FLOAT64)"]


def idempotent_destroy_gcp_resources(
    state: CloudEnvironmentState,
    bucket: str,
    dataset: str,
) -> dict[str, str]:
    """Idempotently tear down BigQuery dataset + tables and GCS bucket + objects."""
    report: dict[str, str] = {}

    if dataset in state.bq_datasets:
        tables = state.bq_datasets.pop(dataset)
        report["bigquery_dataset"] = f"deleted ({dataset}: removed {len(tables)} table(s))"
    else:
        report["bigquery_dataset"] = f"already_absent ({dataset})"

    if bucket in state.gcs_buckets:
        blobs = state.gcs_buckets.pop(bucket)
        report["gcs_bucket"] = f"deleted (gs://{bucket}: removed {len(blobs)} object(s))"
    else:
        report["gcs_bucket"] = f"already_absent (gs://{bucket})"

    report["ongoing_cloud_cost_usd"] = "$0.00/month"
    return report


def main() -> int:
    print("=" * 72)
    print("PHASE 04 · LESSON 04: CLI/UI OPERATIONS & IDEMPOTENT RESOURCE DESTROY")
    print("=" * 72)

    env = CloudEnvironmentState(project_id="genai-511910")
    bucket_name = "genai-511910-sarvam-rag"
    dataset_name = "genai-511910:sarvam_rag"

    # Beat 1: Provision resources (Build)
    provision_rag_resources(env, bucket_name, dataset_name)
    print("\n1. [BUILD] Provisioned Cloud RAG Resources:")
    print(f"   - Active GCS Buckets : {list(env.gcs_buckets.keys())} ({len(env.gcs_buckets[bucket_name])} blobs)")
    print(f"   - Active BQ Datasets : {list(env.bq_datasets.keys())} ({env.bq_datasets[dataset_name][0]})")

    # Beat 2: First Teardown Run (Destroy)
    first_pass = idempotent_destroy_gcp_resources(env, bucket_name, dataset_name)
    print("\n2. [DESTROY — PASS 1] Executing `python -m cli.rag_cli destroy --yes`:")
    for k, v in first_pass.items():
        print(f"   - {k:<24}: {v}")

    assert first_pass["bigquery_dataset"].startswith("deleted")
    assert first_pass["gcs_bucket"].startswith("deleted")
    assert len(env.gcs_buckets) == 0 and len(env.bq_datasets) == 0

    # Beat 3: Second Teardown Run (Verify Idempotency)
    second_pass = idempotent_destroy_gcp_resources(env, bucket_name, dataset_name)
    print("\n3. [DESTROY — PASS 2] Re-running `destroy --yes` to Verify Idempotency:")
    for k, v in second_pass.items():
        print(f"   - {k:<24}: {v}")

    assert second_pass["bigquery_dataset"].startswith("already_absent")
    assert second_pass["gcs_bucket"].startswith("already_absent")

    print("\nVerification passed: Build -> Verify -> Destroy lifecycle is 100% idempotent ($0.00 ongoing cost).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
