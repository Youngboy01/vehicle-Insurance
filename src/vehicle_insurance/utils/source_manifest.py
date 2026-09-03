"""Helpers for immutable source-snapshot manifests.

The manifest is deliberately kept outside the model-bound columns.  It is the
contract between the controlled MongoDB extraction and every later pipeline
stage that needs to prove which source was used.
"""

from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd


MANIFEST_VERSION = "1.0.0"
HASH_ALGORITHM = "SHA-256"


def canonical_json(value: Any) -> str:
    """Return deterministic JSON for metadata hashing and audit output."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    )


def canonical_dataframe_bytes(dataframe: pd.DataFrame) -> bytes:
    """Serialize a source snapshot independently of host newline settings."""

    return dataframe.to_csv(index=False, lineterminator="\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_snapshot_id(manifest: dict[str, Any]) -> str:
    """Build a content-and-read-boundary identity for one extraction."""

    identity = deepcopy(manifest)
    identity.pop("snapshot_id", None)
    return f"mongodb-sha256:{sha256_bytes(canonical_json(identity).encode('utf-8'))}"


def finalize_source_manifest(
    base_manifest: dict[str, Any],
    dataframe: pd.DataFrame,
    snapshot_file_name: str,
) -> dict[str, Any]:
    """Add file, row-range, content, and immutable identity fields."""

    if dataframe.empty:
        raise ValueError("Cannot create a source manifest for an empty snapshot")
    if "id" not in dataframe.columns:
        raise ValueError("Source snapshot must contain the id column")

    canonical_bytes = canonical_dataframe_bytes(dataframe)
    content_hash = sha256_bytes(canonical_bytes)
    manifest = deepcopy(base_manifest)
    manifest.update(
        {
            "manifest_version": MANIFEST_VERSION,
            "snapshot_version": 1,
            "snapshot_format": "csv",
            "snapshot_file_name": snapshot_file_name,
            "immutable": True,
            "row_count": int(len(dataframe)),
            "id_range": {
                "field": "id",
                "min": int(dataframe["id"].min()),
                "max": int(dataframe["id"].max()),
            },
            "content_hash_algorithm": HASH_ALGORITHM,
            "content_hash": content_hash,
            "snapshot_file_sha256": content_hash,
        }
    )
    manifest["snapshot_id"] = build_snapshot_id(manifest)
    return manifest


def validate_source_manifest(
    manifest: Any,
    dataframe: pd.DataFrame | None = None,
) -> None:
    """Fail closed when source provenance is absent, malformed, or altered."""

    if not isinstance(manifest, dict):
        raise ValueError("Source manifest must be a dictionary")

    required_fields = (
        "manifest_version",
        "snapshot_version",
        "snapshot_id",
        "snapshot_format",
        "immutable",
        "source",
        "query",
        "projection",
        "ordering",
        "read_boundary",
        "read_timestamp_utc",
        "database_state_retry_policy",
        "extracted_at_utc",
        "row_count",
        "id_range",
        "content_hash_algorithm",
        "content_hash",
        "snapshot_file_sha256",
    )
    missing = [
        field
        for field in required_fields
        if field not in manifest or manifest[field] in (None, "")
    ]
    if missing:
        raise ValueError(f"Source manifest fields missing: {missing}")
    if manifest["manifest_version"] != MANIFEST_VERSION:
        raise ValueError(
            f"Unsupported source manifest version: {manifest['manifest_version']}"
        )
    if manifest["snapshot_version"] != 1:
        raise ValueError("Unsupported source snapshot version")
    if manifest["snapshot_format"] != "csv":
        raise ValueError("Source snapshot format must be csv")
    if manifest["immutable"] is not True:
        raise ValueError("Source snapshot must be marked immutable")
    if not isinstance(manifest["snapshot_id"], str) or not manifest[
        "snapshot_id"
    ].startswith("mongodb-sha256:"):
        raise ValueError("Source snapshot ID must use the mongodb-sha256 format")

    source = manifest["source"]
    if not isinstance(source, dict):
        raise ValueError("Source identity must be a dictionary")
    for field in ("system", "database", "collection", "namespace"):
        if not str(source.get(field, "")).strip():
            raise ValueError(f"Source identity field missing: {field}")

    query = manifest["query"]
    if not isinstance(query, dict) or "filter" not in query:
        raise ValueError("Source manifest must record the MongoDB filter")

    projection = manifest["projection"]
    if not isinstance(projection, dict):
        raise ValueError("Source projection metadata must be a dictionary")
    projection_fields = projection.get("fields")
    mongo_projection = projection.get("mongo_projection")
    if (
        not isinstance(projection_fields, list)
        or not projection_fields
        or not isinstance(mongo_projection, dict)
        or mongo_projection.get("_id") != 0
    ):
        raise ValueError("Source manifest has invalid explicit projection metadata")
    if "id" not in projection_fields or "Response" not in projection_fields:
        raise ValueError("Training projection must include id and Response")

    ordering = manifest["ordering"]
    if not isinstance(ordering, dict):
        raise ValueError("Source ordering metadata must be a dictionary")
    if (
        ordering.get("field") != "id"
        or ordering.get("direction") != "ascending"
        or ordering.get("index_required") is not True
        or not str(ordering.get("index_name", "")).strip()
    ):
        raise ValueError("Source manifest does not prove indexed id ordering")

    boundary = manifest["read_boundary"]
    if not isinstance(boundary, dict):
        raise ValueError("Read-boundary metadata must be a dictionary")
    if boundary.get("mode") != "snapshot_session":
        raise ValueError("Source read was not performed in a MongoDB snapshot session")
    if boundary.get("read_concern") != "snapshot":
        raise ValueError("Source read concern must be snapshot")
    if not boundary.get("cluster_time") and not boundary.get("operation_time"):
        raise ValueError("Source manifest has no MongoDB cluster or operation time")

    retry_policy = manifest["database_state_retry_policy"]
    if not isinstance(retry_policy, dict) or retry_policy != {
        "same_snapshot_read_failure": "abort_extraction",
        "new_extraction": "new_snapshot_id",
    }:
        raise ValueError("Source manifest has an invalid database-state retry policy")

    if not isinstance(manifest["row_count"], int) or manifest["row_count"] <= 0:
        raise ValueError("Source row_count must be a positive integer")
    id_range = manifest["id_range"]
    if (
        not isinstance(id_range, dict)
        or id_range.get("field") != "id"
        or not isinstance(id_range.get("min"), int)
        or not isinstance(id_range.get("max"), int)
        or id_range["min"] > id_range["max"]
    ):
        raise ValueError("Source manifest has an invalid ID range")

    for field in ("content_hash", "snapshot_file_sha256"):
        value = manifest[field]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value.lower())
        ):
            raise ValueError(f"Source manifest has an invalid {field}")
    if manifest["content_hash_algorithm"] != HASH_ALGORITHM:
        raise ValueError("Source manifest content hash algorithm must be SHA-256")
    if manifest["content_hash"] != manifest["snapshot_file_sha256"]:
        raise ValueError("Source content and snapshot-file hashes do not match")
    if manifest["snapshot_id"] != build_snapshot_id(manifest):
        raise ValueError("Source snapshot ID does not match its manifest identity")

    if dataframe is not None:
        if len(dataframe) != manifest["row_count"]:
            raise ValueError("Snapshot row count does not match its manifest")
        if "id" not in dataframe.columns:
            raise ValueError("Snapshot dataframe does not contain id")
        observed_range = {
            "min": int(dataframe["id"].min()),
            "max": int(dataframe["id"].max()),
        }
        if observed_range != {
            "min": id_range["min"],
            "max": id_range["max"],
        }:
            raise ValueError("Snapshot ID range does not match its manifest")
        if sha256_bytes(canonical_dataframe_bytes(dataframe)) != manifest[
            "content_hash"
        ]:
            raise ValueError("Snapshot contents do not match its manifest hash")


def write_json(path: str | os.PathLike[str], payload: dict[str, Any]) -> None:
    """Write deterministic JSON; callers use a temporary path for atomicity."""

    with open(path, "w", encoding="utf-8", newline="\n") as output_file:
        json.dump(payload, output_file, indent=2, sort_keys=True, default=str)
        output_file.write("\n")


def relative_snapshot_name(path: str | os.PathLike[str]) -> str:
    return Path(path).name
