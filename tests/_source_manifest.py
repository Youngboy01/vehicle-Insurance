from vehicle_insurance.utils.source_manifest import build_snapshot_id


def build_source_manifest(dataset_hash: str) -> dict:
    fields = [
        "id",
        "Gender",
        "Age",
        "Driving_License",
        "Region_Code",
        "Previously_Insured",
        "Vehicle_Age",
        "Vehicle_Damage",
        "Annual_Premium",
        "Policy_Sales_Channel",
        "Vintage",
        "Response",
    ]
    manifest = {
        "manifest_version": "1.0.0",
        "snapshot_version": 1,
        "snapshot_format": "csv",
        "snapshot_file_name": "data.csv",
        "immutable": True,
        "source": {
            "system": "MongoDB",
            "database": "INSURANCE",
            "collection": "INSURANCE_DATA",
            "namespace": "INSURANCE.INSURANCE_DATA",
        },
        "query": {"filter": {}, "filter_canonical_json": "{}"},
        "projection": {
            "fields": fields,
            "mongo_projection": {"_id": 0, **dict.fromkeys(fields, 1)},
        },
        "ordering": {
            "field": "id",
            "direction": "ascending",
            "index_required": True,
            "index_name": "id_1",
        },
        "read_boundary": {
            "mode": "snapshot_session",
            "read_concern": "snapshot",
            "cluster_time": {"seconds": 10, "increment": 1},
            "operation_time": {"seconds": 10, "increment": 2},
        },
        "read_timestamp_utc": "2026-09-03T00:00:00+00:00",
        "database_state_retry_policy": {
            "same_snapshot_read_failure": "abort_extraction",
            "new_extraction": "new_snapshot_id",
        },
        "extracted_at_utc": "2026-09-03T00:00:00+00:00",
        "row_count": 10,
        "id_range": {"field": "id", "min": 1, "max": 10},
        "content_hash_algorithm": "SHA-256",
        "content_hash": dataset_hash,
        "snapshot_file_sha256": dataset_hash,
    }
    manifest["snapshot_id"] = build_snapshot_id(manifest)
    return manifest
