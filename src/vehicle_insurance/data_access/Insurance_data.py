import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import pymongo

from vehicle_insurance.configurations.mongodb_connection import MongoDBConnection
from vehicle_insurance.constants import DATABASE_NAME
from vehicle_insurance.exceptions.handle_exception import VehicleInsuranceException
from vehicle_insurance.logger.logger_config import setup_logger
from vehicle_insurance.utils.source_manifest import (
    MANIFEST_VERSION,
    canonical_json,
    finalize_source_manifest,
)

logger = setup_logger()

FETCH_BATCH_SIZE = 10_000

DEFAULT_TRAINING_PROJECTION_FIELDS = (
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
)

_RETRYABLE_MONGO_ERRORS = tuple(
    error_type
    for error_type in (
        pymongo.errors.AutoReconnect,
        pymongo.errors.NetworkTimeout,
        pymongo.errors.ServerSelectionTimeoutError,
        getattr(pymongo.errors, "_OperationCancelled", None),
    )
    if error_type is not None
)


@dataclass(frozen=True)
class MongoExtractionResult:
    dataframe: pd.DataFrame
    manifest: dict[str, Any]


def _serialize_mongo_time(value: Any) -> dict[str, int] | None:
    """Convert BSON Timestamp values into stable JSON metadata."""

    if value is None:
        return None
    if isinstance(value, dict) and "clusterTime" in value:
        value = value["clusterTime"]
    seconds = getattr(value, "time", None)
    increment = getattr(value, "inc", None)
    if seconds is not None and increment is not None:
        return {"seconds": int(seconds), "increment": int(increment)}
    if isinstance(value, dict) and {"seconds", "increment"}.issubset(value):
        return {
            "seconds": int(value["seconds"]),
            "increment": int(value["increment"]),
        }
    return None


class InsuranceData(MongoDBConnection):
    def __init__(self, database_name=DATABASE_NAME):
        super().__init__(database_name)

    @staticmethod
    def _projection_metadata(projection: dict[str, Any]) -> dict[str, Any]:
        fields = [
            field
            for field, included in projection.items()
            if included and field != "_id"
        ]
        return {
            "fields": fields,
            "mongo_projection": dict(projection),
        }

    @staticmethod
    def _resolve_ordering_index(collection: Any, sort_field: str) -> str:
        """Require a supporting index instead of silently doing an in-memory sort."""

        index_information = collection.index_information()
        for index_name, metadata in index_information.items():
            keys = metadata.get("key") or metadata.get("keys") or []
            if list(keys)[:1] == [(sort_field, pymongo.ASCENDING)]:
                return str(index_name)
        raise RuntimeError(
            f"MongoDB collection has no ascending index beginning with {sort_field!r}; "
            "create the supporting index before extraction"
        )

    def extract_snapshot(
        self,
        collection_name: str,
        database_name: str | None = None,
        query: dict[str, Any] | None = None,
        projection: dict[str, Any] | None = None,
        sort_field: str = "id",
    ) -> MongoExtractionResult:
        """Read one immutable MongoDB snapshot with a single consistent boundary.

        A retryable cursor failure aborts the extraction. Re-running the same
        artifact directory can only reuse a verified local snapshot; a fresh
        pipeline run creates a new snapshot identity.
        """

        try:
            query = dict(query or {})
            if projection is None:
                projection = {
                    "_id": 0,
                    **{field: 1 for field in DEFAULT_TRAINING_PROJECTION_FIELDS},
                }
            else:
                projection = dict(projection)
            if projection.get("_id") != 0:
                raise ValueError(
                    "MongoDB extraction requires an explicit _id: 0 projection"
                )
            if sort_field not in projection:
                raise ValueError(
                    f"MongoDB extraction projection must include sort field {sort_field!r}"
                )

            db = self.client[database_name] if database_name else self.database
            collection = db[collection_name]
            index_name = self._resolve_ordering_index(collection, sort_field)
            extraction_started_at = datetime.now(timezone.utc).isoformat()
            cursor = None

            # Snapshot sessions give every read in this extraction one
            # server-defined atClusterTime. There is intentionally no fallback
            # to a moving read concern because that would destroy reproducibility.
            try:
                with self.client.start_session(snapshot=True) as session:
                    cursor = collection.find(
                        query,
                        projection,
                        batch_size=FETCH_BATCH_SIZE,
                        session=session,
                    )
                    cursor = cursor.sort([(sort_field, pymongo.ASCENDING)]).hint(
                        index_name
                    )
                    records = []
                    for record_number, document in enumerate(cursor, start=1):
                        records.append(document)
                        if record_number % FETCH_BATCH_SIZE == 0:
                            logger.info(
                                "MongoDB snapshot extraction progress: fetched %d records",
                                record_number,
                            )
                    cluster_time = _serialize_mongo_time(
                        getattr(session, "cluster_time", None)
                    )
                    operation_time = _serialize_mongo_time(
                        getattr(session, "operation_time", None)
                    )
            except _RETRYABLE_MONGO_ERRORS as error:
                raise RuntimeError(
                    "MongoDB snapshot read failed after the read began; extraction "
                    "aborted without retrying so a changed database state cannot "
                    "be mixed into the snapshot. Start a new extraction for a "
                    "new snapshot_id."
                ) from error
            finally:
                if cursor is not None:
                    cursor.close()

            if not records:
                raise ValueError("MongoDB snapshot extraction returned no records")

            dataframe = pd.DataFrame.from_records(records)
            if dataframe[sort_field].duplicated().any() or not dataframe[
                sort_field
            ].is_monotonic_increasing:
                raise ValueError(
                    "MongoDB returned records outside the required unique ascending "
                    f"{sort_field} ordering"
                )
            dataframe = dataframe.map(
                lambda value: np.nan
                if isinstance(value, str) and value == "na"
                else value
            ).infer_objects(copy=False)

            read_timestamp = datetime.now(timezone.utc).isoformat()
            extracted_at = datetime.now(timezone.utc).isoformat()
            manifest = {
                "manifest_version": MANIFEST_VERSION,
                "snapshot_format": "csv",
                "source": {
                    "system": "MongoDB",
                    "database": str(db.name),
                    "collection": collection_name,
                    "namespace": f"{db.name}.{collection_name}",
                },
                "query": {
                    "filter": query,
                    "filter_canonical_json": canonical_json(query),
                },
                "projection": self._projection_metadata(projection),
                "ordering": {
                    "field": sort_field,
                    "direction": "ascending",
                    "index_required": True,
                    "index_name": index_name,
                },
                "read_boundary": {
                    "mode": "snapshot_session",
                    "read_concern": "snapshot",
                    "cluster_time": cluster_time,
                    "operation_time": operation_time,
                },
                "read_timestamp_utc": read_timestamp,
                "extraction_started_at_utc": extraction_started_at,
                "extracted_at_utc": extracted_at,
                "database_state_retry_policy": {
                    "same_snapshot_read_failure": "abort_extraction",
                    "new_extraction": "new_snapshot_id",
                },
            }
            # The CSV is written by DataIngestion using the same canonical
            # serializer. The content hash therefore becomes the immutable
            # snapshot-file identity before any training stage starts.
            manifest = finalize_source_manifest(
                manifest,
                dataframe,
                snapshot_file_name="data.csv",
            )
            logger.info(
                "MongoDB snapshot extracted: namespace=%s rows=%d snapshot_id=%s",
                manifest["source"]["namespace"],
                manifest["row_count"],
                manifest["snapshot_id"],
            )
            return MongoExtractionResult(dataframe=dataframe, manifest=manifest)
        except Exception as error:
            raise VehicleInsuranceException(error, sys) from error
