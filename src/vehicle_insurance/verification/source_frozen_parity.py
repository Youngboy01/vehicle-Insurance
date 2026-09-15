"""Source-frozen verification against the frozen notebook.

The standalone verifier certifies one candidate artifact against the historical
notebook final run for the same source snapshot, split protocol, preprocessing
contract, and model recipe. The training pipeline also invokes this verifier as
a formal stage before final evaluation. A successful run produces a durable
JSON report under the candidate artifact directory. Local/MLflow lineage
mutation is opt-in via ``--apply-lineage`` and ``--log-mlflow-evidence``.

Run from the repository root, for example::

    uv run python -m vehicle_insurance.verification.source_frozen_parity ^
        --candidate-artifact-dir <candidate-artifact-dir>

The source split IDs are reconstructed with the exact split sequence from
``notebooks/experimentation_v2.ipynb`` because the historical notebook run did
not persist a split-ID manifest. The report explicitly records this source.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from vehicle_insurance.constants import (
    MODEL_SPEC_FILE_PATH,
    SCHEMA_FILE_PATH,
    TARGET_COLUMN,
)
from vehicle_insurance.exceptions.handle_exception import VehicleInsuranceException
from vehicle_insurance.logger.logger_config import setup_logger
from vehicle_insurance.utils import load_object, read_yaml_file
from vehicle_insurance.utils.report_output import (
    protected_artifact_paths,
    resolve_report_path,
    validate_report_destination,
    write_json_atomic,
)
from vehicle_insurance.utils.source_manifest import validate_source_manifest

logger = setup_logger()


class SourceFrozenParityCertification:
    """Compare one production candidate with the frozen notebook reference."""

    def __init__(
        self,
        candidate_artifact_dir: str | Path,
        reference_run_id: str | None = None,
        reference_model_uri: str | None = None,
        source_data_path: str | Path | None = None,
        report_path: str | Path | None = None,
        prediction_tolerance: float = 1.0e-12,
        metric_tolerance: float = 1.0e-6,
        apply_lineage: bool = False,
        log_mlflow_evidence: bool = False,
    ) -> None:
        self.candidate_artifact_dir = Path(candidate_artifact_dir).resolve()
        self.model_spec = read_yaml_file(MODEL_SPEC_FILE_PATH)
        self.schema_config = read_yaml_file(SCHEMA_FILE_PATH)

        tracking_config = self.model_spec["artifacts"]["mlflow_tracking"]
        configured_reference_model_uri = str(
            tracking_config.get("reference_final_model_uri", "")
        ).strip()
        self.reference_run_id = (
            reference_run_id or tracking_config["reference_final_run_id"]
        )
        self.reference_model_uri = (
            reference_model_uri
            or configured_reference_model_uri
            or f"runs:/{self.reference_run_id}/model"
        )
        self.configured_reference_model_uri = configured_reference_model_uri
        self.configured_reference_model_id = str(
            tracking_config.get("reference_final_model_id", "")
        ).strip()
        self.source_data_path = Path(source_data_path).resolve() if source_data_path else (
            self.candidate_artifact_dir
            / "data_ingestion"
            / "feature_store"
            / "data.csv"
        )
        self.source_manifest_path = self.source_data_path.with_name(
            "source_manifest.json"
        )
        self.report_path_input = (
            Path(report_path).expanduser()
            if report_path
            else self.candidate_artifact_dir
            / "source_frozen_parity"
            / "parity_report.json"
        )
        self.report_path = self.report_path_input.resolve(strict=False)
        self.prediction_tolerance = self._positive_finite(
            prediction_tolerance,
            "prediction_tolerance",
        )
        self.metric_tolerance = self._positive_finite(
            metric_tolerance,
            "metric_tolerance",
        )
        self.apply_lineage = apply_lineage
        self.log_mlflow_evidence = log_mlflow_evidence
        self.certification_attempt_id = uuid4().hex
        self._report_destination_owned_path: Path | None = None

        self.errors: list[dict[str, Any]] = []
        self.warnings: list[dict[str, Any]] = []
        self.checks: dict[str, dict[str, Any]] = {}
        self.candidate_metrics_payload: dict[str, Any] = {}
        self.candidate_lineage: dict[str, Any] = {}
        self.source_manifest: dict[str, Any] = {}
        self.candidate_splits: dict[str, pd.DataFrame] = {}
        self.reference_splits: dict[str, dict[str, list[Any]]] = {}
        self.candidate_model: Any = None
        self.candidate_model_path: Path | None = None
        self.candidate_identity: dict[str, Any] = {}
        self.reference_model: Any = None
        self.candidate_predictions: np.ndarray | None = None
        self.reference_predictions: np.ndarray | None = None
        self._report: dict[str, Any] | None = None

    @staticmethod
    def _positive_finite(value: float, field_name: str) -> float:
        try:
            converted = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{field_name} must be a finite positive number") from error
        if not math.isfinite(converted) or converted <= 0:
            raise ValueError(f"{field_name} must be a finite positive number")
        return converted

    @staticmethod
    def _canonical_id(value: Any) -> str:
        if isinstance(value, (np.integer, int)):
            return str(int(value))
        if isinstance(value, (np.floating, float)):
            if not math.isfinite(float(value)):
                raise ValueError(f"ID is not finite: {value!r}")
            if float(value).is_integer():
                return str(int(value))
        return str(value)

    @classmethod
    def _id_sequence(cls, dataframe: pd.DataFrame) -> list[str]:
        if "id" not in dataframe.columns:
            raise ValueError("Split is missing required identifier column 'id'")
        values = [cls._canonical_id(value) for value in dataframe["id"].tolist()]
        if len(values) != len(set(values)):
            raise ValueError("Split contains duplicate IDs")
        return values

    @staticmethod
    def _sha256_bytes(payload: bytes) -> str:
        return hashlib.sha256(payload).hexdigest()

    @classmethod
    def _sequence_digest(cls, values: list[str], sort_values: bool = False) -> str:
        materialized = sorted(values) if sort_values else values
        return cls._sha256_bytes("\n".join(materialized).encode("utf-8"))

    @staticmethod
    def _sha256_file(file_path: Path) -> str:
        digest = hashlib.sha256()
        with file_path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _dataset_hash(splits: list[pd.DataFrame]) -> str:
        combined = pd.concat(splits, ignore_index=True).sort_values("id")
        canonical_bytes = combined.to_csv(index=False, lineterminator="\n").encode(
            "utf-8"
        )
        return hashlib.sha256(canonical_bytes).hexdigest()

    def _record_error(self, stage: str, error: Exception) -> None:
        """Record a structured error and preserve the project's exception trace."""
        if sys.exc_info()[0] is not None:
            logger.exception("Source-frozen parity stage failed: %s", stage)
        else:
            logger.error("Source-frozen parity stage failed: %s", stage)
        try:
            wrapped_error = VehicleInsuranceException(error, sys)
            traceable_message = str(wrapped_error)
        except Exception as wrapping_error:  # pragma: no cover - defensive only
            traceable_message = f"{type(error).__name__}: {error}; wrapper failed: {wrapping_error}"

        self.errors.append(
            {
                "stage": stage,
                "error_type": type(error).__name__,
                "message": str(error),
                "traceable_exception": traceable_message,
            }
        )

    def _record_warning(self, code: str, message: str, **details: Any) -> None:
        logger.warning("Source-frozen parity warning: %s - %s", code, message)
        self.warnings.append({"code": code, "message": message, **details})

    def _certificate_report_relative_path(self) -> str:
        """Return a relocation-safe certificate path relative to the candidate."""

        try:
            relative_path = self.report_path.relative_to(self.candidate_artifact_dir)
        except ValueError as error:
            raise ValueError(
                "A lineage-bound parity certificate must be stored inside the "
                "candidate artifact directory"
            ) from error
        if not relative_path.parts or ".." in relative_path.parts:
            raise ValueError("Parity certificate path must remain inside the candidate artifact")
        return relative_path.as_posix()

    def _recording_mode(self) -> str:
        if not getattr(self, "apply_lineage", False):
            return "audit_only"
        return "local_and_mlflow" if getattr(self, "log_mlflow_evidence", False) else "local_only"

    def _report_destination_policy(
        self,
        *,
        claim_destination: bool = False,
    ) -> tuple[Path, tuple[Path, ...]]:
        """Return a validated parity destination and protected-input set."""

        canonical_certificate = (
            self.candidate_artifact_dir
            / "source_frozen_parity"
            / "parity_report.json"
        )
        parity_evidence = (
            self.candidate_lineage.get("source_frozen_parity")
            if isinstance(getattr(self, "candidate_lineage", None), dict)
            else None
        )
        extra_paths: list[Path] = [
            path
            for path in (
                getattr(self, "source_data_path", None),
                getattr(self, "source_manifest_path", None),
            )
            if path is not None
        ]
        if isinstance(parity_evidence, dict):
            certificate_path = parity_evidence.get("certificate_report_path")
            if isinstance(certificate_path, str) and certificate_path:
                certificate = Path(certificate_path).expanduser()
                extra_paths.append(
                    certificate
                    if certificate.is_absolute()
                    else self.candidate_artifact_dir / certificate
                )
        protected = list(
            protected_artifact_paths(
                self.candidate_artifact_dir,
                extra_paths=extra_paths,
            )
        )
        # The canonical parity certificate is this writer's own transactional
        # output. An existing custom file is validated but never claimed; a
        # later lineage load may identify it as protected evidence. A fresh
        # custom destination becomes owned only after validation; bind that
        # exemption to the exact path claimed by this attempt so a later path
        # mutation cannot unprotect another file.
        destination = getattr(self, "report_path_input", self.report_path)
        destination_preexisting = Path(destination).expanduser().is_file()
        destination_resolved = resolve_report_path(destination)
        owned_destination = getattr(self, "_report_destination_owned_path", None)
        owned_destination_resolved = (
            resolve_report_path(owned_destination)
            if owned_destination is not None
            else None
        )
        is_owned_destination = destination_resolved == owned_destination_resolved
        is_canonical_destination = destination_resolved == resolve_report_path(
            canonical_certificate
        )
        if is_owned_destination or is_canonical_destination:
            exempt_destination = (
                owned_destination if is_owned_destination else canonical_certificate
            )
            protected = [
                path
                for path in protected
                if resolve_report_path(path) != resolve_report_path(exempt_destination)
            ]
        target = validate_report_destination(
            destination,
            protected_paths=protected,
        )
        if claim_destination and not destination_preexisting:
            self._report_destination_owned_path = target
        return target, tuple(protected)

    def _validate_report_destination(self) -> Path:
        return self._report_destination_policy(claim_destination=True)[0]

    def _metrics_destination_policy(self) -> tuple[Path, tuple[Path, ...]]:
        """Return the canonical metrics output and protected-input set."""

        metrics_path = self.candidate_artifact_dir / "model_trainer" / "metrics.json"
        protected = list(protected_artifact_paths(self.candidate_artifact_dir))
        # Metrics are this transaction's owned state output. Only the exact
        # canonical entry is exempted; a symlink/hard-link alias to another
        # protected artifact is still rejected by the shared validator.
        protected = [
            path
            for path in protected
            if resolve_report_path(path) != resolve_report_path(metrics_path)
        ]
        target = validate_report_destination(
            metrics_path,
            protected_paths=protected,
        )
        return target, tuple(protected)

    def _run_stage(
        self,
        stage: str,
        function: Callable[[], Any],
    ) -> Any:
        try:
            logger.info("Source-frozen parity stage started: %s", stage)
            result = function()
            logger.info("Source-frozen parity stage completed: %s", stage)
            return result
        except Exception as error:
            self._record_error(stage, error)
            return None

    def _validate_configuration(self) -> None:
        expected_contract_version = str(self.schema_config["version"])
        if str(self.model_spec.get("contract_version")) != expected_contract_version:
            raise ValueError(
                "Model specification and schema contract versions do not match: "
                f"model_spec={self.model_spec.get('contract_version')!r}, "
                f"schema={expected_contract_version!r}"
            )

        protocol = self.model_spec["data_protocol"]
        expected_protocol = "stratified_80_10_10"
        if protocol.get("split_protocol") != expected_protocol:
            raise ValueError(
                f"Unsupported split protocol: {protocol.get('split_protocol')!r}"
            )

        ratios = (
            float(protocol["train_ratio"]),
            float(protocol["validation_ratio"]),
            float(protocol["test_ratio"]),
        )
        if ratios != (0.8, 0.1, 0.1):
            raise ValueError(f"Expected frozen 80/10/10 ratios, found {ratios}")
        if int(protocol["random_state"]) != 42:
            raise ValueError(
                f"Expected frozen random_state=42, found {protocol['random_state']}"
            )

        raw_features = self.model_spec["features"]["raw_inputs"]
        schema_features = self.schema_config["features"]["canonical_order"]
        if raw_features != schema_features:
            raise ValueError(
                "Model specification and schema canonical feature order differ"
            )

        self.raw_features = list(raw_features)
        self.random_state = int(protocol["random_state"])
        self.expected_reference_metrics = dict(
            self.model_spec["evaluation_policy"]["frozen_reference_metrics"]
        )

        if self.apply_lineage:
            self._certificate_report_relative_path()

        if (
            self.configured_reference_model_uri
            and self.configured_reference_model_uri != self.reference_model_uri
        ):
            self._record_warning(
                "reference_uri_overridden",
                "The command-line reference model URI overrides the URI recorded "
                "in model_spec.yaml.",
                configured_uri=self.configured_reference_model_uri,
                resolved_uri=self.reference_model_uri,
            )

    def _load_candidate_artifacts(self) -> None:
        if not self.candidate_artifact_dir.is_dir():
            raise FileNotFoundError(
                f"Candidate artifact directory does not exist: {self.candidate_artifact_dir}"
            )

        ingestion_dir = self.candidate_artifact_dir / "data_ingestion"
        trainer_dir = self.candidate_artifact_dir / "model_trainer"
        required_paths = {
            "training_split": ingestion_dir / "ingested" / "train.csv",
            "validation_split": ingestion_dir / "ingested" / "validation.csv",
            "test_split": ingestion_dir / "ingested" / "test.csv",
            "candidate_model": trainer_dir / "model.pkl",
            "candidate_metrics": trainer_dir / "metrics.json",
            "source_manifest": self.source_manifest_path,
        }
        missing_paths = {
            name: str(path)
            for name, path in required_paths.items()
            if not path.is_file()
        }
        if missing_paths:
            raise FileNotFoundError(
                f"Candidate artifact is missing required files: {missing_paths}"
            )

        self.candidate_splits = {
            "train": pd.read_csv(required_paths["training_split"]),
            "validation": pd.read_csv(required_paths["validation_split"]),
            "test": pd.read_csv(required_paths["test_split"]),
        }
        for split_name, dataframe in self.candidate_splits.items():
            if dataframe.empty:
                raise ValueError(f"Candidate {split_name} split is empty")
            required_columns = {"id", TARGET_COLUMN, *self.raw_features}
            missing_columns = sorted(required_columns - set(dataframe.columns))
            if missing_columns:
                raise ValueError(
                    f"Candidate {split_name} split is missing columns: {missing_columns}"
                )
            extra_columns = sorted(set(dataframe.columns) - required_columns)
            if extra_columns:
                raise ValueError(
                    f"Candidate {split_name} split contains unexpected columns: "
                    f"{extra_columns}"
                )
            self._id_sequence(dataframe)

        with required_paths["candidate_metrics"].open(encoding="utf-8") as file:
            payload = json.load(file)
        if not isinstance(payload, dict):
            raise ValueError("Candidate metrics.json must contain a JSON object")
        self.candidate_metrics_payload = payload
        lineage = payload.get("lineage")
        if not isinstance(lineage, dict):
            raise ValueError("Candidate metrics.json does not contain lineage")
        self.candidate_lineage = lineage

        with required_paths["source_manifest"].open(encoding="utf-8") as file:
            source_manifest = json.load(file)
        if not isinstance(source_manifest, dict):
            raise ValueError("Candidate source_manifest.json must contain a JSON object")
        self.source_manifest = source_manifest

        self.candidate_model_path = required_paths["candidate_model"]
        self.candidate_model = load_object(str(required_paths["candidate_model"]))
        if not hasattr(self.candidate_model, "predict_proba"):
            raise TypeError("Candidate model does not expose predict_proba")

        candidate_run_id = str(
            payload.get("mlflow_run_id") or lineage.get("mlflow_run_id") or ""
        )
        if not candidate_run_id:
            raise ValueError("Candidate MLflow run ID is missing from metrics/lineage")

        self.candidate_identity = self._build_candidate_identity()

    def _build_candidate_identity(self) -> dict[str, Any]:
        """Return the immutable identity bound to one candidate artifact."""

        payload = self.candidate_metrics_payload
        lineage = self.candidate_lineage
        model_path = self.candidate_model_path or (
            self.candidate_artifact_dir / "model_trainer" / "model.pkl"
        )
        model_file_sha256 = self._sha256_file(model_path)
        return {
            "mlflow_run_id": str(
                payload.get("mlflow_run_id")
                or lineage.get("mlflow_run_id")
                or ""
            ),
            "model_id": str(payload.get("model_id") or lineage.get("model_id") or ""),
            "model_uri": str(payload.get("model_uri") or lineage.get("model_uri") or ""),
            "dataset_snapshot_id": str(lineage.get("dataset_snapshot_id") or ""),
            "dataset_content_hash": str(lineage.get("dataset_content_hash") or ""),
            "model_file_sha256": model_file_sha256,
        }

    def _reconstruct_notebook_splits(self) -> None:
        if not self.source_data_path.is_file():
            raise FileNotFoundError(
                f"Source snapshot file does not exist: {self.source_data_path}"
            )

        source_dataframe = pd.read_csv(self.source_data_path)
        if source_dataframe.empty:
            raise ValueError("Source snapshot is empty")
        expected_columns = {"id", TARGET_COLUMN, *self.raw_features}
        missing_columns = sorted(expected_columns - set(source_dataframe.columns))
        if missing_columns:
            raise ValueError(
                f"Source snapshot is missing required columns: {missing_columns}"
            )
        extra_columns = sorted(set(source_dataframe.columns) - expected_columns)
        if extra_columns:
            raise ValueError(
                f"Source snapshot contains unexpected columns: {extra_columns}"
            )
        source_ids = self._id_sequence(source_dataframe)
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("Source snapshot contains duplicate IDs")
        validate_source_manifest(self.source_manifest, source_dataframe)
        lineage_manifest = self.candidate_lineage.get("source_manifest")
        if not isinstance(lineage_manifest, dict):
            raise ValueError("Candidate lineage does not contain source manifest")
        if lineage_manifest.get("snapshot_id") != self.source_manifest.get(
            "snapshot_id"
        ):
            raise ValueError(
                "Candidate lineage source snapshot ID does not match the source "
                "manifest beside the feature-store CSV"
            )

        candidate_hash = self._dataset_hash(list(self.candidate_splits.values()))
        source_hash = self._dataset_hash([source_dataframe])
        expected_lineage_hash = str(
            self.candidate_lineage.get("dataset_content_hash", "")
        )
        self.checks["dataset_snapshot"] = {
            "passed": source_hash == candidate_hash == expected_lineage_hash,
            "source_data_path": str(self.source_data_path),
            "source_content_hash": source_hash,
            "candidate_split_content_hash": candidate_hash,
            "candidate_lineage_content_hash": expected_lineage_hash,
        }
        if not self.checks["dataset_snapshot"]["passed"]:
            raise ValueError(
                "Source snapshot hash does not match the candidate split/lineage hash: "
                f"source={source_hash}, candidate={candidate_hash}, "
                f"lineage={expected_lineage_hash}"
            )

        # This is the exact sequence in experimentation_v2.ipynb cell 24.
        # Preserve source row order; do not sort here because row order is part
        # of the notebook's deterministic split input.
        features = source_dataframe.drop(columns=[TARGET_COLUMN, "id"])
        target = source_dataframe[TARGET_COLUMN]
        identifiers = source_dataframe["id"]
        _, _, y_train, y_remainder, id_train, id_remainder = train_test_split(
            features,
            target,
            identifiers,
            test_size=0.2,
            random_state=self.random_state,
            stratify=target,
        )
        _, _, y_test, y_validation, id_test, id_validation = train_test_split(
            source_dataframe.loc[id_remainder.index, self.raw_features],
            y_remainder,
            id_remainder,
            test_size=0.5,
            random_state=self.random_state,
            stratify=y_remainder,
        )

        self.reference_splits = {
            "train": {
                "ids": [self._canonical_id(value) for value in id_train.tolist()],
                "target": [int(value) for value in y_train.tolist()],
            },
            "validation": {
                "ids": [
                    self._canonical_id(value) for value in id_validation.tolist()
                ],
                "target": [int(value) for value in y_validation.tolist()],
            },
            "test": {
                "ids": [self._canonical_id(value) for value in id_test.tolist()],
                "target": [int(value) for value in y_test.tolist()],
            },
        }
        self.checks["reference_split_source"] = {
            "passed": True,
            "method": "exact_notebook_cell_24_split_sequence",
            "source_manifest_available": True,
            "source_manifest_path": str(
                getattr(
                    self,
                    "source_manifest_path",
                    self.source_data_path.with_name("source_manifest.json"),
                )
            ),
            "source_snapshot_id": self.source_manifest["snapshot_id"],
            "source_data_path": str(self.source_data_path),
            "random_state": self.random_state,
            "split_protocol": "stratified_80_10_10",
        }

    def _compare_split_ids_and_targets(self) -> None:
        split_checks: dict[str, Any] = {}
        all_set_matches = True
        all_order_matches = True
        all_target_matches = True

        for split_name, candidate_dataframe in self.candidate_splits.items():
            candidate_ids = self._id_sequence(candidate_dataframe)
            expected = self.reference_splits[split_name]
            expected_ids = expected["ids"]
            candidate_target = [
                int(value) for value in candidate_dataframe[TARGET_COLUMN].tolist()
            ]
            expected_target = expected["target"]
            set_match = set(candidate_ids) == set(expected_ids)
            order_match = candidate_ids == expected_ids
            target_match = candidate_target == expected_target
            all_set_matches = all_set_matches and set_match
            all_order_matches = all_order_matches and order_match
            all_target_matches = all_target_matches and target_match

            split_checks[split_name] = {
                "candidate_row_count": len(candidate_ids),
                "reference_row_count": len(expected_ids),
                "id_set_match": set_match,
                "id_order_match": order_match,
                "target_order_match": target_match,
                "candidate_id_set_sha256": self._sequence_digest(
                    candidate_ids,
                    sort_values=True,
                ),
                "reference_id_set_sha256": self._sequence_digest(
                    expected_ids,
                    sort_values=True,
                ),
                "candidate_id_order_sha256": self._sequence_digest(candidate_ids),
                "reference_id_order_sha256": self._sequence_digest(expected_ids),
                "first_candidate_ids": candidate_ids[:5],
                "first_reference_ids": expected_ids[:5],
            }

            if not set_match:
                split_checks[split_name]["candidate_only_id_count"] = len(
                    set(candidate_ids) - set(expected_ids)
                )
                split_checks[split_name]["reference_only_id_count"] = len(
                    set(expected_ids) - set(candidate_ids)
                )

        self.checks["split_ids"] = {
            "passed": all_set_matches and all_order_matches,
            "split_id_sets_match": all_set_matches,
            "split_id_order_match": all_order_matches,
            "prediction_order_match": split_checks["test"]["id_order_match"],
            "splits": split_checks,
        }
        self.checks["target_order"] = {
            "passed": all_target_matches,
            "all_split_target_orders_match": all_target_matches,
        }

    @staticmethod
    def _extract_preprocessor(model: Any) -> Any:
        named_steps = getattr(model, "named_steps", None)
        if not named_steps:
            raise TypeError("Reference/candidate model is not a named sklearn pipeline")
        for step_name in ("preprocessor", "transformer"):
            if step_name in named_steps:
                return named_steps[step_name]
        raise KeyError("Pipeline does not contain a preprocessor/transformer step")

    @staticmethod
    def _feature_names(preprocessor: Any) -> list[str]:
        if not hasattr(preprocessor, "get_feature_names_out"):
            raise TypeError("Fitted preprocessor does not expose feature names")
        return [str(value) for value in preprocessor.get_feature_names_out()]

    @staticmethod
    def _semantic_feature_names(feature_names: list[str]) -> list[str]:
        """Remove sklearn display prefixes without changing column order.

        The notebook uses sklearn's default ``verbose_feature_names_out=True``
        and emits names such as ``categorical__Gender_Female``. The production
        contract uses ``False`` and emits ``Gender_Female``. The prefix is
        representation metadata; the underlying feature position and value are
        the same. Comparing the normalized sequence still catches real order
        differences.
        """
        return [name.split("__", 1)[-1] for name in feature_names]

    @staticmethod
    def _raw_feature_names(preprocessor: Any) -> list[str] | None:
        values = getattr(preprocessor, "feature_names_in_", None)
        if values is None:
            return None
        return [str(value) for value in values.tolist()]

    @staticmethod
    def _as_dense_array(value: Any) -> np.ndarray:
        if hasattr(value, "toarray"):
            value = value.toarray()
        array = np.asarray(value, dtype=np.float64)
        if array.ndim != 2:
            raise ValueError(f"Transformed data must be 2-dimensional, found {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError("Transformed data contains non-finite values")
        return array

    @staticmethod
    def _prediction_checksum(predictions: np.ndarray) -> str:
        normalized = np.ascontiguousarray(predictions, dtype=np.float64)
        return hashlib.sha256(normalized.tobytes()).hexdigest()

    def _load_reference_model(self) -> None:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if not tracking_uri:
            raise ValueError(
                "MLFLOW_TRACKING_URI is not set; the historical reference model "
                "must be loaded from MLflow"
            )
        mlflow.set_tracking_uri(tracking_uri)
        logger.info("Loading source-frozen reference model from %s", self.reference_model_uri)
        self.reference_model = mlflow.sklearn.load_model(self.reference_model_uri)
        if not hasattr(self.reference_model, "predict_proba"):
            raise TypeError("Reference MLflow model does not expose predict_proba")

        model_info = mlflow.models.get_model_info(self.reference_model_uri)
        observed_model_id = str(getattr(model_info, "model_id", "") or "")
        observed_run_id = str(getattr(model_info, "run_id", "") or "")
        model_id_match = (
            not self.configured_reference_model_id
            or observed_model_id == self.configured_reference_model_id
        )
        run_id_match = not observed_run_id or observed_run_id == self.reference_run_id
        self.checks["reference_model_identity"] = {
            "passed": model_id_match and run_id_match,
            "configured_model_id": self.configured_reference_model_id,
            "observed_model_id": observed_model_id,
            "expected_run_id": self.reference_run_id,
            "observed_run_id": observed_run_id,
            "model_id_match": model_id_match,
            "run_id_match": run_id_match,
            "model_uri": self.reference_model_uri,
        }
        if not model_id_match or not run_id_match:
            raise ValueError(
                "Reference model identity does not match model_spec lineage: "
                f"configured_model_id={self.configured_reference_model_id!r}, "
                f"observed_model_id={observed_model_id!r}, "
                f"expected_run_id={self.reference_run_id!r}, "
                f"observed_run_id={observed_run_id!r}"
            )

    def _compare_models_and_predictions(self) -> None:
        candidate_preprocessor = self._extract_preprocessor(self.candidate_model)
        reference_preprocessor = self._extract_preprocessor(self.reference_model)
        candidate_features = self._feature_names(candidate_preprocessor)
        reference_features = self._feature_names(reference_preprocessor)
        candidate_semantic_features = self._semantic_feature_names(candidate_features)
        reference_semantic_features = self._semantic_feature_names(reference_features)
        candidate_raw_features = self._raw_feature_names(candidate_preprocessor)
        reference_raw_features = self._raw_feature_names(reference_preprocessor)

        self.checks["raw_feature_order"] = {
            "passed": candidate_raw_features == reference_raw_features == self.raw_features,
            "expected": self.raw_features,
            "candidate": candidate_raw_features,
            "reference": reference_raw_features,
        }
        self.checks["transformed_feature_order"] = {
            "passed": candidate_semantic_features == reference_semantic_features,
            "exact_name_match": candidate_features == reference_features,
            "comparison": "semantic_order_after_removing_sklearn_transformer_prefix",
            "candidate_count": len(candidate_features),
            "reference_count": len(reference_features),
            "candidate": candidate_features,
            "reference": reference_features,
            "candidate_semantic": candidate_semantic_features,
            "reference_semantic": reference_semantic_features,
        }

        candidate_model = self.candidate_model.named_steps.get("model")
        reference_model = self.reference_model.named_steps.get("model")
        if candidate_model is None or reference_model is None:
            raise KeyError("Candidate/reference pipeline must contain a 'model' step")
        if not hasattr(candidate_model, "get_params") or not hasattr(
            reference_model, "get_params"
        ):
            raise TypeError("Candidate/reference model does not expose get_params")

        configured_parameters = dict(self.model_spec["model"]["hyperparameters"])
        configured_parameters["scale_pos_weight"] = float(
            self.model_spec["model"]["class_weighting"]["scale_pos_weight"]
        )
        candidate_parameters = candidate_model.get_params()
        reference_parameters = reference_model.get_params()
        parameter_checks: dict[str, Any] = {}
        all_parameters_match = True
        for parameter_name, expected_value in configured_parameters.items():
            candidate_value = candidate_parameters.get(parameter_name)
            reference_value = reference_parameters.get(parameter_name)
            candidate_match = self._values_match(candidate_value, expected_value)
            reference_match = self._values_match(reference_value, expected_value)
            reference_candidate_match = self._values_match(
                candidate_value,
                reference_value,
            )
            passed = candidate_match and reference_match and reference_candidate_match
            all_parameters_match = all_parameters_match and passed
            parameter_checks[parameter_name] = {
                "expected": expected_value,
                "candidate": candidate_value,
                "reference": reference_value,
                "candidate_matches_expected": candidate_match,
                "reference_matches_expected": reference_match,
                "candidate_matches_reference": reference_candidate_match,
                "passed": passed,
            }
        self.checks["model_parameters"] = {
            "passed": all_parameters_match,
            "candidate_class": type(candidate_model).__name__,
            "reference_class": type(reference_model).__name__,
            "checks": parameter_checks,
        }

        test_dataframe = self.candidate_splits["test"].loc[:, self.raw_features]
        candidate_transformed = self._as_dense_array(
            candidate_preprocessor.transform(test_dataframe)
        )
        reference_transformed = self._as_dense_array(
            reference_preprocessor.transform(test_dataframe)
        )
        if candidate_transformed.shape != reference_transformed.shape:
            transformed_error = math.inf
        else:
            transformed_error = float(
                np.max(np.abs(candidate_transformed - reference_transformed))
            )
        self.checks["transformed_values"] = {
            "passed": (
                candidate_transformed.shape == reference_transformed.shape
                and transformed_error <= self.prediction_tolerance
            ),
            "candidate_shape": list(candidate_transformed.shape),
            "reference_shape": list(reference_transformed.shape),
            "maximum_absolute_error": transformed_error,
            "maximum_allowed_error": self.prediction_tolerance,
        }

        candidate_predictions = np.asarray(
            self.candidate_model.predict_proba(test_dataframe)
        )[:, 1].astype(np.float64)
        reference_predictions = np.asarray(
            self.reference_model.predict_proba(test_dataframe)
        )[:, 1].astype(np.float64)
        if candidate_predictions.shape != reference_predictions.shape:
            prediction_error = math.inf
        else:
            prediction_error = float(
                np.max(np.abs(candidate_predictions - reference_predictions))
            )
        self.candidate_predictions = candidate_predictions
        self.reference_predictions = reference_predictions
        self.checks["prediction_array"] = {
            "passed": (
                candidate_predictions.shape == reference_predictions.shape
                and prediction_error <= self.prediction_tolerance
            ),
            "candidate_shape": list(candidate_predictions.shape),
            "reference_shape": list(reference_predictions.shape),
            "maximum_absolute_error": prediction_error,
            "maximum_allowed_error": self.prediction_tolerance,
            "candidate_sha256": self._prediction_checksum(candidate_predictions),
            "reference_sha256": self._prediction_checksum(reference_predictions),
            "evaluated_in_test_id_order": self.checks["split_ids"][
                "prediction_order_match"
            ],
        }

    def _ranking_metrics(
        self,
        predictions: np.ndarray,
        dataframe: pd.DataFrame,
    ) -> dict[str, float]:
        target = dataframe[TARGET_COLUMN].to_numpy(dtype=int)
        identifiers = dataframe["id"].to_numpy()
        if len(predictions) != len(target):
            raise ValueError("Prediction and target lengths differ")

        ranking = pd.DataFrame(
            {"id": identifiers, "score": predictions, "target": target}
        ).sort_values(
            by=["score", "id"],
            ascending=[False, True],
            kind="mergesort",
        )
        k = math.floor(len(ranking) * 0.10)
        if k <= 0:
            precision_at_k = recall_at_k = lift_at_k = 0.0
        else:
            selected_positive_count = float(ranking.head(k)["target"].sum())
            positive_count = float(target.sum())
            precision_at_k = selected_positive_count / k
            recall_at_k = selected_positive_count / positive_count if positive_count else 0.0
            prevalence = positive_count / len(target)
            lift_at_k = precision_at_k / prevalence if prevalence else 0.0

        return {
            "pr_auc": float(average_precision_score(target, predictions)),
            "roc_auc": float(roc_auc_score(target, predictions)),
            "precision_at_10pct": float(precision_at_k),
            "recall_at_10pct": float(recall_at_k),
            "lift_at_10pct": float(lift_at_k),
        }

    def _compare_metrics(self) -> None:
        if self.candidate_predictions is None or self.reference_predictions is None:
            raise ValueError("Prediction arrays are unavailable for metric comparison")

        test_dataframe = self.candidate_splits["test"]
        candidate_metrics = self._ranking_metrics(
            self.candidate_predictions,
            test_dataframe,
        )
        reference_metrics = self._ranking_metrics(
            self.reference_predictions,
            test_dataframe,
        )
        recorded_metrics = self.candidate_metrics_payload.get("test_metrics", {})
        metric_checks: dict[str, Any] = {}
        all_metrics_match = True
        for metric_name in self.expected_reference_metrics:
            candidate_value = candidate_metrics.get(metric_name)
            reference_value = reference_metrics.get(metric_name)
            frozen_value = float(self.expected_reference_metrics[metric_name])
            recorded_value = recorded_metrics.get(metric_name)
            candidate_reference_error = (
                abs(candidate_value - reference_value)
                if candidate_value is not None and reference_value is not None
                else math.inf
            )
            candidate_frozen_error = (
                abs(candidate_value - frozen_value)
                if candidate_value is not None
                else math.inf
            )
            recorded_error = (
                abs(float(recorded_value) - candidate_value)
                if recorded_value is not None and candidate_value is not None
                else math.inf
            )
            passed = (
                candidate_reference_error <= self.metric_tolerance
                and candidate_frozen_error <= self.metric_tolerance
                and recorded_error <= self.metric_tolerance
            )
            all_metrics_match = all_metrics_match and passed
            metric_checks[metric_name] = {
                "candidate_recomputed": candidate_value,
                "reference_recomputed": reference_value,
                "frozen_notebook_value": frozen_value,
                "candidate_recorded": recorded_value,
                "candidate_reference_abs_error": candidate_reference_error,
                "candidate_frozen_abs_error": candidate_frozen_error,
                "candidate_recorded_abs_error": recorded_error,
                "maximum_allowed_error": self.metric_tolerance,
                "passed": passed,
            }

        self.checks["metrics"] = {
            "passed": all_metrics_match,
            "metric_tolerance": self.metric_tolerance,
            "candidate_recomputed": candidate_metrics,
            "reference_recomputed": reference_metrics,
            "checks": metric_checks,
        }

    @classmethod
    def _values_match(cls, left: Any, right: Any) -> bool:
        if left is None or right is None:
            return left is right
        if isinstance(left, (bool, np.bool_)) or isinstance(
            right, (bool, np.bool_)
        ):
            return bool(left) == bool(right)
        try:
            left_float = float(left)
            right_float = float(right)
            if math.isfinite(left_float) and math.isfinite(right_float):
                return math.isclose(
                    left_float,
                    right_float,
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
        except (TypeError, ValueError):
            pass
        return left == right

    def _build_report(self) -> dict[str, Any]:
        parity_checks = {
            "dataset_snapshot": self.checks.get("dataset_snapshot", {"passed": False}),
            "split_ids": self.checks.get("split_ids", {"passed": False}),
            "target_order": self.checks.get("target_order", {"passed": False}),
            "raw_feature_order": self.checks.get("raw_feature_order", {"passed": False}),
            "transformed_feature_order": self.checks.get(
                "transformed_feature_order", {"passed": False}
            ),
            "reference_model_identity": self.checks.get(
                "reference_model_identity", {"passed": False}
            ),
            "model_parameters": self.checks.get(
                "model_parameters", {"passed": False}
            ),
            "transformed_values": self.checks.get("transformed_values", {"passed": False}),
            "prediction_array": self.checks.get("prediction_array", {"passed": False}),
            "metrics": self.checks.get("metrics", {"passed": False}),
        }
        all_checks_passed = bool(parity_checks) and all(
            bool(check.get("passed")) for check in parity_checks.values()
        )
        status = "verified" if all_checks_passed and not self.errors else "failed"
        recording_mode = self._recording_mode()
        # A lineage-bound certificate is not release evidence while its
        # recording transaction is in progress.  The status becomes verified
        # only after the final certificate and metrics commit succeeds.
        if status == "verified" and getattr(self, "apply_lineage", False):
            status = "pending"
        if getattr(self, "apply_lineage", False):
            recording_state = "pending" if status == "pending" else "failed"
        else:
            recording_state = "not_requested"
        candidate_identity = dict(getattr(self, "candidate_identity", {}) or {})
        attempt_id = str(
            getattr(self, "certification_attempt_id", "not_recorded")
        )
        return {
            "certificate_type": "source_frozen_notebook_to_production_parity",
            "certificate_version": "1.0.0",
            "status": status,
            "verified_at_utc": datetime.now(timezone.utc).isoformat(),
            "candidate_artifact_dir": str(self.candidate_artifact_dir),
            "candidate_identity": candidate_identity,
            "certification_attempt_id": attempt_id,
            "candidate_run_id": self.candidate_lineage.get("mlflow_run_id")
            or self.candidate_metrics_payload.get("mlflow_run_id"),
            "reference_run_id": self.reference_run_id,
            "reference_model_id": getattr(self, "configured_reference_model_id", ""),
            "reference_model_uri": self.reference_model_uri,
            "source_data_path": str(self.source_data_path),
            "source_manifest_path": str(
                getattr(
                    self,
                    "source_manifest_path",
                    self.source_data_path.with_name("source_manifest.json"),
                )
            ),
            "source_manifest": self.candidate_lineage.get("source_manifest")
            or getattr(self, "source_manifest", {}),
            "source_split_method": self.checks.get("reference_split_source"),
            "tolerances": {
                "prediction_max_absolute_error": self.prediction_tolerance,
                "metric_max_absolute_error": self.metric_tolerance,
            },
            "checks": parity_checks,
            "warnings": self.warnings,
            "errors": self.errors,
            "lineage_recording": {
                "requested": getattr(self, "apply_lineage", False),
                "recording_mode": recording_mode,
                "recording_state": recording_state,
                "certification_attempt_id": attempt_id,
                "candidate_identity": candidate_identity,
                "certificate_report_path": (
                    self._certificate_report_relative_path()
                    if getattr(self, "apply_lineage", False)
                    else None
                ),
                "local_metrics_json_updated": False,
                "mlflow_evidence_logged": False,
                "mlflow_tags_recorded": False,
                "required_steps": {
                    "certificate_report": getattr(self, "apply_lineage", False),
                    "local_metrics_json": getattr(self, "apply_lineage", False),
                    "mlflow_evidence": getattr(self, "log_mlflow_evidence", False),
                    "mlflow_tags": getattr(self, "log_mlflow_evidence", False),
                },
            },
        }

    def _write_report(self, report: dict[str, Any]) -> None:
        target, protected_paths = self._report_destination_policy(
            claim_destination=True
        )
        write_json_atomic(
            target,
            report,
            protected_paths=protected_paths,
        )
        logger.info("Source-frozen parity report saved to %s", self.report_path)

    def _apply_local_lineage(
        self,
        report: dict[str, Any],
        recording_state: str = "pending",
    ) -> None:
        """Persist pending or complete candidate-bound parity evidence.

        ``verified`` is written only during the final commit after the
        certificate report and all requested remote evidence have succeeded.
        The pending write intentionally supersedes older evidence so replay
        cannot accept a partially recorded newer attempt.
        """

        if recording_state not in {"pending", "complete"}:
            raise ValueError("Parity lineage recording state must be pending or complete")
        metrics_path = self.candidate_artifact_dir / "model_trainer" / "metrics.json"
        if not metrics_path.is_file():
            raise FileNotFoundError(f"Candidate metrics file not found: {metrics_path}")
        with metrics_path.open(encoding="utf-8") as file:
            payload = json.load(file)
        lineage = payload.get("lineage")
        if not isinstance(lineage, dict):
            raise ValueError("Candidate metrics file has no lineage object")

        recording = report.get("lineage_recording")
        if not isinstance(recording, dict):
            raise ValueError("Parity report has no lineage_recording object")
        certificate_report_path = self._certificate_report_relative_path()
        certificate_report_sha256 = (
            self._sha256_file(self.report_path)
            if recording_state == "complete" and self.report_path.is_file()
            else None
        )
        candidate_identity = dict(
            recording.get("candidate_identity")
            or getattr(self, "candidate_identity", {})
            or {}
        )
        current_attempt_id = str(
            recording.get(
                "certification_attempt_id",
                getattr(self, "certification_attempt_id", "not_recorded"),
            )
        )
        previous_evidence = lineage.get("source_frozen_parity")
        if recording_state == "complete":
            if not isinstance(previous_evidence, dict) or str(
                previous_evidence.get("certification_attempt_id", "")
            ) != current_attempt_id:
                raise ValueError(
                    "Parity completion attempt was superseded before the final commit"
                )
            if previous_evidence.get("candidate_identity") != candidate_identity:
                raise ValueError(
                    "Parity completion candidate identity changed before the final commit"
                )
        evidence = {
            "status": "verified" if recording_state == "complete" else "pending",
            "recording_state": recording_state,
            "recording_mode": recording.get(
                "recording_mode", self._recording_mode()
            ),
            "certification_attempt_id": current_attempt_id,
            "candidate_identity": candidate_identity,
            "split_id_sets_match": report["checks"]["split_ids"][
                "split_id_sets_match"
            ],
            "prediction_order_match": report["checks"]["split_ids"][
                "prediction_order_match"
            ],
            "maximum_absolute_prediction_error": report["checks"][
                "prediction_array"
            ]["maximum_absolute_error"],
            "prediction_tolerance": self.prediction_tolerance,
            "transformed_feature_order_match": report["checks"][
                "transformed_feature_order"
            ]["passed"],
            "transformed_values_maximum_absolute_error": report["checks"][
                "transformed_values"
            ]["maximum_absolute_error"],
            "reference_run_id": self.reference_run_id,
            "reference_model_id": getattr(self, "configured_reference_model_id", ""),
            "reference_model_uri": self.reference_model_uri,
            "certificate_report_path": certificate_report_path,
            "certificate_report_sha256": certificate_report_sha256,
            "verified_at_utc": report["verified_at_utc"],
            "recording_steps": {
                "certificate_report_written": (
                    recording_state == "complete" and self.report_path.is_file()
                ),
                "local_metrics_json_committed": recording_state == "complete",
                "mlflow_evidence_logged": bool(
                    recording.get("mlflow_evidence_logged")
                ),
                "mlflow_tags_recorded": bool(
                    recording.get("mlflow_tags_recorded")
                ),
            },
        }
        lineage["source_frozen_parity"] = evidence
        payload["lineage"] = lineage
        metrics_target, protected_paths = self._metrics_destination_policy()
        write_json_atomic(
            metrics_target,
            payload,
            protected_paths=protected_paths,
        )
        report["lineage_recording"]["local_metrics_json_updated"] = True
        logger.info("Local candidate lineage updated at %s", metrics_path)

    def _log_mlflow_evidence(
        self,
        report: dict[str, Any],
        *,
        finalize: bool = False,
    ) -> None:
        candidate_run_id = str(report.get("candidate_run_id") or "")
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if not candidate_run_id:
            raise ValueError("Candidate run ID is unavailable for MLflow evidence logging")
        if not tracking_uri:
            raise ValueError("MLFLOW_TRACKING_URI is not set for evidence logging")
        mlflow.set_tracking_uri(tracking_uri)

        # The status is part of the persisted certificate. Prepare the exact
        # remote payload before opening the MLflow run, while keeping the
        # caller's report false until the artifact and every tag succeed.
        mlflow_report = deepcopy(report)
        remote_recording = mlflow_report["lineage_recording"]
        remote_recording.setdefault(
            "certification_attempt_id",
            str(getattr(self, "certification_attempt_id", "")),
        )
        remote_recording.setdefault(
            "candidate_identity",
            dict(getattr(self, "candidate_identity", {}) or {}),
        )
        remote_recording.setdefault("recording_mode", self._recording_mode())
        remote_recording["recording_state"] = (
            "complete" if finalize else "remote_recorded_pending_local_commit"
        )
        # The first artifact copy describes the state before tags are
        # attempted. A second copy after the tag loop records the complete
        # remote state. This makes a log-success/tag-failure partial state
        # explicit instead of falsely claiming tag completion.
        remote_recording["mlflow_evidence_logged"] = False
        remote_recording["mlflow_tags_recorded"] = False
        mlflow_report["status"] = "verified" if finalize else "pending"
        with mlflow.start_run(run_id=candidate_run_id):
            mlflow.log_dict(mlflow_report, "lineage/source_frozen_parity.json")
            tags = {
                "source_frozen_parity": "verified" if finalize else "pending",
                "source_frozen_parity_certificate_version": "1.0.0",
                "source_frozen_parity_certificate_attempt_id": str(
                    remote_recording.get("certification_attempt_id", "")
                ),
                "source_frozen_parity_candidate_model_id": str(
                    (
                        remote_recording.get("candidate_identity") or {}
                    ).get("model_id", "")
                ),
                "source_frozen_parity_recording_mode": str(
                    remote_recording.get("recording_mode", "")
                ),
            }
            for key, value in tags.items():
                mlflow.set_tag(key, value)
            remote_recording["mlflow_evidence_logged"] = True
            remote_recording["mlflow_tags_recorded"] = True
            mlflow.log_dict(mlflow_report, "lineage/source_frozen_parity.json")
        report["lineage_recording"]["mlflow_evidence_logged"] = True
        report["lineage_recording"]["mlflow_tags_recorded"] = True
        logger.info("Source-frozen parity evidence logged to MLflow run %s", candidate_run_id)

    def run(self) -> dict[str, Any]:
        """Run certification and return the machine-readable report."""
        load_dotenv(override=False)
        # Reject a dangerous destination before reading candidate inputs or
        # contacting MLflow. Later writes revalidate after lineage discovery.
        self._validate_report_destination()
        self._run_stage("configuration", self._validate_configuration)
        if not self.errors:
            self._run_stage("candidate_artifact_loading", self._load_candidate_artifacts)
        if not self.errors:
            self._run_stage("notebook_split_reconstruction", self._reconstruct_notebook_splits)
        if not self.errors:
            self._run_stage("split_id_and_target_comparison", self._compare_split_ids_and_targets)
        if not self.errors:
            self._run_stage("reference_model_loading", self._load_reference_model)
        if not self.errors:
            self._run_stage("feature_and_prediction_comparison", self._compare_models_and_predictions)
        if not self.errors:
            self._run_stage("metric_comparison", self._compare_metrics)

        report = self._build_report()

        # A lineage-bound run is a small durable transaction.  The pending
        # metrics evidence is written before the certificate and any remote
        # call so a certificate-write failure cannot leave an older verified
        # metrics entry authoritative. Only the final certificate write,
        # requested remote evidence, and complete metrics commit expose
        # ``verified`` evidence. Any interruption between these steps leaves a
        # persisted state that the evaluator rejects.
        if report["status"] == "pending" and getattr(self, "apply_lineage", False):
            self._run_stage(
                "local_lineage_pending_recording",
                lambda: self._apply_local_lineage(report, "pending"),
            )
            if not self.errors:
                self._run_stage(
                    "pending_certificate_write",
                    lambda: self._write_report(report),
                )
            if not self.errors and getattr(self, "log_mlflow_evidence", False):
                self._run_stage(
                    "mlflow_evidence_recording",
                    lambda: self._log_mlflow_evidence(report),
                )

            if not self.errors:
                recording = report["lineage_recording"]
                report["status"] = "verified"
                recording["recording_state"] = "complete"
                self._run_stage(
                    "final_certificate_write",
                    lambda: self._write_report(report),
                )
            if not self.errors and getattr(self, "log_mlflow_evidence", False):
                self._run_stage(
                    "mlflow_evidence_finalization",
                    lambda: self._log_mlflow_evidence(report, finalize=True),
                )
            if not self.errors:
                self._run_stage(
                    "local_lineage_complete_commit",
                    lambda: self._apply_local_lineage(report, "complete"),
                )

            if self.errors:
                # Best effort only: rejection must come from the durable
                # pending/failed evidence and certificate validation, not from
                # successfully writing a failure marker after storage failed.
                report["errors"] = self.errors
                report["status"] = "failed"
                report["lineage_recording"]["recording_state"] = "failed"
                try:
                    self._write_report(report)
                except Exception as error:
                    self._record_error("failure_certificate_write", error)
            self._report = report
            return report

        if getattr(self, "log_mlflow_evidence", False):
            self._record_error(
                "mlflow_evidence_recording",
                ValueError(
                    "--log-mlflow-evidence requires --apply-lineage and a "
                    "numerically verified report"
                ),
            )
            report["errors"] = self.errors
            report["status"] = "failed"

        self._report = report
        try:
            self._write_report(report)
        except Exception as error:
            self._record_error("report_writing", error)
            raise VehicleInsuranceException(error, sys) from error
        return report


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Certify a production candidate against the frozen notebook model."
    )
    parser.add_argument(
        "--candidate-artifact-dir",
        required=True,
        help="Timestamped artifact directory containing ingestion and model_trainer outputs.",
    )
    parser.add_argument(
        "--reference-run-id",
        default=None,
        help="Historical notebook final MLflow run; defaults to model_spec.yaml.",
    )
    parser.add_argument(
        "--reference-model-uri",
        default=None,
        help="Reference model URI; defaults to runs:/<reference-run-id>/model.",
    )
    parser.add_argument(
        "--source-data-path",
        default=None,
        help="Full frozen source CSV; defaults to the candidate feature-store data.csv.",
    )
    parser.add_argument(
        "--report-path",
        default=None,
        help="Parity report path; defaults to source_frozen_parity/parity_report.json.",
    )
    parser.add_argument(
        "--prediction-tolerance",
        type=float,
        default=1.0e-12,
        help="Maximum absolute prediction/transformation error (default: 1e-12).",
    )
    parser.add_argument(
        "--metric-tolerance",
        type=float,
        default=1.0e-6,
        help="Maximum absolute metric error (default: 1e-6).",
    )
    parser.add_argument(
        "--apply-lineage",
        action="store_true",
        help="After verification, update the candidate model_trainer/metrics.json lineage.",
    )
    parser.add_argument(
        "--log-mlflow-evidence",
        action="store_true",
        help="After verification, log the certificate to the candidate MLflow run.",
    )
    return parser


def main() -> int:
    args = _build_argument_parser().parse_args()
    try:
        verifier = SourceFrozenParityCertification(
            candidate_artifact_dir=args.candidate_artifact_dir,
            reference_run_id=args.reference_run_id,
            reference_model_uri=args.reference_model_uri,
            source_data_path=args.source_data_path,
            report_path=args.report_path,
            prediction_tolerance=args.prediction_tolerance,
            metric_tolerance=args.metric_tolerance,
            apply_lineage=args.apply_lineage,
            log_mlflow_evidence=args.log_mlflow_evidence,
        )
        report = verifier.run()
        print(json.dumps(report, indent=2, default=str))
        return 0 if report["status"] == "verified" else 1
    except Exception as error:
        logger.exception("Source-frozen parity certification failed unexpectedly")
        try:
            raise VehicleInsuranceException(error, sys) from error
        except VehicleInsuranceException as wrapped_error:
            print(str(wrapped_error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
