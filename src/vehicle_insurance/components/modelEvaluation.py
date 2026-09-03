from __future__ import annotations

import json
import math
import os
import sys
from typing import Any

from mlflow.exceptions import MlflowException

from vehicle_insurance.constants import (
    MLFLOW_CHAMPION_ALIAS,
    MLFLOW_REGISTER_MODEL_NAME,
    MODEL_SPEC_FILE_PATH,
    SCHEMA_FILE_PATH,
)
from vehicle_insurance.entities.artifact_entity import (
    ModelEvaluationArtifact,
    ModelTrainerArtifact,
)
from vehicle_insurance.entities.config_entity import ModelEvaluationConfig
from vehicle_insurance.exceptions.handle_exception import VehicleInsuranceException
from vehicle_insurance.logger.logger_config import setup_logger
from vehicle_insurance.utils import get_mlflow_client, read_yaml_file
from vehicle_insurance.utils.source_manifest import validate_source_manifest

logger = setup_logger()


class ModelEvaluation:
    """Evaluate a trainer candidate against the frozen release policy.

    Candidate acceptance uses validation metrics and repeated-CV stability.
    Final-test metrics are copied into the report for transparency only and
    never affect the decision. MLflow is read-only while promotion is disabled
    in the model specification.
    """

    _REQUIRED_LINEAGE_FIELDS = (
        "contract_version",
        "model_spec_version",
        "git_provenance",
        "git_branch",
        "git_commit",
        "python_version",
        "dependency_lock_hash",
        "lightgbm_version",
        "scikit_learn_version",
        "mlflow_version",
        "schema_version",
        "dataset_snapshot_id",
        "dataset_content_hash",
        "source_manifest",
        "final_transformed_feature_count",
        "final_transformed_feature_names",
        "model_input_feature_order",
        "candidate_fit_data",
        "final_fit_data",
        "test_evaluation_count",
    )

    def __init__(
        self,
        model_evaluation_config: ModelEvaluationConfig,
        model_trainer_artifact: ModelTrainerArtifact,
        evaluation_context: dict[str, Any] | None = None,
        require_champion_comparison: bool | None = None,
    ) -> None:
        try:
            self.model_trainer_artifact = model_trainer_artifact
            self.model_evaluation_config = model_evaluation_config
            if evaluation_context is not None and not isinstance(
                evaluation_context, dict
            ):
                raise TypeError("evaluation_context must be a dictionary when provided")
            self.evaluation_context = dict(evaluation_context or {"mode": "pipeline"})
            self.model_spec = read_yaml_file(MODEL_SPEC_FILE_PATH)
            self.schema_config = read_yaml_file(SCHEMA_FILE_PATH)
            self._validate_policy()
            configured_comparison = self.model_spec["evaluation_policy"][
                "champion_challenger_workflow"
            ]["enabled"]
            if require_champion_comparison is None:
                self.require_champion_comparison = configured_comparison
            else:
                if not isinstance(require_champion_comparison, bool):
                    raise TypeError("require_champion_comparison must be boolean")
                self.require_champion_comparison = require_champion_comparison
        except Exception as error:
            raise VehicleInsuranceException(error, sys) from error

    def _validate_policy(self) -> None:
        if self.model_spec.get("contract_version") != str(
            self.schema_config.get("version")
        ):
            raise ValueError(
                "Model specification and schema contract versions do not match"
            )

        policy = self.model_spec.get("evaluation_policy") or {}
        thresholds = policy.get("acceptance_thresholds") or {}
        required_threshold_groups = (
            "candidate_validation_minimums",
            "stability_mean_minimums",
            "stability_maximum_std",
            "champion_comparison",
        )
        missing_groups = [
            group for group in required_threshold_groups if group not in thresholds
        ]
        if missing_groups:
            raise ValueError(
                f"Evaluation policy is missing threshold groups: {missing_groups}"
            )

        workflow_config = policy.get("champion_challenger_workflow") or {}
        if not isinstance(workflow_config.get("enabled"), bool):
            raise ValueError(
                "evaluation_policy.champion_challenger_workflow.enabled must be boolean"
            )

        promotion_enabled = policy.get("promotion", {}).get("enabled")
        if not isinstance(promotion_enabled, bool):
            raise ValueError("evaluation_policy.promotion.enabled must be boolean")

        git_policy = policy.get("lineage", {}).get("git")
        if not isinstance(git_policy, dict):
            raise ValueError("evaluation_policy.lineage.git must be a dictionary")
        invalid_git_policy_fields = [
            field
            for field in (
                "require_clean_tree",
                "fail_if_unavailable",
                "fail_on_untracked_files",
            )
            if not isinstance(git_policy.get(field), bool)
        ]
        if invalid_git_policy_fields:
            raise ValueError(
                "Git provenance policy fields must be boolean: "
                f"{invalid_git_policy_fields}"
            )

        source_snapshot_policy = policy.get("lineage", {}).get("source_snapshot")
        if not isinstance(source_snapshot_policy, dict) or any(
            source_snapshot_policy.get(field) is not True
            for field in (
                "require_manifest",
                "require_immutable_snapshot",
                "require_snapshot_read_boundary",
                "require_explicit_projection",
                "require_indexed_ordering",
            )
        ):
            raise ValueError(
                "evaluation_policy.lineage.source_snapshot must require all "
                "source provenance controls"
            )

    @staticmethod
    def _finite_float(value: Any) -> float | None:
        if isinstance(value, bool):
            return None
        try:
            converted = float(value)
        except (TypeError, ValueError):
            return None
        return converted if math.isfinite(converted) else None

    def _required_metric_names(self) -> list[str]:
        policy = self.model_spec["evaluation_policy"]
        names = [
            *policy.get("primary_metrics", []),
            *policy.get("supporting_metrics", []),
        ]
        return list(dict.fromkeys(names))

    def _check_validation_metrics(self, metrics: Any) -> dict:
        thresholds = self.model_spec["evaluation_policy"]["acceptance_thresholds"]
        minimums = thresholds["candidate_validation_minimums"]
        metric_values = metrics if isinstance(metrics, dict) else {}
        checks: dict[str, dict] = {}
        reason_codes: list[str] = []

        for metric_name in self._required_metric_names():
            value = self._finite_float(metric_values.get(metric_name))
            minimum = self._finite_float(minimums.get(metric_name))
            passed = value is not None and minimum is not None and value >= minimum
            checks[metric_name] = {
                "observed": value,
                "minimum": minimum,
                "passed": passed,
            }
            if not passed:
                reason_codes.append(f"validation_metric_below_minimum:{metric_name}")

        return {
            "passed": not reason_codes,
            "checks": checks,
            "reason_codes": reason_codes,
        }

    def _check_stability_metrics(self, stability_metrics: Any) -> dict:
        policy = self.model_spec["evaluation_policy"]
        stability_policy = policy["stability"]
        thresholds = policy["acceptance_thresholds"]
        if not stability_policy.get("required", False):
            return {
                "passed": True,
                "required": False,
                "reason_codes": [],
            }

        stability = stability_metrics if isinstance(stability_metrics, dict) else {}
        reason_codes: list[str] = []
        expected_folds = int(stability_policy["folds"])
        expected_repeats = int(stability_policy["repeats"])
        expected_count = expected_folds * expected_repeats
        actual_count = stability.get("evaluation_count")
        if (
            stability.get("required") is not True
            or actual_count != expected_count
        ):
            reason_codes.append("stability_evaluation_count_mismatch")

        summary = stability.get("metrics")
        if not isinstance(summary, dict):
            summary = {}
            reason_codes.append("stability_metrics_missing")

        mean_minimums = thresholds["stability_mean_minimums"]
        maximum_stds = thresholds["stability_maximum_std"]
        checks: dict[str, dict] = {}
        for metric_name in stability_policy.get("selection_metrics", []):
            metric_summary = summary.get(metric_name)
            if not isinstance(metric_summary, dict):
                checks[metric_name] = {"passed": False}
                reason_codes.append(f"stability_metric_missing:{metric_name}")
                continue

            mean = self._finite_float(metric_summary.get("mean"))
            std = self._finite_float(metric_summary.get("std"))
            minimum = self._finite_float(mean_minimums.get(metric_name))
            maximum_std = self._finite_float(maximum_stds.get(metric_name))
            passed = (
                mean is not None
                and std is not None
                and minimum is not None
                and maximum_std is not None
                and mean >= minimum
                and std <= maximum_std
            )
            checks[metric_name] = {
                "mean": mean,
                "minimum_mean": minimum,
                "std": std,
                "maximum_std": maximum_std,
                "passed": passed,
            }
            if not passed:
                reason_codes.append(f"stability_gate_failed:{metric_name}")

        return {
            "passed": not reason_codes,
            "required": True,
            "expected_folds": expected_folds,
            "expected_repeats": expected_repeats,
            "expected_evaluation_count": expected_count,
            "actual_evaluation_count": actual_count,
            "checks": checks,
            "reason_codes": reason_codes,
        }

    def _candidate_lineage(self) -> dict:
        lineage = getattr(self.model_trainer_artifact, "lineage", None)
        if isinstance(lineage, dict) and lineage:
            return dict(lineage)

        # Compatibility fallback for artifacts created before lineage became a
        # first-class ModelTrainerArtifact field. New trainer runs populate the
        # artifact directly; this fallback is read-only.
        metrics_path = getattr(self.model_trainer_artifact, "metrics_file_path", "")
        if metrics_path and os.path.isfile(metrics_path):
            try:
                with open(metrics_path, encoding="utf-8") as metrics_file:
                    metrics_payload = json.load(metrics_file)
                lineage = metrics_payload.get("lineage", {})
                if isinstance(lineage, dict):
                    return lineage
            except (OSError, json.JSONDecodeError) as error:
                logger.warning("Could not load trainer lineage from metrics: %s", error)
        return {}

    def _check_lineage(self, lineage: dict) -> dict:
        expected_contract_version = str(self.schema_config["version"])
        expected_model_spec_version = str(self.model_spec["version"])
        missing_fields = [
            field
            for field in self._REQUIRED_LINEAGE_FIELDS
            if field not in lineage or lineage[field] in (None, "", "not_recorded")
        ]
        reason_codes = [f"lineage_field_missing:{field}" for field in missing_fields]

        if lineage.get("contract_version") != expected_contract_version:
            reason_codes.append("contract_version_mismatch")
        if lineage.get("schema_version") != expected_contract_version:
            reason_codes.append("schema_version_mismatch")
        if lineage.get("model_spec_version") != expected_model_spec_version:
            reason_codes.append("model_spec_version_mismatch")

        git_provenance = lineage.get("git_provenance")
        if not isinstance(git_provenance, dict):
            reason_codes.append("git_provenance_invalid")
        else:
            required_git_fields = (
                "repository_root",
                "branch",
                "commit",
                "is_dirty",
                "status_sha256",
                "patch_sha256",
                "model_spec_sha256",
                "schema_sha256",
            )
            for field in required_git_fields:
                if field not in git_provenance or git_provenance[field] in (
                    None,
                    "",
                ):
                    reason_codes.append(f"git_provenance_field_missing:{field}")

            if not isinstance(git_provenance.get("is_dirty"), bool):
                reason_codes.append("git_provenance_dirty_state_invalid")

            for field in (
                "status_sha256",
                "patch_sha256",
                "model_spec_sha256",
                "schema_sha256",
            ):
                value = git_provenance.get(field)
                if not isinstance(value, str) or len(value) != 64 or any(
                    character not in "0123456789abcdef"
                    for character in value.lower()
                ):
                    reason_codes.append(f"git_provenance_hash_invalid:{field}")

            git_policy = self.model_spec["evaluation_policy"]["lineage"]["git"]
            untracked_files = git_provenance.get("untracked_files") or []
            if git_policy["fail_on_untracked_files"] and untracked_files:
                reason_codes.append("git_untracked_files_present")
            if git_policy["require_clean_tree"] and git_provenance.get(
                "is_dirty"
            ):
                reason_codes.append("git_worktree_dirty")

        content_hash = str(lineage.get("dataset_content_hash", ""))
        snapshot_id = str(lineage.get("dataset_snapshot_id", ""))
        if len(content_hash) != 64 or any(
            character not in "0123456789abcdef" for character in content_hash.lower()
        ):
            reason_codes.append("dataset_content_hash_invalid")
        source_manifest = lineage.get("source_manifest")
        if not isinstance(source_manifest, dict):
            reason_codes.append("source_manifest_invalid")
        else:
            try:
                validate_source_manifest(source_manifest)
            except ValueError as error:
                reason_codes.append(f"source_manifest_invalid:{error}")
            source_contract = self.schema_config.get("source", {})
            source_identity = source_manifest.get("source", {})
            for field in ("database", "collection"):
                if source_identity.get(field) != source_contract.get(field):
                    reason_codes.append(f"source_manifest_{field}_mismatch")
            expected_projection = list(
                self.schema_config["schemas"]["training"]["required_columns"]
            )
            if source_manifest.get("projection", {}).get("fields") != expected_projection:
                reason_codes.append("source_manifest_projection_mismatch")
            expected_filter = self.model_spec["data_protocol"]["source_snapshot"].get(
                "filter", {}
            )
            if source_manifest.get("query", {}).get("filter") != expected_filter:
                reason_codes.append("source_manifest_filter_mismatch")
            if source_manifest.get("snapshot_id") != snapshot_id:
                reason_codes.append("dataset_snapshot_hash_mismatch")
            if source_manifest.get("content_hash") != content_hash:
                reason_codes.append("dataset_content_hash_manifest_mismatch")

        expected_final_count = int(
            self.model_spec["features"]["preprocessing"][
                "final_transformed_feature_count"
            ]
        )
        feature_names = lineage.get("final_transformed_feature_names")
        model_input_order = lineage.get("model_input_feature_order")
        final_count = lineage.get("final_transformed_feature_count")
        if final_count != expected_final_count:
            reason_codes.append("final_transformed_feature_count_mismatch")
        if not isinstance(feature_names, list) or not feature_names:
            reason_codes.append("final_transformed_feature_order_missing")
        elif len(feature_names) != expected_final_count or len(set(feature_names)) != len(
            feature_names
        ):
            reason_codes.append("final_transformed_feature_order_invalid")
        if model_input_order != feature_names:
            reason_codes.append("model_input_feature_order_mismatch")

        if lineage.get("candidate_fit_data") != "train_only":
            reason_codes.append("candidate_fit_scope_invalid")
        if lineage.get("final_fit_data") != "train_plus_validation":
            reason_codes.append("final_fit_scope_invalid")
        if lineage.get("test_evaluation_count") != 1:
            reason_codes.append("test_evaluation_count_invalid")

        return {
            "passed": not reason_codes,
            "required_fields": list(self._REQUIRED_LINEAGE_FIELDS),
            "expected_contract_version": expected_contract_version,
            "expected_model_spec_version": expected_model_spec_version,
            "observed_contract_version": lineage.get("contract_version"),
            "observed_model_spec_version": lineage.get("model_spec_version"),
            "observed_dataset_snapshot_id": lineage.get("dataset_snapshot_id"),
            "observed_source_snapshot_id": (
                source_manifest.get("snapshot_id")
                if isinstance(source_manifest, dict)
                else None
            ),
            "observed_final_transformed_feature_count": final_count,
            "reason_codes": reason_codes,
        }

    def _check_source_frozen_parity(self, lineage: dict) -> dict:
        parity_policy = self.model_spec["evaluation_policy"]["parity"]
        if not parity_policy.get("source_frozen_prediction_parity_required", False):
            return {
                "passed": True,
                "required": False,
                "reason_codes": [],
            }

        evidence = lineage.get("source_frozen_parity")
        if not isinstance(evidence, dict):
            return {
                "passed": False,
                "required": True,
                "status": "missing",
                "reason_codes": ["source_frozen_parity_evidence_missing"],
            }

        error = self._finite_float(evidence.get("maximum_absolute_prediction_error"))
        allowed_error = self._finite_float(
            parity_policy.get("maximum_absolute_prediction_error")
        )
        checks = {
            "status_verified": evidence.get("status") == "verified",
            "split_id_sets_match": evidence.get("split_id_sets_match") is True,
            "prediction_order_match": evidence.get("prediction_order_match") is True,
            "maximum_absolute_prediction_error": {
                "observed": error,
                "maximum_allowed": allowed_error,
                "passed": (
                    error is not None
                    and allowed_error is not None
                    and error <= allowed_error
                ),
            },
        }
        reason_codes: list[str] = []
        if not checks["status_verified"]:
            reason_codes.append("source_frozen_parity_not_verified")
        if not checks["split_id_sets_match"]:
            reason_codes.append("source_frozen_split_id_parity_failed")
        if not checks["prediction_order_match"]:
            reason_codes.append("source_frozen_prediction_order_parity_failed")
        if not checks["maximum_absolute_prediction_error"]["passed"]:
            reason_codes.append("source_frozen_prediction_error_exceeded")

        return {
            "passed": not reason_codes,
            "required": True,
            "status": evidence.get("status"),
            "checks": checks,
            "reason_codes": reason_codes,
        }

    def _get_champion_info(self, client) -> dict:
        """Read valid Champion validation evidence without mutating MLflow."""
        try:
            champion_version_obj = client.get_model_version_by_alias(
                MLFLOW_REGISTER_MODEL_NAME,
                MLFLOW_CHAMPION_ALIAS,
            )
            champion_run_id = str(champion_version_obj.run_id)
            run = client.get_run(champion_run_id)
            run_metrics = dict(run.data.metrics)
            required_metrics = self._required_metric_names()
            validation_metrics = {
                metric_name: run_metrics.get(f"validation_{metric_name}")
                for metric_name in required_metrics
            }
            missing_metrics = [
                metric_name
                for metric_name, value in validation_metrics.items()
                if self._finite_float(value) is None
            ]
            if missing_metrics:
                logger.warning(
                    "Champion alias exists but lacks valid validation metrics: %s",
                    missing_metrics,
                )
                return {
                    "status": "invalid",
                    "reason_codes": [
                        f"champion_validation_metric_missing:{metric}"
                        for metric in missing_metrics
                    ],
                    "run_id": champion_run_id,
                    "model_version": str(champion_version_obj.version),
                }

            tags = dict(run.data.tags)
            champion_lineage = {
                "contract_version": tags.get("contract_version"),
                "model_spec_version": tags.get("model_spec_version"),
                "dataset_snapshot_id": tags.get("dataset_snapshot_id"),
                "final_transformed_feature_count": tags.get(
                    "final_transformed_feature_count"
                ),
            }
            download_artifacts = getattr(client, "download_artifacts", None)
            if callable(download_artifacts):
                try:
                    lineage_path = download_artifacts(
                        champion_run_id,
                        "lineage/lineage.json",
                    )
                    with open(lineage_path, encoding="utf-8") as lineage_file:
                        downloaded_lineage = json.load(lineage_file)
                    if isinstance(downloaded_lineage, dict):
                        champion_lineage.update(downloaded_lineage)
                except (OSError, json.JSONDecodeError, MlflowException) as error:
                    logger.warning(
                        "Champion lineage artifact could not be loaded: %s",
                        error,
                    )
            logger.info(
                "Valid Champion validation evidence found: version=%s run_id=%s",
                champion_version_obj.version,
                champion_run_id,
            )
            return {
                "status": "valid",
                "run_id": champion_run_id,
                "model_version": str(champion_version_obj.version),
                "validation_metrics": validation_metrics,
                "lineage": champion_lineage,
                "reason_codes": [],
            }
        except MlflowException as error:
            error_code = getattr(error, "error_code", None)
            if error_code in {"RESOURCE_DOES_NOT_EXIST", "NOT_FOUND"}:
                logger.info("No registered Champion alias was found in MLflow")
                return {
                    "status": "not_found",
                    "validation_metrics": None,
                    "reason_codes": ["champion_not_found"],
                }
            logger.exception("MLflow failed while reading the Champion alias")
            return {
                "status": "unavailable",
                "validation_metrics": None,
                "reason_codes": [f"champion_lookup_failed:{error_code}"],
            }
        except Exception as error:
            logger.exception("Unable to read the registered Champion evidence")
            return {
                "status": "unavailable",
                "validation_metrics": None,
                "reason_codes": [f"champion_lookup_failed:{type(error).__name__}"],
            }

    def _load_champion_info(self) -> dict:
        try:
            client = get_mlflow_client()
        except Exception as error:
            logger.warning("Champion lookup unavailable: %s", error)
            return {
                "status": "unavailable",
                "validation_metrics": None,
                "reason_codes": [f"champion_lookup_failed:{type(error).__name__}"],
            }
        return self._get_champion_info(client)

    def _check_champion_comparison(
        self,
        candidate_metrics: dict,
        champion_info: dict,
    ) -> dict:
        status = champion_info.get("status")
        if status == "not_found":
            return {
                "passed": True,
                "status": status,
                "comparison_performed": False,
                "reason_codes": ["no_valid_champion_for_comparison"],
            }
        if status != "valid":
            return {
                "passed": False,
                "status": status,
                "comparison_performed": False,
                "reason_codes": champion_info.get("reason_codes")
                or ["champion_evidence_unavailable"],
            }

        comparison_policy = self.model_spec["evaluation_policy"][
            "acceptance_thresholds"
        ]["champion_comparison"]
        tolerance = comparison_policy["non_inferiority_tolerance"]
        champion_validation = champion_info["validation_metrics"]
        metric_checks: dict[str, dict] = {}
        reason_codes: list[str] = []

        candidate_lineage = self._candidate_lineage()
        champion_lineage = champion_info.get("lineage") or {}
        lineage_checks: dict[str, dict] = {}
        for field_name in (
            "contract_version",
            "model_spec_version",
            "dataset_snapshot_id",
            "final_transformed_feature_count",
            "final_transformed_feature_names",
            "model_input_feature_order",
        ):
            candidate_value = candidate_lineage.get(field_name)
            champion_value = champion_lineage.get(field_name)
            passed = candidate_value == champion_value and champion_value not in (
                None,
                "",
                "not_recorded",
            )
            lineage_checks[field_name] = {
                "candidate": candidate_value,
                "champion": champion_value,
                "passed": passed,
            }
            if not passed:
                reason_codes.append(f"champion_lineage_mismatch:{field_name}")

        for metric_name, allowed_drop in tolerance.items():
            candidate_value = self._finite_float(candidate_metrics.get(metric_name))
            champion_value = self._finite_float(champion_validation.get(metric_name))
            tolerance_value = self._finite_float(allowed_drop)
            delta = (
                candidate_value - champion_value
                if candidate_value is not None and champion_value is not None
                else None
            )
            passed = (
                delta is not None
                and tolerance_value is not None
                and delta >= -tolerance_value
            )
            metric_checks[metric_name] = {
                "candidate": candidate_value,
                "champion": champion_value,
                "delta": delta,
                "maximum_allowed_drop": tolerance_value,
                "passed": passed,
            }
            if not passed:
                reason_codes.append(f"champion_non_inferiority_failed:{metric_name}")

        improvement_checks: dict[str, dict] = {}
        improved_primary_metrics: list[str] = []
        for metric_name, required_improvement in comparison_policy[
            "minimum_primary_improvement"
        ].items():
            candidate_value = self._finite_float(candidate_metrics.get(metric_name))
            champion_value = self._finite_float(champion_validation.get(metric_name))
            minimum_improvement = self._finite_float(required_improvement)
            delta = (
                candidate_value - champion_value
                if candidate_value is not None and champion_value is not None
                else None
            )
            passed = (
                delta is not None
                and minimum_improvement is not None
                and delta >= minimum_improvement
            )
            if passed:
                improved_primary_metrics.append(metric_name)
            improvement_checks[metric_name] = {
                "candidate": candidate_value,
                "champion": champion_value,
                "delta": delta,
                "minimum_improvement": minimum_improvement,
                "passed": passed,
            }

        improvement_required = comparison_policy.get(
            "primary_improvement_required", True
        )
        if improvement_required and not improved_primary_metrics:
            reason_codes.append("no_primary_metric_practical_improvement")

        return {
            "passed": not reason_codes,
            "status": status,
            "comparison_performed": True,
            "lineage_checks": lineage_checks,
            "non_inferiority_checks": metric_checks,
            "primary_improvement_checks": improvement_checks,
            "improved_primary_metrics": improved_primary_metrics,
            "reason_codes": reason_codes,
        }

    def _save_report(self, report: dict) -> None:
        try:
            report_directory = os.path.dirname(
                self.model_evaluation_config.report_file_path
            ) or "."
            os.makedirs(report_directory, exist_ok=True)
            with open(
                self.model_evaluation_config.report_file_path,
                "w",
                encoding="utf-8",
            ) as report_file:
                json.dump(report, report_file, indent=2, default=str)
            logger.info(
                "Model evaluation report saved to %s",
                self.model_evaluation_config.report_file_path,
            )
        except Exception as error:
            raise VehicleInsuranceException(error, sys) from error

    def evaluate_model(self) -> ModelEvaluationArtifact:
        try:
            logger.info("Starting validation-based model evaluation")
            candidate = self.model_trainer_artifact
            candidate_lineage = self._candidate_lineage()
            candidate_validation = (
                candidate.validation_metrics
                if isinstance(candidate.validation_metrics, dict)
                else {}
            )
            candidate_test = (
                candidate.test_metrics if isinstance(candidate.test_metrics, dict) else {}
            )
            candidate_stability = (
                candidate.stability_metrics
                if isinstance(candidate.stability_metrics, dict)
                else {}
            )

            candidate_identity_reasons: list[str] = []
            for field_name in ("run_id", "model_id", "model_uri"):
                if not str(getattr(candidate, field_name, "") or "").strip():
                    candidate_identity_reasons.append(f"candidate_{field_name}_missing")

            validation_gate = self._check_validation_metrics(candidate_validation)
            stability_gate = self._check_stability_metrics(candidate_stability)
            lineage_gate = self._check_lineage(candidate_lineage)
            parity_gate = self._check_source_frozen_parity(candidate_lineage)
            if self.require_champion_comparison:
                champion_info = self._load_champion_info()
                champion_gate = self._check_champion_comparison(
                    candidate_validation,
                    champion_info,
                )
            else:
                disabled_reason = "champion_comparison_disabled_by_policy"
                if self.evaluation_context.get("mode") == "bootstrap":
                    disabled_reason = "first_champion_bootstrap"
                champion_info = {
                    "status": "not_required",
                    "validation_metrics": None,
                    "reason_codes": [disabled_reason],
                }
                champion_gate = {
                    "passed": True,
                    "status": "not_required",
                    "comparison_performed": False,
                    "reason_codes": [disabled_reason],
                }

            gate_results = {
                "candidate_identity": {
                    "passed": not candidate_identity_reasons,
                    "reason_codes": candidate_identity_reasons,
                },
                "validation_metrics": validation_gate,
                "stability": stability_gate,
                "lineage": lineage_gate,
                "source_frozen_parity": parity_gate,
                "champion_comparison": champion_gate,
            }
            reason_codes: list[str] = []
            for gate in gate_results.values():
                if not gate["passed"]:
                    reason_codes.extend(gate.get("reason_codes", []))

            evaluation_passed = not reason_codes
            promotion_enabled = bool(
                self.model_spec["evaluation_policy"]["promotion"]["enabled"]
            )
            if not evaluation_passed:
                decision = "rejected"
                promotion_status = "blocked_by_evaluation_gates"
            elif not promotion_enabled:
                decision = "eligible_but_promotion_disabled"
                promotion_status = "disabled"
            else:
                # Registration/alias mutation is deliberately not part of this
                # evaluator until an explicit promotion implementation is
                # approved and tested. A config-only switch cannot silently
                # change the served model.
                decision = "eligible_pending_explicit_promotion_workflow"
                promotion_status = "not_performed"

            report = {
                "evaluation_execution": self.evaluation_context,
                "evaluation_policy": {
                    "model_spec_version": str(self.model_spec["version"]),
                    "contract_version": str(self.schema_config["version"]),
                    "primary_metrics": self.model_spec["evaluation_policy"][
                        "primary_metrics"
                    ],
                    "supporting_metrics": self.model_spec["evaluation_policy"][
                        "supporting_metrics"
                    ],
                    "acceptance_thresholds": self.model_spec["evaluation_policy"][
                        "acceptance_thresholds"
                    ],
                    "promotion_enabled": promotion_enabled,
                    "champion_comparison_required": self.require_champion_comparison,
                },
                "candidate": {
                    "run_id": candidate.run_id,
                    "model_id": candidate.model_id,
                    "model_uri": candidate.model_uri,
                    "registered_model_name": candidate.registered_model_name,
                    "registered_model_version": candidate.registered_model_version,
                    "validation_metrics": candidate_validation,
                    "stability_metrics": candidate_stability,
                    "test_metrics": candidate_test,
                    "test_metrics_usage": "reporting_only",
                    "lineage": candidate_lineage,
                },
                "source_manifest": candidate_lineage.get("source_manifest"),
                "champion": champion_info,
                "gates": gate_results,
                "decision": {
                    "evaluation_passed": evaluation_passed,
                    "decision": decision,
                    "promotion_status": promotion_status,
                    "promotion_performed": False,
                    "reason_codes": reason_codes,
                },
            }
            self._save_report(report)

            artifact = ModelEvaluationArtifact(
                evaluation_passed=evaluation_passed,
                decision=decision,
                promotion_status=promotion_status,
                promotion_performed=False,
                candidate_run_id=candidate.run_id,
                candidate_model_id=candidate.model_id,
                candidate_model_uri=candidate.model_uri,
                champion_run_id=champion_info.get("run_id"),
                champion_model_version=champion_info.get("model_version"),
                candidate_validation_metrics=candidate_validation,
                champion_validation_metrics=champion_info.get("validation_metrics"),
                candidate_test_metrics=candidate_test,
                stability_metrics=candidate_stability,
                metric_comparison=champion_gate,
                gate_results=gate_results,
                reason_codes=reason_codes,
                lineage=candidate_lineage,
                evaluation_report_file_path=self.model_evaluation_config.report_file_path,
            )
            logger.info(
                "Model evaluation artifact created: candidate_run_id=%s "
                "evaluation_passed=%s promotion_status=%s report=%s",
                artifact.candidate_run_id,
                artifact.evaluation_passed,
                artifact.promotion_status,
                artifact.evaluation_report_file_path,
            )
            logger.info(
                "Model evaluation completed: decision=%s promotion_status=%s",
                decision,
                promotion_status,
            )
            return artifact
        except Exception as error:
            logger.exception("Model evaluation failed")
            raise VehicleInsuranceException(error, sys) from error
