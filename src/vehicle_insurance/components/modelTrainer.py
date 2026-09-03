from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from math import floor

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline

from vehicle_insurance.constants import (
    MLFLOW_EXPERIMENT_NAME,
    MODEL_SPEC_FILE_PATH,
    SCHEMA_FILE_PATH,
    TARGET_COLUMN,
)
from vehicle_insurance.entities.artifact_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from vehicle_insurance.entities.config_entity import ModelTrainerConfig
from vehicle_insurance.exceptions.handle_exception import VehicleInsuranceException
from vehicle_insurance.logger.logger_config import setup_logger
from vehicle_insurance.utils import (
    capture_git_provenance,
    load_numpy_array_data,
    load_object,
    read_yaml_file,
    save_object,
    resolve_repository_root,
    validate_git_provenance,
)
from vehicle_insurance.utils.source_manifest import validate_source_manifest

logger = setup_logger()


class ModelTrainer:
    """Train and record the frozen LightGBM production candidate."""

    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
        data_transformation_artifact: DataTransformationArtifact,
        data_ingestion_artifact: DataIngestionArtifact,
        git_provenance: dict | None = None,
    ) -> None:
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_spec = read_yaml_file(MODEL_SPEC_FILE_PATH)
            self.schema_config = read_yaml_file(SCHEMA_FILE_PATH)
            self._validate_model_spec()
            self.git_provenance = (
                dict(git_provenance)
                if git_provenance is not None
                else capture_git_provenance(resolve_repository_root())
            )
        except Exception as error:
            raise VehicleInsuranceException(error, sys) from error

    def _validate_model_spec(self) -> None:
        if self.model_spec.get("contract_version") != str(
            self.schema_config.get("version")
        ):
            raise ValueError(
                "Model specification and schema contract versions do not match"
            )

        data_protocol = self.model_spec["data_protocol"]
        if data_protocol["split_protocol"] != "stratified_80_10_10":
            raise ValueError(
                "Unsupported model-spec split protocol: "
                f"{data_protocol['split_protocol']}"
            )

        expected_ratios = (0.8, 0.1, 0.1)
        actual_ratios = (
            float(data_protocol["train_ratio"]),
            float(data_protocol["validation_ratio"]),
            float(data_protocol["test_ratio"]),
        )
        if actual_ratios != expected_ratios:
            raise ValueError(
                "Model specification does not contain the frozen 80/10/10 ratios"
            )

        if self.model_spec["model"]["class_weighting"]["method"] != (
            "scale_pos_weight"
        ):
            raise ValueError("Frozen model requires scale_pos_weight class weighting")

        if (
            self.model_spec["features"]["raw_inputs"]
            != self.schema_config["features"]["canonical_order"]
        ):
            raise ValueError(
                "Model specification and schema canonical feature order do not match"
            )

        mlflow_tracking = self.model_spec["artifacts"]["mlflow_tracking"]
        required_reference_fields = (
            "source_tuning_run_id",
            "reference_final_run_id",
            "reference_final_model_id",
            "reference_final_model_uri",
        )
        missing_reference_fields = [
            field
            for field in required_reference_fields
            if not str(mlflow_tracking.get(field, "")).strip()
        ]
        if missing_reference_fields:
            raise ValueError(
                "Model specification is missing required MLflow reference lineage: "
                f"{missing_reference_fields}"
            )

        git_policy = (
            self.model_spec.get("evaluation_policy", {})
            .get("lineage", {})
            .get("git")
        )
        if not isinstance(git_policy, dict):
            raise ValueError(
                "evaluation_policy.lineage.git must be a dictionary"
            )

        source_snapshot = data_protocol.get("source_snapshot")
        if not isinstance(source_snapshot, dict) or source_snapshot.get(
            "required"
        ) is not True:
            raise ValueError("data_protocol.source_snapshot must be required")
        source_lineage_policy = (
            self.model_spec.get("evaluation_policy", {})
            .get("lineage", {})
            .get("source_snapshot")
        )
        if not isinstance(source_lineage_policy, dict) or any(
            source_lineage_policy.get(field) is not True
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
    def _load_and_split_data(file_path: str) -> tuple[np.ndarray, np.ndarray]:
        try:
            array = load_numpy_array_data(file_path)
            if array.ndim != 2 or array.shape[1] < 2:
                raise ValueError(
                    f"Transformed array must be 2-dimensional with a target column: {file_path}"
                )
            return array[:, :-1], array[:, -1]
        except Exception as error:
            raise VehicleInsuranceException(error, sys) from error

    @staticmethod
    def _read_raw_split(file_path: str) -> pd.DataFrame:
        try:
            dataframe = pd.read_csv(file_path)
            if dataframe.empty:
                raise ValueError(f"Raw split is empty: {file_path}")
            return dataframe
        except Exception as error:
            raise VehicleInsuranceException(error, sys) from error

    @staticmethod
    def _package_version(package_name: str) -> str:
        try:
            return version(package_name)
        except PackageNotFoundError:
            return "not_installed"

    @staticmethod
    def _dataset_hash(
        train_df: pd.DataFrame,
        validation_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> str:
        combined = pd.concat(
            [train_df, validation_df, test_df],
            ignore_index=True,
        ).sort_values("id")
        canonical_bytes = combined.to_csv(index=False, lineterminator="\n").encode(
            "utf-8"
        )
        return hashlib.sha256(canonical_bytes).hexdigest()

    def _source_manifest(self) -> dict:
        manifest = getattr(self.data_ingestion_artifact, "source_manifest", None)
        if not isinstance(manifest, dict):
            manifest_path = getattr(
                self.data_ingestion_artifact,
                "source_manifest_file_path",
                None,
            )
            if manifest_path and os.path.isfile(manifest_path):
                with open(manifest_path, encoding="utf-8") as manifest_file:
                    manifest = json.load(manifest_file)
        validate_source_manifest(manifest)
        return dict(manifest)

    def _build_lineage(
        self,
        train_df: pd.DataFrame,
        validation_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> dict:
        dataset_hash = self._dataset_hash(train_df, validation_df, test_df)
        source_manifest = self._source_manifest()
        if dataset_hash != source_manifest["content_hash"]:
            raise ValueError(
                "Reassembled raw split content does not match the immutable "
                "MongoDB source snapshot manifest"
            )
        return {
            "contract_version": str(self.schema_config["version"]),
            "model_spec_version": str(self.model_spec["version"]),
            "git_branch": self.git_provenance["branch"],
            "git_commit": self.git_provenance["commit"],
            "git_provenance": dict(self.git_provenance),
            "python_version": platform.python_version(),
            "dependency_lock_hash": self.git_provenance.get(
                "dependency_lock_sha256"
            )
            or "not_recorded",
            "lightgbm_version": self._package_version("lightgbm"),
            "scikit_learn_version": self._package_version("scikit-learn"),
            "mlflow_version": self._package_version("mlflow"),
            "schema_version": str(self.schema_config["version"]),
            "dataset_snapshot_id": source_manifest["snapshot_id"],
            "dataset_content_hash": source_manifest["content_hash"],
            "source_manifest": source_manifest,
            "split_protocol": self.model_spec["data_protocol"]["split_protocol"],
            "random_state": self.model_spec["data_protocol"]["random_state"],
            "source_tuning_run_id": self.model_spec["artifacts"][
                "mlflow_tracking"
            ]["source_tuning_run_id"],
            "reference_final_run_id": self.model_spec["artifacts"][
                "mlflow_tracking"
            ]["reference_final_run_id"],
            "reference_final_model_id": self.model_spec["artifacts"][
                "mlflow_tracking"
            ]["reference_final_model_id"],
            "reference_final_model_uri": self.model_spec["artifacts"][
                "mlflow_tracking"
            ]["reference_final_model_uri"],
            "source_frozen_parity": {
                "status": "not_verified",
                "reason": (
                    "Direct split-ID and prediction-array comparison against the "
                    "source-frozen reference is pending"
                ),
            },
        }

    def _model_parameters(self) -> dict:
        model_config = self.model_spec["model"]
        parameters = dict(model_config["hyperparameters"])
        parameters["scale_pos_weight"] = float(
            model_config["class_weighting"]["scale_pos_weight"]
        )
        if "is_unbalance" in parameters:
            raise ValueError("is_unbalance is forbidden by the frozen model spec")
        return parameters

    def _expected_feature_count(self, field_name: str, description: str) -> int:
        preprocessing_config = self.model_spec["features"]["preprocessing"]
        configured_count = preprocessing_config.get(field_name)
        if configured_count is None:
            raise ValueError(f"Model specification must define {description}")
        feature_count = int(configured_count)
        if feature_count <= 0:
            raise ValueError(f"{description.capitalize()} must be positive")
        return feature_count

    def _expected_train_only_feature_count(self) -> int:
        return self._expected_feature_count(
            "train_only_transformed_feature_count",
            "the train-only transformed feature count",
        )

    def _expected_final_feature_count(self) -> int:
        return self._expected_feature_count(
            "final_transformed_feature_count",
            "the final transformed feature count",
        )

    def _ranking_metrics(
        self,
        positive_scores: np.ndarray,
        y_true: np.ndarray,
        ids: np.ndarray,
    ) -> dict:
        selection_config = self.model_spec["selection"]
        percentage = float(
            selection_config.get(
                "top_k_percentage",
                selection_config.get("default_percentage"),
            )
        )
        if not 0 < percentage <= 100:
            raise ValueError("Top-K percentage must be greater than 0 and at most 100")
        if selection_config["k_calculation"] != "floor":
            raise ValueError("Only floor K calculation is supported")

        ranking = pd.DataFrame(
            {
                "id": ids,
                "score": positive_scores,
                "target": y_true,
            }
        ).sort_values(
            by=["score", "id"],
            ascending=[False, True],
            kind="mergesort",
        )
        k = floor(len(ranking) * percentage / 100.0)
        if k <= 0:
            precision_at_k = 0.0
            recall_at_k = 0.0
            lift_at_k = 0.0
        else:
            selected = ranking.head(k)["target"]
            positive_count = float(np.sum(y_true == 1))
            selected_positive_count = float(np.sum(selected == 1))
            precision_at_k = selected_positive_count / k
            recall_at_k = (
                selected_positive_count / positive_count if positive_count else 0.0
            )
            prevalence = positive_count / len(y_true)
            lift_at_k = precision_at_k / prevalence if prevalence else 0.0

        return {
            "precision_at_10pct": round(float(precision_at_k), 6),
            "recall_at_10pct": round(float(recall_at_k), 6),
            "lift_at_10pct": round(float(lift_at_k), 6),
        }

    def _compute_metrics(
        self,
        model: LGBMClassifier | Pipeline,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray,
        ids: np.ndarray,
    ) -> dict:
        try:
            y_true = np.asarray(y).ravel().astype(int)
            id_values = np.asarray(ids).ravel()
            if len(X) != len(y_true) or len(X) != len(id_values):
                raise ValueError("X, y, and id lengths do not match")

            y_proba = np.asarray(model.predict_proba(X))[:, 1].ravel()
            y_pred = (y_proba >= 0.5).astype(int)
            metrics = {
                "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
                "precision": round(
                    float(precision_score(y_true, y_pred, zero_division=0)),
                    6,
                ),
                "recall": round(
                    float(recall_score(y_true, y_pred, zero_division=0)),
                    6,
                ),
                "f1_score": round(
                    float(f1_score(y_true, y_pred, zero_division=0)),
                    6,
                ),
                "pr_auc": round(float(average_precision_score(y_true, y_proba)), 6),
                "roc_auc": round(float(roc_auc_score(y_true, y_proba)), 6),
            }
            metrics.update(self._ranking_metrics(y_proba, y_true, id_values))
            return metrics
        except Exception as error:
            raise VehicleInsuranceException(error, sys) from error

    def _compute_stability_metrics(
        self,
        model: LGBMClassifier,
        X: pd.DataFrame,
        y: np.ndarray,
        ids: np.ndarray,
        preprocessor: Pipeline,
    ) -> dict:
        stability_config = self.model_spec["evaluation_policy"]["stability"]
        if not stability_config["required"]:
            return {"required": False}

        folds = int(stability_config["folds"])
        repeats = int(stability_config["repeats"])
        if folds < 2 or repeats < 1:
            raise ValueError("Stability CV requires folds >= 2 and repeats >= 1")

        cv = RepeatedStratifiedKFold(
            n_splits=folds,
            n_repeats=repeats,
            random_state=int(self.model_spec["data_protocol"]["random_state"]),
        )
        metric_names = stability_config["selection_metrics"]
        fold_metrics = []
        for fold_number, (train_indices, validation_indices) in enumerate(
            cv.split(X, y),
            start=1,
        ):
            fold_preprocessor = clone(preprocessor)
            if hasattr(fold_preprocessor, "set_output"):
                fold_preprocessor.set_output(transform="pandas")
            fold_pipeline = Pipeline(
                steps=[
                    ("preprocessor", fold_preprocessor),
                    ("model", clone(model)),
                ]
            )
            fold_pipeline.fit(
                self._select_rows(X, train_indices),
                np.asarray(y)[train_indices],
            )
            fold_result = self._compute_metrics(
                fold_pipeline,
                self._select_rows(X, validation_indices),
                y[validation_indices],
                ids[validation_indices],
            )
            fold_metrics.append(
                {
                    "fold": fold_number,
                    **{name: fold_result[name] for name in metric_names},
                }
            )

        summary = {}
        for metric_name in metric_names:
            values = np.asarray(
                [result[metric_name] for result in fold_metrics],
                dtype=float,
            )
            summary[metric_name] = {
                "mean": round(float(values.mean()), 6),
                "std": round(float(values.std(ddof=0)), 6),
                "minimum": round(float(values.min()), 6),
                "maximum": round(float(values.max()), 6),
            }
        return {
            "required": True,
            "method": "repeated_stratified_cross_validation",
            "folds": folds,
            "repeats": repeats,
            "evaluation_count": folds * repeats,
            "data_scope": "training_split_only",
            "preprocessing_scope": "fit_inside_each_fold_on_fold_training_rows",
            "metrics": summary,
            "fold_results": fold_metrics,
        }

    @staticmethod
    def _select_rows(
        dataframe_or_array: np.ndarray | pd.DataFrame,
        indices: np.ndarray,
    ) -> np.ndarray | pd.DataFrame:
        if isinstance(dataframe_or_array, pd.DataFrame):
            return dataframe_or_array.iloc[indices]
        return dataframe_or_array[indices]

    def _set_up_mlflow(self) -> None:
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
        if not tracking_uri:
            raise ValueError("MLFLOW_TRACKING_URI is not set in environment variables")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        logger.info("MLflow tracking URI set to: %s", tracking_uri)
        logger.info("MLflow experiment set to: %s", MLFLOW_EXPERIMENT_NAME)

    @staticmethod
    def _validate_raw_transformed_alignment(
        raw_dataframe: pd.DataFrame,
        transformed_target: np.ndarray,
        split_name: str,
    ) -> None:
        raw_target = raw_dataframe[TARGET_COLUMN].to_numpy().astype(int)
        transformed_target = np.asarray(transformed_target).ravel().astype(int)
        if raw_dataframe["id"].nunique() != len(raw_dataframe):
            raise ValueError(f"{split_name} raw data contains duplicate IDs")
        if not np.array_equal(raw_target, transformed_target):
            raise ValueError(f"{split_name} transformed target order is inconsistent")

    def train(self) -> ModelTrainerArtifact:
        try:
            validate_git_provenance(
                self.git_provenance,
                self.model_spec["evaluation_policy"]["lineage"]["git"],
            )
            logger.info("Model Trainer started")

            # The transformed artifacts are retained for the train-only candidate
            # phase. They also provide an independent row/target alignment check
            # against the raw ingestion splits.
            X_train_transformed, y_train = self._load_and_split_data(
                self.data_transformation_artifact.transformed_train_file_path
            )
            X_validation_transformed, y_validation = self._load_and_split_data(
                self.data_transformation_artifact.transformed_validation_file_path
            )
            X_test_transformed, y_test = self._load_and_split_data(
                self.data_transformation_artifact.transformed_test_file_path
            )

            raw_train = self._read_raw_split(
                self.data_ingestion_artifact.trained_file_path
            )
            raw_validation = self._read_raw_split(
                self.data_ingestion_artifact.validation_file_path
            )
            raw_test = self._read_raw_split(self.data_ingestion_artifact.test_file_path)

            raw_features = self.model_spec["features"]["raw_inputs"]
            X_train_raw = raw_train.loc[:, raw_features]
            X_validation_raw = raw_validation.loc[:, raw_features]
            X_test_raw = raw_test.loc[:, raw_features]
            train_ids = raw_train["id"].to_numpy()
            validation_ids = raw_validation["id"].to_numpy()
            test_ids = raw_test["id"].to_numpy()

            if (
                len(X_train_transformed) != len(X_train_raw)
                or len(X_validation_transformed) != len(X_validation_raw)
                or len(X_test_transformed) != len(X_test_raw)
            ):
                raise ValueError("Transformed and raw split row counts do not match")

            self._validate_raw_transformed_alignment(raw_train, y_train, "training")
            self._validate_raw_transformed_alignment(
                raw_validation,
                y_validation,
                "validation",
            )
            self._validate_raw_transformed_alignment(raw_test, y_test, "test")

            train_only_preprocessor = load_object(
                self.data_transformation_artifact.transformer_object_file_path
            )
            candidate_feature_names = [
                str(column)
                for column in train_only_preprocessor.get_feature_names_out()
            ]
            candidate_feature_count = len(candidate_feature_names)
            if candidate_feature_count <= 0:
                raise ValueError("Train-only preprocessor produced no features")
            expected_train_only_feature_count = self._expected_train_only_feature_count()
            if candidate_feature_count != expected_train_only_feature_count:
                raise ValueError(
                    "Train-only preprocessor feature count does not match the "
                    "model spec: "
                    f"expected={expected_train_only_feature_count}, "
                    f"actual={candidate_feature_count}"
                )

            candidate_arrays = {
                "train": X_train_transformed,
                "validation": X_validation_transformed,
                "test": X_test_transformed,
            }
            candidate_widths = {
                split_name: array.shape[1]
                for split_name, array in candidate_arrays.items()
            }
            if any(
                width != candidate_feature_count for width in candidate_widths.values()
            ):
                raise ValueError(
                    "Transformed arrays do not match the train-only fitted "
                    f"preprocessor: expected={candidate_feature_count}, "
                    f"actual={candidate_widths}"
                )

            X_train_candidate = pd.DataFrame(
                X_train_transformed,
                columns=candidate_feature_names,
            )
            X_validation_candidate = pd.DataFrame(
                X_validation_transformed,
                columns=candidate_feature_names,
            )

            parameters = self._model_parameters()
            candidate_model = LGBMClassifier(**parameters)
            logger.info(
                "Training train-only LightGBM candidate using the frozen model specification"
            )
            candidate_model.fit(X_train_candidate, y_train)

            train_metrics = self._compute_metrics(
                candidate_model,
                X_train_candidate,
                y_train,
                train_ids,
            )
            validation_metrics = self._compute_metrics(
                candidate_model,
                X_validation_candidate,
                y_validation,
                validation_ids,
            )
            stability_metrics = self._compute_stability_metrics(
                candidate_model,
                X_train_raw,
                y_train,
                train_ids,
                train_only_preprocessor,
            )

            # Match experimentation_v2 cell 130: the production artifact is
            # refit only after candidate/operating-point decisions are complete.
            X_final_raw = pd.concat(
                [X_train_raw, X_validation_raw],
                axis=0,
                ignore_index=True,
            )
            y_final = np.concatenate([y_train, y_validation])
            final_preprocessor = clone(train_only_preprocessor)
            if hasattr(final_preprocessor, "set_output"):
                final_preprocessor.set_output(transform="pandas")
            final_model = clone(candidate_model)
            final_pipeline = Pipeline(
                steps=[
                    ("preprocessor", final_preprocessor),
                    ("model", final_model),
                ]
            )
            logger.info(
                "Fitting final frozen pipeline on training plus validation data"
            )
            final_pipeline.fit(X_final_raw, y_final)

            final_feature_names = [
                str(column)
                for column in final_pipeline.named_steps[
                    "preprocessor"
                ].get_feature_names_out()
            ]
            final_feature_count = len(final_feature_names)
            expected_final_feature_count = self._expected_final_feature_count()
            if final_feature_count != expected_final_feature_count:
                raise ValueError(
                    "Final refit feature count does not match the model spec: "
                    f"expected={expected_final_feature_count}, "
                    f"actual={final_feature_count}"
                )
            if (
                final_pipeline.named_steps["model"].n_features_in_
                != final_feature_count
            ):
                raise ValueError(
                    "Final model input width does not match the final preprocessor "
                    f"width: model={final_pipeline.named_steps['model'].n_features_in_}, "
                    f"preprocessor={final_feature_count}"
                )

            # This is the only test evaluation. It is deliberately performed
            # through the final raw-input pipeline, not the candidate model.
            test_metrics = self._compute_metrics(
                final_pipeline,
                X_test_raw,
                y_test,
                test_ids,
            )
            logger.info("Candidate training metrics: %s", train_metrics)
            logger.info("Candidate validation metrics: %s", validation_metrics)
            logger.info("Final test metrics: %s", test_metrics)

            lineage = self._build_lineage(raw_train, raw_validation, raw_test)
            lineage.update(
                {
                    "candidate_fit_data": "train_only",
                    "candidate_transformed_feature_count": candidate_feature_count,
                    "final_fit_data": "train_plus_validation",
                    "final_fit_rows": len(X_final_raw),
                    "final_transformed_feature_count": final_feature_count,
                    "final_transformed_feature_names": final_feature_names,
                    "model_input_feature_order": final_feature_names,
                    "test_evaluation_count": 1,
                    "test_evaluation_model": "final_pipeline",
                }
            )

            self._set_up_mlflow()
            with mlflow.start_run(run_name="frozen_lightgbm_candidate") as run:
                run_id = run.info.run_id
                mlflow.log_params(
                    {f"model_{key}": value for key, value in parameters.items()}
                )
                mlflow.log_params(
                    {
                        "candidate_fit_data": "train_only",
                        "final_fit_data": "train_plus_validation",
                        "candidate_transformed_feature_count": candidate_feature_count,
                        "final_transformed_feature_count": final_feature_count,
                        "test_evaluation_count": 1,
                    }
                )
                mlflow.set_tags(
                    {
                        "contract_version": lineage["contract_version"],
                        "model_spec_version": lineage["model_spec_version"],
                        "split_protocol": lineage["split_protocol"],
                        "dataset_snapshot_id": lineage["dataset_snapshot_id"],
                        "source_database": lineage["source_manifest"]["source"][
                            "database"
                        ],
                        "source_collection": lineage["source_manifest"]["source"][
                            "collection"
                        ],
                        "source_row_count": str(
                            lineage["source_manifest"]["row_count"]
                        ),
                        "source_id_min": str(
                            lineage["source_manifest"]["id_range"]["min"]
                        ),
                        "source_id_max": str(
                            lineage["source_manifest"]["id_range"]["max"]
                        ),
                        "source_read_boundary": lineage["source_manifest"][
                            "read_boundary"
                        ]["mode"],
                        "candidate_fit_data": "train_only",
                        "final_fit_data": "train_plus_validation",
                        "candidate_transformed_feature_count": candidate_feature_count,
                        "final_transformed_feature_count": final_feature_count,
                        "test_evaluation": "one_time_final_pipeline",
                        "source_tuning_run_id": lineage["source_tuning_run_id"],
                        "reference_final_run_id": lineage["reference_final_run_id"],
                        "promotion_status": "disabled",
                        "git_commit": lineage["git_commit"],
                        "git_is_dirty": str(
                            lineage["git_provenance"]["is_dirty"]
                        ).lower(),
                        "git_patch_sha256": lineage["git_provenance"][
                            "patch_sha256"
                        ],
                    }
                )
                mlflow.log_metrics(
                    {f"train_{key}": value for key, value in train_metrics.items()}
                )
                mlflow.log_metrics(
                    {
                        f"validation_{key}": value
                        for key, value in validation_metrics.items()
                    }
                )
                mlflow.log_metrics(
                    {f"test_{key}": value for key, value in test_metrics.items()}
                )
                for metric_name, values in stability_metrics.get("metrics", {}).items():
                    mlflow.log_metrics(
                        {
                            f"stability_{metric_name}_mean": values["mean"],
                            f"stability_{metric_name}_std": values["std"],
                        }
                    )

                signature = mlflow.models.infer_signature(
                    X_final_raw.head(3),
                    pd.DataFrame(
                        {
                            "response_propensity_score": final_pipeline.predict_proba(
                                X_final_raw.head(3)
                            )[:, 1]
                        }
                    ),
                )
                model_info = mlflow.sklearn.log_model(
                    sk_model=final_pipeline,
                    name="model",
                    signature=signature,
                    input_example=X_final_raw.head(3),
                )
                mlflow.log_artifact(
                    MODEL_SPEC_FILE_PATH,
                    artifact_path="specification",
                )
                mlflow.log_artifact(
                    SCHEMA_FILE_PATH,
                    artifact_path="specification",
                )
                mlflow.log_dict(
                    {
                        "raw_feature_order": raw_features,
                        "candidate_transformed_feature_order": candidate_feature_names,
                        "final_transformed_feature_order": final_feature_names,
                    },
                    "specification/feature_order.json",
                )

                model_id = getattr(model_info, "model_id", None)
                model_uri = getattr(model_info, "model_uri", None)
                if model_uri is None and model_id:
                    model_uri = f"models:/{model_id}"
                if not model_id or not model_uri:
                    raise RuntimeError(
                        "MLflow did not return a model ID and model URI; "
                        "lineage cannot be recorded safely"
                    )
                lineage.update(
                    {
                        "mlflow_run_id": run_id,
                        "model_id": model_id,
                        "model_uri": model_uri,
                        "trainer_candidate_run_id": run_id,
                        "trainer_candidate_model_id": model_id,
                        "trainer_candidate_model_uri": model_uri,
                        "registered_model_name": None,
                        "registered_model_version": None,
                    }
                )
                mlflow.log_dict(lineage, "lineage/lineage.json")
                mlflow.log_dict(
                    lineage["source_manifest"],
                    "lineage/source_manifest.json",
                )

            save_object(
                self.model_trainer_config.trained_model_file_path,
                final_pipeline,
            )
            metrics = {
                "mlflow_run_id": run_id,
                "model_id": lineage["model_id"],
                "model_uri": lineage["model_uri"],
                "registered_model_name": None,
                "registered_model_version": None,
                "promotion_status": "disabled",
                "train_metrics": train_metrics,
                "validation_metrics": validation_metrics,
                "test_metrics": test_metrics,
                "stability_metrics": stability_metrics,
                "lineage": lineage,
            }
            os.makedirs(
                os.path.dirname(self.model_trainer_config.metrics_file_path),
                exist_ok=True,
            )
            with open(
                self.model_trainer_config.metrics_file_path,
                "w",
                encoding="utf-8",
            ) as metrics_file:
                json.dump(metrics, metrics_file, indent=2, default=str)

            artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metrics_file_path=self.model_trainer_config.metrics_file_path,
                train_metrics=train_metrics,
                validation_metrics=validation_metrics,
                test_metrics=test_metrics,
                run_id=run_id,
                model_id=lineage["model_id"],
                model_uri=lineage["model_uri"],
                registered_model_name=None,
                registered_model_version=None,
                stability_metrics=stability_metrics,
                lineage=lineage,
            )
            logger.info("Model Trainer completed with artifact: %s", artifact)
            return artifact
        except Exception as error:
            logger.exception("Model Trainer failed")
            raise VehicleInsuranceException(error, sys) from error
