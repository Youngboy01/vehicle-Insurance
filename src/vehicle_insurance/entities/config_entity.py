import os
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4

from vehicle_insurance.constants import *


def _new_run_name() -> str:
    """Create a filesystem-safe run name that cannot collide at second precision."""
    utc_timestamp = datetime.now(timezone.utc).strftime("%m_%d_%Y_%H_%M_%S_%f")
    return f"{utc_timestamp}_{uuid4().hex}"


def _resolve_artifact_root(
    artifact_dir: str | None,
    stage_dir: str | None = None,
) -> str:
    if artifact_dir:
        return os.fspath(artifact_dir)
    if stage_dir:
        return os.path.dirname(os.fspath(stage_dir))
    return os.path.join(ARTIFACT_DIR, _new_run_name())


@dataclass
class TrainingPipelineConfig:
    """Run identity shared by every stage in one training execution."""

    pipeline_name: str = PIPELINE_NAME
    artifact_dir: str | None = None
    timestamp: str | None = None

    def __post_init__(self) -> None:
        run_name = self.timestamp or _new_run_name()
        self.timestamp = run_name
        if not self.artifact_dir:
            self.artifact_dir = os.path.join(ARTIFACT_DIR, run_name)


@dataclass
class DataIngestionConfig:
    data_ingestion_dir: str | None = None
    feature_store_file_path: str | None = None
    training_file_path: str | None = None
    validation_file_path: str | None = None
    testing_file_path: str | None = None
    collection_name: str = DATA_INGESTION_COLLECTION_NAME
    artifact_dir: str | None = None

    def __post_init__(self) -> None:
        artifact_root = _resolve_artifact_root(
            self.artifact_dir,
            self.data_ingestion_dir,
        )
        self.artifact_dir = artifact_root
        self.data_ingestion_dir = self.data_ingestion_dir or os.path.join(
            artifact_root, DATA_INGESTION_DIR_NAME
        )
        self.feature_store_file_path = self.feature_store_file_path or os.path.join(
            self.data_ingestion_dir,
            DATA_INGESTION_FEATURE_STORE_DIR,
            FILE_NAME,
        )
        self.training_file_path = self.training_file_path or os.path.join(
            self.data_ingestion_dir,
            DATA_INGESTION_INGESTED_DIR,
            TRAIN_FILE_NAME,
        )
        self.validation_file_path = self.validation_file_path or os.path.join(
            self.data_ingestion_dir,
            DATA_INGESTION_INGESTED_DIR,
            VALIDATION_FILE_NAME,
        )
        self.testing_file_path = self.testing_file_path or os.path.join(
            self.data_ingestion_dir,
            DATA_INGESTION_INGESTED_DIR,
            TEST_FILE_NAME,
        )


@dataclass
class DataValidationConfig:
    data_validation_dir: str | None = None
    report_file_path: str | None = None
    artifact_dir: str | None = None

    def __post_init__(self) -> None:
        artifact_root = _resolve_artifact_root(
            self.artifact_dir,
            self.data_validation_dir,
        )
        self.artifact_dir = artifact_root
        self.data_validation_dir = self.data_validation_dir or os.path.join(
            artifact_root, DATA_VALIDATION_DIR_NAME
        )
        self.report_file_path = self.report_file_path or os.path.join(
            self.data_validation_dir,
            "report",
            DATA_VALIDATION_REPORT_FILE_NAME,
        )


@dataclass
class DataTransformationConfig:
    data_transformation_dir: str | None = None
    transformed_train_file_path: str | None = None
    transformed_validation_file_path: str | None = None
    transformed_test_file_path: str | None = None
    transformer_object_file_path: str | None = None
    artifact_dir: str | None = None

    def __post_init__(self) -> None:
        artifact_root = _resolve_artifact_root(
            self.artifact_dir,
            self.data_transformation_dir,
        )
        self.artifact_dir = artifact_root
        self.data_transformation_dir = self.data_transformation_dir or os.path.join(
            artifact_root, DATA_TRANSFORMATION_DIR_NAME
        )
        transformed_data_dir = os.path.join(
            self.data_transformation_dir,
            DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
        )
        self.transformed_train_file_path = (
            self.transformed_train_file_path
            or os.path.join(
                transformed_data_dir,
                DATA_TRANSFORMATION_TRANSFORMED_TRAIN_FILE_NAME,
            )
        )
        self.transformed_validation_file_path = (
            self.transformed_validation_file_path
            or os.path.join(
                transformed_data_dir,
                DATA_TRANSFORMATION_TRANSFORMED_VALIDATION_FILE_NAME,
            )
        )
        self.transformed_test_file_path = (
            self.transformed_test_file_path
            or os.path.join(
                transformed_data_dir,
                DATA_TRANSFORMATION_TRANSFORMED_TEST_FILE_NAME,
            )
        )
        self.transformer_object_file_path = (
            self.transformer_object_file_path
            or os.path.join(
                self.data_transformation_dir,
                DATA_TRANSFORMATION_TRANSFORMER_OBJECT_FILE_NAME,
            )
        )


@dataclass
class ModelTrainerConfig:
    model_trainer_dir: str | None = None
    trained_model_file_path: str | None = None
    metrics_file_path: str | None = None
    artifact_dir: str | None = None

    def __post_init__(self) -> None:
        artifact_root = _resolve_artifact_root(
            self.artifact_dir,
            self.model_trainer_dir,
        )
        self.artifact_dir = artifact_root
        self.model_trainer_dir = self.model_trainer_dir or os.path.join(
            artifact_root, MODEL_TRAINER_DIR_NAME
        )
        self.trained_model_file_path = self.trained_model_file_path or os.path.join(
            self.model_trainer_dir,
            MODEL_TRAINER_TRAINED_MODEL_FILE_NAME,
        )
        self.metrics_file_path = self.metrics_file_path or os.path.join(
            self.model_trainer_dir,
            MODEL_TRAINER_METRICS_FILE_NAME,
        )


@dataclass
class ModelEvaluationConfig:
    model_evaluation_dir: str | None = None
    report_file_path: str | None = None
    artifact_dir: str | None = None

    def __post_init__(self) -> None:
        artifact_root = _resolve_artifact_root(
            self.artifact_dir,
            self.model_evaluation_dir,
        )
        self.artifact_dir = artifact_root
        self.model_evaluation_dir = self.model_evaluation_dir or os.path.join(
            artifact_root, MODEL_EVALUATION_DIR_NAME
        )
        self.report_file_path = self.report_file_path or os.path.join(
            self.model_evaluation_dir,
            MODEL_EVALUATION_REPORT_FILE_NAME,
        )
