from dataclasses import dataclass
from typing import Optional


@dataclass
class DataIngestionArtifact:
    trained_file_path: str
    validation_file_path: str
    test_file_path: str


@dataclass
class DataValidationArtifact:
    report_file_path: str
    message: str
    status: bool


@dataclass
class DataTransformationArtifact:
    transformed_train_file_path: str
    transformed_test_file_path: str
    transformer_object_file_path: str


@dataclass
class ModelTrainerArtifact:
    trained_model_file_path: str
    metrics_file_path: str
    train_metrics: dict
    test_metrics: dict
    run_id: str


@dataclass
class ModelEvaluationArtifact:
    is_model_accepted: bool
    champion_model_version: Optional[str]
    challenger_model_version: str
    champion_recall: Optional[float]
    challenger_recall: float
    evaluation_report_file_path: str
