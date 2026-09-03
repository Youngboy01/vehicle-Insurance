from dataclasses import dataclass
from typing import Optional


@dataclass
class DataIngestionArtifact:
    trained_file_path: str
    validation_file_path: str
    test_file_path: str
    feature_store_file_path: Optional[str] = None
    source_manifest_file_path: Optional[str] = None
    source_manifest: Optional[dict] = None


@dataclass
class DataValidationArtifact:
    report_file_path: str
    message: str
    status: bool


@dataclass
class DataTransformationArtifact:
    transformed_train_file_path: str
    transformed_validation_file_path: str
    transformed_test_file_path: str
    transformer_object_file_path: str


@dataclass
class ModelTrainerArtifact:
    trained_model_file_path: str
    metrics_file_path: str
    train_metrics: dict
    validation_metrics: dict
    test_metrics: dict
    run_id: str
    model_id: str
    model_uri: str
    registered_model_name: Optional[str]
    registered_model_version: Optional[str]
    stability_metrics: dict
    lineage: dict


@dataclass
class ModelEvaluationArtifact:
    evaluation_passed: bool
    decision: str
    promotion_status: str
    promotion_performed: bool
    candidate_run_id: str
    candidate_model_id: str
    candidate_model_uri: str
    champion_run_id: Optional[str]
    champion_model_version: Optional[str]
    candidate_validation_metrics: dict
    champion_validation_metrics: Optional[dict]
    candidate_test_metrics: dict
    stability_metrics: dict
    metric_comparison: dict
    gate_results: dict
    reason_codes: list[str]
    lineage: dict
    evaluation_report_file_path: str
