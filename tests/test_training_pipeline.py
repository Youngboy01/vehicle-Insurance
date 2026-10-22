import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from vehicle_insurance.entities.artifact_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
    DataValidationArtifact,
)
from vehicle_insurance.exceptions.handle_exception import VehicleInsuranceException
from vehicle_insurance.pipeline import training_pipeline as training_pipeline_module
from vehicle_insurance.pipeline.training_pipeline import TrainingPipeline


def test_run_pipeline_executes_stages_in_order_and_returns_evaluation_artifact(
    monkeypatch,
):
    pipeline = TrainingPipeline()
    monkeypatch.setattr(pipeline, "_validate_git_preflight", lambda: None)
    calls = []
    ingestion_artifact = DataIngestionArtifact("train", "validation", "test")
    validation_artifact = DataValidationArtifact("report", "valid", True)
    transformation_artifact = DataTransformationArtifact(
        "train.npy",
        "validation.npy",
        "test.npy",
        "preprocessor.pkl",
    )
    trainer_artifact = SimpleNamespace()
    evaluation_artifact = SimpleNamespace(
        evaluation_passed=False,
        promotion_status="blocked_by_evaluation_gates",
        decision="rejected",
        evaluation_report_file_path="artifact/report.json",
    )

    monkeypatch.setattr(
        pipeline,
        "start_data_ingestion",
        lambda: calls.append("ingestion") or ingestion_artifact,
    )
    monkeypatch.setattr(
        pipeline,
        "start_data_validation",
        lambda **_: calls.append("validation") or validation_artifact,
    )
    monkeypatch.setattr(
        pipeline,
        "start_data_transformation",
        lambda **_: calls.append("transformation") or transformation_artifact,
    )
    monkeypatch.setattr(
        pipeline,
        "start_model_training",
        lambda **_: calls.append("training") or trainer_artifact,
    )
    monkeypatch.setattr(
        pipeline,
        "start_source_frozen_parity",
        lambda: (
            calls.append("parity")
            or {
                "status": "verified",
                "report_path": "artifact/parity_report.json",
                "lineage_recording": {"local_metrics_json_updated": False},
            }
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "start_model_evaluation",
        lambda **_: calls.append("evaluation") or evaluation_artifact,
    )

    result = pipeline.run_pipeline()

    assert result is evaluation_artifact
    assert calls == [
        "ingestion",
        "validation",
        "transformation",
        "training",
        "parity",
        "evaluation",
    ]


def test_run_pipeline_blocks_downstream_stages_when_validation_fails(monkeypatch):
    pipeline = TrainingPipeline()
    monkeypatch.setattr(pipeline, "_validate_git_preflight", lambda: None)
    calls = []

    monkeypatch.setattr(
        pipeline,
        "start_data_ingestion",
        lambda: calls.append("ingestion") or DataIngestionArtifact("", "", ""),
    )
    monkeypatch.setattr(
        pipeline,
        "start_data_validation",
        lambda **_: (
            calls.append("validation")
            or DataValidationArtifact("artifact/report.json", "invalid", False)
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "start_data_transformation",
        lambda **_: calls.append("transformation"),
    )

    with pytest.raises(VehicleInsuranceException):
        pipeline.run_pipeline()

    assert calls == ["ingestion", "validation"]


def test_run_pipeline_blocks_before_ingestion_when_git_is_dirty(monkeypatch):
    pipeline = TrainingPipeline()
    pipeline.git_provenance["is_dirty"] = True
    pipeline.git_provenance["status_porcelain"] = " M src/example.py"
    calls = []

    monkeypatch.setattr(
        pipeline,
        "start_data_ingestion",
        lambda: calls.append("ingestion"),
    )

    with pytest.raises(VehicleInsuranceException):
        pipeline.run_pipeline()

    assert calls == []


def test_source_frozen_parity_stage_uses_candidate_artifact_and_applies_lineage(
    monkeypatch,
    tmp_path,
):
    pipeline = TrainingPipeline()
    pipeline.model_trainer_config = SimpleNamespace(
        model_trainer_dir=str(tmp_path / "model_trainer")
    )
    captured = {}

    class _FakeCertification:
        def __init__(
            self,
            *,
            candidate_artifact_dir,
            apply_lineage,
            log_mlflow_evidence,
        ):
            captured["candidate_artifact_dir"] = candidate_artifact_dir
            captured["apply_lineage"] = apply_lineage
            captured["log_mlflow_evidence"] = log_mlflow_evidence
            self.report_path = (
                Path(candidate_artifact_dir)
                / "source_frozen_parity"
                / "parity_report.json"
            )

        def run(self):
            return {
                "status": "verified",
                "lineage_recording": {
                    "requested": True,
                    "local_metrics_json_updated": True,
                },
            }

    monkeypatch.setattr(
        training_pipeline_module,
        "SourceFrozenParityCertification",
        _FakeCertification,
    )

    report = pipeline.start_source_frozen_parity()

    assert report["status"] == "verified"
    assert report["report_path"].endswith("source_frozen_parity\\parity_report.json")
    assert captured["candidate_artifact_dir"] == (tmp_path).resolve()
    assert captured["apply_lineage"] is True
    assert captured["log_mlflow_evidence"] is True


def test_reload_lineage_refreshes_in_memory_trainer_artifact(tmp_path):
    metrics_path = tmp_path / "model_trainer" / "metrics.json"
    metrics_path.parent.mkdir()
    metrics_path.write_text(
        json.dumps(
            {
                "lineage": {
                    "source_frozen_parity": {"status": "verified"},
                }
            }
        ),
        encoding="utf-8",
    )
    trainer_artifact = SimpleNamespace(
        metrics_file_path=str(metrics_path),
        lineage={"source_frozen_parity": {"status": "not_verified"}},
    )

    TrainingPipeline._reload_lineage_from_metrics(trainer_artifact)

    assert trainer_artifact.lineage["source_frozen_parity"]["status"] == "verified"
