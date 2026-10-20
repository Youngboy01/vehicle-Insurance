from contextlib import nullcontext
from copy import deepcopy

import pandas as pd
import pytest

from vehicle_insurance.verification.source_frozen_parity import (
    SourceFrozenParityCertification,
)


def _split_frames(*, reorder_test: bool = False) -> dict[str, pd.DataFrame]:
    test_ids = [5, 6]
    test_targets = [0, 1]
    if reorder_test:
        test_ids = [6, 5]

    return {
        "train": pd.DataFrame({"id": [1, 2], "Response": [0, 1]}),
        "validation": pd.DataFrame({"id": [3, 4], "Response": [1, 0]}),
        "test": pd.DataFrame({"id": test_ids, "Response": test_targets}),
    }


def _certification_for_splits(
    candidate_splits: dict[str, pd.DataFrame],
    reference_splits: dict[str, dict[str, list]],
) -> SourceFrozenParityCertification:
    certification = SourceFrozenParityCertification.__new__(
        SourceFrozenParityCertification
    )
    certification.candidate_splits = candidate_splits
    certification.reference_splits = reference_splits
    certification.checks = {}
    return certification


def _reference_split_payload(
    splits: dict[str, pd.DataFrame],
) -> dict[str, dict[str, list]]:
    return {
        split_name: {
            "ids": [str(value) for value in dataframe["id"].tolist()],
            "target": [int(value) for value in dataframe["Response"].tolist()],
        }
        for split_name, dataframe in splits.items()
    }


def test_split_comparison_requires_id_sets_and_order_to_match():
    reference_frames = _split_frames()
    candidate_frames = _split_frames(reorder_test=True)
    certification = _certification_for_splits(
        candidate_frames,
        _reference_split_payload(reference_frames),
    )

    certification._compare_split_ids_and_targets()

    assert certification.checks["split_ids"]["split_id_sets_match"] is True
    assert certification.checks["split_ids"]["prediction_order_match"] is False
    assert certification.checks["split_ids"]["passed"] is False
    assert certification.checks["target_order"]["passed"] is True


def test_split_comparison_rejects_duplicate_candidate_ids():
    candidate_frames = _split_frames()
    candidate_frames["test"] = pd.DataFrame({"id": [5, 5], "Response": [0, 1]})
    certification = _certification_for_splits(
        candidate_frames,
        _reference_split_payload(_split_frames()),
    )

    with pytest.raises(ValueError, match="duplicate IDs"):
        certification._compare_split_ids_and_targets()


def test_certificate_status_requires_every_parity_check_to_pass(tmp_path):
    certification = SourceFrozenParityCertification.__new__(
        SourceFrozenParityCertification
    )
    check_names = (
        "dataset_snapshot",
        "split_ids",
        "target_order",
        "raw_feature_order",
        "transformed_feature_order",
        "reference_model_identity",
        "model_parameters",
        "transformed_values",
        "prediction_array",
        "metrics",
    )
    certification.checks = {check_name: {"passed": True} for check_name in check_names}
    certification.errors = []
    certification.warnings = []
    certification.candidate_artifact_dir = tmp_path
    certification.candidate_lineage = {"mlflow_run_id": "candidate-run"}
    certification.candidate_metrics_payload = {}
    certification.reference_run_id = "reference-run"
    certification.reference_model_uri = "models:/reference-model"
    certification.source_data_path = tmp_path / "data.csv"
    certification.prediction_tolerance = 1.0e-12
    certification.metric_tolerance = 1.0e-6
    certification.apply_lineage = False

    verified_report = certification._build_report()
    assert verified_report["status"] == "verified"

    certification.checks["prediction_array"]["passed"] = False
    failed_report = certification._build_report()
    assert failed_report["status"] == "failed"


def test_certificate_error_prevents_verified_status(tmp_path):
    certification = SourceFrozenParityCertification.__new__(
        SourceFrozenParityCertification
    )
    certification.checks = {
        check_name: {"passed": True}
        for check_name in (
            "dataset_snapshot",
            "split_ids",
            "target_order",
            "raw_feature_order",
            "transformed_feature_order",
            "reference_model_identity",
            "model_parameters",
            "transformed_values",
            "prediction_array",
            "metrics",
        )
    }
    certification.errors = [{"stage": "test", "message": "failure"}]
    certification.warnings = []
    certification.candidate_artifact_dir = tmp_path
    certification.candidate_lineage = {}
    certification.candidate_metrics_payload = {}
    certification.reference_run_id = "reference-run"
    certification.reference_model_uri = "models:/reference-model"
    certification.source_data_path = tmp_path / "data.csv"
    certification.prediction_tolerance = 1.0e-12
    certification.metric_tolerance = 1.0e-6
    certification.apply_lineage = False

    report = certification._build_report()

    assert report["status"] == "failed"
    assert report["errors"][0]["stage"] == "test"


def test_mlflow_evidence_logs_pending_then_complete_payload_snapshots(
    monkeypatch, tmp_path
):
    certification = SourceFrozenParityCertification.__new__(
        SourceFrozenParityCertification
    )
    certification.candidate_lineage = {}
    certification.errors = []
    certification.warnings = []
    certification.reference_run_id = "reference-run"
    certification.report_path = tmp_path / "parity_report.json"
    certification.source_data_path = tmp_path / "data.csv"
    certification.prediction_tolerance = 1.0e-12
    certification.metric_tolerance = 1.0e-6
    certification.apply_lineage = True
    certification.log_mlflow_evidence = True
    certification.certification_attempt_id = "attempt-1"
    certification.candidate_identity = {"model_id": "candidate-model"}

    captured = {"reports": [], "artifact_files": [], "tags": []}

    class _FakeMlflow:
        def set_tracking_uri(self, value):
            captured["tracking_uri"] = value

        def start_run(self, *, run_id):
            captured["run_id"] = run_id
            return nullcontext()

        def log_dict(self, value, artifact_file):
            captured["reports"].append(deepcopy(value))
            captured["artifact_files"].append(artifact_file)

        def set_tag(self, key, value):
            captured["tags"].append((key, value))

    monkeypatch.setattr(
        "vehicle_insurance.verification.source_frozen_parity.mlflow",
        _FakeMlflow(),
    )
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "file:./mlartifacts")

    report = {
        "candidate_run_id": "candidate-run",
        "lineage_recording": {
            "requested": True,
            "local_metrics_json_updated": True,
            "mlflow_evidence_logged": False,
        },
    }

    certification._log_mlflow_evidence(report)
    report["status"] = "verified"
    report["lineage_recording"]["recording_state"] = "complete"
    certification._log_mlflow_evidence(report, finalize=True)

    assert captured["tracking_uri"] == "file:./mlartifacts"
    assert captured["run_id"] == "candidate-run"
    assert captured["artifact_files"] == [
        "lineage/source_frozen_parity.json"
    ] * 4
    assert [payload["status"] for payload in captured["reports"]] == [
        "pending",
        "pending",
        "verified",
        "verified",
    ]
    recording = [payload["lineage_recording"] for payload in captured["reports"]]
    assert [item["recording_state"] for item in recording] == [
        "remote_recorded_pending_local_commit",
        "remote_recorded_pending_local_commit",
        "complete",
        "complete",
    ]
    assert [item["mlflow_evidence_logged"] for item in recording] == [
        False,
        True,
        False,
        True,
    ]
    assert [item["mlflow_tags_recorded"] for item in recording] == [
        False,
        True,
        False,
        True,
    ]
    assert report["lineage_recording"]["mlflow_evidence_logged"] is True
    assert [
        value
        for key, value in captured["tags"]
        if key == "source_frozen_parity"
    ] == ["pending", "verified"]
    for key, expected in {
        "source_frozen_parity_certificate_version": "1.0.0",
        "source_frozen_parity_certificate_attempt_id": "attempt-1",
        "source_frozen_parity_candidate_model_id": "candidate-model",
        "source_frozen_parity_recording_mode": "local_and_mlflow",
    }.items():
        assert [value for tag, value in captured["tags"] if tag == key] == [
            expected,
            expected,
        ]
