"""Persisted-state regression tests for source-frozen parity completion."""

from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

from vehicle_insurance.components.modelEvaluation import ModelEvaluation
from vehicle_insurance.entities.config_entity import ModelEvaluationConfig
from vehicle_insurance.verification.replay_model_evaluation import (
    load_model_trainer_artifact,
)
from vehicle_insurance.verification.source_frozen_parity import (
    SourceFrozenParityCertification,
)

_CHECK_NAMES = (
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


def _candidate(tmp_path: Path) -> Path:
    root = tmp_path / "candidate"
    trainer = root / "model_trainer"
    trainer.mkdir(parents=True)
    (trainer / "model.pkl").write_bytes(b"candidate-model")
    payload = {
        "mlflow_run_id": "candidate-run",
        "model_id": "candidate-model",
        "model_uri": "models:/candidate-model",
        "train_metrics": {},
        "validation_metrics": {},
        "test_metrics": {},
        "stability_metrics": {},
        "lineage": {
            "mlflow_run_id": "candidate-run",
            "model_id": "candidate-model",
            "model_uri": "models:/candidate-model",
            "dataset_snapshot_id": "snapshot-1",
            "dataset_content_hash": "d" * 64,
            "source_frozen_parity": {"status": "not_verified"},
        },
    }
    (trainer / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")
    return root


def _certification(
    root: Path,
    *,
    remote: bool = False,
    report_path: Path | None = None,
):
    certification = SourceFrozenParityCertification(
        candidate_artifact_dir=root,
        apply_lineage=True,
        log_mlflow_evidence=remote,
        report_path=report_path,
    )
    certification._validate_configuration = lambda: None
    certification._load_candidate_artifacts = lambda: None
    certification._reconstruct_notebook_splits = lambda: None
    certification._compare_split_ids_and_targets = lambda: None
    certification._load_reference_model = lambda: None
    certification._compare_models_and_predictions = lambda: None
    certification._compare_metrics = lambda: None
    certification.checks = {name: {"passed": True} for name in _CHECK_NAMES}
    certification.checks["split_ids"].update(
        {"split_id_sets_match": True, "prediction_order_match": True}
    )
    certification.checks["prediction_array"].update({"maximum_absolute_error": 0.0})
    certification.checks["transformed_values"].update({"maximum_absolute_error": 0.0})
    model_path = root / "model_trainer" / "model.pkl"
    certification.candidate_metrics_payload = {
        "mlflow_run_id": "candidate-run",
        "model_id": "candidate-model",
        "model_uri": "models:/candidate-model",
    }
    certification.candidate_lineage = {
        "mlflow_run_id": "candidate-run",
        "model_id": "candidate-model",
        "model_uri": "models:/candidate-model",
        "dataset_snapshot_id": "snapshot-1",
        "dataset_content_hash": "d" * 64,
    }
    certification.candidate_model_path = model_path
    certification.candidate_identity = {
        "mlflow_run_id": "candidate-run",
        "model_id": "candidate-model",
        "model_uri": "models:/candidate-model",
        "dataset_snapshot_id": "snapshot-1",
        "dataset_content_hash": "d" * 64,
        "model_file_sha256": certification._sha256_file(model_path),
    }
    return certification


def _parity_gate(root: Path):
    artifact = load_model_trainer_artifact(root)
    evaluator = ModelEvaluation(
        ModelEvaluationConfig(report_file_path=str(root / "gate.json")),
        artifact,
    )
    return evaluator._check_source_frozen_parity(artifact.lineage)


def _patch_remote_mlflow(monkeypatch, *, log_dict=None, set_tag=None):
    def start_run(*, run_id):
        assert run_id == "candidate-run"
        return nullcontext()

    monkeypatch.setattr(
        "vehicle_insurance.verification.source_frozen_parity.mlflow",
        SimpleNamespace(
            set_tracking_uri=lambda _value: None,
            start_run=start_run,
            log_dict=log_dict or (lambda *_args: None),
            set_tag=set_tag or (lambda *_args: None),
        ),
    )
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "file:./mlartifacts")


def _run_remote_success(root: Path, monkeypatch):
    certification = _certification(root, remote=True)
    _patch_remote_mlflow(monkeypatch)
    return certification.run()


def test_complete_success_is_persisted_and_replay_gate_accepts(tmp_path, monkeypatch):
    root = _candidate(tmp_path)
    report = _run_remote_success(root, monkeypatch)

    persisted_metrics = json.loads(
        (root / "model_trainer" / "metrics.json").read_text(encoding="utf-8")
    )
    persisted_report = json.loads(
        (root / "source_frozen_parity" / "parity_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["status"] == "verified"
    assert persisted_metrics["lineage"]["source_frozen_parity"]["status"] == "verified"
    assert (
        persisted_metrics["lineage"]["source_frozen_parity"]["recording_state"]
        == "complete"
    )
    assert persisted_report["status"] == "verified"
    assert _parity_gate(root)["passed"] is True


def test_fresh_custom_certificate_completes_pending_to_final(tmp_path):
    root = _candidate(tmp_path)
    certificate = root / "source_frozen_parity" / "fresh-certificate.json"
    certification = _certification(root, report_path=certificate)

    report = certification.run()

    persisted_report = json.loads(certificate.read_text(encoding="utf-8"))
    assert report["status"] == "verified"
    assert persisted_report["status"] == "verified"
    assert persisted_report["lineage_recording"]["recording_state"] == "complete"


def test_local_write_failure_leaves_replay_rejected(tmp_path):
    root = _candidate(tmp_path)
    certification = _certification(root)

    def fail_pending(_report, recording_state="pending"):
        if recording_state == "pending":
            raise OSError("metrics storage is unavailable")
        raise AssertionError("complete commit must not be attempted")

    certification._apply_local_lineage = fail_pending
    report = certification.run()

    assert report["status"] == "failed"
    assert (
        json.loads(
            (root / "model_trainer" / "metrics.json").read_text(encoding="utf-8")
        )["lineage"]["source_frozen_parity"]["status"]
        == "not_verified"
    )
    assert _parity_gate(root)["passed"] is False


def test_remote_tag_failure_leaves_pending_metrics_and_replay_rejected(
    tmp_path, monkeypatch
):
    root = _candidate(tmp_path)
    certification = _certification(root, remote=True)

    def fail_version_tag(key, _value):
        if key.endswith("certificate_version"):
            raise OSError("remote tag write failed")

    _patch_remote_mlflow(monkeypatch, set_tag=fail_version_tag)

    report = certification.run()

    evidence = json.loads(
        (root / "model_trainer" / "metrics.json").read_text(encoding="utf-8")
    )["lineage"]["source_frozen_parity"]
    assert report["status"] == "failed"
    assert evidence["status"] == "pending"
    assert evidence["recording_state"] == "pending"
    assert _parity_gate(root)["passed"] is False


def test_remote_log_failure_leaves_pending_metrics_and_replay_rejected(
    tmp_path, monkeypatch
):
    root = _candidate(tmp_path)
    certification = _certification(root, remote=True)

    def fail_log(_value, _artifact_file):
        raise OSError("remote artifact write failed")

    def unexpected_tag(*_args):
        raise AssertionError("tags must not run after a failed remote log")

    _patch_remote_mlflow(
        monkeypatch,
        log_dict=fail_log,
        set_tag=unexpected_tag,
    )

    report = certification.run()

    evidence = json.loads(
        (root / "model_trainer" / "metrics.json").read_text(encoding="utf-8")
    )["lineage"]["source_frozen_parity"]
    assert report["status"] == "failed"
    assert evidence["status"] == "pending"
    assert _parity_gate(root)["passed"] is False


def test_final_remote_finalization_failure_leaves_pending_metrics(
    tmp_path, monkeypatch
):
    root = _candidate(tmp_path)
    certification = _certification(root, remote=True)

    def fail_verified_payload(value, _artifact_file):
        if value["status"] == "verified":
            raise OSError("final remote certificate write failed")

    _patch_remote_mlflow(monkeypatch, log_dict=fail_verified_payload)

    report = certification.run()

    evidence = json.loads(
        (root / "model_trainer" / "metrics.json").read_text(encoding="utf-8")
    )["lineage"]["source_frozen_parity"]
    assert report["status"] == "failed"
    assert evidence["status"] == "pending"
    assert _parity_gate(root)["passed"] is False


def test_final_certificate_write_failure_leaves_pending_metrics(tmp_path):
    root = _candidate(tmp_path)
    certification = _certification(root)
    original_write = certification._write_report

    def fail_final(report):
        if report.get("status") == "verified":
            raise OSError("certificate storage is unavailable")
        original_write(report)

    certification._write_report = fail_final
    report = certification.run()

    persisted_metrics = json.loads(
        (root / "model_trainer" / "metrics.json").read_text(encoding="utf-8")
    )
    persisted_report = json.loads(
        (root / "source_frozen_parity" / "parity_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["status"] == "failed"
    assert persisted_metrics["lineage"]["source_frozen_parity"]["status"] == "pending"
    assert persisted_report["status"] == "failed"
    assert _parity_gate(root)["passed"] is False


def test_local_only_success_does_not_satisfy_remote_required_policy(tmp_path):
    root = _candidate(tmp_path)
    certification = _certification(root, remote=False)

    report = certification.run()

    evidence = json.loads(
        (root / "model_trainer" / "metrics.json").read_text(encoding="utf-8")
    )["lineage"]["source_frozen_parity"]
    assert report["status"] == "verified"
    assert evidence["status"] == "verified"
    assert evidence["recording_mode"] == "local_only"
    gate = _parity_gate(root)
    assert gate["passed"] is False
    assert "source_frozen_recording_mode_mismatch" in gate["reason_codes"]


def test_newer_attempt_cannot_reuse_an_older_certificate(tmp_path, monkeypatch):
    root = _candidate(tmp_path)
    _run_remote_success(root, monkeypatch)
    metrics_path = root / "model_trainer" / "metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    payload["lineage"]["source_frozen_parity"]["certification_attempt_id"] = (
        "new-failed-attempt"
    )
    metrics_path.write_text(json.dumps(payload), encoding="utf-8")

    gate = _parity_gate(root)
    assert gate["passed"] is False
    assert "source_frozen_certificate_attempt_mismatch" in gate["reason_codes"]


def test_certificate_hash_mutation_is_rejected_after_reload(tmp_path, monkeypatch):
    root = _candidate(tmp_path)
    _run_remote_success(root, monkeypatch)
    certificate_path = root / "source_frozen_parity" / "parity_report.json"
    certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
    certificate["status"] = "pending"
    certificate_path.write_text(json.dumps(certificate), encoding="utf-8")

    gate = _parity_gate(root)
    assert gate["passed"] is False
    assert "source_frozen_certificate_hash_mismatch" in gate["reason_codes"]


def test_certificate_path_traversal_is_rejected_after_reload(tmp_path, monkeypatch):
    root = _candidate(tmp_path)
    _run_remote_success(root, monkeypatch)
    metrics_path = root / "model_trainer" / "metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    payload["lineage"]["source_frozen_parity"]["certificate_report_path"] = (
        "../outside.json"
    )
    metrics_path.write_text(json.dumps(payload), encoding="utf-8")

    gate = _parity_gate(root)
    assert gate["passed"] is False
    assert "source_frozen_certificate_path_not_relocation_safe" in gate["reason_codes"]


def test_relative_certificate_reference_survives_artifact_relocation(
    tmp_path, monkeypatch
):
    original_root = _candidate(tmp_path)
    _run_remote_success(original_root, monkeypatch)
    relocated_root = tmp_path / "relocated_candidate"
    original_root.rename(relocated_root)

    assert _parity_gate(relocated_root)["passed"] is True


def test_older_attempt_cannot_commit_after_newer_pending_attempt(tmp_path):
    root = _candidate(tmp_path)
    older = _certification(root)
    newer = _certification(root)
    older_report = older._build_report()
    newer_report = newer._build_report()

    older._apply_local_lineage(older_report, "pending")
    newer._apply_local_lineage(newer_report, "pending")

    try:
        older._apply_local_lineage(older_report, "complete")
    except ValueError as error:
        assert "superseded" in str(error)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("an older certification attempt committed unexpectedly")
