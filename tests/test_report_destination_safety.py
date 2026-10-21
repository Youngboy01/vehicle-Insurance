from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import vehicle_insurance.utils.report_output as report_output
from vehicle_insurance.components.modelEvaluation import ModelEvaluation
from vehicle_insurance.entities.artifact_entity import ModelTrainerArtifact
from vehicle_insurance.entities.config_entity import ModelEvaluationConfig
from vehicle_insurance.exceptions.handle_exception import VehicleInsuranceException
from vehicle_insurance.utils.report_output import ReportDestinationError
from vehicle_insurance.verification import replay_model_evaluation as replay_module
from vehicle_insurance.verification.source_frozen_parity import (
    SourceFrozenParityCertification,
)


def _candidate(root: Path) -> dict[str, Path]:
    paths = {
        "model": root / "model_trainer" / "model.pkl",
        "metrics": root / "model_trainer" / "metrics.json",
        "source": root / "data_ingestion" / "feature_store" / "data.csv",
        "manifest": root / "data_ingestion" / "feature_store" / "source_manifest.json",
        "certificate": root / "source_frozen_parity" / "parity_report.json",
        "decision": root / "model_evaluation" / "report.json",
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    paths["model"].write_bytes(b"immutable model bytes")
    paths["metrics"].write_text(
        json.dumps(
            {
                "mlflow_run_id": "candidate-run",
                "model_id": "candidate-model",
                "model_uri": "models:/candidate-model",
                "train_metrics": {},
                "validation_metrics": {},
                "test_metrics": {},
                "stability_metrics": {},
                "lineage": {"source_frozen_parity": {"status": "verified"}},
            }
        ),
        encoding="utf-8",
    )
    paths["source"].write_bytes(b"immutable source snapshot")
    paths["manifest"].write_bytes(b"immutable source manifest")
    paths["certificate"].write_bytes(b"immutable parity certificate")
    paths["decision"].write_bytes(b"immutable original decision")
    return paths


def _fake_evaluator(monkeypatch):
    class _Evaluator:
        def __init__(self, *, model_evaluation_config, **_kwargs):
            self.config = model_evaluation_config

        def evaluate_model(self):
            output = Path(self.config.report_file_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"replay report")
            return SimpleNamespace(
                evaluation_passed=False,
                decision="rejected",
                evaluation_report_file_path=str(output),
            )

    monkeypatch.setattr(replay_module, "ModelEvaluation", _Evaluator)


@pytest.mark.parametrize(
    "protected_name",
    ("model", "metrics", "source", "manifest", "decision"),
)
def test_replay_rejects_protected_destination_before_evaluator(
    tmp_path,
    monkeypatch,
    protected_name,
):
    paths = _candidate(tmp_path)

    class _UnexpectedEvaluator:
        def __init__(self, **_kwargs):  # pragma: no cover - defensive assertion
            raise AssertionError("destination must be rejected before evaluation")

    monkeypatch.setattr(replay_module, "ModelEvaluation", _UnexpectedEvaluator)
    original_bytes = paths[protected_name].read_bytes()

    with pytest.raises(VehicleInsuranceException):
        replay_module.replay_model_evaluation(
            tmp_path,
            report_file_path=paths[protected_name],
        )

    assert paths[protected_name].read_bytes() == original_bytes


def test_parity_writer_rejects_existing_noncanonical_certificate(tmp_path):
    paths = _candidate(tmp_path)
    old_certificate = paths["certificate"].with_name("older_certificate.json")
    old_certificate.write_bytes(b"older immutable certificate")
    certification = _parity_writer(tmp_path, old_certificate)
    original_bytes = old_certificate.read_bytes()

    with pytest.raises(ReportDestinationError):
        certification._write_report({"attempt": "unsafe"})

    assert old_certificate.read_bytes() == original_bytes


def test_replay_rejects_same_file_hard_link_alias(tmp_path, monkeypatch):
    paths = _candidate(tmp_path)
    alias = tmp_path / "outside-model-alias.pkl"
    try:
        os.link(paths["model"], alias)
    except OSError as error:  # pragma: no cover - platform/filesystem dependent
        pytest.skip(f"hard links unavailable: {error}")
    _fake_evaluator(monkeypatch)
    original_bytes = paths["model"].read_bytes()

    with pytest.raises(VehicleInsuranceException):
        replay_module.replay_model_evaluation(
            tmp_path,
            report_file_path=alias,
        )

    assert paths["model"].read_bytes() == original_bytes
    assert alias.read_bytes() == original_bytes


def test_replay_rejects_symbolic_link_alias_without_touching_target(
    tmp_path, monkeypatch
):
    paths = _candidate(tmp_path)
    alias = tmp_path / "outside-model-link.pkl"
    try:
        alias.symlink_to(paths["model"])
    except OSError as error:  # pragma: no cover - platform/filesystem dependent
        pytest.skip(f"symbolic links unavailable: {error}")
    _fake_evaluator(monkeypatch)
    original_bytes = paths["model"].read_bytes()

    with pytest.raises(VehicleInsuranceException):
        replay_module.replay_model_evaluation(
            tmp_path,
            report_file_path=alias,
        )

    assert paths["model"].read_bytes() == original_bytes
    assert alias.is_symlink()


def test_replay_second_validation_protects_custom_external_certificate(
    tmp_path,
    monkeypatch,
):
    paths = _candidate(tmp_path)
    external_certificate = tmp_path / "external_certificate.json"
    payload = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    payload["lineage"]["source_frozen_parity"]["certificate_report_path"] = str(
        external_certificate
    )
    paths["metrics"].write_text(json.dumps(payload), encoding="utf-8")

    class _UnexpectedEvaluator:
        def __init__(self, **_kwargs):  # pragma: no cover - defensive assertion
            raise AssertionError("dynamic certificate alias must be rejected")

    constructions = []

    class GuardedEvaluator(_UnexpectedEvaluator):
        def __init__(self, **kwargs):
            constructions.append(kwargs)
            super().__init__(**kwargs)

    monkeypatch.setattr(replay_module, "ModelEvaluation", GuardedEvaluator)

    with pytest.raises(VehicleInsuranceException) as error_info:
        replay_module.replay_model_evaluation(
            tmp_path,
            report_file_path=external_certificate,
        )

    assert isinstance(error_info.value.__cause__, ReportDestinationError)
    assert constructions == []
    assert not external_certificate.exists()


def test_replay_default_and_explicit_reports_are_exclusive(tmp_path, monkeypatch):
    _candidate(tmp_path)
    _fake_evaluator(monkeypatch)

    first = replay_module.replay_model_evaluation(tmp_path)
    default_report = tmp_path / "model_evaluation" / "replay_report.json"
    assert first.evaluation_report_file_path == str(default_report)
    original_bytes = default_report.read_bytes()

    with pytest.raises(VehicleInsuranceException):
        replay_module.replay_model_evaluation(tmp_path)
    assert default_report.read_bytes() == original_bytes

    explicit = tmp_path / "model_evaluation" / "fresh_replay.json"
    replay_module.replay_model_evaluation(tmp_path, report_file_path=explicit)
    explicit_bytes = explicit.read_bytes()
    with pytest.raises(VehicleInsuranceException):
        replay_module.replay_model_evaluation(tmp_path, report_file_path=explicit)
    assert explicit.read_bytes() == explicit_bytes


def _evaluation_artifact(root: Path) -> ModelTrainerArtifact:
    paths = _candidate(root)
    return ModelTrainerArtifact(
        trained_model_file_path=str(paths["model"]),
        metrics_file_path=str(paths["metrics"]),
        train_metrics={},
        validation_metrics={},
        test_metrics={},
        run_id="candidate-run",
        model_id="candidate-model",
        model_uri="models:/candidate-model",
        registered_model_name=None,
        registered_model_version=None,
        stability_metrics={},
        lineage={"source_frozen_parity": {"status": "verified"}},
    )


def test_evaluator_writer_rejects_protected_inputs_and_overwrites_ordinary_report(
    tmp_path,
):
    artifact = _evaluation_artifact(tmp_path)
    protected_bytes = Path(artifact.trained_model_file_path).read_bytes()
    evaluator = ModelEvaluation(
        ModelEvaluationConfig(report_file_path=str(tmp_path / "safe-report.json")),
        artifact,
    )
    evaluator.model_evaluation_config.report_file_path = (
        artifact.trained_model_file_path
    )

    with pytest.raises(VehicleInsuranceException):
        evaluator._save_report({"attempt": "unsafe"})
    assert Path(artifact.trained_model_file_path).read_bytes() == protected_bytes

    ordinary_report = tmp_path / "ordinary-report.json"
    evaluator = ModelEvaluation(
        ModelEvaluationConfig(report_file_path=str(ordinary_report)),
        artifact,
    )
    evaluator._save_report({"attempt": 1})
    evaluator._save_report({"attempt": 2})
    assert json.loads(ordinary_report.read_text(encoding="utf-8")) == {"attempt": 2}


def test_evaluator_normal_decision_report_is_explicitly_replaceable(tmp_path):
    artifact = _evaluation_artifact(tmp_path)
    decision_report = tmp_path / "model_evaluation" / "report.json"
    original_bytes = decision_report.read_bytes()
    evaluator = ModelEvaluation(
        ModelEvaluationConfig(report_file_path=str(decision_report)),
        artifact,
    )

    evaluator._save_report({"decision": "new"})

    assert decision_report.read_bytes() != original_bytes
    assert json.loads(decision_report.read_text(encoding="utf-8")) == {
        "decision": "new"
    }


def test_evaluator_rejects_decision_path_hard_linked_to_model(tmp_path):
    artifact = _evaluation_artifact(tmp_path)
    decision_report = tmp_path / "model_evaluation" / "report.json"
    model_path = Path(artifact.trained_model_file_path)
    decision_report.unlink()
    try:
        os.link(model_path, decision_report)
    except OSError as error:  # pragma: no cover - platform/filesystem dependent
        pytest.skip(f"hard links unavailable: {error}")

    evaluator = ModelEvaluation(
        ModelEvaluationConfig(report_file_path=str(tmp_path / "safe.json")),
        artifact,
    )
    original_model = model_path.read_bytes()
    evaluator.model_evaluation_config.report_file_path = str(decision_report)

    with pytest.raises(VehicleInsuranceException):
        evaluator._save_report({"unsafe": True})

    assert model_path.read_bytes() == original_model
    assert decision_report.read_bytes() == original_model


@pytest.mark.parametrize("link_kind", ("hard", "symbolic"))
def test_evaluator_writer_does_not_follow_predictable_temp_alias(tmp_path, link_kind):
    artifact = _evaluation_artifact(tmp_path)
    output = tmp_path / "ordinary-report.json"
    temporary_alias = output.with_suffix(output.suffix + ".tmp")
    model_path = Path(artifact.trained_model_file_path)
    original_model = model_path.read_bytes()
    try:
        if link_kind == "hard":
            os.link(model_path, temporary_alias)
        else:
            temporary_alias.symlink_to(model_path)
    except OSError as error:  # pragma: no cover - platform/filesystem dependent
        pytest.skip(f"{link_kind} links unavailable: {error}")

    evaluator = ModelEvaluation(
        ModelEvaluationConfig(report_file_path=str(output)),
        artifact,
    )
    evaluator._save_report({"safe": True})

    assert model_path.read_bytes() == original_model
    if link_kind == "hard":
        assert temporary_alias.read_bytes() == original_model
    else:
        assert temporary_alias.is_symlink()
        assert temporary_alias.read_bytes() == original_model


def _parity_writer(root: Path, destination: Path) -> SourceFrozenParityCertification:
    certification = SourceFrozenParityCertification.__new__(
        SourceFrozenParityCertification
    )
    certification.candidate_artifact_dir = root
    certification.report_path = destination
    certification.candidate_lineage = {}
    certification.apply_lineage = False
    certification.log_mlflow_evidence = False
    return certification


def test_parity_run_validates_destination_before_candidate_stages(tmp_path):
    model_path = tmp_path / "model_trainer" / "model.pkl"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"immutable model")
    certification = SourceFrozenParityCertification(
        candidate_artifact_dir=tmp_path,
        report_path=model_path,
    )

    def unexpected_stage():
        raise AssertionError("candidate stages must not run")

    certification._validate_configuration = unexpected_stage
    original_bytes = model_path.read_bytes()

    with pytest.raises(ReportDestinationError):
        certification.run()

    assert model_path.read_bytes() == original_bytes


@pytest.mark.parametrize(
    "protected_name",
    ("model", "metrics", "source", "manifest", "decision"),
)
def test_parity_writer_rejects_protected_inputs_without_mutation(
    tmp_path, protected_name
):
    paths = _candidate(tmp_path)
    certification = _parity_writer(tmp_path, paths[protected_name])
    original_bytes = paths[protected_name].read_bytes()

    with pytest.raises(ReportDestinationError):
        certification._write_report({"attempt": "unsafe"})

    assert paths[protected_name].read_bytes() == original_bytes


def test_parity_writer_uses_unique_temp_and_allows_ordinary_replacement(tmp_path):
    paths = _candidate(tmp_path)
    original_model = paths["model"].read_bytes()
    original_metrics = paths["metrics"].read_bytes()
    temporary_alias = paths["certificate"].with_suffix(
        paths["certificate"].suffix + ".tmp"
    )
    temporary_alias.write_bytes(b"pre-existing temporary evidence")
    certification = _parity_writer(tmp_path, paths["certificate"])

    certification._write_report({"attempt": 1})
    assert json.loads(paths["certificate"].read_text(encoding="utf-8")) == {
        "attempt": 1
    }
    assert paths["model"].read_bytes() == original_model
    assert paths["metrics"].read_bytes() == original_metrics
    assert temporary_alias.read_bytes() == b"pre-existing temporary evidence"

    certification._write_report({"attempt": 2})
    assert json.loads(paths["certificate"].read_text(encoding="utf-8")) == {
        "attempt": 2
    }
    assert paths["model"].read_bytes() == original_model
    assert paths["metrics"].read_bytes() == original_metrics


def test_parity_writer_rejects_existing_lineage_certificate_after_initial_check(
    tmp_path,
):
    paths = _candidate(tmp_path)
    certificate = tmp_path / "custom-certificate.json"
    original = b"immutable lineage certificate"
    certificate.write_bytes(original)
    payload = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    payload["lineage"]["source_frozen_parity"]["certificate_report_path"] = (
        certificate.name
    )
    paths["metrics"].write_text(json.dumps(payload), encoding="utf-8")

    certification = _parity_writer(tmp_path, certificate)
    certification._validate_report_destination()
    certification.candidate_lineage = payload["lineage"]

    with pytest.raises(ReportDestinationError):
        certification._write_report({"status": "replacement"})

    assert certificate.read_bytes() == original


@pytest.mark.parametrize("link_kind", ("hard", "symbolic"))
def test_parity_writer_does_not_follow_predictable_temp_alias(tmp_path, link_kind):
    paths = _candidate(tmp_path)
    temporary_alias = paths["certificate"].with_suffix(
        paths["certificate"].suffix + ".tmp"
    )
    original_model = paths["model"].read_bytes()
    try:
        if link_kind == "hard":
            os.link(paths["model"], temporary_alias)
        else:
            temporary_alias.symlink_to(paths["model"])
    except OSError as error:  # pragma: no cover - platform/filesystem dependent
        pytest.skip(f"{link_kind} links unavailable: {error}")

    certification = _parity_writer(tmp_path, paths["certificate"])
    certification._write_report({"attempt": "safe"})

    assert paths["model"].read_bytes() == original_model
    if link_kind == "hard":
        assert temporary_alias.read_bytes() == original_model
    else:
        assert temporary_alias.is_symlink()
        assert temporary_alias.read_bytes() == original_model


def test_parity_writer_rejects_hard_link_alias_to_model(tmp_path):
    paths = _candidate(tmp_path)
    alias = tmp_path / "parity-hard-link.pkl"
    try:
        os.link(paths["model"], alias)
    except OSError as error:  # pragma: no cover - platform/filesystem dependent
        pytest.skip(f"hard links unavailable: {error}")
    certification = _parity_writer(tmp_path, alias)
    original_bytes = paths["model"].read_bytes()

    with pytest.raises(ReportDestinationError):
        certification._write_report({"attempt": "unsafe"})

    assert paths["model"].read_bytes() == original_bytes
    assert alias.read_bytes() == original_bytes


def test_parity_writer_rejects_canonical_certificate_hard_linked_to_model(tmp_path):
    paths = _candidate(tmp_path)
    certificate = paths["certificate"]
    certificate.unlink()
    try:
        os.link(paths["model"], certificate)
    except OSError as error:  # pragma: no cover - platform/filesystem dependent
        pytest.skip(f"hard links unavailable: {error}")
    certification = _parity_writer(tmp_path, certificate)
    original_model = paths["model"].read_bytes()

    with pytest.raises(ReportDestinationError):
        certification._write_report({"attempt": "unsafe"})

    assert paths["model"].read_bytes() == original_model
    assert certificate.read_bytes() == original_model


def test_require_new_competing_writer_cannot_overwrite_winner(tmp_path, monkeypatch):
    destination = tmp_path / "new-report.json"
    original_link = report_output.os.link

    def competing_writer(source, target):
        Path(target).write_bytes(b'{"winner":"competing"}')
        return original_link(source, target)

    monkeypatch.setattr(report_output.os, "link", competing_writer)

    with pytest.raises(ReportDestinationError):
        report_output.write_json_atomic(
            destination,
            {"winner": "loser"},
            require_new=True,
        )

    assert destination.read_bytes() == b'{"winner":"competing"}'
    assert not list(destination.parent.glob(f".{destination.name}.*.tmp"))


def _lineage_update_fixture(root: Path) -> tuple[SourceFrozenParityCertification, dict]:
    paths = _candidate(root)
    metrics_path = paths["metrics"]
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    certification = SourceFrozenParityCertification.__new__(
        SourceFrozenParityCertification
    )
    certification.candidate_artifact_dir = root
    certification.report_path = paths["certificate"]
    certification.candidate_metrics_payload = payload
    certification.candidate_lineage = payload["lineage"]
    certification.reference_run_id = "reference-run"
    certification.reference_model_uri = "models:/reference-model"
    certification.configured_reference_model_id = "reference-model"
    certification.prediction_tolerance = 1.0e-12
    certification.apply_lineage = True
    certification.log_mlflow_evidence = False
    certification.certification_attempt_id = "attempt-1"
    certification.candidate_identity = {
        "mlflow_run_id": "candidate-run",
        "model_id": "candidate-model",
        "model_uri": "models:/candidate-model",
        "dataset_snapshot_id": "snapshot-1",
        "dataset_content_hash": "d" * 64,
        "model_file_sha256": "model-hash",
    }
    report = {
        "verified_at_utc": "2026-09-15T00:00:00+00:00",
        "checks": {
            "split_ids": {
                "split_id_sets_match": True,
                "prediction_order_match": True,
            },
            "prediction_array": {"maximum_absolute_error": 0.0},
            "transformed_feature_order": {"passed": True},
            "transformed_values": {"maximum_absolute_error": 0.0},
        },
        "lineage_recording": {
            "recording_mode": "local_only",
            "certification_attempt_id": "attempt-1",
            "candidate_identity": certification.candidate_identity,
        },
    }
    return certification, report


def test_parity_metrics_writer_does_not_follow_predictable_temp_alias(tmp_path):
    certification, report = _lineage_update_fixture(tmp_path)
    paths = _candidate(tmp_path)
    temporary_alias = paths["metrics"].with_suffix(paths["metrics"].suffix + ".tmp")
    original_model = paths["model"].read_bytes()
    try:
        os.link(paths["model"], temporary_alias)
    except OSError as error:  # pragma: no cover - platform/filesystem dependent
        pytest.skip(f"hard links unavailable: {error}")

    certification._apply_local_lineage(report, "pending")

    assert paths["model"].read_bytes() == original_model
    assert temporary_alias.read_bytes() == original_model


def test_parity_metrics_writer_rejects_metrics_entry_hard_linked_to_model(tmp_path):
    certification, report = _lineage_update_fixture(tmp_path)
    paths = _candidate(tmp_path)
    metrics = paths["metrics"]
    original_bytes = metrics.read_bytes()
    paths["model"].write_bytes(original_bytes)
    metrics.unlink()
    try:
        os.link(paths["model"], metrics)
    except OSError as error:  # pragma: no cover - platform/filesystem dependent
        pytest.skip(f"hard links unavailable: {error}")
    with pytest.raises(ReportDestinationError):
        certification._apply_local_lineage(report, "pending")

    assert paths["model"].read_bytes() == original_bytes
    assert metrics.read_bytes() == original_bytes
