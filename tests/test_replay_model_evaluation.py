import hashlib
import json
from pathlib import Path

import pytest

from vehicle_insurance.verification import replay_model_evaluation as replay_module


def _metrics_payload() -> dict:
    return {
        "mlflow_run_id": "candidate-run",
        "model_id": "candidate-model",
        "model_uri": "models:/candidate-model",
        "registered_model_name": None,
        "registered_model_version": None,
        "train_metrics": {"pr_auc": 0.40},
        "validation_metrics": {"pr_auc": 0.39},
        "test_metrics": {"pr_auc": 0.38},
        "stability_metrics": {"required": True, "evaluation_count": 9},
        "lineage": {
            "source_frozen_parity": {"status": "verified"},
        },
    }


def _write_candidate_artifact(root: Path) -> None:
    (root / "model_trainer").mkdir(parents=True)
    (root / "model_evaluation").mkdir(parents=True)
    (root / "model_trainer" / "metrics.json").write_text(
        json.dumps(_metrics_payload()),
        encoding="utf-8",
    )
    (root / "model_trainer" / "model.pkl").write_bytes(b"model")
    (root / "model_evaluation" / "report.json").write_text(
        "{}",
        encoding="utf-8",
    )


def test_load_model_trainer_artifact_reconstructs_typed_artifact(tmp_path):
    _write_candidate_artifact(tmp_path)

    artifact = replay_module.load_model_trainer_artifact(tmp_path)

    assert artifact.run_id == "candidate-run"
    assert artifact.model_id == "candidate-model"
    assert artifact.validation_metrics == {"pr_auc": 0.39}
    assert artifact.lineage["source_frozen_parity"]["status"] == "verified"
    assert artifact.trained_model_file_path == str(
        tmp_path / "model_trainer" / "model.pkl"
    )


def test_loader_rejects_missing_model_file(tmp_path):
    (tmp_path / "model_trainer").mkdir(parents=True)
    (tmp_path / "model_trainer" / "metrics.json").write_text(
        json.dumps(_metrics_payload()),
        encoding="utf-8",
    )

    with pytest.raises(replay_module.ReplayArtifactError, match="model file"):
        replay_module.load_model_trainer_artifact(tmp_path)


def test_loader_rejects_missing_structural_metrics_section(tmp_path):
    _write_candidate_artifact(tmp_path)
    payload = _metrics_payload()
    payload.pop("stability_metrics")
    (tmp_path / "model_trainer" / "metrics.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(
        replay_module.ReplayArtifactError,
        match="stability_metrics",
    ):
        replay_module.load_model_trainer_artifact(tmp_path)


def test_replay_context_records_source_and_previous_report_hashes(tmp_path):
    _write_candidate_artifact(tmp_path)
    artifact = replay_module.load_model_trainer_artifact(tmp_path)
    metrics_path = tmp_path / "model_trainer" / "metrics.json"
    previous_report = tmp_path / "model_evaluation" / "report.json"
    metrics_bytes = metrics_path.read_bytes()
    report_bytes = previous_report.read_bytes()

    context = replay_module._build_evaluation_context(tmp_path, artifact)

    assert context["mode"] == "replay"
    assert context["source_metrics_sha256"] == hashlib.sha256(metrics_bytes).hexdigest()
    assert Path(context["previous_evaluation_report"]) == previous_report.resolve()
    assert context["previous_evaluation_report_sha256"] == hashlib.sha256(
        report_bytes
    ).hexdigest()

    previous_report.write_bytes(b'{"updated":true}')
    updated_context = replay_module._build_evaluation_context(tmp_path, artifact)
    assert updated_context["source_metrics_sha256"] == context["source_metrics_sha256"]
    assert updated_context["previous_evaluation_report_sha256"] == hashlib.sha256(
        previous_report.read_bytes()
    ).hexdigest()
    assert (
        updated_context["previous_evaluation_report_sha256"]
        != context["previous_evaluation_report_sha256"]
    )
