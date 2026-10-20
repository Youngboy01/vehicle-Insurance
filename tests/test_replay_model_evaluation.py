import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from vehicle_insurance.entities.artifact_entity import ModelEvaluationArtifact
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


def test_replay_delegates_to_evaluator_and_preserves_previous_report(
    tmp_path,
    monkeypatch,
):
    _write_candidate_artifact(tmp_path)
    captured = {}

    class _FakeEvaluator:
        def __init__(
            self, *, model_evaluation_config, model_trainer_artifact, evaluation_context
        ):
            captured["config"] = model_evaluation_config
            captured["artifact"] = model_trainer_artifact
            captured["context"] = evaluation_context

        def evaluate_model(self):
            return ModelEvaluationArtifact(
                evaluation_passed=True,
                decision="eligible_but_promotion_disabled",
                promotion_status="disabled",
                promotion_performed=False,
                candidate_run_id="candidate-run",
                candidate_model_id="candidate-model",
                candidate_model_uri="models:/candidate-model",
                champion_run_id=None,
                champion_model_version=None,
                candidate_validation_metrics={"pr_auc": 0.39},
                champion_validation_metrics=None,
                candidate_test_metrics={"pr_auc": 0.38},
                stability_metrics={"evaluation_count": 9},
                metric_comparison={},
                gate_results={},
                reason_codes=[],
                lineage={"source_frozen_parity": {"status": "verified"}},
                evaluation_report_file_path=captured["config"].report_file_path,
            )

    monkeypatch.setattr(replay_module, "ModelEvaluation", _FakeEvaluator)

    result = replay_module.replay_model_evaluation(tmp_path)

    assert result.evaluation_passed is True
    assert captured["context"]["mode"] == "replay"
    assert captured["context"]["replay_purpose"] == (
        "re_evaluate_existing_candidate_after_lineage_update"
    )
    assert captured["config"].report_file_path == str(
        tmp_path / "model_evaluation" / "replay_report.json"
    )
    assert (tmp_path / "model_evaluation" / "report.json").read_text() == "{}"


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
