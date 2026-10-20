import json
from types import SimpleNamespace

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST

from _source_manifest import build_source_manifest
from vehicle_insurance.components import modelEvaluation as model_evaluation_module
from vehicle_insurance.components.modelEvaluation import ModelEvaluation
from vehicle_insurance.entities.artifact_entity import ModelTrainerArtifact
from vehicle_insurance.entities.config_entity import ModelEvaluationConfig


def _lineage() -> dict:
    dataset_hash = "a" * 64
    source_manifest = build_source_manifest(dataset_hash)
    feature_names = [f"feature_{index}" for index in range(220)]
    return {
        "contract_version": "1.0.0",
        "model_spec_version": "3",
        "git_branch": "project/updations",
        "git_commit": "a" * 40,
        "git_provenance": {
            "repository_root": "D:/repo",
            "branch": "project/updations",
            "commit": "a" * 40,
            "is_dirty": False,
            "status_porcelain": "",
            "untracked_files": [],
            "status_sha256": "b" * 64,
            "patch_sha256": "c" * 64,
            "model_spec_sha256": "d" * 64,
            "schema_sha256": "e" * 64,
        },
        "python_version": "3.13.0",
        "dependency_lock_hash": "b" * 64,
        "lightgbm_version": "4.6.0",
        "scikit_learn_version": "1.8.0",
        "mlflow_version": "3.10.1",
        "schema_version": "1.0.0",
        "dataset_snapshot_id": source_manifest["snapshot_id"],
        "dataset_content_hash": dataset_hash,
        "source_manifest": source_manifest,
        "candidate_fit_data": "train_only",
        "final_fit_data": "train_plus_validation",
        "final_transformed_feature_count": 220,
        "final_transformed_feature_names": feature_names,
        "model_input_feature_order": feature_names.copy(),
        "test_evaluation_count": 1,
        "source_frozen_parity": {
            "status": "verified",
            "split_id_sets_match": True,
            "prediction_order_match": True,
            "maximum_absolute_prediction_error": 0.0,
        },
    }


def _trainer(tmp_path, *, lineage: dict | None = None) -> ModelTrainerArtifact:
    validation_metrics = {
        "precision_at_10pct": 0.40,
        "lift_at_10pct": 3.25,
        "recall_at_10pct": 0.32,
        "pr_auc": 0.37,
        "roc_auc": 0.85,
    }
    stability_metrics = {
        "required": True,
        "evaluation_count": 9,
        "metrics": {
            metric: {"mean": value, "std": 0.001}
            for metric, value in {
                "precision_at_10pct": 0.40,
                "lift_at_10pct": 3.25,
                "recall_at_10pct": 0.32,
                "pr_auc": 0.37,
            }.items()
        },
    }
    candidate_root = tmp_path / "candidate_artifact"
    trainer_root = candidate_root / "model_trainer"
    trainer_root.mkdir(parents=True, exist_ok=True)
    model_path = trainer_root / "model.pkl"
    model_path.write_bytes(b"model")
    metrics_path = trainer_root / "metrics.json"
    candidate_lineage = _lineage() if lineage is None else lineage
    parity = candidate_lineage.get("source_frozen_parity")
    if isinstance(parity, dict) and parity.get("status") == "verified":
        import hashlib

        candidate_identity = {
            "mlflow_run_id": "candidate-run",
            "model_id": "candidate-model",
            "model_uri": "models:/candidate-model",
            "dataset_snapshot_id": candidate_lineage["dataset_snapshot_id"],
            "dataset_content_hash": candidate_lineage["dataset_content_hash"],
            "model_file_sha256": hashlib.sha256(b"model").hexdigest(),
        }
        parity.update(
            {
                "recording_state": "complete",
                "recording_mode": "local_and_mlflow",
                "certification_attempt_id": "attempt-1",
                "candidate_identity": candidate_identity,
                "reference_run_id": "dc28826afdcc4e5cbce068db62439094",
                "reference_model_id": "m-3aacf46fc018452ab9aab96033eb7e56",
                "reference_model_uri": ("models:/m-3aacf46fc018452ab9aab96033eb7e56"),
                "certificate_report_path": ("source_frozen_parity/parity_report.json"),
                "certificate_report_sha256": None,
                "recording_steps": {
                    "certificate_report_written": True,
                    "local_metrics_json_committed": True,
                    "mlflow_evidence_logged": True,
                    "mlflow_tags_recorded": True,
                },
            }
        )
        parity_report = {
            "certificate_type": "source_frozen_notebook_to_production_parity",
            "certificate_version": "1.0.0",
            "status": "verified",
            "candidate_run_id": "candidate-run",
            "candidate_identity": candidate_identity,
            "certification_attempt_id": "attempt-1",
            "reference_run_id": parity["reference_run_id"],
            "reference_model_id": parity["reference_model_id"],
            "reference_model_uri": parity["reference_model_uri"],
            "lineage_recording": {
                "recording_state": "complete",
                "recording_mode": "local_and_mlflow",
                "certification_attempt_id": "attempt-1",
                "candidate_identity": candidate_identity,
                "certificate_report_path": ("source_frozen_parity/parity_report.json"),
                "local_metrics_json_updated": True,
                "mlflow_evidence_logged": True,
                "mlflow_tags_recorded": True,
                "required_steps": {
                    "certificate_report": True,
                    "local_metrics_json": True,
                    "mlflow_evidence": True,
                    "mlflow_tags": True,
                },
            },
        }
        certificate_path = (
            candidate_root / "source_frozen_parity" / "parity_report.json"
        )
        certificate_path.parent.mkdir(parents=True, exist_ok=True)
        certificate_path.write_text(json.dumps(parity_report), encoding="utf-8")
        parity["certificate_report_sha256"] = hashlib.sha256(
            certificate_path.read_bytes()
        ).hexdigest()

    metrics_path.write_text(
        json.dumps(
            {
                "mlflow_run_id": "candidate-run",
                "model_id": "candidate-model",
                "model_uri": "models:/candidate-model",
                "train_metrics": validation_metrics,
                "validation_metrics": validation_metrics,
                "test_metrics": {"pr_auc": 0.0},
                "stability_metrics": stability_metrics,
                "lineage": candidate_lineage,
            }
        ),
        encoding="utf-8",
    )
    return ModelTrainerArtifact(
        trained_model_file_path=str(model_path),
        metrics_file_path=str(metrics_path),
        train_metrics=validation_metrics,
        validation_metrics=validation_metrics,
        test_metrics={
            "precision_at_10pct": 0.0,
            "lift_at_10pct": 0.0,
            "recall_at_10pct": 0.0,
            "pr_auc": 0.0,
            "roc_auc": 0.0,
        },
        run_id="candidate-run",
        model_id="candidate-model",
        model_uri="models:/candidate-model",
        registered_model_name=None,
        registered_model_version=None,
        stability_metrics=stability_metrics,
        lineage=candidate_lineage,
    )


class _NoChampionClient:
    def get_model_version_by_alias(self, *_args, **_kwargs):
        raise MlflowException(
            "Champion alias does not exist",
            error_code=RESOURCE_DOES_NOT_EXIST,
        )

    def set_registered_model_alias(self, *_args, **_kwargs):
        raise AssertionError("evaluation must not mutate MLflow aliases")

    def delete_registered_model_alias(self, *_args, **_kwargs):
        raise AssertionError("evaluation must not mutate MLflow aliases")


def _forbid_mlflow_lookup(monkeypatch):
    def unexpected_lookup():
        raise AssertionError("candidate-only evaluation must not look up MLflow")

    monkeypatch.setattr(
        model_evaluation_module,
        "get_mlflow_client",
        unexpected_lookup,
    )


def test_evaluation_uses_validation_metrics_and_does_not_promote(
    tmp_path,
    monkeypatch,
):
    _forbid_mlflow_lookup(monkeypatch)
    evaluator = ModelEvaluation(
        ModelEvaluationConfig(report_file_path=str(tmp_path / "report.json")),
        _trainer(tmp_path),
    )

    artifact = evaluator.evaluate_model()

    assert artifact.evaluation_passed is True
    assert artifact.decision == "eligible_but_promotion_disabled"
    assert artifact.promotion_status == "disabled"
    assert artifact.promotion_performed is False
    assert artifact.candidate_test_metrics["pr_auc"] == 0.0
    assert artifact.gate_results["validation_metrics"]["passed"] is True
    assert artifact.gate_results["source_frozen_parity"]["passed"] is True
    assert artifact.champion_run_id is None
    assert artifact.gate_results["champion_comparison"]["status"] == "not_required"
    assert artifact.gate_results["champion_comparison"]["reason_codes"] == [
        "champion_comparison_disabled_by_policy"
    ]
    assert artifact.gate_results["champion_comparison"]["comparison_performed"] is False
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["candidate"]["test_metrics_usage"] == "reporting_only"
    assert (
        report["source_manifest"]["snapshot_id"]
        == artifact.lineage["source_manifest"]["snapshot_id"]
    )


def test_explicit_champion_comparison_override_reenables_future_workflow(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        model_evaluation_module,
        "get_mlflow_client",
        lambda: _NoChampionClient(),
    )

    artifact = ModelEvaluation(
        ModelEvaluationConfig(report_file_path=str(tmp_path / "report.json")),
        _trainer(tmp_path),
        require_champion_comparison=True,
    ).evaluate_model()

    assert artifact.gate_results["champion_comparison"]["status"] == "not_found"
    assert artifact.gate_results["champion_comparison"]["comparison_performed"] is False


def test_evaluation_rejects_unverified_source_frozen_parity(tmp_path, monkeypatch):
    lineage = _lineage()
    lineage["source_frozen_parity"] = {
        "status": "not_verified",
        "reason": "direct comparison is pending",
    }
    _forbid_mlflow_lookup(monkeypatch)
    evaluator = ModelEvaluation(
        ModelEvaluationConfig(report_file_path=str(tmp_path / "report.json")),
        _trainer(tmp_path, lineage=lineage),
    )

    artifact = evaluator.evaluate_model()

    assert artifact.evaluation_passed is False
    assert artifact.decision == "rejected"
    assert "source_frozen_parity_not_verified" in artifact.reason_codes


def test_evaluation_compares_a_valid_champion_on_validation_metrics(
    tmp_path,
    monkeypatch,
):
    lineage = _lineage()
    lineage_path = tmp_path / "lineage.json"
    lineage_path.write_text(json.dumps(lineage), encoding="utf-8")

    candidate = _trainer(tmp_path, lineage=lineage)
    champion_metrics = {
        f"validation_{name}": value
        for name, value in {
            "precision_at_10pct": 0.39,
            "lift_at_10pct": 3.10,
            "recall_at_10pct": 0.32,
            "pr_auc": 0.36,
            "roc_auc": 0.848,
        }.items()
    }

    class _ValidChampionClient:
        def get_model_version_by_alias(self, *_args, **_kwargs):
            return SimpleNamespace(run_id="champion-run", version="7")

        def get_run(self, _run_id):
            return SimpleNamespace(
                data=SimpleNamespace(
                    metrics=champion_metrics,
                    tags={
                        "contract_version": "1.0.0",
                        "model_spec_version": "3",
                        "dataset_snapshot_id": lineage["dataset_snapshot_id"],
                        "final_transformed_feature_count": "220",
                    },
                )
            )

        def download_artifacts(self, *_args, **_kwargs):
            return str(lineage_path)

    monkeypatch.setattr(
        model_evaluation_module,
        "get_mlflow_client",
        lambda: _ValidChampionClient(),
    )
    artifact = ModelEvaluation(
        ModelEvaluationConfig(report_file_path=str(tmp_path / "report.json")),
        candidate,
        require_champion_comparison=True,
    ).evaluate_model()

    assert artifact.metric_comparison["comparison_performed"] is True
    assert artifact.metric_comparison["passed"] is True
    assert artifact.champion_run_id == "champion-run"
    assert artifact.champion_model_version == "7"
    assert artifact.metric_comparison["improved_primary_metrics"] == [
        "precision_at_10pct",
        "lift_at_10pct",
    ]


def test_evaluation_rejects_missing_required_validation_metric(tmp_path, monkeypatch):
    candidate = _trainer(tmp_path)
    candidate.validation_metrics.pop("pr_auc")
    _forbid_mlflow_lookup(monkeypatch)

    artifact = ModelEvaluation(
        ModelEvaluationConfig(report_file_path=str(tmp_path / "report.json")),
        candidate,
    ).evaluate_model()

    assert artifact.evaluation_passed is False
    assert "validation_metric_below_minimum:pr_auc" in artifact.reason_codes


def test_evaluation_rejects_lineage_hash_mismatch_even_when_metrics_pass(
    tmp_path,
    monkeypatch,
):
    lineage = _lineage()
    lineage["dataset_snapshot_id"] = "sha256:" + "c" * 64
    _forbid_mlflow_lookup(monkeypatch)

    artifact = ModelEvaluation(
        ModelEvaluationConfig(report_file_path=str(tmp_path / "report.json")),
        _trainer(tmp_path, lineage=lineage),
    ).evaluate_model()

    assert artifact.evaluation_passed is False
    assert "dataset_snapshot_hash_mismatch" in artifact.reason_codes
    assert artifact.gate_results["validation_metrics"]["passed"] is True
