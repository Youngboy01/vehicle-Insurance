from _source_manifest import build_source_manifest
from vehicle_insurance.components.modelEvaluation import ModelEvaluation
from vehicle_insurance.entities.artifact_entity import ModelTrainerArtifact
from vehicle_insurance.entities.config_entity import ModelEvaluationConfig


def _evaluation(tmp_path) -> ModelEvaluation:
    artifact = ModelTrainerArtifact(
        trained_model_file_path=str(tmp_path / "model.pkl"),
        metrics_file_path=str(tmp_path / "metrics.json"),
        train_metrics={},
        validation_metrics={},
        test_metrics={},
        run_id="candidate-run",
        model_id="candidate-model",
        model_uri="models:/candidate-model",
        registered_model_name=None,
        registered_model_version=None,
        stability_metrics={},
        lineage={},
    )
    return ModelEvaluation(
        model_evaluation_config=ModelEvaluationConfig(
            report_file_path=str(tmp_path / "report.json")
        ),
        model_trainer_artifact=artifact,
    )


def _valid_lineage(evaluation: ModelEvaluation) -> dict:
    dataset_hash = "a" * 64
    source_manifest = build_source_manifest(dataset_hash)
    feature_count = int(
        evaluation.model_spec["features"]["preprocessing"][
            "final_transformed_feature_count"
        ]
    )
    feature_names = [f"feature_{index}" for index in range(feature_count)]
    return {
        "contract_version": str(evaluation.schema_config["version"]),
        "model_spec_version": str(evaluation.model_spec["version"]),
        "git_branch": "project/updations",
        "git_commit": "b" * 40,
        "git_provenance": {
            "repository_root": "D:/repo",
            "branch": "project/updations",
            "commit": "b" * 40,
            "is_dirty": False,
            "status_porcelain": "",
            "untracked_files": [],
            "status_sha256": "c" * 64,
            "patch_sha256": "d" * 64,
            "model_spec_sha256": "e" * 64,
            "schema_sha256": "f" * 64,
        },
        "python_version": "3.13.0",
        "dependency_lock_hash": "c" * 64,
        "lightgbm_version": "4.6.0",
        "scikit_learn_version": "1.8.0",
        "mlflow_version": "3.10.1",
        "schema_version": str(evaluation.schema_config["version"]),
        "dataset_snapshot_id": source_manifest["snapshot_id"],
        "dataset_content_hash": dataset_hash,
        "source_manifest": source_manifest,
        "final_transformed_feature_count": feature_count,
        "final_transformed_feature_names": feature_names,
        "model_input_feature_order": feature_names.copy(),
        "candidate_fit_data": "train_only",
        "final_fit_data": "train_plus_validation",
        "test_evaluation_count": 1,
    }


def test_lineage_gate_accepts_complete_matching_provenance(tmp_path):
    evaluation = _evaluation(tmp_path)

    result = evaluation._check_lineage(_valid_lineage(evaluation))

    assert result["passed"] is True
    assert result["reason_codes"] == []


def test_lineage_gate_rejects_missing_commit_and_invalid_dataset_identity(tmp_path):
    evaluation = _evaluation(tmp_path)
    lineage = _valid_lineage(evaluation)
    lineage.pop("git_commit")
    lineage["dataset_snapshot_id"] = "sha256:" + "d" * 64

    result = evaluation._check_lineage(lineage)

    assert result["passed"] is False
    assert "lineage_field_missing:git_commit" in result["reason_codes"]
    assert "dataset_snapshot_hash_mismatch" in result["reason_codes"]


def test_lineage_gate_rejects_feature_order_drift(tmp_path):
    evaluation = _evaluation(tmp_path)
    lineage = _valid_lineage(evaluation)
    lineage["model_input_feature_order"] = list(
        reversed(lineage["final_transformed_feature_names"])
    )

    result = evaluation._check_lineage(lineage)

    assert result["passed"] is False
    assert "model_input_feature_order_mismatch" in result["reason_codes"]


def test_lineage_gate_rejects_dirty_git_worktree(tmp_path):
    evaluation = _evaluation(tmp_path)
    lineage = _valid_lineage(evaluation)
    lineage["git_provenance"]["is_dirty"] = True
    lineage["git_provenance"]["status_porcelain"] = " M src/example.py"

    result = evaluation._check_lineage(lineage)

    assert result["passed"] is False
    assert "git_worktree_dirty" in result["reason_codes"]


def test_lineage_gate_rejects_untracked_git_files(tmp_path):
    evaluation = _evaluation(tmp_path)
    lineage = _valid_lineage(evaluation)
    lineage["git_provenance"]["untracked_files"] = ["src/untracked.py"]

    result = evaluation._check_lineage(lineage)

    assert result["passed"] is False
    assert "git_untracked_files_present" in result["reason_codes"]


def test_lineage_gate_rejects_invalid_git_identity_hash(tmp_path):
    evaluation = _evaluation(tmp_path)
    lineage = _valid_lineage(evaluation)
    lineage["git_provenance"]["patch_sha256"] = "not-a-hash"

    result = evaluation._check_lineage(lineage)

    assert result["passed"] is False
    assert "git_provenance_hash_invalid:patch_sha256" in result["reason_codes"]


def test_lineage_gate_rejects_changed_source_read_boundary(tmp_path):
    evaluation = _evaluation(tmp_path)
    lineage = _valid_lineage(evaluation)
    lineage["source_manifest"]["read_boundary"]["mode"] = "moving_read"

    result = evaluation._check_lineage(lineage)

    assert result["passed"] is False
    assert any(
        reason.startswith("source_manifest_invalid:")
        for reason in result["reason_codes"]
    )
