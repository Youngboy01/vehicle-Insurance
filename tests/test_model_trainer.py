import json
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.base import clone

from vehicle_insurance.components import modelTrainer as model_trainer_module
from vehicle_insurance.components.dataTransformation import DataTransformation
from vehicle_insurance.components.modelTrainer import ModelTrainer
from vehicle_insurance.entities.artifact_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
)
from vehicle_insurance.entities.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
)
from vehicle_insurance.utils import load_object, save_object
from vehicle_insurance.utils.source_manifest import finalize_source_manifest


def _raw_data(row_count: int = 18) -> pd.DataFrame:
    rows = np.arange(row_count)
    return pd.DataFrame(
        {
            "id": rows + 1,
            "Gender": np.where(rows % 2, "Female", "Male"),
            "Age": 25 + rows,
            "Driving_License": 1,
            "Region_Code": np.where(rows % 2, 29.0, 28.0),
            "Previously_Insured": rows % 2,
            "Vehicle_Age": np.where(rows % 2, "1-2 Year", "< 1 Year"),
            "Vehicle_Damage": np.where(rows % 2, "No", "Yes"),
            "Annual_Premium": 30000.0 + rows,
            "Policy_Sales_Channel": np.where(rows % 2, 26.0, 152.0),
            "Vintage": 100 + rows,
            "Response": rows % 2,
        }
    )


def _trainer() -> ModelTrainer:
    return ModelTrainer(
        model_trainer_config=ModelTrainerConfig(),
        data_transformation_artifact=DataTransformationArtifact("", "", "", ""),
        data_ingestion_artifact=DataIngestionArtifact("", "", ""),
    )


def _clean_git_provenance() -> dict:
    return {
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
        "dependency_lock_sha256": "f" * 64,
    }


def _source_manifest(dataframe: pd.DataFrame) -> dict:
    return finalize_source_manifest(
        {
            "source": {
                "system": "MongoDB",
                "database": "INSURANCE",
                "collection": "INSURANCE_DATA",
                "namespace": "INSURANCE.INSURANCE_DATA",
            },
            "query": {"filter": {}, "filter_canonical_json": "{}"},
            "projection": {
                "fields": list(dataframe.columns),
                "mongo_projection": {
                    "_id": 0,
                    **{field: 1 for field in dataframe.columns},
                },
            },
            "ordering": {
                "field": "id",
                "direction": "ascending",
                "index_required": True,
                "index_name": "id_1",
            },
            "read_boundary": {
                "mode": "snapshot_session",
                "read_concern": "snapshot",
                "cluster_time": {"seconds": 10, "increment": 1},
                "operation_time": {"seconds": 10, "increment": 2},
            },
            "read_timestamp_utc": "2026-09-03T00:00:00+00:00",
            "database_state_retry_policy": {
                "same_snapshot_read_failure": "abort_extraction",
                "new_extraction": "new_snapshot_id",
            },
            "extracted_at_utc": "2026-09-03T00:00:00+00:00",
        },
        dataframe.sort_values("id").reset_index(drop=True),
        "data.csv",
    )


def test_trainer_uses_frozen_model_parameters() -> None:
    parameters = _trainer()._model_parameters()

    assert parameters["scale_pos_weight"] == 7.159039820166988
    assert parameters["n_estimators"] == 700
    assert "is_unbalance" not in parameters


def test_ranking_metrics_apply_floor_and_id_tie_breaking() -> None:
    trainer = _trainer()

    metrics = trainer._ranking_metrics(
        positive_scores=np.array([0.9] * 10),
        y_true=np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
        ids=np.array([10, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    )

    assert metrics["precision_at_10pct"] == 1.0
    assert metrics["recall_at_10pct"] == 1.0
    assert metrics["lift_at_10pct"] == 10.0


def test_trainer_train_writes_lineage_and_self_contained_pipeline(
    tmp_path,
    monkeypatch,
) -> None:
    raw = _raw_data()
    raw_train = raw.iloc[:10].copy()
    raw_validation = raw.iloc[10:14].copy()
    raw_test = raw.iloc[14:].copy()
    # This category is absent from train but present in validation. The final
    # train-plus-validation refit must therefore add one encoded feature.
    raw_validation.loc[raw_validation.index[0], "Region_Code"] = 30.0

    train_path = tmp_path / "train.csv"
    validation_path = tmp_path / "validation.csv"
    test_path = tmp_path / "test.csv"
    raw_train.to_csv(train_path, index=False)
    raw_validation.to_csv(validation_path, index=False)
    raw_test.to_csv(test_path, index=False)

    raw_features = _trainer().model_spec["features"]["raw_inputs"]
    transformation = DataTransformation(
        data_ingestion_artifact=DataIngestionArtifact(
            str(train_path), str(validation_path), str(test_path)
        ),
        data_validation_artifact=SimpleNamespace(status=True),
        data_transformation_config=DataTransformationConfig(),
    )
    preprocessor = transformation.get_data_transformer_object()
    train_x = raw_train.loc[:, raw_features]
    validation_x = raw_validation.loc[:, raw_features]
    test_x = raw_test.loc[:, raw_features]
    train_transformed = preprocessor.fit_transform(train_x)
    validation_transformed = preprocessor.transform(validation_x)
    test_transformed = preprocessor.transform(test_x)

    final_preprocessor = clone(preprocessor)
    final_x = pd.concat([train_x, validation_x], ignore_index=True)
    final_transformed = final_preprocessor.fit_transform(final_x)

    preprocessor_path = tmp_path / "preprocessor.pkl"
    save_object(str(preprocessor_path), preprocessor)
    transformed_train_path = tmp_path / "train.npy"
    transformed_validation_path = tmp_path / "validation.npy"
    transformed_test_path = tmp_path / "test.npy"
    np.save(
        transformed_train_path,
        np.c_[train_transformed, raw_train["Response"].to_numpy()],
    )
    np.save(
        transformed_validation_path,
        np.c_[validation_transformed, raw_validation["Response"].to_numpy()],
    )
    np.save(
        transformed_test_path,
        np.c_[test_transformed, raw_test["Response"].to_numpy()],
    )

    output_model_path = tmp_path / "model.pkl"
    metrics_path = tmp_path / "metrics.json"
    trainer_config = ModelTrainerConfig(
        trained_model_file_path=str(output_model_path),
        metrics_file_path=str(metrics_path),
    )
    transformation_artifact = DataTransformationArtifact(
        str(transformed_train_path),
        str(transformed_validation_path),
        str(transformed_test_path),
        str(preprocessor_path),
    )
    ingestion_artifact = DataIngestionArtifact(
        str(train_path),
        str(validation_path),
        str(test_path),
        source_manifest=_source_manifest(
            pd.concat([raw_train, raw_validation, raw_test], ignore_index=True)
            .sort_values("id")
            .reset_index(drop=True)
        ),
    )
    trainer = ModelTrainer(
        model_trainer_config=trainer_config,
        data_transformation_artifact=transformation_artifact,
        data_ingestion_artifact=ingestion_artifact,
        git_provenance=_clean_git_provenance(),
    )
    trainer.model_spec["features"]["preprocessing"][
        "train_only_transformed_feature_count"
    ] = train_transformed.shape[1]
    trainer.model_spec["features"]["preprocessing"][
        "final_transformed_feature_count"
    ] = final_transformed.shape[1]
    trainer.model_spec["model"]["hyperparameters"].update(
        {
            "n_estimators": 5,
            "num_leaves": 3,
            "min_child_samples": 1,
            "n_jobs": 1,
        }
    )
    trainer.model_spec["evaluation_policy"]["stability"].update(
        {"folds": 2, "repeats": 1}
    )

    monkeypatch.setattr(trainer, "_set_up_mlflow", lambda: None)
    monkeypatch.setattr(
        model_trainer_module.mlflow,
        "start_run",
        lambda **_: nullcontext(
            SimpleNamespace(info=SimpleNamespace(run_id="test-run-id"))
        ),
    )
    monkeypatch.setattr(model_trainer_module.mlflow, "log_params", lambda _: None)
    monkeypatch.setattr(model_trainer_module.mlflow, "set_tags", lambda _: None)
    monkeypatch.setattr(model_trainer_module.mlflow, "log_metrics", lambda _: None)
    monkeypatch.setattr(
        model_trainer_module.mlflow, "log_artifact", lambda *args, **_: None
    )
    monkeypatch.setattr(
        model_trainer_module.mlflow, "log_dict", lambda *args, **_: None
    )
    monkeypatch.setattr(
        model_trainer_module.mlflow.models,
        "infer_signature",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        model_trainer_module.mlflow.sklearn,
        "log_model",
        lambda **_: SimpleNamespace(model_id="m-test", model_uri="models:/m-test"),
    )

    artifact = trainer.train()

    assert artifact.run_id == "test-run-id"
    assert artifact.model_id == "m-test"
    assert artifact.model_uri == "models:/m-test"
    assert artifact.validation_metrics["pr_auc"] >= 0.0
    assert artifact.stability_metrics["required"] is True
    assert artifact.stability_metrics["evaluation_count"] == 2
    assert (
        artifact.stability_metrics["preprocessing_scope"]
        == "fit_inside_each_fold_on_fold_training_rows"
    )
    assert output_model_path.is_file()
    assert metrics_path.is_file()

    final_pipeline = load_object(str(output_model_path))
    assert (
        final_pipeline.named_steps["model"].n_features_in_ == final_transformed.shape[1]
    )

    lineage = json.loads(metrics_path.read_text(encoding="utf-8"))["lineage"]
    assert lineage["final_fit_data"] == "train_plus_validation"
    assert lineage["test_evaluation_count"] == 1
    assert lineage["candidate_transformed_feature_count"] == train_transformed.shape[1]
    assert lineage["final_transformed_feature_count"] == final_transformed.shape[1]
    assert lineage["source_tuning_run_id"] == "e573ee5423434e6485404a96ca5f5731"
    assert lineage["reference_final_run_id"] == "dc28826afdcc4e5cbce068db62439094"
    assert lineage["trainer_candidate_run_id"] == "test-run-id"
    assert isinstance(lineage["git_provenance"], dict)
