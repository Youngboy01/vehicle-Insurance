from pathlib import Path

import numpy as np
import pandas as pd

from vehicle_insurance.components import dataIngestion as data_ingestion_module
from vehicle_insurance.components.dataIngestion import DataIngestion
from vehicle_insurance.components.dataTransformation import DataTransformation
from vehicle_insurance.data_access.Insurance_data import MongoExtractionResult
from vehicle_insurance.entities.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
)
from vehicle_insurance.entities.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
)
from vehicle_insurance.utils import load_object
from vehicle_insurance.utils.source_manifest import finalize_source_manifest


def _valid_training_data(row_count: int = 20) -> pd.DataFrame:
    row_numbers = np.arange(row_count)
    return pd.DataFrame(
        {
            "id": row_numbers + 1,
            "Gender": np.where(row_numbers % 2, "Female", "Male"),
            "Age": 25 + row_numbers,
            "Driving_License": 1,
            "Region_Code": np.where(row_numbers % 2, 29.0, 28.0),
            "Previously_Insured": row_numbers % 2,
            "Vehicle_Age": np.where(row_numbers % 3, "1-2 Year", "< 1 Year"),
            "Vehicle_Damage": np.where(row_numbers % 2, "No", "Yes"),
            "Annual_Premium": 30000.0 + row_numbers,
            "Policy_Sales_Channel": np.where(row_numbers % 2, 26.0, 152.0),
            "Vintage": 100 + row_numbers,
            "Response": row_numbers % 2,
        }
    )


def test_ingestion_creates_disjoint_stratified_three_way_split(
    tmp_path,
    monkeypatch,
) -> None:
    source_data = _valid_training_data()

    def source_manifest():
        return finalize_source_manifest(
            {
                "source": {
                    "system": "MongoDB",
                    "database": "INSURANCE",
                    "collection": "INSURANCE_DATA",
                    "namespace": "INSURANCE.INSURANCE_DATA",
                },
                "query": {
                    "filter": {},
                    "filter_canonical_json": "{}",
                },
                "projection": {
                    "fields": list(source_data.columns),
                    "mongo_projection": {
                        "_id": 0,
                        **{field: 1 for field in source_data.columns},
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
            source_data,
            "data.csv",
        )

    class FakeInsuranceData:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def extract_snapshot(self, collection_name, **kwargs):
            return MongoExtractionResult(
                dataframe=source_data.copy(),
                manifest=source_manifest(),
            )

    monkeypatch.setattr(data_ingestion_module, "InsuranceData", FakeInsuranceData)

    ingestion_dir = tmp_path / "data_ingestion"
    config = DataIngestionConfig(
        data_ingestion_dir=str(ingestion_dir),
        feature_store_file_path=str(ingestion_dir / "feature_store" / "data.csv"),
        training_file_path=str(ingestion_dir / "ingested" / "train.csv"),
        validation_file_path=str(ingestion_dir / "ingested" / "validation.csv"),
        testing_file_path=str(ingestion_dir / "ingested" / "test.csv"),
    )

    artifact = DataIngestion(config).initiate_data_ingestion()

    train = pd.read_csv(artifact.trained_file_path)
    validation = pd.read_csv(artifact.validation_file_path)
    test = pd.read_csv(artifact.test_file_path)

    assert (len(train), len(validation), len(test)) == (16, 2, 2)
    assert set(train["id"]).isdisjoint(validation["id"])
    assert set(train["id"]).isdisjoint(test["id"])
    assert set(validation["id"]).isdisjoint(test["id"])
    assert set(pd.concat([train, validation, test])["id"]) == set(source_data["id"])
    assert set(train["Response"]) == {0, 1}
    assert set(validation["Response"]) == {0, 1}
    assert set(test["Response"]) == {0, 1}

    # The notebook assigns the first second-stage partition to X_test and the
    # second partition to X_val.  This assertion protects the semantic labels,
    # not only the 80/10/10 row counts and disjointness properties.
    _, remainder_set = data_ingestion_module.train_test_split(
        source_data,
        test_size=0.2,
        random_state=42,
        stratify=source_data["Response"],
    )
    expected_test, expected_validation = data_ingestion_module.train_test_split(
        remainder_set,
        test_size=0.5,
        random_state=42,
        stratify=remainder_set["Response"],
    )
    assert set(test["id"]) == set(expected_test["id"])
    assert set(validation["id"]) == set(expected_validation["id"])
    assert Path(artifact.feature_store_file_path) == (
        ingestion_dir / "feature_store" / "data.csv"
    )
    assert Path(artifact.source_manifest_file_path) == (
        ingestion_dir / "feature_store" / "source_manifest.json"
    )
    assert artifact.source_manifest["immutable"] is True
    assert artifact.source_manifest["row_count"] == len(source_data)
    assert (ingestion_dir / "feature_store" / "source_manifest.json").is_file()


def test_ingestion_reuses_verified_snapshot_without_a_new_mongodb_read(
    tmp_path,
    monkeypatch,
):
    source_data = _valid_training_data()
    extraction_calls = []

    class FakeInsuranceData:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def extract_snapshot(self, collection_name, **kwargs):
            extraction_calls.append(collection_name)
            manifest = {
                "source": {
                    "system": "MongoDB",
                    "database": "INSURANCE",
                    "collection": "INSURANCE_DATA",
                    "namespace": "INSURANCE.INSURANCE_DATA",
                },
                "query": {"filter": {}, "filter_canonical_json": "{}"},
                "projection": {
                    "fields": list(source_data.columns),
                    "mongo_projection": {
                        "_id": 0,
                        **{field: 1 for field in source_data.columns},
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
            }
            return MongoExtractionResult(
                dataframe=source_data.copy(),
                manifest=finalize_source_manifest(manifest, source_data, "data.csv"),
            )

    monkeypatch.setattr(data_ingestion_module, "InsuranceData", FakeInsuranceData)
    config = DataIngestionConfig(artifact_dir=str(tmp_path / "artifact"))

    first = DataIngestion(config).initiate_data_ingestion()
    second = DataIngestion(config).initiate_data_ingestion()

    assert extraction_calls == ["INSURANCE_DATA"]
    assert first.source_manifest["snapshot_id"] == second.source_manifest["snapshot_id"]


def test_transformation_pipeline_uses_frozen_preprocessing_rules() -> None:
    source_data = _valid_training_data(6)
    raw_features = source_data.drop(columns=["id", "Response"])
    ingestion_artifact = DataIngestionArtifact("", "", "")
    validation_artifact = DataValidationArtifact("", "", True)

    transformation = DataTransformation(
        data_ingestion_artifact=ingestion_artifact,
        data_validation_artifact=validation_artifact,
        data_transformation_config=DataTransformationConfig(),
    )

    pipeline = transformation.get_data_transformer_object()
    transformed = pipeline.fit_transform(raw_features)
    feature_names = list(pipeline.get_feature_names_out())

    # Two Gender + two Vehicle_Age + two Vehicle_Damage + two Region_Code
    # + two Policy_Sales_Channel categories, followed by five passthrough fields.
    assert transformed.shape == (6, 15)
    assert feature_names[-5:] == [
        "Age",
        "Annual_Premium",
        "Vintage",
        "Driving_License",
        "Previously_Insured",
    ]

    unknown_row = raw_features.iloc[[0]].copy()
    unknown_row["Region_Code"] = 999.0
    unknown_row["Policy_Sales_Channel"] = 999.0
    unknown_transformed = pipeline.transform(unknown_row)

    assert unknown_transformed.shape == (1, 15)
    assert np.isfinite(unknown_transformed).all()

    unknown_region_columns = [
        index
        for index, name in enumerate(feature_names)
        if name.startswith("Region_Code_")
    ]
    unknown_channel_columns = [
        index
        for index, name in enumerate(feature_names)
        if name.startswith("Policy_Sales_Channel_")
    ]
    assert unknown_transformed[0, unknown_region_columns].sum() == 0
    assert unknown_transformed[0, unknown_channel_columns].sum() == 0


def test_transformation_persists_all_three_labeled_outputs(tmp_path) -> None:
    source_data = _valid_training_data(6)
    train = source_data.iloc[[0, 1]]
    validation = source_data.iloc[[2, 3]].copy()
    validation.loc[:, "Region_Code"] = [999.0, 999.0]
    validation.loc[:, "Policy_Sales_Channel"] = [999.0, 999.0]
    test = source_data.iloc[[4, 5]]

    train_path = tmp_path / "train.csv"
    validation_path = tmp_path / "validation.csv"
    test_path = tmp_path / "test.csv"
    train.to_csv(train_path, index=False)
    validation.to_csv(validation_path, index=False)
    test.to_csv(test_path, index=False)

    output_dir = tmp_path / "transformation"
    transformation_config = DataTransformationConfig(
        data_transformation_dir=str(output_dir),
        transformed_train_file_path=str(output_dir / "train.npy"),
        transformed_validation_file_path=str(output_dir / "validation.npy"),
        transformed_test_file_path=str(output_dir / "test.npy"),
        transformer_object_file_path=str(output_dir / "preprocessor.pkl"),
    )
    ingestion_artifact = DataIngestionArtifact(
        trained_file_path=str(train_path),
        validation_file_path=str(validation_path),
        test_file_path=str(test_path),
    )
    validation_artifact = DataValidationArtifact("", "", True)

    transformation = DataTransformation(
        data_ingestion_artifact=ingestion_artifact,
        data_validation_artifact=validation_artifact,
        data_transformation_config=transformation_config,
    )
    # The small fixture has 15 transformed columns; production remains locked
    # to the schema's expected 220-column benchmark feature space.
    transformation._schema_config["preprocessing"][
        "train_only_transformed_feature_count"
    ] = 15

    artifact = transformation.initiate_data_transformation()

    train_array = np.load(artifact.transformed_train_file_path, allow_pickle=False)
    validation_array = np.load(
        artifact.transformed_validation_file_path,
        allow_pickle=False,
    )
    test_array = np.load(artifact.transformed_test_file_path, allow_pickle=False)
    fitted_transformer = load_object(artifact.transformer_object_file_path)

    assert train_array.shape == (2, 16)
    assert validation_array.shape == (2, 16)
    assert test_array.shape == (2, 16)
    for saved, split in (
        (train_array, train),
        (validation_array, validation),
        (test_array, test),
    ):
        np.testing.assert_array_equal(saved[:, -1], split["Response"].to_numpy())
        np.testing.assert_allclose(
            saved[:, :-1],
            fitted_transformer.transform(split.drop(columns="Response")),
        )

    encoder = fitted_transformer.named_steps["preprocessor"].named_transformers_[
        "categorical_encoder"
    ]
    assert all(999.0 not in categories for categories in encoder.categories_)
