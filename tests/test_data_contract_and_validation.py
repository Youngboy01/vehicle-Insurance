import json

import pandas as pd
import pytest

from vehicle_insurance.components.dataValidation import DataValidation
from vehicle_insurance.constants import SCHEMA_FILE_PATH
from vehicle_insurance.contracts.data_contract import DataContractValidator
from vehicle_insurance.entities.artifact_entity import DataIngestionArtifact
from vehicle_insurance.entities.config_entity import DataValidationConfig
from vehicle_insurance.exceptions.handle_exception import (
    VehicleInsuranceException,
)


@pytest.fixture
def validator() -> DataContractValidator:
    return DataContractValidator(SCHEMA_FILE_PATH)


@pytest.fixture
def valid_training_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 2],
            "Gender": ["Male", "Female"],
            "Age": [30, 45],
            "Driving_License": [1, 1],
            "Region_Code": [28.0, 29.0],
            "Previously_Insured": [0, 1],
            "Vehicle_Age": ["1-2 Year", "< 1 Year"],
            "Vehicle_Damage": ["Yes", "No"],
            "Annual_Premium": [30000.0, 40000.0],
            "Policy_Sales_Channel": [152.0, 26.0],
            "Vintage": [100, 200],
            "Response": [0, 1],
        }
    )


def test_valid_training_data_is_accepted(
    validator: DataContractValidator,
    valid_training_data: pd.DataFrame,
) -> None:
    result = validator.validate(valid_training_data, mode="training")

    assert result.shape == (2, 12)
    assert list(result.columns) == [
        "id",
        "Gender",
        "Age",
        "Driving_License",
        "Region_Code",
        "Previously_Insured",
        "Vehicle_Age",
        "Vehicle_Damage",
        "Annual_Premium",
        "Policy_Sales_Channel",
        "Vintage",
        "Response",
    ]


def test_valid_scoring_data_is_accepted(
    validator: DataContractValidator,
    valid_training_data: pd.DataFrame,
) -> None:
    scoring_data = valid_training_data.drop(columns=["Response"])

    result = validator.validate(scoring_data, mode="scoring")

    assert result.shape == (2, 11)
    assert "Response" not in result.columns


def test_scoring_data_rejects_response(
    validator: DataContractValidator,
    valid_training_data: pd.DataFrame,
) -> None:
    with pytest.raises(VehicleInsuranceException):
        validator.validate(valid_training_data, mode="scoring")


def test_extra_column_is_rejected(
    validator: DataContractValidator,
    valid_training_data: pd.DataFrame,
) -> None:
    invalid_data = valid_training_data.assign(unexpected_column=1)

    with pytest.raises(VehicleInsuranceException):
        validator.validate(invalid_data, mode="training")


def test_invalid_age_is_rejected(
    validator: DataContractValidator,
    valid_training_data: pd.DataFrame,
) -> None:
    invalid_data = valid_training_data.copy()
    invalid_data.loc[0, "Age"] = 17

    with pytest.raises(VehicleInsuranceException):
        validator.validate(invalid_data, mode="training")


def test_duplicate_id_is_rejected(
    validator: DataContractValidator,
    valid_training_data: pd.DataFrame,
) -> None:
    invalid_data = valid_training_data.copy()
    invalid_data.loc[1, "id"] = invalid_data.loc[0, "id"]

    with pytest.raises(VehicleInsuranceException):
        validator.validate(invalid_data, mode="training")


def test_data_validation_component_writes_success_report(
    tmp_path,
    valid_training_data: pd.DataFrame,
) -> None:
    train_path = tmp_path / "train.csv"
    validation_path = tmp_path / "validation.csv"
    test_path = tmp_path / "test.csv"
    report_path = tmp_path / "report.json"

    valid_training_data.to_csv(train_path, index=False)
    valid_training_data.to_csv(validation_path, index=False)
    valid_training_data.to_csv(test_path, index=False)

    ingestion_artifact = DataIngestionArtifact(
        trained_file_path=str(train_path),
        validation_file_path=str(validation_path),
        test_file_path=str(test_path),
    )

    validation = DataValidation(
        data_validation_config=DataValidationConfig(
            report_file_path=str(report_path),
        ),
        data_ingestion_artifact=ingestion_artifact,
    )

    artifact = validation.initiate_data_validation()

    assert artifact.status is True
    assert report_path.is_file()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["overall_status"] is True
    assert report["contract_version"] == "1.0.0"
