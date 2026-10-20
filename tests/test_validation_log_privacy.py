"""Regression coverage for redacted data-contract validation diagnostics."""

from __future__ import annotations

import io
import json
import logging
from logging.handlers import RotatingFileHandler

import pandas as pd
import pytest

import app
import vehicle_insurance.contracts.data_contract as data_contract_module
from vehicle_insurance.components.dataValidation import DataValidation
from vehicle_insurance.constants import SCHEMA_FILE_PATH
from vehicle_insurance.contracts.data_contract import DataContractValidator
from vehicle_insurance.entities.artifact_entity import DataIngestionArtifact
from vehicle_insurance.entities.config_entity import DataValidationConfig
from vehicle_insurance.exceptions.handle_exception import VehicleInsuranceException


def _scoring_record(gender: str) -> dict:
    return {
        "id": 1,
        "Gender": gender,
        "Age": 35,
        "Driving_License": 1,
        "Region_Code": 28.0,
        "Previously_Insured": 0,
        "Vehicle_Age": "1-2 Year",
        "Vehicle_Damage": "Yes",
        "Annual_Premium": 25000.0,
        "Policy_Sales_Channel": 1.0,
        "Vintage": 5,
    }


@pytest.fixture
def captured_validation_logs():
    logger = data_contract_module.logger
    root_logger = logging.getLogger()
    file_stream = io.StringIO()
    console_stream = io.StringIO()
    root_stream = io.StringIO()
    root_handler = logging.StreamHandler(root_stream)
    file_handler = next(
        handler
        for handler in logger.handlers
        if isinstance(handler, RotatingFileHandler)
    )
    console_handler = next(
        handler for handler in logger.handlers if type(handler) is logging.StreamHandler
    )
    original_file_stream = file_handler.stream
    original_console_stream = console_handler.stream
    file_handler.stream = file_stream
    console_handler.stream = console_stream
    root_logger.addHandler(root_handler)
    try:
        yield file_stream, console_stream, root_stream
    finally:
        file_handler.stream = original_file_stream
        console_handler.stream = original_console_stream
        root_logger.removeHandler(root_handler)


def test_scoring_validation_redacts_marker_from_file_and_console_logs(
    captured_validation_logs,
):
    marker = "SENSITIVE_MARKER_A05_SCORING"

    with pytest.raises(app.ScoringContractError) as raised:
        app._validated_scoring_dataframe(
            [app.PredictRequest(**_scoring_record(marker))]
        )

    file_output, console_output, root_output = (
        stream.getvalue() for stream in captured_validation_logs
    )
    assert raised.value.fields == ["Gender"]
    assert "failures=1" in file_output
    assert "Gender" in file_output
    assert "allowed_values" in file_output
    assert marker not in file_output
    assert marker not in console_output
    assert marker not in root_output


def test_shape_validation_redacts_unexpected_input_name_from_logs(
    captured_validation_logs,
):
    marker = "SENSITIVE_MARKER_A05_COLUMN"
    dataframe = pd.DataFrame([_scoring_record("Male")]).assign(**{marker: "submitted"})

    with pytest.raises(VehicleInsuranceException):
        DataContractValidator(SCHEMA_FILE_PATH).validate(
            dataframe,
            mode="scoring",
        )

    outputs = [stream.getvalue() for stream in captured_validation_logs]
    assert all(marker not in output for output in outputs)


def test_training_validation_report_redacts_marker_and_keeps_safe_summary(tmp_path):
    marker = "SENSITIVE_MARKER_A05_TRAINING"
    valid = pd.DataFrame(
        {
            "id": [1, 2],
            "Gender": [marker, "Female"],
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
    paths = {name: tmp_path / f"{name}.csv" for name in ("train", "validation", "test")}
    valid.to_csv(paths["train"], index=False)
    valid.assign(Gender=["Male", "Female"]).to_csv(
        paths["validation"],
        index=False,
    )
    valid.assign(Gender=["Male", "Female"]).to_csv(
        paths["test"],
        index=False,
    )
    report_path = tmp_path / "validation-report.json"
    validation = DataValidation(
        data_validation_config=DataValidationConfig(
            report_file_path=str(report_path),
        ),
        data_ingestion_artifact=DataIngestionArtifact(
            trained_file_path=str(paths["train"]),
            validation_file_path=str(paths["validation"]),
            test_file_path=str(paths["test"]),
        ),
    )

    artifact = validation.initiate_data_validation()

    assert artifact.status is False
    report_text = report_path.read_text(encoding="utf-8")
    assert marker not in report_text
    report = json.loads(report_text)
    errors = report["training_data"]["errors"]
    assert errors[0]["column"] == "Gender"
    assert errors[0]["check"] == "allowed_values"
    assert errors[0]["failure_count"] == 1
    assert errors[0]["failure_case"] is None
    assert errors[0]["index"] is None
