import os
import json
import sys

import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema, Check
from pandas import DataFrame

from vehicle_insurance.exceptions.handle_exception import VehicleInsuranceException
from vehicle_insurance.logger.logger_config import setup_logger
from vehicle_insurance.utils import read_yaml_file
from vehicle_insurance.entities.config_entity import DataValidationConfig
from vehicle_insurance.entities.artifact_entity import (
    DataValidationArtifact,
    DataIngestionArtifact,
)
from vehicle_insurance.constants import SCHEMA_FILE_PATH


logger = setup_logger()


class DataValidation:
    schema = DataFrameSchema(
        columns={
            "id": Column(int, Check(lambda x: x > 0)),
            "Gender": Column(str, checks=Check.isin(["Male", "Female"])),
            "Age": Column(int, Check(lambda x: x > 0)),
            "Driving_License": Column(int, checks=Check.isin([0, 1])),
            "Region_Code": Column(float, checks=Check.greater_than_or_equal_to(0)),
            "Previously_Insured": Column(int, checks=Check.isin([0, 1])),
            "Vehicle_Age": Column(
                str, checks=Check.isin(["< 1 Year", "1-2 Year", "> 2 Years"])
            ),
            "Vehicle_Damage": Column(str, checks=Check.isin(["Yes", "No"])),
            "Annual_Premium": Column(float, checks=Check.greater_than(0)),
            "Policy_Sales_Channel": Column(float, checks=Check.greater_than(0)),
            "Vintage": Column(int, checks=Check.greater_than_or_equal_to(0)),
            "Response": Column(int, checks=Check.isin([0, 1])),
        }
    )

    def __init__(
        self,
        data_validation_config: DataValidationConfig,
        data_ingestion_artifact: DataIngestionArtifact,
    ):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            logger.error(f"Error occurred while initializing DataValidation: {e}")
            raise VehicleInsuranceException(e, sys) from e

    @staticmethod
    def read_data(file_path: str):
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            logger.error(f"Error occurred while reading data from {file_path}: {e}")
            raise VehicleInsuranceException(e, sys) from e

    def validate_data(self, dataframe: DataFrame, df_label: str):
        try:
            self.schema.validate(dataframe, lazy=True)
            logger.info(f"{df_label} data validation successful.")
            return True, []
        except pa.errors.SchemaErrors as e:
            failure_df = e.failure_cases[
                ["schema_context", "column", "check", "failure_case", "index"]
            ]
            errors = failure_df.to_dict(orient="records")
            logger.info(
                f"{df_label} : pandera balidation failed with {len(errors)} errors."
            )
            for error in errors:
                logger.info(
                    f"[{df_label}] Column: {error['column']}, Check: {error['check']}, Failure Case: {error['failure_case']}, Row_Index: {error['index']}"
                )
            return False, errors

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            logger.info("Starting data validation process.")
            validation_error_msg = ""
            train_df = DataValidation.read_data(
                self.data_ingestion_artifact.trained_file_path
            )
            is_train_valid, train_errors = self.validate_data(train_df, "Training df")
            if not is_train_valid:
                validation_error_msg += (
                    f"Training data validation failed with {len(train_errors)} errors. "
                )
            else:
                validation_error_msg += "Training data validation successful. "

            test_df = DataValidation.read_data(
                self.data_ingestion_artifact.test_file_path
            )
            is_test_valid, test_errors = self.validate_data(test_df, "Testing df")
            if not is_test_valid:
                validation_error_msg += (
                    f"Testing data validation failed with {len(test_errors)} errors."
                )
            else:
                validation_error_msg += "Testing data validation successful."

            validation_status = is_train_valid and is_test_valid
            data_validation_artifact = DataValidationArtifact(
                report_file_path=self.data_validation_config.report_file_path,
                message=validation_error_msg,
                status=validation_status,
            )

            report_dir = os.path.join(
                self.data_validation_config.data_validation_dir, "report"
            )
            os.makedirs(report_dir, exist_ok=True)

            validation_report = {
                "training_data": {"is_valid": is_train_valid, "errors": train_errors},
                "testing_data": {"is_valid": is_test_valid, "errors": test_errors},
                "overall_status": validation_status,
                "message": validation_error_msg,
            }
            with open(self.data_validation_config.report_file_path, "w") as report_file:
                json.dump(validation_report, report_file)

            logger.info("Data validation process completed.")
            logger.info(
                f"Data validation report saved at: {self.data_validation_config.report_file_path}"
            )
            return data_validation_artifact
        except Exception as e:
            raise VehicleInsuranceException(e, sys) from e
