import json
import os
import sys

import pandas as pd
from pandas import DataFrame

from vehicle_insurance.constants import SCHEMA_FILE_PATH
from vehicle_insurance.contracts.data_contract import (
    DataContractValidator,
)
from vehicle_insurance.entities.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
)
from vehicle_insurance.entities.config_entity import DataValidationConfig
from vehicle_insurance.exceptions.handle_exception import VehicleInsuranceException
from vehicle_insurance.logger.logger_config import setup_logger

logger = setup_logger()


class DataValidation:
    def __init__(
        self,
        data_validation_config: DataValidationConfig,
        data_ingestion_artifact: DataIngestionArtifact,
    ):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self.contract_validator = DataContractValidator(
                schema_path=SCHEMA_FILE_PATH
            )
            logger.info(
                "Data contract loaded successfully: version=%s",
                self.contract_validator.contract_version,
            )
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

    def validate_data(
        self, dataframe: DataFrame, df_label: str, mode: str = "training"
    ):
        try:
            self.contract_validator.validate(dataframe, mode)
            logger.info(
                "%s data contract validation successful",
                df_label,
            )
            return True, []
        except VehicleInsuranceException as error:
            logger.error(
                "%s data contract validation failed: %s",
                df_label,
                error,
            )

            cause = error.__cause__
            failure_cases = getattr(cause, "failure_cases", None)

            if failure_cases is not None:
                errors = (
                    failure_cases.astype(object)
                    .where(
                        failure_cases.notna(),
                        None,
                    )
                    .to_dict(orient="records")
                )
            else:
                errors = [
                    {
                        "schema_context": "data_contract",
                        "column": None,
                        "check": "contract_validation",
                        "failure_case": str(error),
                        "index": None,
                    }
                ]

        return False, errors

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            logger.info("Starting data validation process.")
            validation_error_msg = ""
            train_df = DataValidation.read_data(
                self.data_ingestion_artifact.trained_file_path
            )
            is_train_valid, train_errors = self.validate_data(
                train_df, "Training df", mode="training"
            )
            if not is_train_valid:
                validation_error_msg += (
                    f"Training data validation failed with {len(train_errors)} errors. "
                )
            else:
                validation_error_msg += "Training data validation successful. "

            test_df = DataValidation.read_data(
                self.data_ingestion_artifact.test_file_path
            )
            is_test_valid, test_errors = self.validate_data(
                test_df, "Testing df", mode="training"
            )
            if not is_test_valid:
                validation_error_msg += (
                    f"Testing data validation failed with {len(test_errors)} errors."
                )
            else:
                validation_error_msg += "Testing data validation successful."

            validation_status = is_train_valid and is_test_valid
            os.makedirs(
                os.path.dirname(self.data_validation_config.report_file_path),
                exist_ok=True,
            )
            validation_report = {
                "contract_version": self.contract_validator.contract_version,
                "training_data": {"is_valid": is_train_valid, "errors": train_errors},
                "testing_data": {"is_valid": is_test_valid, "errors": test_errors},
                "overall_status": validation_status,
                "message": validation_error_msg,
            }
            with open(self.data_validation_config.report_file_path, "w") as report_file:
                json.dump(validation_report, report_file, indent=2, default=str)

            data_validation_artifact = DataValidationArtifact(
                report_file_path=self.data_validation_config.report_file_path,
                message=validation_error_msg,
                status=validation_status,
            )

            logger.info("Data validation process completed.")
            logger.info(
                f"Data validation report saved at: {self.data_validation_config.report_file_path}"
            )
            return data_validation_artifact
        except Exception as e:
            raise VehicleInsuranceException(e, sys) from e
