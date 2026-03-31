import sys
from vehicle_insurance.logger.logger_config import setup_logger
from vehicle_insurance.exceptions.handle_exception import VehicleInsuranceException
from vehicle_insurance.components.dataIngestion import DataIngestion
from vehicle_insurance.components.dataValidation import DataValidation
from vehicle_insurance.entities.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
)
from vehicle_insurance.entities.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
)

logger = setup_logger()


class TrainingPipeline:
    def __init__(self):
        try:
            self.data_ingestion_config = DataIngestionConfig()
            self.data_validation_config = DataValidationConfig()
        except Exception as e:
            raise VehicleInsuranceException(e, sys) from e

    def start_data_ingestion(self):
        try:
            logger.info("Starting data ingestion component")
            data_ingestion = DataIngestion(
                data_ingestion_config=self.data_ingestion_config
            )
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logger.info(f"Got the train and test file path: {data_ingestion_artifact}")
            logger.info(
                f"Data ingestion component completed with artifact: {data_ingestion_artifact}"
            )
            return data_ingestion_artifact
        except Exception as e:
            raise VehicleInsuranceException(e, sys) from e

    def start_data_validation(
        self, data_ingestion_artifact: DataIngestionArtifact
    ) -> DataValidationArtifact:
        try:
            logger.info("Starting data validation component")
            data_validation = DataValidation(
                data_validation_config=self.data_validation_config,
                data_ingestion_artifact=data_ingestion_artifact,
            )
            data_validation_artifact = data_validation.initiate_data_validation()
            logger.info(
                f"Data validation component completed with artifact: {data_validation_artifact}"
            )
            return data_validation_artifact
        except Exception as e:
            raise VehicleInsuranceException(e, sys) from e

    def run_pipeline(self):
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(
                data_ingestion_artifact=data_ingestion_artifact
            )
            logger.info(f"Data validation status: {data_validation_artifact.status}")
        except Exception as e:
            raise VehicleInsuranceException(e, sys) from e
