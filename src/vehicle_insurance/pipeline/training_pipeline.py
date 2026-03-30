import sys
from vehicle_insurance.logger.logger_config import setup_logger
from vehicle_insurance.exceptions.handle_exception import VehicleInsuranceException
from vehicle_insurance.components.dataIngestion import DataIngestion
from vehicle_insurance.entities.config_entity import DataIngestionConfig
from vehicle_insurance.entities.artifact_entity import DataIngestionArtifact

logger = setup_logger()


class TrainingPipeline:
    def __init__(self):
        try:
            self.data_ingestion_config = DataIngestionConfig()
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

    def run_pipeline(self):
        try:
            data_ingestion_artifact = self.start_data_ingestion()
        except Exception as e:
            raise VehicleInsuranceException(e, sys) from e
