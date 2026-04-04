import sys
from vehicle_insurance.logger.logger_config import setup_logger
from vehicle_insurance.exceptions.handle_exception import VehicleInsuranceException
from vehicle_insurance.components.dataIngestion import DataIngestion
from vehicle_insurance.components.dataValidation import DataValidation
from vehicle_insurance.components.dataTransformation import DataTransformation
from vehicle_insurance.components.modelTrainer import ModelTrainer
from vehicle_insurance.entities.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
)
from vehicle_insurance.entities.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
)

logger = setup_logger()


class TrainingPipeline:
    def __init__(self):
        try:
            self.data_ingestion_config = DataIngestionConfig()
            self.data_validation_config = DataValidationConfig()
            self.data_transformation_config = DataTransformationConfig()
            self.model_trainer_config = ModelTrainerConfig()
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

    def start_data_transformation(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_artifact: DataValidationArtifact,
    ) -> DataTransformationArtifact:
        try:
            logger.info("Starting data transformation component")
            data_transformation = DataTransformation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_artifact=data_validation_artifact,
                data_transformation_config=self.data_transformation_config,
            )
            data_transformation_artifact = (
                data_transformation.initiate_data_transformation()
            )
            logger.info(
                f"Data transformation component completed with artifact: {data_transformation_artifact}"
            )
            return data_transformation_artifact
        except Exception as e:
            raise VehicleInsuranceException(e, sys) from e

    def start_model_training(
        self, data_transformation_artifact: DataTransformationArtifact
    ) -> ModelTrainerArtifact:
        try:
            logger.info("Starting model training component")
            model_trainer = ModelTrainer(
                model_trainer_config=self.model_trainer_config,
                data_transformation_artifact=data_transformation_artifact,
            )
            model_trainer_artifact = model_trainer.train()
            logger.info(
                f"Model training component completed with artifact: {model_trainer_artifact}"
            )
            return model_trainer_artifact
        except Exception as e:
            raise VehicleInsuranceException(e, sys) from e

    def run_pipeline(self):
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(
                data_ingestion_artifact=data_ingestion_artifact
            )
            data_transformation_artifact = self.start_data_transformation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_artifact=data_validation_artifact,
            )
            model_trainer_artifact = self.start_model_training(
                data_transformation_artifact=data_transformation_artifact
            )
            logger.info(f"Data validation status: {data_validation_artifact.status}")
        except Exception as e:
            raise VehicleInsuranceException(e, sys) from e
