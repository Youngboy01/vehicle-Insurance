import sys

from dotenv import load_dotenv

from vehicle_insurance.components.dataIngestion import DataIngestion
from vehicle_insurance.components.dataTransformation import DataTransformation
from vehicle_insurance.components.dataValidation import DataValidation
from vehicle_insurance.components.modelEvaluation import ModelEvaluation
from vehicle_insurance.components.modelTrainer import ModelTrainer
from vehicle_insurance.entities.artifact_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
    DataValidationArtifact,
    ModelEvaluationArtifact,
    ModelTrainerArtifact,
)
from vehicle_insurance.entities.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    DataValidationConfig,
    ModelEvaluationConfig,
    ModelTrainerConfig,
)
from vehicle_insurance.exceptions.handle_exception import VehicleInsuranceException
from vehicle_insurance.logger.logger_config import setup_logger

logger = setup_logger()


class TrainingPipeline:
    """Orchestrate the complete vehicle-insurance training workflow."""

    def __init__(self):
        try:
            # Load local development configuration without overriding values
            # supplied by the execution environment or deployment platform.
            load_dotenv(override=False)
            self.data_ingestion_config = DataIngestionConfig()
            self.data_validation_config = DataValidationConfig()
            self.data_transformation_config = DataTransformationConfig()
            self.model_trainer_config = ModelTrainerConfig()
            self.model_evaluation_config = ModelEvaluationConfig()
        except Exception as e:
            raise VehicleInsuranceException(e, sys) from e

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logger.info("Starting data ingestion component")
            data_ingestion = DataIngestion(
                data_ingestion_config=self.data_ingestion_config
            )
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logger.info("Received data ingestion artifact: %s", data_ingestion_artifact)
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
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_transformation_artifact: DataTransformationArtifact,
    ) -> ModelTrainerArtifact:
        try:
            logger.info("Starting model training component")
            model_trainer = ModelTrainer(
                model_trainer_config=self.model_trainer_config,
                data_transformation_artifact=data_transformation_artifact,
                data_ingestion_artifact=data_ingestion_artifact,
            )
            model_trainer_artifact = model_trainer.train()
            logger.info(
                f"Model training component completed with artifact: {model_trainer_artifact}"
            )
            return model_trainer_artifact
        except Exception as e:
            raise VehicleInsuranceException(e, sys) from e

    def start_model_evaluation(
        self, model_trainer_artifact: ModelTrainerArtifact
    ) -> ModelEvaluationArtifact:
        try:
            logger.info("Starting model evaluation component")
            model_evaluation = ModelEvaluation(
                model_evaluation_config=self.model_evaluation_config,
                model_trainer_artifact=model_trainer_artifact,
            )
            model_evaluation_artifact = model_evaluation.evaluate_model()
            logger.info(
                f"Model evaluation component completed with artifact: {model_evaluation_artifact}"
            )
            return model_evaluation_artifact
        except Exception as e:
            raise VehicleInsuranceException(e, sys) from e

    def run_pipeline(self) -> ModelEvaluationArtifact:
        """Run all pipeline stages and return the evaluation artifact."""
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(
                data_ingestion_artifact=data_ingestion_artifact
            )
            if not data_validation_artifact.status:
                raise ValueError(
                    "Data validation failed; transformation and training are blocked. "
                    f"See {data_validation_artifact.report_file_path}"
                )
            data_transformation_artifact = self.start_data_transformation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_artifact=data_validation_artifact,
            )
            model_trainer_artifact = self.start_model_training(
                data_ingestion_artifact=data_ingestion_artifact,
                data_transformation_artifact=data_transformation_artifact
            )
            model_evaluation_artifact = self.start_model_evaluation(
                model_trainer_artifact=model_trainer_artifact
            )
            if not model_evaluation_artifact.evaluation_passed:
                logger.info(
                    "Pipeline completed - candidate failed one or more evaluation gates; no promotion was performed."
                )
            elif model_evaluation_artifact.promotion_status == "disabled":
                logger.info(
                    "Pipeline completed - candidate passed evaluation gates, but promotion is disabled."
                )
            else:
                logger.info(
                    "Pipeline completed - candidate passed evaluation gates; promotion was not performed by this evaluator."
                )
            logger.info(
                "End-to-end training pipeline completed: decision=%s report=%s",
                model_evaluation_artifact.decision,
                model_evaluation_artifact.evaluation_report_file_path,
            )
            return model_evaluation_artifact
        except Exception as e:
            raise VehicleInsuranceException(e, sys) from e


if __name__ == "__main__":
    TrainingPipeline().run_pipeline()
