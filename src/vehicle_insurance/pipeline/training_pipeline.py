import json
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from vehicle_insurance.components.dataIngestion import DataIngestion
from vehicle_insurance.components.dataTransformation import DataTransformation
from vehicle_insurance.components.dataValidation import DataValidation
from vehicle_insurance.components.modelEvaluation import ModelEvaluation
from vehicle_insurance.components.modelTrainer import ModelTrainer
from vehicle_insurance.constants import MODEL_SPEC_FILE_PATH
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
    TrainingPipelineConfig,
)
from vehicle_insurance.exceptions.handle_exception import VehicleInsuranceException
from vehicle_insurance.logger.logger_config import setup_logger
from vehicle_insurance.utils import (
    capture_git_provenance,
    read_yaml_file,
    resolve_repository_root,
    validate_git_provenance,
)
from vehicle_insurance.verification.source_frozen_parity import (
    SourceFrozenParityCertification,
)

logger = setup_logger()


class TrainingPipeline:
    """Orchestrate the complete vehicle-insurance training workflow."""

    def __init__(self):
        try:
            # Load local development configuration without overriding values
            # supplied by the execution environment or deployment platform.
            load_dotenv(override=False)
            self.git_provenance = capture_git_provenance(
                resolve_repository_root()
            )
            self.training_pipeline_config = TrainingPipelineConfig()
            artifact_dir = self.training_pipeline_config.artifact_dir
            self.data_ingestion_config = DataIngestionConfig(
                artifact_dir=artifact_dir
            )
            self.data_validation_config = DataValidationConfig(
                artifact_dir=artifact_dir
            )
            self.data_transformation_config = DataTransformationConfig(
                artifact_dir=artifact_dir
            )
            self.model_trainer_config = ModelTrainerConfig(artifact_dir=artifact_dir)
            self.model_evaluation_config = ModelEvaluationConfig(
                artifact_dir=artifact_dir
            )
            self.source_frozen_parity_report: dict[str, Any] | None = None
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
                git_provenance=self.git_provenance,
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
            evaluation_context: dict[str, Any] = {"mode": "pipeline"}
            if self.source_frozen_parity_report is not None:
                evaluation_context["source_frozen_parity_stage"] = {
                    "status": self.source_frozen_parity_report.get("status"),
                    "report_path": self.source_frozen_parity_report.get(
                        "report_path"
                    ),
                    "lineage_recording": self.source_frozen_parity_report.get(
                        "lineage_recording", {}
                    ),
                }
            model_evaluation = ModelEvaluation(
                model_evaluation_config=self.model_evaluation_config,
                model_trainer_artifact=model_trainer_artifact,
                evaluation_context=evaluation_context,
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
            self._validate_git_preflight()
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
            self.source_frozen_parity_report = self.start_source_frozen_parity()
            if (
                self.source_frozen_parity_report.get("status") == "verified"
                and self.source_frozen_parity_report.get("lineage_recording", {}).get(
                    "local_metrics_json_updated"
                )
            ):
                self._reload_lineage_from_metrics(model_trainer_artifact)
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

    def _validate_git_preflight(self) -> None:
        """Block release-quality runs before ingestion or model creation."""
        model_spec = read_yaml_file(MODEL_SPEC_FILE_PATH)
        git_policy = (
            model_spec.get("evaluation_policy", {})
            .get("lineage", {})
            .get("git")
        )
        if not isinstance(git_policy, dict):
            raise ValueError(
                "evaluation_policy.lineage.git must be a dictionary"
            )
        validate_git_provenance(self.git_provenance, git_policy)
        logger.info(
            "Git provenance preflight passed: commit=%s branch=%s",
            self.git_provenance["commit"],
            self.git_provenance["branch"],
        )

    def start_source_frozen_parity(self) -> dict[str, Any]:
        """Certify and attach source-frozen parity before final evaluation.

        A failed certificate is still written and passed to the canonical
        evaluator as evidence. A verified certificate updates the trainer
        metrics lineage on disk; ``run_pipeline`` reloads that lineage into the
        in-memory trainer artifact before evaluation.
        """
        try:
            candidate_artifact_dir = (
                Path(self.model_trainer_config.model_trainer_dir).resolve().parent
            )
            logger.info(
                "Starting source-frozen parity stage for candidate artifact: %s",
                candidate_artifact_dir,
            )
            certification = SourceFrozenParityCertification(
                candidate_artifact_dir=candidate_artifact_dir,
                apply_lineage=True,
                log_mlflow_evidence=True,
            )
            report = certification.run()
            report["report_path"] = str(certification.report_path)
            logger.info(
                "Source-frozen parity stage completed: status=%s report=%s",
                report.get("status"),
                certification.report_path,
            )
            return report
        except Exception as error:
            raise VehicleInsuranceException(error, sys) from error

    @staticmethod
    def _reload_lineage_from_metrics(
        model_trainer_artifact: ModelTrainerArtifact,
    ) -> None:
        """Refresh the in-memory artifact after parity updates metrics.json."""
        metrics_path = Path(model_trainer_artifact.metrics_file_path)
        try:
            with metrics_path.open(encoding="utf-8") as metrics_file:
                payload = json.load(metrics_file)
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                "Could not reload trainer lineage after parity certification: "
                f"{metrics_path}"
            ) from error

        lineage = payload.get("lineage")
        if not isinstance(lineage, dict):
            raise ValueError(
                "Trainer metrics has no lineage after parity certification: "
                f"{metrics_path}"
            )
        model_trainer_artifact.lineage = lineage


if __name__ == "__main__":
    TrainingPipeline().run_pipeline()
