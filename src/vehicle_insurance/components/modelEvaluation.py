import os
import sys
import json
from mlflow.exceptions import MlflowException

from vehicle_insurance.constants import (
    MLFLOW_CHALLENGER_ALIAS,
    MLFLOW_CHAMPION_ALIAS,
    MLFLOW_REGISTER_MODEL_NAME,
)
from vehicle_insurance.utils import get_mlflow_client
from vehicle_insurance.entities.artifact_entity import (
    ModelEvaluationArtifact,
    ModelTrainerArtifact,
)
from vehicle_insurance.entities.config_entity import ModelEvaluationConfig
from vehicle_insurance.exceptions.handle_exception import VehicleInsuranceException
from vehicle_insurance.logger.logger_config import setup_logger

logger = setup_logger()


class ModelEvaluation:
    def __init__(
        self,
        model_evaluation_config: ModelEvaluationConfig,
        model_trainer_artifact: ModelTrainerArtifact,
    ):
        try:
            self.model_trainer_artifact = model_trainer_artifact
            self.model_evaluation_config = model_evaluation_config
        except Exception as e:
            raise VehicleInsuranceException(e, sys) from e

    def _get_champion_info(self, client):
        try:
            champion_version_obj = client.get_model_version_by_alias(
                MLFLOW_REGISTER_MODEL_NAME, MLFLOW_CHAMPION_ALIAS
            )
            champion_version = champion_version_obj.version
            champion_run_id = champion_version_obj.run_id
            logger.info(
                f"Champion model version: {champion_version}, run_id: {champion_run_id}"
            )
            run = client.get_run(champion_run_id)
            champion_metrics = run.data.metrics
            champion_recall = champion_metrics.get("test_recall", 0)
            logger.info(f"Champion model recall: {champion_recall} derived from MLflow")
            return champion_version, champion_recall
        except MlflowException as e:
            logger.warning(f"No champion model found in MLflow: {e}")
            return None, None
        except Exception as e:
            raise VehicleInsuranceException(e, sys) from e

    def _save_report(self, report: dict):
        try:
            os.makedirs(
                os.path.dirname(self.model_evaluation_config.report_file_path),
                exist_ok=True,
            )
            with open(
                self.model_evaluation_config.report_file_path, "w"
            ) as report_file:
                json.dump(report, report_file, indent=4)

            logger.info(
                f"Model evaluation report saved to {self.model_evaluation_config.report_file_path}"
            )
        except Exception as e:
            raise VehicleInsuranceException(e, sys) from e

    def evaluate_model(self):
        try:
            logger.info("Starting model evaluation process")
            client = get_mlflow_client()
            logger.info("Getting the challenger version and recall from MLflow")
            challenger_version_info = client.get_model_version_by_alias(
                MLFLOW_REGISTER_MODEL_NAME, MLFLOW_CHALLENGER_ALIAS
            )
            challenger_version = challenger_version_info.version
            challenger_recall = self.model_trainer_artifact.test_metrics["recall"]
            logger.info(
                f"Challenger model version: {challenger_version}, recall: {challenger_recall}"
            )
            champion_version, champion_recall = self._get_champion_info(client)
            threshold = self.model_evaluation_config.changed_threshold_score

            if champion_version is None:
                logger.info("No champion model found, accepting challenger by default")
                is_model_accepted = True
                final_champion_version = challenger_version
                client.set_registered_model_alias(
                    name=MLFLOW_REGISTER_MODEL_NAME,
                    version=challenger_version,
                    alias=MLFLOW_CHAMPION_ALIAS,
                )
                logger.info(
                    f"Challenger model version {challenger_version} set as champion in MLflow"
                )

                client.delete_registered_model_alias(
                    name=MLFLOW_REGISTER_MODEL_NAME, alias=MLFLOW_CHALLENGER_ALIAS
                )
                logger.info(
                    f"Removed challenger alias for version {challenger_version} in MLflow"
                )

            elif (challenger_recall - champion_recall) > threshold:
                logger.info(
                    f"Challenger model recall {challenger_recall} is better than champion recall {champion_recall} by {(challenger_recall - champion_recall):.4f}."
                )
                logger.info("Accepting challenger model and promoting to champion")
                is_model_accepted = True
                final_champion_version = challenger_version
                client.set_registered_model_alias(
                    name=MLFLOW_REGISTER_MODEL_NAME,
                    version=challenger_version,
                    alias=MLFLOW_CHAMPION_ALIAS,
                )
                logger.info(
                    f"Challenger model version {challenger_version} set as champion in MLflow"
                )
                client.delete_registered_model_alias(
                    name=MLFLOW_REGISTER_MODEL_NAME, alias=MLFLOW_CHALLENGER_ALIAS
                )
                logger.info(
                    f"Removed challenger alias for version {challenger_version} in MLflow"
                )

            else:
                logger.info(
                    f"Challenger model recall {challenger_recall} is not better than champion recall {champion_recall} by more than threshold {threshold}."
                )
                logger.info("Rejecting challenger model")
                is_model_accepted = False
                final_champion_version = champion_version
                client.delete_registered_model_alias(
                    name=MLFLOW_REGISTER_MODEL_NAME, alias=MLFLOW_CHALLENGER_ALIAS
                )
                logger.info(
                    f"Removed challenger alias for version {challenger_version} in MLflow"
                )

            report = {
                "is_model_accepted": is_model_accepted,
                "champion_model_version": final_champion_version,
                "challenger_model_version": challenger_version,
                "champion_recall": champion_recall,
                "challenger_recall": challenger_recall,
                "threshold": threshold,
                "decision": "challenger prompted"
                if is_model_accepted and champion_version is not None
                else "challenger accepted by default"
                if is_model_accepted
                else "champion retained",
            }
            self._save_report(report)
            artifact = ModelEvaluationArtifact(
                is_model_accepted=is_model_accepted,
                champion_model_version=final_champion_version,
                challenger_model_version=challenger_version,
                champion_recall=champion_recall,
                challenger_recall=challenger_recall,
                evaluation_report_file_path=self.model_evaluation_config.report_file_path,
            )
            logger.info(f"Model Evaluation artifact: {artifact}")
            logger.info("Model evaluation process completed")
            return artifact
        except Exception as e:
            raise VehicleInsuranceException(e, sys) from e
