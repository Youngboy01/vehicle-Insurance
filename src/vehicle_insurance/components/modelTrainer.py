import os
import sys
import json
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    recall_score,
    precision_score,
    f1_score,
)

from vehicle_insurance.entities.config_entity import ModelTrainerConfig
from vehicle_insurance.entities.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from vehicle_insurance.exceptions.handle_exception import VehicleInsuranceException
from vehicle_insurance.logger.logger_config import setup_logger
from vehicle_insurance.utils import (
    load_object,
    load_numpy_array_data,
    save_object,
)

logger = setup_logger()


class ModelTrainer:
    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
        data_transformation_artifact: DataTransformationArtifact,
    ):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise VehicleInsuranceException(e, sys) from e

    def _load_and_split_data(self, file_path: str):
        try:
            arr = load_numpy_array_data(file_path)
            X = arr[:, :-1]
            y = arr[:, -1]
            return X, y
        except Exception as e:
            raise VehicleInsuranceException(e, sys) from e

    def _compute_metrics(self, model: LGBMClassifier, X, y):
        try:
            y_true = np.asarray(y).ravel()
            y_pred = np.asarray(model.predict(X)).ravel()
            y_proba = np.asarray(model.predict_proba(X))[:, 1].ravel()
            metrics = {
                "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
                "precision": round(float(precision_score(y_true, y_pred)), 4),
                "recall": round(float(recall_score(y_true, y_pred)), 4),
                "f1_score": round(float(f1_score(y_true, y_pred)), 4),
                "roc_auc": round(float(roc_auc_score(y_true, y_proba)), 4),
            }
            return metrics
        except Exception as e:
            raise VehicleInsuranceException(e, sys) from e

    def train(self) -> ModelTrainerArtifact:
        try:
            logger.info("Model Trainer started.")
            logger.info("Loading transformed train and test arrays ")
            X_train, y_train = self._load_and_split_data(
                self.data_transformation_artifact.transformed_train_file_path
            )
            X_test, y_test = self._load_and_split_data(
                self.data_transformation_artifact.transformed_test_file_path
            )
            logger.info(
                f"Loading completed with shapes: X_train: {X_train.shape}, y_train: {y_train.shape}, X_test: {X_test.shape}, y_test: {y_test.shape}"
            )

            logger.info(
                f"Loading experiment bundle configs from {self.model_trainer_config.experiment_pkl_path} "
            )

            bundle = load_object(self.model_trainer_config.experiment_pkl_path)

            best_params = bundle["metadata"]["best_params"]
            experiment_test_metrics = bundle["metadata"]["test_metrics"]
            logger.info(f"Best hyperparameters: {best_params}")
            logger.info(f"Experiment test metrics: {experiment_test_metrics}")

            logger.info("Training LightGBM with is_unbalanced=True")
            model = LGBMClassifier(
                **best_params, is_unbalance=True, random_state=42, verbosity=-1
            )
            model.fit(X_train, y_train)
            logger.info("Model training completed.")

            # lets evaluate
            logger.info("Evaluating on trian and test data")
            train_metrics = self._compute_metrics(model, X_train, y_train)
            test_metrics = self._compute_metrics(model, X_test, y_test)
            logger.info(f"Train metrics: {train_metrics}")
            logger.info(f"Test metrics: {test_metrics}")

            # lets now save the model and the metrics
            logger.info("Saving the trained model")
            save_object(self.model_trainer_config.trained_model_file_path, model)
            logger.info("Saving the metrics")
            metrics = {
                "best_params": best_params,
                "experiment_test_metrics": experiment_test_metrics,
                "pipeline_train_metrics": train_metrics,
                "pipeline_test_metrics": test_metrics,
            }

            os.makedirs(
                os.path.dirname(self.model_trainer_config.metrics_file_path),
                exist_ok=True,
            )
            with open(self.model_trainer_config.metrics_file_path, "w") as f:
                json.dump(metrics, f, indent=4)

            logger.info(
                f"Metrics saved to {self.model_trainer_config.metrics_file_path}"
            )

            # build the artifact and return
            artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metrics_file_path=self.model_trainer_config.metrics_file_path,
                train_metrics=train_metrics,
                test_metrics=test_metrics,
            )
            logger.info(f"Model Trainer artifact: {artifact}")
            logger.info("Model Trainer completed.")
            return artifact
        except Exception as e:
            raise VehicleInsuranceException(e, sys) from e
