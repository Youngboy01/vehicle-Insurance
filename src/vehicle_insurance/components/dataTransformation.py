import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from vehicle_insurance.constants import SCHEMA_FILE_PATH, TARGET_COLUMN
from vehicle_insurance.contracts.data_contract import DataContractValidator
from vehicle_insurance.entities.artifact_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
    DataValidationArtifact,
)
from vehicle_insurance.entities.config_entity import DataTransformationConfig
from vehicle_insurance.exceptions.handle_exception import VehicleInsuranceException
from vehicle_insurance.logger.logger_config import setup_logger
from vehicle_insurance.utils import read_yaml_file, save_numpy_array_data, save_object

logger = setup_logger()


class DataTransformation:
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_artifact: DataValidationArtifact,
        data_transformation_config: DataTransformationConfig,
    ) -> None:
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
            self.contract_validator = DataContractValidator(SCHEMA_FILE_PATH)
        except Exception as e:
            raise VehicleInsuranceException(e, sys) from e

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise VehicleInsuranceException(e, sys) from e

    def get_data_transformer_object(self):
        try:
            logger.info("Creating data transformer object")
            preprocessing_config = self._schema_config["preprocessing"]
            numeric_features = preprocessing_config["numeric_features"]
            binary_features = preprocessing_config["binary_features"]
            categorical_features = preprocessing_config["categorical_features"]
            one_hot_drop = preprocessing_config["one_hot_drop"]
            handle_unknown = preprocessing_config["handle_unknown"]
            remainder = preprocessing_config["remainder"]

            if preprocessing_config["numeric_scaling"] != "none":
                raise ValueError("Frozen preprocessing requires numeric_scaling='none'")

            passthrough_features = numeric_features + binary_features

            logger.info("Numeric features: %s", numeric_features)
            logger.info("Binary passthrough features: %s", binary_features)
            logger.info("Categorical features: %s", categorical_features)
            logger.info("Passthrough features: %s", passthrough_features)
            preprocessor = ColumnTransformer(
                transformers=[
                    (
                        "categorical_encoder",
                        OneHotEncoder(
                            drop=one_hot_drop,
                            handle_unknown=handle_unknown,
                            sparse_output=False,
                        ),
                        categorical_features,
                    ),
                    ("passthrough", "passthrough", passthrough_features),
                ],
                remainder=remainder,
                verbose_feature_names_out=False,
            )
            pipeline = Pipeline(steps=[("preprocessor", preprocessor)])
            logger.info("Data transformer object created successfully")
            return pipeline
        except Exception as e:
            raise VehicleInsuranceException(e, sys) from e

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logger.info("Initiating data transformation")
            if not self.data_validation_artifact.status:
                raise ValueError(
                    "Data validation failed. Cannot proceed with data transformation."
                )
            logger.info("Data validation status: Passed. ")
            logger.info("Reading training, validation, and testing data")

            train_df = self.read_data(self.data_ingestion_artifact.trained_file_path)
            validation_df = self.read_data(
                self.data_ingestion_artifact.validation_file_path
            )
            test_df = self.read_data(self.data_ingestion_artifact.test_file_path)
            logger.info(
                "Successfully read training, validation, and testing data with "
                f"shapes {train_df.shape}, {validation_df.shape}, and "
                f"{test_df.shape} respectively."
            )

            validated_train_df = self.contract_validator.validate(
                train_df,
                mode="training",
            )
            validated_validation_df = self.contract_validator.validate(
                validation_df,
                mode="training",
            )
            validated_test_df = self.contract_validator.validate(
                test_df,
                mode="training",
            )

            raw_features = self._schema_config["features"]["canonical_order"]
            logger.info("Canonical raw features: %s", raw_features)

            x_train = validated_train_df.loc[:, raw_features]
            y_train = validated_train_df[TARGET_COLUMN].to_numpy()
            x_validation = validated_validation_df.loc[:, raw_features]
            y_validation = validated_validation_df[TARGET_COLUMN].to_numpy()
            x_test = validated_test_df.loc[:, raw_features]
            y_test = validated_test_df[TARGET_COLUMN].to_numpy()

            logger.info(
                f"train class distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}"
            )
            logger.info(
                "validation class distribution: %s",
                dict(zip(*np.unique(y_validation, return_counts=True))),
            )
            logger.info(
                f"test class distribution: {dict(zip(*np.unique(y_test, return_counts=True)))}"
            )

            logger.info("Fitting transformer pipeline on training data")
            preprocessor = self.get_data_transformer_object()
            x_train_transformed = preprocessor.fit_transform(x_train)
            logger.info(
                f"Train transformation completed successfully with shape: {x_train_transformed.shape}"
            )
            logger.info(
                "Transforming validation and testing data using the fitted "
                "training transformer pipeline"
            )
            x_validation_transformed = preprocessor.transform(x_validation)
            x_test_transformed = preprocessor.transform(x_test)
            logger.info(
                "Validation and test transformation completed successfully with "
                f"shapes: {x_validation_transformed.shape} and "
                f"{x_test_transformed.shape}"
            )

            expected_feature_count = self._schema_config["preprocessing"][
                "transformed_feature_count"
            ]
            transformed_feature_names = preprocessor.get_feature_names_out()
            actual_feature_count = len(transformed_feature_names)

            if actual_feature_count != expected_feature_count:
                raise ValueError(
                    "Transformed feature count does not match the frozen contract: "
                    f"expected={expected_feature_count}, "
                    f"actual={actual_feature_count}"
                )

            transformed_widths = {
                "train": x_train_transformed.shape[1],
                "validation": x_validation_transformed.shape[1],
                "test": x_test_transformed.shape[1],
            }
            if len(set(transformed_widths.values())) != 1:
                raise ValueError(
                    "Train, validation, and test transformed feature widths "
                    f"differ: {transformed_widths}"
                )

            logger.info(
                "Fitted pipeline produced %d transformed features",
                actual_feature_count,
            )

            train_arr = np.c_[x_train_transformed, y_train]
            validation_arr = np.c_[x_validation_transformed, y_validation]
            test_arr = np.c_[x_test_transformed, y_test]

            logger.info(
                "Final arrays for train, validation, and test data created "
                f"successfully with shapes {train_arr.shape}, "
                f"{validation_arr.shape}, and {test_arr.shape}."
            )

            save_numpy_array_data(
                self.data_transformation_config.transformed_train_file_path, train_arr
            )
            save_numpy_array_data(
                self.data_transformation_config.transformed_validation_file_path,
                validation_arr,
            )
            save_numpy_array_data(
                self.data_transformation_config.transformed_test_file_path, test_arr
            )
            save_object(
                self.data_transformation_config.transformer_object_file_path,
                preprocessor,
            )

            logger.info(
                "Transformed train, validation, and test data saved successfully "
                f"at {self.data_transformation_config.transformed_train_file_path}, "
                f"{self.data_transformation_config.transformed_validation_file_path}, "
                f"and {self.data_transformation_config.transformed_test_file_path}."
            )
            logger.info(
                f"Transformer object saved successfully at {self.data_transformation_config.transformer_object_file_path}"
            )

            data_transformation_artifact = DataTransformationArtifact(
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_validation_file_path=self.data_transformation_config.transformed_validation_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                transformer_object_file_path=self.data_transformation_config.transformer_object_file_path,
            )

            logger.info(
                f"Data transformation artifact created successfully with details: {data_transformation_artifact}"
            )
            return data_transformation_artifact
        except Exception as e:
            raise VehicleInsuranceException(e, sys) from e
