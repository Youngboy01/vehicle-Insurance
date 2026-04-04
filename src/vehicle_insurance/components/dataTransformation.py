import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from vehicle_insurance.exceptions.handle_exception import VehicleInsuranceException
from vehicle_insurance.logger.logger_config import setup_logger
from vehicle_insurance.entities.config_entity import DataTransformationConfig
from vehicle_insurance.entities.artifact_entity import (
    DataTransformationArtifact,
    DataIngestionArtifact,
    DataValidationArtifact,
)
from vehicle_insurance.utils import read_yaml_file, save_object, save_numpy_array_data
from vehicle_insurance.constants import TARGET_COLUMN, SCHEMA_FILE_PATH
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


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
            num_features = self._schema_config["num_features"]
            rs_columns = self._schema_config["rs_columns"]
            cat_features = self._schema_config["cat_features"]

            logger.info(f"Numerical features: {num_features}")
            logger.info(f"Robust scaler features: {rs_columns}")
            logger.info(f"Categorical features: {cat_features}")
            preprocessor = ColumnTransformer(
                transformers=[
                    ("standard_scaler", StandardScaler(), num_features),
                    ("robust_scaler", RobustScaler(), rs_columns),
                    (
                        "one_hot_encoder",
                        OneHotEncoder(
                            drop="first", handle_unknown="ignore", sparse_output=False
                        ),
                        cat_features,
                    ),
                ],
                remainder="passthrough",
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
                raise VehicleInsuranceException(
                    Exception(
                        "Data validation failed. Cannot proceed with data transformation."
                    ),
                    sys,
                )
            logger.info("Data validation status: Passed. ")
            logger.info("Reading training and testing data")

            train_df = self.read_data(self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(self.data_ingestion_artifact.test_file_path)
            logger.info(
                f"Successfully read training and testing data with shapes {train_df.shape} and {test_df.shape} respectively."
            )

            drop_columns = self._schema_config["drop_columns"]
            logger.info(f"Dropping columns: {drop_columns}")
            x_train = train_df.drop(columns=drop_columns + [TARGET_COLUMN])
            y_train = train_df[TARGET_COLUMN].values
            x_test = test_df.drop(columns=drop_columns + [TARGET_COLUMN])
            y_test = test_df[TARGET_COLUMN].values

            logger.info(
                f"train class distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}"
            )
            logger.info(
                f"test class distribution: {dict(zip(*np.unique(y_test, return_counts=True)))}"
            )

            logger.info("Fitting transformer peipeline on training data")
            preprocessor = self.get_data_transformer_object()
            x_train_transformed = preprocessor.fit_transform(x_train)
            logger.info(
                f"Train tranformation completed successfully with shape: {x_train_transformed.shape}"
            )
            logger.info(
                "Transforming testing data using the fitted transformer pipeline"
            )
            x_test_transformed = preprocessor.transform(x_test)
            logger.info(
                f"Test transformation completed successfully with shape: {x_test_transformed.shape}"
            )

            train_arr = np.c_[x_train_transformed, y_train]
            test_arr = np.c_[x_test_transformed, y_test]

            logger.info(
                f"Final arrays for train and test data created successfully with shapes {train_arr.shape} and {test_arr.shape} respectively."
            )

            save_numpy_array_data(
                self.data_transformation_config.transformed_train_file_path, train_arr
            )
            save_numpy_array_data(
                self.data_transformation_config.transformed_test_file_path, test_arr
            )
            save_object(
                self.data_transformation_config.transformer_object_file_path,
                preprocessor,
            )

            logger.info(
                f"Transformed train and test data saved successfully at {self.data_transformation_config.transformed_train_file_path} and {self.data_transformation_config.transformed_test_file_path} respectively."
            )
            logger.info(
                f"Transformer object saved successfully at {self.data_transformation_config.transformer_object_file_path}"
            )

            data_transformation_artifact = DataTransformationArtifact(
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                transformer_object_file_path=self.data_transformation_config.transformer_object_file_path,
            )

            logger.info(
                f"Data transformation artifact created successfully with details: {data_transformation_artifact}"
            )
            return data_transformation_artifact
        except Exception as e:
            raise VehicleInsuranceException(e, sys) from e
