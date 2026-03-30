import os
import sys
from sklearn.model_selection import train_test_split

from vehicle_insurance.entities.config_entity import DataIngestionConfig
from vehicle_insurance.entities.artifact_entity import DataIngestionArtifact
from vehicle_insurance.data_access.Insurance_data import InsuranceData
from vehicle_insurance.exceptions.handle_exception import VehicleInsuranceException
from vehicle_insurance.logger.logger_config import setup_logger


logger = setup_logger()


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig) -> None:
        self.data_ingestion_config = data_ingestion_config

    def export_data_into_feature_store(self):
        try:
            logger.info("Exporting data from mongodb to feature store")
            insurance_data = InsuranceData()
            df = insurance_data.get_collection(
                collection_name=self.data_ingestion_config.collection_name
            )
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)
            df.to_csv(feature_store_file_path, index=False, header=True)
            logger.info(
                f"Data exported successfully from mongodb to feature store at path: {feature_store_file_path}"
            )
            logger.info(f"Dataframe shape: {df.shape}")
            return df
        except Exception as e:
            raise VehicleInsuranceException(e, sys) from e

    def split_data_as_train_test(self, df):
        try:
            logger.info("Splitting data into train and test set")
            train_set, test_set = train_test_split(
                df,
                test_size=self.data_ingestion_config.train_test_split_ratio,
                random_state=42,
            )
            logger.info(
                f"Data split into train and test set successfully with train set shape: {train_set.shape} and test set shape: {test_set.shape}"
            )
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)
            train_set.to_csv(
                self.data_ingestion_config.training_file_path, index=False, header=True
            )
            test_set.to_csv(
                self.data_ingestion_config.testing_file_path, index=False, header=True
            )
            logger.info(
                f"Train and test set saved successfully at path: {self.data_ingestion_config.training_file_path} and {self.data_ingestion_config.testing_file_path}"
            )
            return train_set, test_set
        except Exception as e:
            raise VehicleInsuranceException(e, sys) from e

    def initiate_data_ingestion(self):
        try:
            df = self.export_data_into_feature_store()
            logger.info("Exporting data into feature store")
            train_set, test_set = self.split_data_as_train_test(df)
            logger.info("splitted data into train and test set")
            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path,
            )
            logger.info(
                f"Data ingestion artifact created successfully with data ingestion artifact: {data_ingestion_artifact}"
            )
            return data_ingestion_artifact
        except Exception as e:
            raise VehicleInsuranceException(e, sys) from e
