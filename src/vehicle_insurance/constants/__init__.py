import os

DATABASE_NAME = "INSURANCE"
COLLECTION_NAME = "INSURANCE_DATA"
MONGODB_URL_KEY = "MONGODB_URL"
PIPELINE_NAME: str = ""
ARTIFACT_DIR: str = "artifact"

FILE_NAME: str = "data.csv"
TRAIN_FILE_NAME: str = "train.csv"
VALIDATION_FILE_NAME: str = "validation.csv"
TEST_FILE_NAME: str = "test.csv"
SCHEMA_FILE_PATH: str = os.path.join("configs", "schema.yaml")
MODEL_SPEC_FILE_PATH: str = os.path.join("configs", "model_spec.yaml")

# data ingestion related constants
DATA_INGESTION_COLLECTION_NAME: str = "INSURANCE_DATA"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"


# data validation related constants
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_REPORT_FILE_NAME: str = "report.json"

# data transformation related constants
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed_data"
DATA_TRANSFORMATION_TRANSFORMED_TRAIN_FILE_NAME: str = "train.npy"
DATA_TRANSFORMATION_TRANSFORMED_VALIDATION_FILE_NAME: str = "validation.npy"
DATA_TRANSFORMATION_TRANSFORMED_TEST_FILE_NAME: str = "test.npy"
DATA_TRANSFORMATION_TRANSFORMER_OBJECT_FILE_NAME: str = "preprocessor.pkl"
TARGET_COLUMN: str = "Response"


# model trainer related constants
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_FILE_NAME: str = "model.pkl"
MODEL_TRAINER_METRICS_FILE_NAME: str = "metrics.json"

# mlflow related constants
MLFLOW_TRACKING_URI_KEY: str = "MLFLOW_TRACKING_URI"
MLFLOW_TRACKING_USERNAME_KEY: str = "MLFLOW_TRACKING_USERNAME"
MLFLOW_TRACKING_PASSWORD_KEY: str = "MLFLOW_TRACKING_PASSWORD"
MLFLOW_EXPERIMENT_NAME: str = "Vehicle_Insurance"
MLFLOW_REGISTER_MODEL_NAME: str = "Vehicle_Insurance_LGBM"
MLFLOW_CHAMPION_ALIAS: str = "Champion"
MLFLOW_CHALLENGER_ALIAS: str = "Challenger"

# model evaluation related constants
MODEL_EVALUATION_DIR_NAME: str = "model_evaluation"
MODEL_EVALUATION_REPORT_FILE_NAME: str = "report.json"
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE: float = 0.0
