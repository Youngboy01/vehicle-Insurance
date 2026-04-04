import os

DATABASE_NAME = "INSURANCE"
COLLECTION_NAME = "INSURANCE_DATA"
MONGODB_URL_KEY = "MONGODB_URL"
PIPELINE_NAME: str = ""
ARTIFACT_DIR: str = "artifact"

FILE_NAME: str = "data.csv"
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"
SCHEMA_FILE_PATH: str = os.path.join("configs", "schema.yaml")

# data ingestion related constants
DATA_INGESTION_COLLECTION_NAME: str = "INSURANCE_DATA"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.25


# data validation related constants
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_REPORT_FILE_NAME: str = "report.yaml"

# data transformation related constants
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed_data"
DATA_TRANSFORMATION_TRANSFORMED_TRAIN_FILE_NAME: str = "train.npy"
DATA_TRANSFORMATION_TRANSFORMED_TEST_FILE_NAME: str = "test.npy"
DATA_TRANSFORMATION_TRANSFORMER_OBJECT_FILE_NAME: str = "preprocessor.pkl"
TARGET_COLUMN: str = "Response"


# model trainer related constants
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_FILE_NAME: str = "model.pkl"
MODEL_TRAINER_METRICS_FILE_NAME: str = "metrics.json"
MODEL_TRAINER_EXPERIMENT_PKL_PATH: str = os.path.join(
    "artifacts", "models", "best_model_Round3_ClassWeight_LightGBM_is_unbalance.pkl"
)
MODEL_TRAINER_EXPECTED_SCORE: float = 0.7
