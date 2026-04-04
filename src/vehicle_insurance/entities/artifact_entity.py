from dataclasses import dataclass


@dataclass
class DataIngestionArtifact:
    trained_file_path: str
    test_file_path: str


@dataclass
class DataValidationArtifact:
    report_file_path: str
    message: str
    status: bool
    
    
@dataclass
class DataTransformationArtifact:
    transformed_train_file_path: str
    transformed_test_file_path: str
    transformer_object_file_path: str
