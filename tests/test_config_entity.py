from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from vehicle_insurance.entities.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    DataValidationConfig,
    ModelEvaluationConfig,
    ModelTrainerConfig,
    TrainingPipelineConfig,
)
from vehicle_insurance.pipeline.training_pipeline import TrainingPipeline


def test_pipeline_run_configs_get_unique_artifact_roots():
    first = TrainingPipelineConfig()
    second = TrainingPipelineConfig()

    assert first.artifact_dir != second.artifact_dir
    assert first.timestamp != second.timestamp
    assert first.artifact_dir is not None
    assert second.artifact_dir is not None
    assert Path(first.artifact_dir).name != Path(second.artifact_dir).name


def test_independent_stage_configs_do_not_reuse_import_time_artifact_root():
    first = DataIngestionConfig()
    second = DataIngestionConfig()

    assert first.artifact_dir != second.artifact_dir
    assert first.data_ingestion_dir != second.data_ingestion_dir
    assert first.feature_store_file_path != second.feature_store_file_path


def test_one_training_pipeline_shares_one_artifact_root_across_all_stages():
    pipeline = TrainingPipeline()
    expected_root = pipeline.training_pipeline_config.artifact_dir

    assert expected_root is not None
    assert pipeline.data_ingestion_config.artifact_dir == expected_root
    assert pipeline.data_validation_config.artifact_dir == expected_root
    assert pipeline.data_transformation_config.artifact_dir == expected_root
    assert pipeline.model_trainer_config.artifact_dir == expected_root
    assert pipeline.model_evaluation_config.artifact_dir == expected_root

    assert Path(pipeline.data_ingestion_config.data_ingestion_dir).parent == Path(
        expected_root
    )
    assert Path(pipeline.data_validation_config.data_validation_dir).parent == Path(
        expected_root
    )
    assert Path(
        pipeline.data_transformation_config.data_transformation_dir
    ).parent == Path(expected_root)
    assert Path(pipeline.model_trainer_config.model_trainer_dir).parent == Path(
        expected_root
    )
    assert Path(pipeline.model_evaluation_config.model_evaluation_dir).parent == Path(
        expected_root
    )


def test_run_root_generation_remains_unique_for_parallel_config_creation():
    with ThreadPoolExecutor(max_workers=8) as executor:
        configs = list(executor.map(lambda _: TrainingPipelineConfig(), range(32)))

    artifact_roots = [config.artifact_dir for config in configs]
    timestamps = [config.timestamp for config in configs]

    assert len(set(artifact_roots)) == len(configs)
    assert len(set(timestamps)) == len(configs)


def test_explicit_stage_paths_remain_unchanged(tmp_path):
    ingestion_dir = tmp_path / "ingestion"
    validation_dir = tmp_path / "validation"
    transformation_dir = tmp_path / "transformation"
    trainer_dir = tmp_path / "trainer"
    evaluation_dir = tmp_path / "evaluation"

    ingestion = DataIngestionConfig(data_ingestion_dir=str(ingestion_dir))
    validation = DataValidationConfig(data_validation_dir=str(validation_dir))
    transformation = DataTransformationConfig(
        data_transformation_dir=str(transformation_dir)
    )
    trainer = ModelTrainerConfig(model_trainer_dir=str(trainer_dir))
    evaluation = ModelEvaluationConfig(model_evaluation_dir=str(evaluation_dir))

    assert ingestion.data_ingestion_dir == str(ingestion_dir)
    assert validation.data_validation_dir == str(validation_dir)
    assert transformation.data_transformation_dir == str(transformation_dir)
    assert trainer.model_trainer_dir == str(trainer_dir)
    assert evaluation.model_evaluation_dir == str(evaluation_dir)
