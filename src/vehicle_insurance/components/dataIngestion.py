import json
import os
import sys
from datetime import datetime, timezone
from math import isclose

import pandas as pd
from sklearn.model_selection import train_test_split

from vehicle_insurance.constants import SCHEMA_FILE_PATH, TARGET_COLUMN
from vehicle_insurance.contracts.data_contract import DataContractValidator
from vehicle_insurance.data_access.Insurance_data import InsuranceData
from vehicle_insurance.entities.artifact_entity import DataIngestionArtifact
from vehicle_insurance.entities.config_entity import DataIngestionConfig
from vehicle_insurance.exceptions.handle_exception import VehicleInsuranceException
from vehicle_insurance.logger.logger_config import setup_logger
from vehicle_insurance.utils.source_manifest import (
    canonical_dataframe_bytes,
    finalize_source_manifest,
    sha256_file,
    validate_source_manifest,
    write_json,
)

logger = setup_logger()


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig) -> None:
        try:
            self.data_ingestion_config = data_ingestion_config
            self.contract_validator = DataContractValidator(
                schema_path=SCHEMA_FILE_PATH
            )
            self.source_manifest: dict | None = None
        except Exception as error:
            logger.exception("Failed to initialize data ingestion")
            raise VehicleInsuranceException(error, sys) from error

    def _load_existing_snapshot(self):
        feature_store_path = self.data_ingestion_config.feature_store_file_path
        manifest_path = self.data_ingestion_config.source_manifest_file_path
        abort_marker_path = os.path.join(
            os.path.dirname(manifest_path),
            "extraction_aborted.json",
        )
        if os.path.isfile(abort_marker_path):
            raise RuntimeError(
                "This artifact already recorded an aborted MongoDB extraction; "
                "use a new artifact directory for a new snapshot"
            )
        feature_store_exists = os.path.isfile(feature_store_path)
        manifest_exists = os.path.isfile(manifest_path)
        if feature_store_exists != manifest_exists:
            raise RuntimeError(
                "Source snapshot is incomplete: data.csv and source_manifest.json "
                "must be created and retained together"
            )
        if not feature_store_exists:
            return None

        with open(manifest_path, encoding="utf-8") as manifest_file:
            manifest = json.load(manifest_file)
        dataframe = pd.read_csv(feature_store_path)
        validate_source_manifest(manifest, dataframe)
        dataframe = self.contract_validator.validate(
            dataframe=dataframe,
            mode="training",
        )
        if sha256_file(feature_store_path) != manifest["snapshot_file_sha256"]:
            raise RuntimeError(
                "Immutable source snapshot file hash does not match its manifest"
            )
        logger.info(
            "Reusing verified immutable source snapshot: snapshot_id=%s",
            manifest["snapshot_id"],
        )
        return dataframe, manifest

    def _record_aborted_extraction(self, error: Exception) -> None:
        manifest_path = self.data_ingestion_config.source_manifest_file_path
        marker_path = os.path.join(
            os.path.dirname(manifest_path),
            "extraction_aborted.json",
        )
        if os.path.exists(marker_path):
            return
        marker_directory = os.path.dirname(marker_path) or "."
        os.makedirs(marker_directory, exist_ok=True)
        temporary_marker_path = f"{marker_path}.tmp"
        try:
            write_json(
                temporary_marker_path,
                {
                    "status": "aborted",
                    "aborted_at_utc": datetime.now(timezone.utc).isoformat(),
                    "reason": str(error),
                    "retry_policy": "new_artifact_directory_required",
                },
            )
            os.replace(temporary_marker_path, marker_path)
        finally:
            if os.path.exists(temporary_marker_path):
                os.remove(temporary_marker_path)

    def _persist_snapshot(self, dataframe: pd.DataFrame, base_manifest: dict) -> dict:
        feature_store_path = self.data_ingestion_config.feature_store_file_path
        manifest_path = self.data_ingestion_config.source_manifest_file_path
        if os.path.exists(feature_store_path) or os.path.exists(manifest_path):
            raise RuntimeError(
                "A source snapshot already exists at this artifact location; "
                "refusing to overwrite an immutable snapshot"
            )

        manifest = finalize_source_manifest(
            base_manifest,
            dataframe,
            snapshot_file_name=os.path.basename(feature_store_path),
        )
        directory = os.path.dirname(feature_store_path) or "."
        os.makedirs(directory, exist_ok=True)
        temporary_csv_path = f"{feature_store_path}.tmp"
        temporary_manifest_path = f"{manifest_path}.tmp"
        try:
            with open(temporary_csv_path, "wb") as snapshot_file:
                snapshot_file.write(canonical_dataframe_bytes(dataframe))
            if sha256_file(temporary_csv_path) != manifest["snapshot_file_sha256"]:
                raise RuntimeError("Source snapshot file hash changed during write")
            write_json(temporary_manifest_path, manifest)
            os.replace(temporary_csv_path, feature_store_path)
            os.replace(temporary_manifest_path, manifest_path)
        finally:
            for temporary_path in (temporary_csv_path, temporary_manifest_path):
                if os.path.exists(temporary_path):
                    os.remove(temporary_path)

        validate_source_manifest(manifest, pd.read_csv(feature_store_path))
        return manifest

    def export_data_into_feature_store(self):
        try:
            logger.info("Exporting data from mongodb to feature store")
            existing_snapshot = self._load_existing_snapshot()
            if existing_snapshot is not None:
                dataframe, self.source_manifest = existing_snapshot
                return dataframe

            insurance_data = InsuranceData()
            projection_fields = list(
                self.contract_validator.contract["schemas"]["training"][
                    "required_columns"
                ]
            )
            projection = {"_id": 0, **{field: 1 for field in projection_fields}}
            try:
                extraction = insurance_data.extract_snapshot(
                    collection_name=self.data_ingestion_config.collection_name,
                    query={},
                    projection=projection,
                    sort_field="id",
                )
            except Exception as error:
                self._record_aborted_extraction(error)
                raise
            extraction.manifest.update(
                {
                    "contract_version": self.contract_validator.contract_version,
                    "schema_version": self.contract_validator.contract_version,
                }
            )
            df = extraction.dataframe
            df = self.contract_validator.validate(
                dataframe=df,
                mode="training",
            )
            self.source_manifest = self._persist_snapshot(df, extraction.manifest)
            logger.info(
                "Data exported successfully from MongoDB to immutable feature "
                "store snapshot at path: %s",
                self.data_ingestion_config.feature_store_file_path,
            )
            logger.info(f"Dataframe shape: {df.shape}")
            return df
        except Exception as e:
            raise VehicleInsuranceException(e, sys) from e

    def split_data_as_train_validation_test(self, df):
        try:
            train_ratio, validation_ratio, test_ratio, random_state = (
                self._get_split_config()
            )
            logger.info(
                "Splitting data using stratified protocol: train=%.4f "
                "validation=%.4f test=%.4f random_state=%d",
                train_ratio,
                validation_ratio,
                test_ratio,
                random_state,
            )

            train_set, remainder_set = train_test_split(
                df,
                test_size=1.0 - train_ratio,
                random_state=random_state,
                stratify=df[TARGET_COLUMN],
            )

            # Match experimentation_v2.ipynb exactly.  sklearn returns the
            # first partition as the retained portion of remainder_set; the
            # notebook names that partition X_test and the second partition
            # X_val.  Keep those semantic names here so downstream artifacts
            # and metrics use the same customer IDs as the reference run.
            test_set, validation_set = train_test_split(
                remainder_set,
                test_size=test_ratio / (validation_ratio + test_ratio),
                random_state=random_state,
                stratify=remainder_set[TARGET_COLUMN],
            )

            self._assert_split_integrity(
                df,
                train_set,
                validation_set,
                test_set,
            )

            logger.info(
                "Data split successfully: train=%s validation=%s test=%s",
                train_set.shape,
                validation_set.shape,
                test_set.shape,
            )
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)
            train_set.to_csv(
                self.data_ingestion_config.training_file_path, index=False, header=True
            )
            validation_set.to_csv(
                self.data_ingestion_config.validation_file_path,
                index=False,
                header=True,
            )
            test_set.to_csv(
                self.data_ingestion_config.testing_file_path, index=False, header=True
            )
            logger.info(
                "Train, validation, and test files saved at: %s, %s, %s",
                self.data_ingestion_config.training_file_path,
                self.data_ingestion_config.validation_file_path,
                self.data_ingestion_config.testing_file_path,
            )
            return train_set, validation_set, test_set
        except Exception as error:
            logger.exception("Failed to create train/validation/test split")
            raise VehicleInsuranceException(error, sys) from error

    def _get_split_config(self) -> tuple[float, float, float, int]:
        protocol = self.contract_validator.contract["data_protocol"]
        if protocol.get("split_protocol") != "stratified_80_10_10":
            raise ValueError(
                f"Unsupported split protocol: {protocol.get('split_protocol')}"
            )

        try:
            train_ratio = float(protocol["train_ratio"])
            validation_ratio = float(protocol["validation_ratio"])
            test_ratio = float(protocol["test_ratio"])
            random_state = int(protocol["random_state"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "data_protocol must define numeric train_ratio, "
                "validation_ratio, test_ratio, and integer random_state"
            ) from error

        ratios = (train_ratio, validation_ratio, test_ratio)
        if any(ratio <= 0 or ratio >= 1 for ratio in ratios):
            raise ValueError("Split ratios must each be greater than 0 and less than 1")
        if not isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                "train_ratio + validation_ratio + test_ratio must equal 1.0"
            )

        return train_ratio, validation_ratio, test_ratio, random_state

    @staticmethod
    def _assert_split_integrity(
        source_df: pd.DataFrame,
        train_set: pd.DataFrame,
        validation_set: pd.DataFrame,
        test_set: pd.DataFrame,
    ) -> None:
        train_ids = set(train_set["id"])
        validation_ids = set(validation_set["id"])
        test_ids = set(test_set["id"])
        source_ids = set(source_df["id"])

        if len(source_ids) != len(source_df):
            raise ValueError("Source data contains duplicate IDs")

        if train_ids & validation_ids:
            raise ValueError("Training and validation IDs overlap")
        if train_ids & test_ids:
            raise ValueError("Training and test IDs overlap")
        if validation_ids & test_ids:
            raise ValueError("Validation and test IDs overlap")

        split_ids = train_ids | validation_ids | test_ids
        if split_ids != source_ids:
            raise ValueError(
                "Train, validation, and test IDs do not cover the source data"
            )
        if len(train_set) + len(validation_set) + len(test_set) != len(source_df):
            raise ValueError(
                "Train, validation, and test row counts do not cover the source data"
            )

        for name, split in (
            ("training", train_set),
            ("validation", validation_set),
            ("test", test_set),
        ):
            if set(split[TARGET_COLUMN].unique()) != {0, 1}:
                raise ValueError(f"{name} split does not contain both target classes")

    def initiate_data_ingestion(self):
        try:
            df = self.export_data_into_feature_store()
            logger.info("Exporting data into feature store")
            train_set, validation_set, test_set = (
                self.split_data_as_train_validation_test(df)
            )
            logger.info("Created train, validation, and test datasets")
            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                validation_file_path=self.data_ingestion_config.validation_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path,
                feature_store_file_path=self.data_ingestion_config.feature_store_file_path,
                source_manifest_file_path=self.data_ingestion_config.source_manifest_file_path,
                source_manifest=self.source_manifest,
            )
            logger.info(
                f"Data ingestion artifact created successfully with data ingestion artifact: {data_ingestion_artifact}"
            )
            return data_ingestion_artifact
        except Exception as e:
            raise VehicleInsuranceException(e, sys) from e
