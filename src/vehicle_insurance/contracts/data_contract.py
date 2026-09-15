from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import yaml
from pandera.pandas import Check, Column, DataFrameSchema
from pandera.errors import SchemaError, SchemaErrors

from vehicle_insurance.exceptions.handle_exception import (
    VehicleInsuranceException,
)
from vehicle_insurance.logger.logger_config import setup_logger

logger = setup_logger()

ValidationMode = Literal["training", "scoring"]


class DataContractError(ValueError):
    """Raised when data violates the project data contract."""


class DataContractValidationException(VehicleInsuranceException):
    """Validation failure carrying only safe field/check metadata."""

    def __init__(self, summary: dict[str, Any], error_detail: sys) -> None:
        self.validation_summary = summary
        safe_error = DataContractError(
            "Data contract validation failed: "
            f"mode={summary.get('mode', 'unknown')} "
            f"failures={int(summary.get('failure_count', 0))}"
        )
        super().__init__(safe_error, error_detail)


class DataContractValidator:
    """
    Validates model-bound training and scoring data.

    Operational metadata must be supplied separately in a metadata envelope
    or lineage manifest. It must not be included in the DataFrame.
    """

    def __init__(self, schema_path: str | Path) -> None:
        try:
            self.schema_path = Path(schema_path)

            if not self.schema_path.is_file():
                raise FileNotFoundError(
                    f"Schema file does not exist: {self.schema_path}"
                )

            with self.schema_path.open("r", encoding="utf-8") as file:
                self.contract = yaml.safe_load(file)

            self.contract_version = str(
                self.contract.get(
                    "contract_version",
                    self.contract.get("version", ""),
                )
            )

            if not self.contract_version:
                raise DataContractError("schema.yaml must define a contract version")
            self.definitions = self.contract["features"]["definitions"]

        except Exception as error:
            logger.exception("Failed to load data contract")
            raise VehicleInsuranceException(error, sys) from error

    def validate(
        self,
        dataframe: pd.DataFrame,
        mode: ValidationMode,
    ) -> pd.DataFrame:
        """
        Validate and return model-bound data in canonical raw-column order.

        Raises:
            VehicleInsuranceException: if validation fails.
        """

        try:
            if mode not in {"training", "scoring"}:
                raise DataContractError(f"Unsupported validation mode: {mode}")

            if not isinstance(dataframe, pd.DataFrame):
                raise TypeError("Input must be a pandas DataFrame")

            if dataframe.empty:
                raise DataContractError("Input data must contain at least one row")

            required_columns = list(self.contract["schemas"][mode]["required_columns"])

            self._validate_forbidden_fields(dataframe, mode)

            self._validate_columns(
                dataframe=dataframe,
                required_columns=required_columns,
                mode=mode,
            )

            # Validate in the contract-defined raw-column order.
            ordered_dataframe = dataframe.loc[:, required_columns]

            pandera_schema = self._build_pandera_schema(mode)
            validated_dataframe = pandera_schema.validate(
                ordered_dataframe,
                lazy=True,
            )

            self._validate_global_rules(validated_dataframe)

            if mode == "training":
                self._validate_training_target(validated_dataframe)

            logger.info(
                "Data contract validation successful: mode=%s rows=%d "
                "contract_version=%s",
                mode,
                len(validated_dataframe),
                self.contract_version,
            )

            return validated_dataframe

        except (SchemaError, SchemaErrors) as error:
            summary = self._log_pandera_failure(error, mode)
            raise DataContractValidationException(summary, sys) from None

        except DataContractError as error:
            summary = self._summarize_contract_error(error, mode)
            self._log_validation_summary(summary)
            raise DataContractValidationException(summary, sys) from None

        except Exception as error:
            logger.exception(
                "Data contract validation failed: mode=%s",
                mode,
            )
            raise VehicleInsuranceException(error, sys) from error

    def _validate_forbidden_fields(
        self,
        dataframe: pd.DataFrame,
        mode: ValidationMode,
    ) -> None:
        validation_config = self.contract["validation"]

        if mode == "scoring":
            forbidden_fields = set(validation_config["forbidden_scoring_columns"])
        else:
            forbidden_fields = set(
                validation_config["forbidden_training_outcome_columns"]
            )

        present_forbidden_fields = sorted(
            forbidden_fields.intersection(dataframe.columns)
        )

        if present_forbidden_fields:
            raise DataContractError(
                f"{mode} data contains forbidden outcome or leakage "
                f"fields: {present_forbidden_fields}"
            )

    def _validate_columns(
        self,
        dataframe: pd.DataFrame,
        required_columns: list[str],
        mode: ValidationMode,
    ) -> None:
        if dataframe.columns.duplicated().any():
            duplicated = dataframe.columns[dataframe.columns.duplicated()].tolist()

            raise DataContractError(
                f"Duplicate column names are not allowed: {duplicated}"
            )

        actual_columns = set(dataframe.columns)
        expected_columns = set(required_columns)

        missing_columns = sorted(expected_columns - actual_columns)
        extra_columns = sorted(actual_columns - expected_columns)

        if missing_columns:
            raise DataContractError(
                f"{mode} data is missing required columns: {missing_columns}"
            )

        if extra_columns:
            raise DataContractError(
                f"{mode} data contains unexpected columns: {extra_columns}"
            )

    def _build_pandera_schema(
        self,
        mode: ValidationMode,
    ) -> DataFrameSchema:
        required_columns = self.contract["schemas"][mode]["required_columns"]

        columns: dict[str, Column] = {}

        for column_name in required_columns:
            if column_name == "id":
                identifier_spec = self.contract["identifier"]
                identifier_checks = [
                    Check.greater_than_or_equal_to(
                        identifier_spec["minimum"]
                    )
                ]
                if "maximum" in identifier_spec:
                    identifier_checks.append(
                        Check.less_than_or_equal_to(
                            identifier_spec["maximum"]
                        )
                    )
                columns[column_name] = Column(
                    int,
                    checks=identifier_checks,
                    nullable=False,
                    unique=bool(identifier_spec.get("unique_within_input", True)),
                )
                continue

            if column_name == "Response":
                field_spec = self.contract["features"]["target"]
            else:
                field_spec = self.definitions[column_name]

            columns[column_name] = self._build_column(
                column_name,
                field_spec,
            )

        return DataFrameSchema(
            columns=columns,
            strict=True,
            coerce=False,
        )

    @staticmethod
    def _build_column(
        column_name: str,
        field_spec: dict,
    ) -> Column:
        field_type = field_spec["type"]
        checks = []

        allowed_values = field_spec.get("allowed_values")
        if allowed_values is not None:
            checks.append(Check.isin(allowed_values))

        if "minimum" in field_spec:
            checks.append(Check.greater_than_or_equal_to(field_spec["minimum"]))

        if "maximum" in field_spec:
            checks.append(Check.less_than_or_equal_to(field_spec["maximum"]))

        if "minimum_exclusive" in field_spec:
            checks.append(Check.greater_than(field_spec["minimum_exclusive"]))

        if field_type == "integer":
            dtype = int
        elif field_type == "string":
            dtype = str
        elif field_type in {"number", "numeric_category"}:
            # Numeric contract fields may be represented as int or float.
            dtype = None
            checks.append(
                Check(
                    lambda values: pd.api.types.is_numeric_dtype(values),
                    error=f"{column_name} must be numeric",
                )
            )
        else:
            raise DataContractError(
                f"Unsupported contract type for {column_name}: {field_type}"
            )

        return Column(
            dtype=dtype,
            checks=checks,
            nullable=False,
        )

    def _validate_global_rules(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        if dataframe.isna().any().any():
            raise DataContractError("Null, missing, or NaN values are not permitted")

        numeric_columns = dataframe.select_dtypes(include=[np.number])

        if not np.isfinite(numeric_columns.to_numpy()).all():
            raise DataContractError("Infinite numeric values are not permitted")

        string_columns = dataframe.select_dtypes(include=["object", "string"])

        for column in string_columns.columns:
            empty_values = string_columns[column].astype(str).str.strip()

            if empty_values.eq("").any():
                raise DataContractError(
                    f"{column} contains empty or whitespace-only values"
                )

    def _validate_training_target(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        target_values = set(dataframe["Response"].unique())

        if not target_values.issubset({0, 1}):
            raise DataContractError("Response must contain only 0 and 1")

        if len(target_values) < 2:
            raise DataContractError("Training data must contain both target classes")

    def _summarize_pandera_failure(
        self,
        error: SchemaError | SchemaErrors,
        mode: ValidationMode,
    ) -> dict[str, Any]:
        failure_cases = getattr(error, "failure_cases", None)
        if not isinstance(failure_cases, pd.DataFrame):
            return self._build_validation_summary(
                mode,
                1,
                {"unknown": 1},
                {"contract_validation": 1},
                {("data_contract", "unknown", "contract_validation"): 1},
            )

        expected_fields = set(
            self.contract["schemas"][mode]["required_columns"]
        )
        metadata_columns = [
            column
            for column in ("schema_context", "column", "check")
            if column in failure_cases.columns
        ]
        if not metadata_columns:
            failure_count = max(1, int(len(failure_cases)))
            return self._build_validation_summary(
                mode,
                failure_count,
                {"unknown": failure_count},
                {"contract_validation": failure_count},
                {
                    ("data_contract", "unknown", "contract_validation"): failure_count
                },
            )
        field_counts: dict[str, int] = {}
        check_counts: dict[str, int] = {}
        group_counts: dict[tuple[str, str, str], int] = {}
        for values in failure_cases.loc[:, metadata_columns].itertuples(
            index=False,
            name=None,
        ):
            metadata = dict(zip(metadata_columns, values))
            raw_field = metadata.get("column")
            field = (
                raw_field
                if isinstance(raw_field, str) and raw_field in expected_fields
                else "unknown"
            )
            raw_context = metadata.get("schema_context")
            schema_context = (
                raw_context
                if isinstance(raw_context, str)
                and raw_context in {"Column", "DataFrameSchema"}
                else "data_contract"
            )
            raw_check = metadata.get("check")
            check = self._safe_check_identifier(raw_check)
            field_counts[field] = field_counts.get(field, 0) + 1
            check_counts[check] = check_counts.get(check, 0) + 1
            group = (schema_context, field, check)
            group_counts[group] = group_counts.get(group, 0) + 1

        return self._build_validation_summary(
            mode,
            max(1, int(len(failure_cases))),
            field_counts,
            check_counts,
            group_counts,
        )

    @staticmethod
    def _build_validation_summary(
        mode: str,
        failure_count: int,
        field_counts: dict[str, int],
        check_counts: dict[str, int],
        group_counts: dict[tuple[str, str, str], int],
    ) -> dict[str, Any]:
        safe_mode = mode if mode in {"training", "scoring"} else "unknown"
        groups = [
            {
                "schema_context": schema_context,
                "column": field,
                "check": check,
                "failure_count": count,
            }
            for (schema_context, field, check), count in sorted(group_counts.items())
        ]
        return {
            "mode": safe_mode,
            "failure_count": int(failure_count),
            "field_counts": dict(sorted(field_counts.items())),
            "check_counts": dict(sorted(check_counts.items())),
            "groups": groups,
        }

    def _summarize_contract_error(
        self,
        error: DataContractError,
        mode: ValidationMode,
    ) -> dict[str, Any]:
        """Return allowlisted field metadata for non-Pandera contract errors."""

        schema = self.contract.get("schemas", {}).get(mode, {})
        expected_fields = schema.get("required_columns", [])
        message = str(error)
        fields = {
            field: 1
            for field in expected_fields
            if field in message
        }
        if not fields:
            fields = {"unknown": 1}
        failure_count = sum(fields.values())
        return self._build_validation_summary(
            mode,
            failure_count,
            fields,
            {"contract_validation": failure_count},
            {
                ("data_contract", field, "contract_validation"): count
                for field, count in fields.items()
            },
        )

    @staticmethod
    def _safe_check_identifier(raw_check: object) -> str:
        """Map validator checks to stable identifiers, never raw text."""

        if not isinstance(raw_check, str):
            return "contract_validation"
        check = raw_check.strip().lower()
        if check.startswith("isin("):
            return "allowed_values"
        if check.startswith("greater_than_or_equal_to("):
            return "minimum"
        if check.startswith("less_than_or_equal_to("):
            return "maximum"
        if check.startswith("greater_than("):
            return "minimum_exclusive"
        if "numeric" in check:
            return "numeric_type"
        if check.startswith("dtype("):
            return "dtype"
        return "contract_validation"

    def _log_pandera_failure(
        self,
        error: SchemaError | SchemaErrors,
        mode: ValidationMode,
    ) -> dict[str, Any]:
        """Log validation metadata without submitted values or raw errors."""

        summary = self._summarize_pandera_failure(error, mode)
        self._log_validation_summary(summary)
        return summary

    @staticmethod
    def _log_validation_summary(summary: dict[str, Any]) -> None:
        logger.error(
            "Data contract validation failed: mode=%s failures=%d "
            "fields=%s checks=%s",
            summary["mode"],
            summary["failure_count"],
            summary["field_counts"],
            summary["check_counts"],
        )
