import asyncio
import hmac
import os
import sys
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from math import floor
from pathlib import Path
from typing import Literal, TypeVar

import anyio
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import uvicorn
from dotenv import load_dotenv
from fastapi import (
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Request,
)
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    StrictStr,
    model_validator,
)
from sklearn.pipeline import Pipeline

from vehicle_insurance.constants import (
    MODEL_SPEC_FILE_PATH,
    SCHEMA_FILE_PATH,
)
from vehicle_insurance.exceptions.handle_exception import (
    VehicleInsuranceException,
)
from vehicle_insurance.contracts.data_contract import (
    DataContractValidationException,
    DataContractValidator,
)
from vehicle_insurance.logger.logger_config import setup_logger
from vehicle_insurance.utils import get_mlflow_client, read_yaml_file

load_dotenv(override=False)
logger = setup_logger()

APPLICATION_ROOT = Path(__file__).resolve().parent


def _utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp for API and model metadata."""

    return datetime.now(timezone.utc)


def _utc_isoformat() -> str:
    """Return an ISO-8601 timestamp carrying an explicit UTC offset."""

    return _utc_now().isoformat()

SCHEMA_PATH = APPLICATION_ROOT / SCHEMA_FILE_PATH
MODEL_SPEC_PATH = APPLICATION_ROOT / MODEL_SPEC_FILE_PATH

SCHEMA_CONFIG = read_yaml_file(SCHEMA_PATH)
MODEL_CONFIG = read_yaml_file(MODEL_SPEC_PATH)

MLFLOW_TRACKING_CONFIG = MODEL_CONFIG["artifacts"]["mlflow_tracking"]
MAX_CUSTOMER_ID = int(SCHEMA_CONFIG["identifier"]["maximum"])

EXPECTED_RAW_FEATURES = tuple(
    str(feature) for feature in SCHEMA_CONFIG["features"]["canonical_order"]
)
EXPECTED_FINAL_FEATURE_COUNT = int(
    MODEL_CONFIG["features"]["preprocessing"]["final_transformed_feature_count"]
)

SCORING_REQUIRED_COLUMNS = tuple(
    str(column)
    for column in SCHEMA_CONFIG["schemas"]["scoring"]["required_columns"]
)
SCORING_FORBIDDEN_COLUMNS = frozenset(
    str(column)
    for column in SCHEMA_CONFIG["validation"]["forbidden_scoring_columns"]
)
SCORING_VALIDATOR = DataContractValidator(SCHEMA_PATH)


def _load_optional_operating_threshold() -> float | None:
    """Load an optional, versioned diagnostic threshold from model_spec.yaml."""

    threshold_config = MODEL_CONFIG.get("selection", {}).get(
        "operating_threshold",
        {},
    )
    if threshold_config is None:
        return None
    if not isinstance(threshold_config, dict):
        raise ValueError("selection.operating_threshold must be a mapping")
    if not threshold_config.get("enabled", False):
        return None

    configured_value = threshold_config.get("value")
    if configured_value is None:
        raise ValueError(
            "selection.operating_threshold.value is required when the threshold is enabled"
        )

    threshold = float(configured_value)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "selection.operating_threshold.value must be between 0.0 and 1.0"
        )
    return threshold


OPERATING_THRESHOLD = _load_optional_operating_threshold()


@dataclass(frozen=True)
class ImmutableModelReference:
    """One release-pinned identity used for loading and metadata lookup."""

    model_id: str
    model_uri: str
    run_id: str

    def validate(self) -> None:
        expected_uri = f"models:/{self.model_id}"
        if self.model_uri != expected_uri:
            raise ValueError(
                "Configured model URI must be the immutable logged-model URI "
                f"{expected_uri!r}; aliases are not allowed."
            )
        if not self.model_id or not self.run_id:
            raise ValueError(
                "Immutable model configuration must include model ID and run ID"
            )


CURRENT_MODEL_REFERENCE = ImmutableModelReference(
    model_id=str(MLFLOW_TRACKING_CONFIG["serving_candidate_model_id"]),
    model_uri=str(MLFLOW_TRACKING_CONFIG["serving_candidate_model_uri"]),
    run_id=str(MLFLOW_TRACKING_CONFIG["serving_candidate_run_id"]),
)
CURRENT_MODEL_REFERENCE.validate()

# Compatibility constants retained for callers that import the old names.
CURRENT_MODEL_URI = CURRENT_MODEL_REFERENCE.model_uri
CURRENT_MODEL_ID = CURRENT_MODEL_REFERENCE.model_id
CURRENT_MODEL_RUN_ID = CURRENT_MODEL_REFERENCE.run_id


def _read_bounded_positive_int(
    name: str,
    default: int,
    *,
    maximum: int | None = None,
) -> int:
    """Read a positive serving limit while rejecting unsafe values."""

    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if value < 1 or (maximum is not None and value > maximum):
        maximum_message = f" and at most {maximum}" if maximum is not None else ""
        raise ValueError(
            f"{name} must be a positive integer{maximum_message}"
        )
    return value


SERVING_CPU_COUNT = os.cpu_count() or 1
SCORING_MAX_CONCURRENCY = _read_bounded_positive_int(
    "VEHICLE_INSURANCE_SCORING_CONCURRENCY",
    1,
    maximum=SERVING_CPU_COUNT,
)
SERVING_MODEL_THREADS = _read_bounded_positive_int(
    "VEHICLE_INSURANCE_MODEL_THREADS",
    1,
    maximum=SERVING_CPU_COUNT,
)
MAX_REQUEST_BODY_BYTES = _read_bounded_positive_int(
    "VEHICLE_INSURANCE_MAX_REQUEST_BODY_BYTES",
    4 * 1024 * 1024,
    maximum=64 * 1024 * 1024,
)

# These limiters are deliberately process-local. A deployment using multiple
# workers gets one model and one capacity budget per worker and must account for
# that multiplication explicitly.
SCORING_OPERATION_LIMITER = anyio.CapacityLimiter(SCORING_MAX_CONCURRENCY)
RELOAD_OPERATION_LIMITER = anyio.CapacityLimiter(1)

_OperationResult = TypeVar("_OperationResult")

EVALUATION_METRIC_NAMES = (
    "pr_auc",
    "precision_at_10pct",
    "recall_at_10pct",
    "lift_at_10pct",
    "roc_auc",
)


class PredictionEngine:
    def __init__(
        self,
        model,
        model_reference: ImmutableModelReference,
        evaluation_metrics: dict[str, dict[str, float | None]],
        lineage: dict[str, str | None],
        raw_feature_names,
        operating_threshold,
    ):
        self.model = model
        self.model_id = model_reference.model_id
        self.model_version = model_reference.model_id  # compatibility field
        self.model_uri = model_reference.model_uri
        self.source_run_id = model_reference.run_id
        self.evaluation_metrics = evaluation_metrics
        self.lineage = lineage
        # Compatibility with the old response field. This is optional and is
        # never used as the serving or promotion metric.
        self.recall = evaluation_metrics.get("test", {}).get(
            "recall_at_10pct"
        )
        self.raw_feature_names = tuple(raw_feature_names)
        self.operating_threshold = operating_threshold
        self.loaded_at = _utc_now()


current_engine: PredictionEngine | None = None
model_load_error: str | None = None
model_load_failure_code: str | None = None
model_load_attempted_at: datetime | None = None


def _safe_metric(value) -> float | None:
    """Return a finite metric value, or None for absent/legacy metadata."""

    if value is None:
        return None
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None
    return numeric_value if np.isfinite(numeric_value) else None


def _extract_evaluation_metrics(run) -> dict[str, dict[str, float | None]]:
    """Read versioned validation/test metrics without making them load gates."""

    metrics = getattr(run.data, "metrics", {}) or {}
    return {
        split: {
            metric_name: _safe_metric(
                metrics.get(f"{split}_{metric_name}")
            )
            for metric_name in EVALUATION_METRIC_NAMES
        }
        for split in ("validation", "test")
    }


def _extract_lineage(run, model_reference: ImmutableModelReference) -> dict[str, str | None]:
    """Extract release lineage tags while tolerating older model runs."""

    tags = getattr(run.data, "tags", {}) or {}

    def tag_value(name: str) -> str | None:
        value = tags.get(name)
        if value is None or str(value).strip() == "":
            return None
        return str(value)

    return {
        "model_id": model_reference.model_id,
        "model_uri": model_reference.model_uri,
        "source_run_id": model_reference.run_id,
        "model_spec_version": tag_value("model_spec_version"),
        "contract_version": tag_value("contract_version"),
        "git_commit": tag_value("git_commit"),
        "dataset_snapshot_id": tag_value("dataset_snapshot_id"),
    }


def _metadata_status(
    evaluation_metrics: dict[str, dict[str, float | None]],
    lineage: dict[str, str | None],
) -> tuple[str, list[str]]:
    """Classify metadata completeness without preventing compatible serving."""

    missing: list[str] = []
    for field_name in (
        "model_spec_version",
        "contract_version",
        "git_commit",
        "dataset_snapshot_id",
    ):
        if lineage.get(field_name) is None:
            missing.append(field_name)
    for split, metrics in evaluation_metrics.items():
        for metric_name, value in metrics.items():
            if value is None:
                missing.append(f"{split}_{metric_name}")

    all_metric_values = [
        value
        for split_metrics in evaluation_metrics.values()
        for value in split_metrics.values()
    ]
    lineage_missing = [
        field_name
        for field_name in (
            "model_spec_version",
            "contract_version",
            "git_commit",
            "dataset_snapshot_id",
        )
        if lineage.get(field_name) is None
    ]

    if len(missing) == 0:
        return "complete", missing
    if lineage_missing and all(value is None for value in all_metric_values):
        return "unavailable", missing
    return "partial", missing


def _load_engine_from_mlflow(
    model_reference: ImmutableModelReference = CURRENT_MODEL_REFERENCE,
):
    try:
        # Validate and freeze the release identity before making any MLflow call.
        # No alias is resolved during loading or reload.
        model_reference.validate()
        client = get_mlflow_client()
        run = client.get_run(run_id=model_reference.run_id)
        returned_run_id = getattr(getattr(run, "info", None), "run_id", None)
        if returned_run_id and str(returned_run_id) != model_reference.run_id:
            raise ValueError(
                "MLflow metadata lookup returned a different source run ID"
            )

        model_info = mlflow.models.get_model_info(
            model_uri=model_reference.model_uri
        )
        resolved_model_uri = getattr(model_info, "model_uri", None)
        resolved_model_id = getattr(model_info, "model_id", None)
        resolved_run_id = getattr(model_info, "run_id", None)
        if resolved_model_uri and str(resolved_model_uri) != model_reference.model_uri:
            raise ValueError(
                "MLflow model URI does not match the configured immutable model URI"
            )
        if resolved_model_id and str(resolved_model_id) != model_reference.model_id:
            raise ValueError(
                "MLflow model identity does not match the configured immutable model ID"
            )
        if resolved_run_id and str(resolved_run_id) != model_reference.run_id:
            raise ValueError(
                "MLflow model source run does not match the configured immutable run ID"
            )

        model = mlflow.sklearn.load_model(model_uri=model_reference.model_uri)
        if not isinstance(model, Pipeline):
            raise TypeError("Loaded MLflow artifact is not a sklearn Pipeline")
        required_steps = {"preprocessor", "model"}
        actual_steps = set(model.named_steps)

        if not required_steps.issubset(actual_steps):
            raise ValueError(
                "Loaded pipeline must contain both 'preprocessor' and 'model' steps"
            )

        if not callable(getattr(model, "predict_proba", None)):
            raise TypeError(
                "Loaded pipeline does not expose the public predict_proba() interface"
            )

        raw_feature_names = tuple(
            str(feature) for feature in getattr(model, "feature_names_in_", ())
        )

        if raw_feature_names != EXPECTED_RAW_FEATURES:
            raise ValueError(
                "Loaded pipeline raw feature contract does not match schema: "
                f"expected={EXPECTED_RAW_FEATURES}, "
                f"actual={raw_feature_names}"
            )
        fitted_preprocessor = model.named_steps["preprocessor"]

        transformed_feature_names = tuple(
            str(feature) for feature in fitted_preprocessor.get_feature_names_out()
        )

        if len(transformed_feature_names) != EXPECTED_FINAL_FEATURE_COUNT:
            raise ValueError(
                "Loaded pipeline transformed feature width does not match model spec: "
                f"expected={EXPECTED_FINAL_FEATURE_COUNT}, "
                f"actual={len(transformed_feature_names)}"
            )

        model_step = model.named_steps["model"]
        set_params = getattr(model_step, "set_params", None)
        if not callable(set_params):
            raise TypeError(
                "Loaded model does not expose set_params for serving thread limits"
            )
        set_params(n_jobs=SERVING_MODEL_THREADS)
        model_classes = list(getattr(model_step, "classes_", ()))
        if model_classes != [0, 1]:
            raise ValueError(
                "Loaded model must expose binary classes in the [0, 1] order; "
                f"actual={model_classes}"
            )
        model_feature_count = int(getattr(model_step, "n_features_in_", -1))

        if model_feature_count != EXPECTED_FINAL_FEATURE_COUNT:
            raise ValueError(
                "Loaded model input width does not match final pipeline width: "
                f"expected={EXPECTED_FINAL_FEATURE_COUNT}, "
                f"actual={model_feature_count}"
            )
        logger.info(
            "Immutable candidate model loaded: model_id=%s run_id=%s uri=%s "
            "raw_features=%d transformed_features=%d native_threads=%d",
            model_reference.model_id,
            model_reference.run_id,
            model_reference.model_uri,
            len(raw_feature_names),
            len(transformed_feature_names),
            SERVING_MODEL_THREADS,
        )

        evaluation_metrics = _extract_evaluation_metrics(run)
        lineage = _extract_lineage(run, model_reference)

        return PredictionEngine(
            model=model,
            model_reference=model_reference,
            evaluation_metrics=evaluation_metrics,
            lineage=lineage,
            raw_feature_names=raw_feature_names,
            operating_threshold=OPERATING_THRESHOLD,
        )
    except Exception as e:
        raise VehicleInsuranceException(e, sys) from e


@asynccontextmanager
async def lifespan(app: FastAPI):
    global current_engine, model_load_error, model_load_failure_code, model_load_attempted_at
    logger.info("App is starting up. Loading immutable candidate model from MLflow.")
    model_load_attempted_at = _utc_now()
    try:
        current_engine = _load_engine_from_mlflow()
        model_load_error = None
        model_load_failure_code = None
        logger.info(
            f"Immutable candidate model loaded successfully with ID: {current_engine.model_id}"
        )
    except Exception as e:
        current_engine = None
        model_load_error = "model_load_failed"
        model_load_failure_code = (
            "DEPENDENCY_UNAVAILABLE"
            if _is_temporary_model_dependency_failure(e)
            else "MODEL_LOAD_FAILED"
        )
        logger.error(f"Error occurred while loading candidate model: {e}")
        logger.warning(
            "Application will remain live but not ready until an immutable model "
            "can be loaded."
        )

    yield
    logger.info("App is shutting down.")


app = FastAPI(
    title="Vehicle Insurance Cross-Sell Response API",
    description=(
        "API for estimating a customer's response propensity to a vehicle-insurance "
        "cross-sell offer and managing model reload operations. "
        "Predictions are intended for marketing prioritization only and must not be "
        "used for claims, pricing, underwriting, fraud detection, eligibility, "
        "policy cancellation, or automatic denial decisions."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

STATIC_PATH = APPLICATION_ROOT / "static"
TEMPLATES_PATH = APPLICATION_ROOT / "templates"
app.mount("/static", StaticFiles(directory=str(STATIC_PATH)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_PATH))


def _stylesheet_url() -> str:
    """Return a cache-busted stylesheet URL based on the file's current mtime."""

    stylesheet_path = STATIC_PATH / "style.css"
    try:
        version = stylesheet_path.stat().st_mtime_ns
    except OSError:
        # Keep the page renderable if the asset is temporarily unavailable; the
        # static-file request will still expose the underlying deployment issue.
        version = "unavailable"
    return f"/static/style.css?v={version}"


def verify_reload_token(x_reload_token: str = Header(...)):
    expected = os.environ.get("RELOAD_TOKEN")
    if not expected:
        raise APIError(
            status_code=500,
            error_code="INTERNAL_ERROR",
            message="The reload operation is not configured on the server.",
        )
    if not hmac.compare_digest(x_reload_token, expected):
        raise APIError(
            status_code=401,
            error_code="AUTHENTICATION_FAILED",
            message="Authentication failed.",
        )


class PredictRequest(BaseModel):
    """Customer information available before a cross-sell campaign."""

    id: StrictInt = Field(
        ge=1,
        le=MAX_CUSTOMER_ID,
        description=(
            "Positive customer identifier in the browser-safe range "
            f"1-{MAX_CUSTOMER_ID}."
        ),
    )
    Age: StrictInt
    Gender: StrictStr = Field(max_length=64)
    Driving_License: StrictInt
    Region_Code: StrictInt | StrictFloat
    Previously_Insured: StrictInt
    Vehicle_Age: StrictStr = Field(max_length=64)
    Vehicle_Damage: StrictStr = Field(max_length=64)
    Annual_Premium: StrictInt | StrictFloat
    Policy_Sales_Channel: StrictInt | StrictFloat
    Vintage: StrictInt

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "id": 38113,
                "Age": 35,
                "Gender": "Male",
                "Driving_License": 1,
                "Region_Code": 28.0,
                "Previously_Insured": 0,
                "Vehicle_Age": "1-2 Year",
                "Vehicle_Damage": "Yes",
                "Annual_Premium": 25000.0,
                "Policy_Sales_Channel": 1.0,
                "Vintage": 5,
            }
        },
        strict=True,
    )


class PredictResponse(BaseModel):
    """Estimated response propensity returned by the candidate model."""

    response_propensity_score: float = Field(
        description=(
            "Uncalibrated positive-class response-propensity score used for ranking."
        )
    )
    probability: float = Field(
        deprecated=True,
        description=(
            "Deprecated compatibility alias for response_propensity_score. "
            "This is not a calibrated probability."
        )
    )
    prediction: int | None = Field(
        default=None,
        description=(
            "Optional diagnostic binary decision. It is null until an approved "
            "operating threshold is configured."
        ),
    )
    operating_threshold: float | None = Field(
        default=None,
        description=(
            "Configured diagnostic threshold, or null when no threshold is enabled."
        ),
    )
    model_version: str = Field(
        description="Compatibility field containing the immutable candidate model ID."
    )
    api_timestamp: str = Field(
        description=(
            "Timezone-aware ISO-8601 UTC timestamp at which the API generated "
            "the prediction."
        )
    )


class RankRequest(BaseModel):
    """Raw customer records and an optional campaign capacity override."""

    records: list[PredictRequest] = Field(
        min_length=1,
        max_length=10_000,
        description="Customers to score and rank before campaign selection.",
    )
    top_k_percentage: StrictInt | StrictFloat | None = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description=(
            "Optional percentage of the batch to select. If omitted, the "
            "versioned model-spec percentage is used."
        ),
    )
    top_k_count: StrictInt | None = Field(
        default=None,
        ge=0,
        description=(
            "Optional fixed number of customers to select. If omitted, the "
            "percentage capacity is used."
        ),
    )

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_capacity_override(self):
        if (
            self.top_k_percentage is not None
            and self.top_k_count is not None
        ):
            raise ValueError(
                "Provide only one of top_k_percentage or top_k_count"
            )
        return self


class RankedCustomer(BaseModel):
    """One selected customer in deterministic campaign rank order."""

    id: StrictInt
    rank: int
    response_propensity_score: float = Field(
        description="Uncalibrated positive-class score used for ranking."
    )


class RankResponse(BaseModel):
    """Top-K campaign selection produced from a scored customer batch."""

    selected_customers: list[RankedCustomer]
    total_scored: int
    selected_count: int
    capacity_type: Literal["top_k_percentage", "top_k_count"]
    capacity_value: float | int
    capacity_warning: str | None = None
    cutoff_score: float | None = None
    model_id: str
    model_uri: str
    source_run_id: str
    model_version: str
    operating_threshold: float | None = Field(
        description="Optional diagnostic threshold; it does not select customers."
    )
    api_timestamp: str = Field(
        description="Timezone-aware ISO-8601 UTC timestamp for this ranking response."
    )


class LivenessResponse(BaseModel):
    """Process-level liveness; deliberately independent of model availability."""

    status: Literal["alive"]


class HealthResponse(BaseModel):
    """Readiness and immutable model/evaluation metadata for the serving process."""

    status: str
    model_id: str | None = None
    model_uri: str | None = None
    source_run_id: str | None = None
    model_version: str | None = Field(
        default=None,
        deprecated=True,
        description="Deprecated compatibility alias for model_id.",
    )
    model_spec_version: str | None = None
    contract_version: str | None = None
    git_commit: str | None = None
    dataset_snapshot_id: str | None = None
    validation_metrics: dict[str, float | None] = Field(default_factory=dict)
    test_metrics: dict[str, float | None] = Field(default_factory=dict)
    test_metrics_reporting_only: bool = True
    metadata_status: Literal["complete", "partial", "unavailable"] = "unavailable"
    metadata_missing: list[str] = Field(default_factory=list)
    recall: float | None = Field(
        default=None,
        deprecated=True,
        description=(
            "Deprecated compatibility alias for test Recall@10%; this is not "
            "the primary business metric or live campaign performance."
        ),
    )
    loaded_at: str | None = Field(
        default=None,
        description="Timezone-aware ISO-8601 UTC model-load timestamp.",
    )


class ReloadResponse(HealthResponse):
    """Result of loading the current immutable MLflow candidate model."""

    new_model_version: str | None = Field(
        default=None,
        deprecated=True,
        description="Deprecated compatibility alias for model_id.",
    )
    new_recall: float | None = Field(
        default=None,
        deprecated=True,
        description="Deprecated compatibility alias for test Recall@10%.",
    )
    reloaded_at: str = Field(
        description="Timezone-aware ISO-8601 UTC reload-completion timestamp."
    )


@app.get("/", tags=["UI"])
async def root(request: Request):
    """Serves the frontend UI."""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"stylesheet_url": _stylesheet_url()},
    )


class ScoringContractError(Exception):
    """Safe structured scoring-contract failure for API clients."""

    def __init__(self, error_code: str, message: str, fields: list[str]):
        super().__init__(message)
        self.error_code = error_code
        self.message = message
        self.fields = fields


class APIError(Exception):
    """Safe, structured error raised by an API operation."""

    def __init__(
        self,
        status_code: int,
        error_code: str,
        message: str,
        fields: list[str] | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code
        self.message = message
        self.fields = fields or []


class APIErrorResponse(BaseModel):
    """Stable error envelope shared by all API failure responses."""

    error_code: str
    message: str
    fields: list[str] = Field(default_factory=list)


def _api_error_response(
    status_code: int,
    error_code: str,
    message: str,
    fields: list[str] | None = None,
) -> JSONResponse:
    """Build an error response without exposing internal exception details."""

    return JSONResponse(
        status_code=status_code,
        content=APIErrorResponse(
            error_code=error_code,
            message=message,
            fields=fields or [],
        ).model_dump(),
    )


class RequestBodyLimitMiddleware:
    """Reject oversized bodies before FastAPI parses JSON into request models."""

    def __init__(self, app, max_body_bytes: int):
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        content_length = next(
            (
                value
                for name, value in scope.get("headers", [])
                if name.lower() == b"content-length"
            ),
            None,
        )
        try:
            declared_length = int(content_length) if content_length else None
        except (TypeError, ValueError):
            declared_length = None
        if declared_length is not None and declared_length > self.max_body_bytes:
            await _api_error_response(
                status_code=413,
                error_code="REQUEST_BODY_TOO_LARGE",
                message="Request body exceeds the configured size limit.",
            )(scope, receive, send)
            return

        received_bytes = 0
        request_messages = []
        while True:
            message = await receive()
            request_messages.append(message)
            if message.get("type") == "http.request":
                received_bytes += len(message.get("body", b""))
                if received_bytes > self.max_body_bytes:
                    await _api_error_response(
                        status_code=413,
                        error_code="REQUEST_BODY_TOO_LARGE",
                        message="Request body exceeds the configured size limit.",
                    )(scope, receive, send)
                    return
                if not message.get("more_body", False):
                    break
            elif message.get("type") == "http.disconnect":
                break

        message_index = 0

        async def replay_receive():
            nonlocal message_index
            if message_index < len(request_messages):
                message = request_messages[message_index]
                message_index += 1
                return message
            return {"type": "http.request", "body": b"", "more_body": False}

        await self.app(scope, replay_receive, send)


app.add_middleware(
    RequestBodyLimitMiddleware,
    max_body_bytes=MAX_REQUEST_BODY_BYTES,
)


def _validated_scoring_dataframe(
    records: list[PredictRequest],
) -> pd.DataFrame:
    """Validate raw API records using the shared scoring contract."""

    scoring_dataframe = pd.DataFrame(
        [record.model_dump() for record in records],
        columns=SCORING_REQUIRED_COLUMNS,
    )
    try:
        return SCORING_VALIDATOR.validate(
            scoring_dataframe,
            mode="scoring",
        )
    except DataContractValidationException as error:
        summary = error.validation_summary
        failure_fields = sorted(
            str(field)
            for field in summary.get("field_counts", {})
            if field != "unknown"
        )
        raise ScoringContractError(
            error_code="SCORING_VALUE_INVALID",
            message=(
                "Scoring rejected: one or more values violate the scoring "
                "data contract."
            ),
            fields=failure_fields,
        ) from None
    except VehicleInsuranceException as error:
        message = str(error)
        failure_fields = [
            field
            for field in SCORING_REQUIRED_COLUMNS
            if field in message
        ]
        raise ScoringContractError(
            error_code="SCORING_VALUE_INVALID",
            message=(
                "Scoring rejected: one or more values violate the scoring "
                "data contract."
            ),
            fields=failure_fields,
        ) from error


def _score_raw_dataframe(
    engine: PredictionEngine,
    validated_dataframe: pd.DataFrame,
) -> np.ndarray:
    """Score canonical raw features through the complete sklearn pipeline."""

    input_dataframe = validated_dataframe.loc[:, engine.raw_feature_names]
    probabilities = np.asarray(engine.model.predict_proba(input_dataframe))
    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise ValueError(
            "Loaded pipeline predict_proba() must return two class scores"
        )
    scores = probabilities[:, 1].astype(float)
    if not np.isfinite(scores).all() or ((scores < 0.0) | (scores > 1.0)).any():
        raise ValueError(
            "Loaded pipeline returned a non-finite or out-of-range response score"
        )
    return scores


async def _run_bounded_operation(
    limiter: anyio.CapacityLimiter,
    operation: Callable[..., _OperationResult],
    args: tuple[object, ...],
    *,
    busy_error_code: str,
    busy_message: str,
) -> _OperationResult:
    """Run one synchronous operation without queueing beyond its capacity."""

    borrower = object()
    try:
        limiter.acquire_on_behalf_of_nowait(borrower)
    except anyio.WouldBlock:
        raise APIError(
            status_code=503,
            error_code=busy_error_code,
            message=busy_message,
        ) from None

    released = False

    def release_capacity() -> None:
        nonlocal released
        if not released:
            released = True
            limiter.release_on_behalf_of(borrower)

    try:
        # Keep the worker in a child task so cancellation of the HTTP request
        # cannot release admission while the synchronous operation continues.
        worker_task = asyncio.create_task(
            anyio.to_thread.run_sync(
                operation,
                *args,
                abandon_on_cancel=False,
            )
        )

        def finish_worker(task: asyncio.Task[_OperationResult]) -> None:
            release_capacity()
            if not task.cancelled():
                # Retrieve an unobserved worker exception if the request was
                # cancelled before the child task completed.
                task.exception()

        worker_task.add_done_callback(finish_worker)
    except BaseException:
        release_capacity()
        raise

    try:
        result = await asyncio.shield(worker_task)
    except asyncio.CancelledError:
        raise
    else:
        release_capacity()
        return result


def _resolve_top_k(
    request: RankRequest,
    total_rows: int,
) -> tuple[int, Literal["top_k_percentage", "top_k_count"], float | int, str | None]:
    """Resolve and safely clamp a batch campaign capacity to the batch size."""

    selection_config = MODEL_CONFIG.get("selection", {})
    configured_percentage = float(selection_config.get("top_k_percentage", 10))
    if not 0.0 <= configured_percentage <= 100.0:
        raise ValueError(
            "selection.top_k_percentage must be between 0.0 and 100.0"
        )

    if request.top_k_count is not None:
        requested_k = int(request.top_k_count)
        selected_k = min(requested_k, total_rows)
        warning = (
            "Warning: requested fixed capacity exceeds the available population. "
            "All valid customers were selected."
            if requested_k > total_rows
            else None
        )
        return selected_k, "top_k_count", requested_k, warning

    percentage = (
        configured_percentage
        if request.top_k_percentage is None
        else float(request.top_k_percentage)
    )
    requested_k = floor(total_rows * percentage / 100.0)
    warning = (
        "Warning: the requested percentage is too small for this population and "
        "calculated K is 0. No customers were selected. Use a fixed-count capacity "
        "if at least one customer must be selected."
        if requested_k == 0 and total_rows > 0
        else None
    )
    return requested_k, "top_k_percentage", percentage, warning


def _build_prediction_response(
    engine: PredictionEngine,
    request: PredictRequest,
) -> PredictResponse:
    """Perform the complete synchronous single-record scoring operation."""

    try:
        validated_dataframe = _validated_scoring_dataframe([request])
    except ScoringContractError:
        raise
    except Exception as error:
        logger.exception("Unexpected prediction preparation failure: %s", error)
        raise APIError(
            status_code=500,
            error_code="INTERNAL_ERROR",
            message="The API could not prepare the prediction request.",
        ) from error

    try:
        # Pass raw canonical features directly to the self-contained pipeline.
        response_propensity_score = float(
            _score_raw_dataframe(engine, validated_dataframe)[0]
        )
    except Exception as error:
        logger.exception("Model inference failed during prediction: %s", error)
        raise APIError(
            status_code=500,
            error_code="MODEL_INFERENCE_FAILED",
            message="The model could not score this request.",
        ) from error

    try:
        prediction = (
            None
            if engine.operating_threshold is None
            else int(response_propensity_score >= engine.operating_threshold)
        )
        return PredictResponse(
            response_propensity_score=round(response_propensity_score, 4),
            probability=round(response_propensity_score, 4),
            prediction=prediction,
            operating_threshold=engine.operating_threshold,
            model_version=engine.model_version,
            api_timestamp=_utc_isoformat(),
        )
    except Exception as error:
        logger.exception("Prediction response construction failed: %s", error)
        raise APIError(
            status_code=500,
            error_code="INTERNAL_ERROR",
            message="The API could not build the prediction response.",
        ) from error


def _build_rank_response(
    engine: PredictionEngine,
    request: RankRequest,
) -> RankResponse:
    """Perform validation, inference, ranking, and response construction together."""

    try:
        validated_dataframe = _validated_scoring_dataframe(request.records)
    except ScoringContractError:
        raise
    except Exception as error:
        logger.exception("Unexpected ranking preparation failure: %s", error)
        raise APIError(
            status_code=500,
            error_code="INTERNAL_ERROR",
            message="The API could not prepare the ranking request.",
        ) from error

    try:
        scores = _score_raw_dataframe(engine, validated_dataframe)
    except Exception as error:
        logger.exception("Model inference failed during ranking: %s", error)
        raise APIError(
            status_code=500,
            error_code="MODEL_INFERENCE_FAILED",
            message="The model could not score this request.",
        ) from error

    try:
        ranked_dataframe = validated_dataframe.loc[:, ["id"]].copy()
        ranked_dataframe["response_propensity_score"] = scores
        ranked_dataframe = ranked_dataframe.sort_values(
            by=["response_propensity_score", "id"],
            ascending=[False, True],
            kind="mergesort",
        ).reset_index(drop=True)

        selected_k, capacity_type, capacity_value, capacity_warning = (
            _resolve_top_k(request, len(ranked_dataframe))
        )
        selected_dataframe = ranked_dataframe.head(selected_k)
        selected_customers = [
            RankedCustomer(
                id=int(customer_id),
                rank=rank,
                response_propensity_score=round(
                    float(score),
                    4,
                ),
            )
            for rank, (customer_id, score) in enumerate(
                selected_dataframe[
                    ["id", "response_propensity_score"]
                ].itertuples(index=False, name=None),
                start=1,
            )
        ]

        cutoff_score = (
            float(selected_dataframe.iloc[-1]["response_propensity_score"])
            if not selected_dataframe.empty
            else None
        )
        return RankResponse(
            selected_customers=selected_customers,
            total_scored=len(ranked_dataframe),
            selected_count=len(selected_customers),
            capacity_type=capacity_type,
            capacity_value=capacity_value,
            capacity_warning=capacity_warning,
            cutoff_score=(
                round(cutoff_score, 4)
                if cutoff_score is not None
                else None
            ),
            model_id=engine.model_id,
            model_uri=engine.model_uri,
            source_run_id=engine.source_run_id,
            model_version=engine.model_version,
            operating_threshold=engine.operating_threshold,
            api_timestamp=_utc_isoformat(),
        )
    except Exception as error:
        logger.exception("Ranking response construction failed: %s", error)
        raise APIError(
            status_code=500,
            error_code="INTERNAL_ERROR",
            message="The API could not build the ranking response.",
        ) from error


@app.exception_handler(ScoringContractError)
async def scoring_contract_error_handler(
    request: Request,
    exc: ScoringContractError,
):
    return _api_error_response(
        status_code=422,
        error_code=exc.error_code,
        message=exc.message,
        fields=exc.fields,
    )


@app.exception_handler(APIError)
async def api_error_handler(request: Request, exc: APIError):
    return _api_error_response(
        status_code=exc.status_code,
        error_code=exc.error_code,
        message=exc.message,
        fields=exc.fields,
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Normalize framework/authentication errors to the public API envelope."""

    status_to_code = {
        401: "AUTHENTICATION_FAILED",
        403: "FORBIDDEN",
        404: "NOT_FOUND",
        405: "METHOD_NOT_ALLOWED",
        422: "REQUEST_SCHEMA_INVALID",
        500: "INTERNAL_ERROR",
        503: "SERVICE_UNAVAILABLE",
    }
    status_to_message = {
        401: "Authentication failed.",
        403: "The caller is not allowed to perform this operation.",
        404: "The requested API resource was not found.",
        405: "The HTTP method is not allowed for this endpoint.",
        422: "The request does not match the endpoint contract.",
        500: "The API could not complete the request.",
        503: "The service is temporarily unavailable.",
    }
    detail = exc.detail if isinstance(exc.detail, dict) else None
    if detail and detail.get("error_code"):
        raw_fields = detail.get("fields", [])
        if not isinstance(raw_fields, list):
            raw_fields = []
        return _api_error_response(
            status_code=exc.status_code,
            error_code=str(detail["error_code"]),
            message=str(detail.get("message", "The request failed.")),
            fields=[str(field) for field in raw_fields],
        )
    return _api_error_response(
        status_code=exc.status_code,
        error_code=status_to_code.get(exc.status_code, "API_ERROR"),
        message=status_to_message.get(
            exc.status_code,
            "The API could not complete the request.",
        ),
    )


@app.exception_handler(RequestValidationError)
async def request_validation_error_handler(
    request: Request,
    exc: RequestValidationError,
):
    fields = sorted(
        {
            str(location[-1])
            for error in exc.errors()
            if (location := error.get("loc")) and len(location) > 1
        }
    )
    if request.url.path not in {"/predict", "/rank"}:
        return _api_error_response(
            status_code=422,
            error_code="REQUEST_SCHEMA_INVALID",
            message="The request does not match the endpoint contract.",
            fields=fields,
        )

    forbidden_fields = sorted(
        set(fields).intersection(SCORING_FORBIDDEN_COLUMNS)
    )
    error_code = (
        "SCORING_TARGET_FORBIDDEN"
        if forbidden_fields
        else "SCORING_SCHEMA_INVALID"
    )
    message = (
        "Scoring rejected: target or outcome fields are forbidden."
        if forbidden_fields
        else "Scoring rejected: request does not match the scoring data contract."
    )
    return _api_error_response(
        status_code=422,
        error_code=error_code,
        message=message,
        fields=forbidden_fields or fields,
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Keep unexpected failures safe and consistent for API clients."""

    logger.exception("Unhandled API exception on %s: %s", request.url.path, exc)
    return _api_error_response(
        status_code=500,
        error_code="INTERNAL_ERROR",
        message="The API could not complete the request.",
    )


def _model_unavailable_error() -> APIError:
    """Return the most specific safe readiness error available."""

    error_code = model_load_failure_code or "MODEL_NOT_READY"
    message = (
        "A model-serving dependency is temporarily unavailable."
        if error_code == "DEPENDENCY_UNAVAILABLE"
        else "The immutable candidate model could not be loaded."
        if error_code == "MODEL_LOAD_FAILED"
        else "The scoring model is not ready."
    )
    return APIError(
        status_code=503,
        error_code=error_code,
        message=message,
    )


def _health_response(
    engine: PredictionEngine,
    status: str,
) -> HealthResponse:
    metadata_status, metadata_missing = _metadata_status(
        engine.evaluation_metrics,
        engine.lineage,
    )
    return HealthResponse(
        status=status,
        model_id=engine.model_id,
        model_uri=engine.model_uri,
        source_run_id=engine.source_run_id,
        model_version=engine.model_version,
        model_spec_version=engine.lineage.get("model_spec_version"),
        contract_version=engine.lineage.get("contract_version"),
        git_commit=engine.lineage.get("git_commit"),
        dataset_snapshot_id=engine.lineage.get("dataset_snapshot_id"),
        validation_metrics=engine.evaluation_metrics.get("validation", {}),
        test_metrics=engine.evaluation_metrics.get("test", {}),
        test_metrics_reporting_only=True,
        metadata_status=metadata_status,
        metadata_missing=metadata_missing,
        recall=engine.recall,
        loaded_at=engine.loaded_at.isoformat(),
    )


@app.get(
    "/health/live",
    response_model=LivenessResponse,
    tags=["Health Check"],
    summary="Check process liveness",
)
@app.get("/livez", response_model=LivenessResponse, include_in_schema=False)
async def liveness_check():
    """Return process liveness without depending on MLflow or model state."""

    return LivenessResponse(status="alive")


@app.get(
    "/health/ready",
    response_model=HealthResponse,
    tags=["Health Check"],
    summary="Check model readiness",
)
@app.get("/readyz", response_model=HealthResponse, include_in_schema=False)
async def readiness_check():
    """Return readiness only when a validated immutable model is loaded."""

    engine = current_engine
    if engine is None:
        raise _model_unavailable_error()
    return _health_response(engine, status="ready")


@app.get("/health", response_model=HealthResponse, tags=["Health Check"])
async def health_check():
    """Backward-compatible aggregate health endpoint; equivalent to readiness."""

    engine = current_engine
    if engine is None:
        raise _model_unavailable_error()
    return _health_response(engine, status="healthy")


@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["Prediction"],
    summary="Estimate cross-sell response propensity",
    description=(
        "Scores one customer using information available before a marketing "
        "campaign. The result is intended to support customer prioritization for "
        "vehicle-insurance cross-sell outreach."
    ),
)
async def predict(request: PredictRequest):
    engine = current_engine
    if engine is None:
        raise _model_unavailable_error()
    try:
        return await _run_bounded_operation(
            SCORING_OPERATION_LIMITER,
            _build_prediction_response,
            (engine, request),
            busy_error_code="SCORING_BUSY",
            busy_message=(
                "Scoring capacity is busy. Retry when the active operation "
                "has completed."
            ),
        )
    except (ScoringContractError, APIError):
        raise
    except Exception as error:
        logger.exception("Unexpected prediction operation failure: %s", error)
        raise APIError(
            status_code=500,
            error_code="INTERNAL_ERROR",
            message="The API could not complete the prediction request.",
        ) from error


@app.post(
    "/rank",
    response_model=RankResponse,
    tags=["Prediction"],
    summary="Rank customers and select a top-K campaign audience",
    description=(
        "Validates and scores a batch of pre-campaign customer records, ranks "
        "them by descending response propensity with ascending ID tie-breaking, "
        "and returns only the highest-scoring top-K customers. An optional "
        "operating threshold is diagnostic and never controls campaign capacity."
    ),
)
async def rank(request: RankRequest):
    engine = current_engine
    if engine is None:
        raise _model_unavailable_error()
    try:
        return await _run_bounded_operation(
            SCORING_OPERATION_LIMITER,
            _build_rank_response,
            (engine, request),
            busy_error_code="SCORING_BUSY",
            busy_message=(
                "Scoring capacity is busy. Retry when the active operation "
                "has completed."
            ),
        )
    except (ScoringContractError, APIError):
        raise
    except Exception as error:
        logger.exception("Unexpected ranking operation failure: %s", error)
        raise APIError(
            status_code=500,
            error_code="INTERNAL_ERROR",
            message="The API could not complete the ranking request.",
        ) from error


def _is_temporary_model_dependency_failure(error: Exception) -> bool:
    """Identify external model-store failures without exposing exception details."""

    candidate = error
    visited: set[int] = set()
    while candidate is not None and id(candidate) not in visited:
        visited.add(id(candidate))
        if isinstance(candidate, (ConnectionError, TimeoutError)):
            return True
        module_name = type(candidate).__module__.lower()
        if module_name.startswith("mlflow") or "requests" in module_name:
            return True
        candidate = candidate.__cause__ or candidate.__context__
    return False


@app.post("/reload", response_model=ReloadResponse, tags=["Model Management"])
async def reload_model(_: None = Depends(verify_reload_token)):
    global current_engine, model_load_error, model_load_failure_code, model_load_attempted_at
    model_load_attempted_at = _utc_now()
    try:
        # Build and validate the complete immutable candidate engine first. The
        # currently served engine is left untouched if loading fails.
        new_engine = await _run_bounded_operation(
            RELOAD_OPERATION_LIMITER,
            _load_engine_from_mlflow,
            (),
            busy_error_code="RELOAD_BUSY",
            busy_message=(
                "A model reload is already in progress. Retry when it has "
                "completed."
            ),
        )

    except APIError:
        raise
    except Exception as e:
        model_load_error = "reload_failed"
        logger.exception("/reload failed: %s", e)
        if _is_temporary_model_dependency_failure(e):
            error_code = "DEPENDENCY_UNAVAILABLE"
            message = "A model-serving dependency is temporarily unavailable."
        else:
            error_code = "MODEL_RELOAD_FAILED"
            message = "The immutable candidate model could not be loaded."
        model_load_failure_code = error_code
        raise APIError(
            status_code=503,
            error_code=error_code,
            message=message,
        ) from e

    # Swap only after all model, feature, identity, and metadata checks pass.
    current_engine = new_engine
    model_load_error = None
    model_load_failure_code = None
    logger.info(
        "Engine swapped. New model_id=%s run_id=%s uri=%s",
        current_engine.model_id,
        current_engine.source_run_id,
        current_engine.model_uri,
    )
    health_payload = _health_response(current_engine, status="success")
    return ReloadResponse(
        **health_payload.model_dump(),
        new_model_version=current_engine.model_version,
        new_recall=current_engine.recall,
        reloaded_at=_utc_isoformat(),
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080, reload=False)
