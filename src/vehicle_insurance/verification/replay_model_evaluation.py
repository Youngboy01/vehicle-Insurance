"""Replay production evaluation for an existing local trainer artifact.

This module deliberately does not retrain a model, download a model, or mutate
MLflow registry aliases.  It reconstructs the typed ``ModelTrainerArtifact``
from the recorded ``model_trainer/metrics.json`` file and delegates all release
gates to the canonical ``ModelEvaluation`` component.

Run from the repository root with the project's ``uv`` environment::

    uv run python -m vehicle_insurance.verification.replay_model_evaluation \
        --candidate-artifact-dir artifact\\08_31_2026_01_05_59

Exit codes:

* ``0``: evaluation completed and the candidate passed all configured gates;
* ``2``: evaluation completed and the candidate was rejected by one or more
  gates; and
* ``1``: replay could not be executed because the artifact or configuration was
  invalid, unavailable, or failed unexpectedly.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from dotenv import load_dotenv

from vehicle_insurance.components.modelEvaluation import ModelEvaluation
from vehicle_insurance.entities.artifact_entity import ModelEvaluationArtifact, ModelTrainerArtifact
from vehicle_insurance.entities.config_entity import ModelEvaluationConfig
from vehicle_insurance.exceptions.handle_exception import VehicleInsuranceException
from vehicle_insurance.logger.logger_config import setup_logger

logger = setup_logger()

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
METRICS_RELATIVE_PATH = Path("model_trainer") / "metrics.json"
MODEL_RELATIVE_PATH = Path("model_trainer") / "model.pkl"
DEFAULT_REPLAY_REPORT_RELATIVE_PATH = (
    Path("model_evaluation") / "replay_report.json"
)
REQUIRED_METRICS_SECTIONS = (
    "train_metrics",
    "validation_metrics",
    "test_metrics",
    "stability_metrics",
    "lineage",
)
REQUIRED_IDENTITY_FIELDS = ("mlflow_run_id", "model_id", "model_uri")


class ReplayArtifactError(ValueError):
    """Raised when a local trainer artifact cannot be replayed safely."""


def _resolve_repository_path(path_value: str | os.PathLike[str]) -> Path:
    """Resolve a CLI path relative to the repository, never the process temp dir."""

    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return path.resolve()


def _read_json_object(file_path: Path) -> dict[str, Any]:
    try:
        with file_path.open("r", encoding="utf-8") as input_file:
            payload = json.load(input_file)
    except (OSError, json.JSONDecodeError) as error:
        raise ReplayArtifactError(
            f"Could not read JSON artifact file: {file_path}: {error}"
        ) from error

    if not isinstance(payload, dict):
        raise ReplayArtifactError(
            f"Expected a JSON object in artifact file: {file_path}"
        )
    return payload


def _require_mapping(payload: Mapping[str, Any], field_name: str) -> dict[str, Any]:
    value = payload.get(field_name)
    if not isinstance(value, dict):
        raise ReplayArtifactError(
            f"Trainer metrics field '{field_name}' must be a JSON object"
        )
    return dict(value)


def _optional_string(payload: Mapping[str, Any], field_name: str) -> str | None:
    value = payload.get(field_name)
    if value is None:
        return None
    if not isinstance(value, (str, int, float)) or isinstance(value, bool):
        raise ReplayArtifactError(
            f"Trainer metrics field '{field_name}' must be text or null"
        )
    return str(value)


def load_model_trainer_artifact(
    candidate_artifact_dir: str | os.PathLike[str],
) -> ModelTrainerArtifact:
    """Load and validate the recorded trainer artifact without fitting anything.

    The loader intentionally preserves empty identity values so that the
    canonical evaluator can report them as failed candidate-identity gates. It
    does reject missing structural fields, malformed JSON, missing model files,
    and non-object metric sections because those conditions mean that the
    artifact is not replayable evidence.
    """

    candidate_root = _resolve_repository_path(candidate_artifact_dir)
    if not candidate_root.is_dir():
        raise ReplayArtifactError(
            f"Candidate artifact directory does not exist: {candidate_root}"
        )

    metrics_file_path = candidate_root / METRICS_RELATIVE_PATH
    model_file_path = candidate_root / MODEL_RELATIVE_PATH
    if not metrics_file_path.is_file():
        raise ReplayArtifactError(
            f"Candidate trainer metrics file does not exist: {metrics_file_path}"
        )
    if not model_file_path.is_file():
        raise ReplayArtifactError(
            f"Candidate trained model file does not exist: {model_file_path}"
        )

    payload = _read_json_object(metrics_file_path)
    missing_fields = [
        field_name
        for field_name in (*REQUIRED_IDENTITY_FIELDS, *REQUIRED_METRICS_SECTIONS)
        if field_name not in payload
    ]
    if missing_fields:
        raise ReplayArtifactError(
            "Trainer metrics file is missing required fields: "
            f"{missing_fields}; file={metrics_file_path}"
        )

    identity = {
        field_name: _optional_string(payload, field_name)
        for field_name in REQUIRED_IDENTITY_FIELDS
    }
    if identity["mlflow_run_id"] is None:
        raise ReplayArtifactError(
            f"Trainer metrics field 'mlflow_run_id' must be text: {metrics_file_path}"
        )
    if identity["model_id"] is None:
        raise ReplayArtifactError(
            f"Trainer metrics field 'model_id' must be text: {metrics_file_path}"
        )
    if identity["model_uri"] is None:
        raise ReplayArtifactError(
            f"Trainer metrics field 'model_uri' must be text: {metrics_file_path}"
        )

    lineage = _require_mapping(payload, "lineage")
    artifact = ModelTrainerArtifact(
        trained_model_file_path=str(model_file_path),
        metrics_file_path=str(metrics_file_path),
        train_metrics=_require_mapping(payload, "train_metrics"),
        validation_metrics=_require_mapping(payload, "validation_metrics"),
        test_metrics=_require_mapping(payload, "test_metrics"),
        run_id=identity["mlflow_run_id"] or "",
        model_id=identity["model_id"] or "",
        model_uri=identity["model_uri"] or "",
        registered_model_name=_optional_string(payload, "registered_model_name"),
        registered_model_version=_optional_string(payload, "registered_model_version"),
        stability_metrics=_require_mapping(payload, "stability_metrics"),
        lineage=lineage,
    )
    logger.info(
        "Loaded existing trainer artifact for evaluation replay: "
        "artifact_dir=%s run_id=%s model_id=%s source_frozen_parity=%s",
        candidate_root,
        artifact.run_id,
        artifact.model_id,
        lineage.get("source_frozen_parity", {}).get("status")
        if isinstance(lineage.get("source_frozen_parity"), dict)
        else "missing",
    )
    return artifact


def _sha256_file(file_path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with file_path.open("rb") as input_file:
            for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ReplayArtifactError(
            f"Could not hash replay source file: {file_path}: {error}"
        ) from error
    return digest.hexdigest()


def _build_evaluation_context(
    candidate_artifact_dir: str | os.PathLike[str],
    artifact: ModelTrainerArtifact,
) -> dict[str, Any]:
    candidate_root = _resolve_repository_path(candidate_artifact_dir)
    metrics_path = Path(artifact.metrics_file_path).resolve()
    previous_report_path = candidate_root / "model_evaluation" / "report.json"
    context: dict[str, Any] = {
        "mode": "replay",
        "replay_purpose": "re_evaluate_existing_candidate_after_lineage_update",
        "candidate_artifact_dir": str(candidate_root),
        "source_metrics_file": str(metrics_path),
        "source_metrics_sha256": _sha256_file(metrics_path),
        "source_model_file": str(Path(artifact.trained_model_file_path).resolve()),
        "replayed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if previous_report_path.is_file():
        context["previous_evaluation_report"] = str(previous_report_path.resolve())
        context["previous_evaluation_report_sha256"] = _sha256_file(
            previous_report_path
        )
    return context


def replay_model_evaluation(
    candidate_artifact_dir: str | os.PathLike[str],
    report_file_path: str | os.PathLike[str] | None = None,
) -> ModelEvaluationArtifact:
    """Replay the canonical evaluator against a previously generated candidate.

    No data ingestion, preprocessing, model fitting, model download, or model
    registry mutation happens here. The only output is a fresh evaluation report
    at the requested path, defaulting to ``model_evaluation/replay_report.json``
    within the candidate artifact directory.
    """

    try:
        candidate_root = _resolve_repository_path(candidate_artifact_dir)
        candidate_artifact = load_model_trainer_artifact(candidate_root)
        evaluation_context = _build_evaluation_context(
            candidate_root,
            candidate_artifact,
        )
        if report_file_path is None:
            output_report_path = candidate_root / DEFAULT_REPLAY_REPORT_RELATIVE_PATH
        else:
            output_report_path = _resolve_repository_path(report_file_path)
        output_report_path.parent.mkdir(parents=True, exist_ok=True)

        evaluator = ModelEvaluation(
            model_evaluation_config=ModelEvaluationConfig(
                report_file_path=str(output_report_path)
            ),
            model_trainer_artifact=candidate_artifact,
            evaluation_context=evaluation_context,
        )
        evaluation_artifact = evaluator.evaluate_model()
        logger.info(
            "Evaluation replay completed: decision=%s passed=%s report=%s",
            evaluation_artifact.decision,
            evaluation_artifact.evaluation_passed,
            evaluation_artifact.evaluation_report_file_path,
        )
        return evaluation_artifact
    except VehicleInsuranceException:
        raise
    except Exception as error:
        logger.exception("Evaluation replay failed")
        raise VehicleInsuranceException(error, sys) from error


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay validation-based model evaluation for an existing local "
            "ModelTrainerArtifact without retraining."
        )
    )
    parser.add_argument(
        "--candidate-artifact-dir",
        required=True,
        help=(
            "Existing timestamped artifact directory, relative to the repository "
            "root or an absolute path."
        ),
    )
    parser.add_argument(
        "--report-file-path",
        default=None,
        help=(
            "Optional output report path. Defaults to "
            "<candidate-artifact-dir>/model_evaluation/replay_report.json."
        ),
    )
    parser.add_argument(
        "--fail-on-rejection",
        action="store_true",
        help="Return exit code 1 instead of 2 when evaluation gates reject the candidate.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the replay CLI and return a machine-readable process status."""

    load_dotenv(override=False)
    parser = _build_argument_parser()
    args = parser.parse_args(argv)
    try:
        evaluation_artifact = replay_model_evaluation(
            candidate_artifact_dir=args.candidate_artifact_dir,
            report_file_path=args.report_file_path,
        )
        summary = {
            "evaluation_passed": evaluation_artifact.evaluation_passed,
            "decision": evaluation_artifact.decision,
            "promotion_status": evaluation_artifact.promotion_status,
            "promotion_performed": evaluation_artifact.promotion_performed,
            "candidate_run_id": evaluation_artifact.candidate_run_id,
            "candidate_model_id": evaluation_artifact.candidate_model_id,
            "report_file_path": evaluation_artifact.evaluation_report_file_path,
            "reason_codes": evaluation_artifact.reason_codes,
        }
        print(json.dumps(summary, indent=2))
        if evaluation_artifact.evaluation_passed:
            return 0
        return 1 if args.fail_on_rejection else 2
    except Exception as error:
        logger.exception("Evaluation replay command failed")
        print(f"Evaluation replay failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
