"""Safe, atomic JSON report output helpers.

Report paths are user-controlled in the replay and parity command-line tools.
This module keeps those destinations from aliasing candidate evidence and gives
callers an exclusive temporary-file protocol that never truncates a predictable
``<destination>.tmp`` path.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable


class ReportDestinationError(ValueError):
    """Raised when a report destination is unsafe or unavailable."""


def resolve_report_path(path_value: str | os.PathLike[str]) -> Path:
    """Resolve a report path without requiring the destination to exist."""

    try:
        return Path(path_value).expanduser().resolve(strict=False)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ReportDestinationError(
            f"Could not resolve report destination {path_value!r}"
        ) from error


def _path_key(path_value: str | os.PathLike[str]) -> str:
    return os.path.normcase(str(resolve_report_path(path_value)))


def paths_alias(
    first: str | os.PathLike[str],
    second: str | os.PathLike[str],
) -> bool:
    """Return whether two paths name the same entry or filesystem object."""

    first_path = Path(first)
    second_path = Path(second)
    if _path_key(first_path) == _path_key(second_path):
        return True
    try:
        if os.path.exists(first_path) and os.path.exists(second_path):
            return os.path.samefile(first_path, second_path)
    except OSError:
        # The lexical/real-path check above remains useful when metadata is
        # unavailable (for example, a broken link or a transient filesystem).
        return False
    return False


def validate_report_destination(
    destination: str | os.PathLike[str],
    *,
    protected_paths: Iterable[str | os.PathLike[str]] = (),
    require_new: bool = False,
) -> Path:
    """Validate and return a report destination before any report work.

    Existing symlinks and non-regular files are rejected so writers never
    follow a caller-provided link. Existing ordinary files are allowed only
    when ``require_new`` is false; callers that permit replacement must still
    list all protected inputs explicitly.
    """

    target = resolve_report_path(destination)
    raw_target = Path(destination).expanduser()
    try:
        target_exists = os.path.lexists(raw_target)
    except (OSError, TypeError) as error:
        raise ReportDestinationError(
            f"Could not inspect report destination {destination!r}"
        ) from error

    if target_exists:
        if raw_target.is_symlink():
            raise ReportDestinationError(
                f"Report destination must not be a symbolic link: {raw_target}"
            )
        if raw_target.is_dir() or not raw_target.is_file():
            raise ReportDestinationError(
                f"Report destination is not a regular file: {raw_target}"
            )
        if require_new:
            raise ReportDestinationError(
                f"Report destination already exists; choose a fresh path: {target}"
            )

    for protected in protected_paths:
        if paths_alias(target, protected):
            raise ReportDestinationError(
                "Report destination aliases protected evidence: "
                f"destination={target}, protected={resolve_report_path(protected)}"
            )

    parent = target.parent
    try:
        if os.path.lexists(parent) and (parent.is_symlink() or not parent.is_dir()):
            raise ReportDestinationError(
                f"Report destination parent is not a directory: {parent}"
            )
    except OSError as error:
        raise ReportDestinationError(
            f"Could not inspect report destination parent: {parent}"
        ) from error
    return target


def protected_artifact_paths(
    candidate_root: str | os.PathLike[str],
    *,
    extra_paths: Iterable[str | os.PathLike[str]] = (),
) -> tuple[Path, ...]:
    """Return known candidate inputs/evidence that report writers must protect."""

    root = resolve_report_path(candidate_root)
    repository_root = Path(__file__).resolve().parents[3]
    paths: list[Path] = [
        repository_root / "configs" / "model_spec.yaml",
        repository_root / "configs" / "schema.yaml",
        root / "model_trainer" / "model.pkl",
        root / "model_trainer" / "metrics.json",
        root / "data_ingestion" / "feature_store" / "data.csv",
        root / "data_ingestion" / "feature_store" / "source_manifest.json",
        root / "data_ingestion" / "ingested" / "train.csv",
        root / "data_ingestion" / "ingested" / "validation.csv",
        root / "data_ingestion" / "ingested" / "test.csv",
        root / "data_validation" / "report.json",
        root / "data_transformation" / "transformed_data" / "train.npy",
        root / "data_transformation" / "transformed_data" / "validation.npy",
        root / "data_transformation" / "transformed_data" / "test.npy",
        root / "data_transformation" / "preprocessor.pkl",
        root / "model_evaluation" / "report.json",
        root / "source_frozen_parity" / "parity_report.json",
    ]

    # Protect additional certificates/snapshots already present in their
    # candidate directories without forbidding a new ordinary report there.
    for relative_directory in (
        "data_ingestion",
        "data_transformation",
        "model_trainer",
        "source_frozen_parity",
    ):
        directory = root / relative_directory
        if not directory.is_dir():
            continue
        try:
            paths.extend(
                candidate
                for candidate in directory.rglob("*")
                if candidate.is_file() or candidate.is_symlink()
            )
        except (OSError, RuntimeError):
            # The fixed paths above still protect the canonical evidence when
            # a directory cannot be enumerated.
            continue

    paths.extend(Path(path).expanduser() for path in extra_paths)
    return tuple(paths)


def write_json_atomic(
    destination: str | os.PathLike[str],
    payload: Any,
    *,
    protected_paths: Iterable[str | os.PathLike[str]] = (),
    require_new: bool = False,
) -> Path:
    """Serialize JSON through a unique exclusive temporary file and replace."""

    protected = tuple(protected_paths)
    target = validate_report_destination(
        destination,
        protected_paths=protected,
        require_new=require_new,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    file_descriptor: int | None = None
    try:
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=str(target.parent),
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(
            file_descriptor,
            "w",
            encoding="utf-8",
            newline="\n",
        ) as report_file:
            file_descriptor = None
            json.dump(payload, report_file, indent=2, default=str)
            report_file.flush()
            os.fsync(report_file.fileno())

        # Recheck link/protected-file state after serialization. ``os.replace``
        # replaces the destination entry rather than following a symlink.
        validate_report_destination(
            target,
            protected_paths=protected,
            require_new=require_new,
        )
        os.replace(temporary_path, target)
        temporary_path = None
        return target
    except Exception:
        if file_descriptor is not None:
            try:
                os.close(file_descriptor)
            except OSError:
                pass
        raise
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
