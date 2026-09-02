from __future__ import annotations

import hashlib
import os
import pickle
import subprocess
import sys
from pathlib import Path

import mlflow
import numpy as np
import yaml
from mlflow.tracking import MlflowClient

from vehicle_insurance.exceptions.handle_exception import VehicleInsuranceException
from vehicle_insurance.logger.logger_config import setup_logger

logger = setup_logger()


def read_yaml_file(file_path: str):
    try:
        with open(file_path, "r") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise VehicleInsuranceException(e, sys) from e


def save_object(file_path: str, obj):
    try:
        logger.info(f"Saving object to {file_path}")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logger.info(f"Object saved successfully to {file_path}")
    except Exception as e:
        raise VehicleInsuranceException(e, sys) from e


def load_object(file_path: str):
    try:
        logger.info(f"Loading object from {file_path}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        with open(file_path, "rb") as file_obj:
            obj = pickle.load(file_obj)
        logger.info(f"Object loaded successfully from {file_path}")
        return obj
    except Exception as e:
        raise VehicleInsuranceException(e, sys) from e


def save_numpy_array_data(file_path: str, array: np.ndarray):
    try:
        logger.info(f"Saving numpy array to {file_path}")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
        logger.info(f"Numpy array saved successfully to {file_path}")
    except Exception as e:
        raise VehicleInsuranceException(e, sys) from e


def load_numpy_array_data(file_path: str) -> np.ndarray:
    try:
        logger.info(f"Loading numpy array from {file_path}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        with open(file_path, "rb") as file_obj:
            array = np.load(file_obj, allow_pickle=False)
        logger.info(f"Numpy array loaded successfully from {file_path}")
        return array
    except Exception as e:
        raise VehicleInsuranceException(e, sys) from e


def get_mlflow_client() -> MlflowClient:
    try:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if not tracking_uri:
            raise ValueError("MLFLOW_TRACKING_URI is not set in environment variables.")
        mlflow.set_tracking_uri(tracking_uri)
        return MlflowClient(tracking_uri=tracking_uri)
    except Exception as e:
        raise VehicleInsuranceException(e, sys) from e


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _run_git(repo_root: Path, *arguments: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(repo_root), *arguments],
        check=True,
        capture_output=True,
    )
    return result.stdout


def resolve_repository_root(start_directory: Path | None = None) -> Path:
    start_directory = start_directory or Path.cwd()

    output = _run_git(
        start_directory,
        "rev-parse",
        "--show-toplevel",
    )

    return Path(output.decode("utf-8").strip()).resolve()


def capture_git_provenance(repo_root: Path) -> dict:
    status_bytes = _run_git(
        repo_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )

    patch_bytes = _run_git(
        repo_root,
        "diff",
        "--no-ext-diff",
        "--binary",
        "HEAD",
    )

    branch = (
        _run_git(
            repo_root,
            "branch",
            "--show-current",
        )
        .decode("utf-8")
        .strip()
    )

    commit = (
        _run_git(
            repo_root,
            "rev-parse",
            "HEAD",
        )
        .decode("utf-8")
        .strip()
    )

    model_spec_path = repo_root / "configs" / "model_spec.yaml"
    schema_path = repo_root / "configs" / "schema.yaml"
    lock_path = repo_root / "uv.lock"
    status_text = status_bytes.decode("utf-8", errors="replace")
    untracked_files = [
        line[3:]
        for line in status_text.splitlines()
        if line.startswith("?? ")
    ]

    return {
        "repository_root": str(repo_root),
        "branch": branch or "detached",
        "commit": commit,
        "is_dirty": bool(status_bytes),
        "status_porcelain": status_text,
        "untracked_files": untracked_files,
        "status_sha256": _sha256_bytes(status_bytes),
        "patch_sha256": _sha256_bytes(patch_bytes),
        "model_spec_sha256": _sha256_file(model_spec_path),
        "schema_sha256": _sha256_file(schema_path),
        "dependency_lock_sha256": (
            _sha256_file(lock_path) if lock_path.is_file() else None
        ),
    }


def validate_git_provenance(
    git_provenance: dict,
    git_policy: dict,
) -> None:
    """Fail closed when release Git provenance is unavailable or unsafe."""
    if not isinstance(git_provenance, dict):
        raise ValueError("Git provenance must be a dictionary")
    if not isinstance(git_policy, dict):
        raise ValueError("Git provenance policy must be a dictionary")

    policy_fields = (
        "require_clean_tree",
        "fail_if_unavailable",
        "fail_on_untracked_files",
    )
    invalid_policy_fields = [
        field
        for field in policy_fields
        if not isinstance(git_policy.get(field), bool)
    ]
    if invalid_policy_fields:
        raise ValueError(
            "Git provenance policy fields must be boolean: "
            f"{invalid_policy_fields}"
        )

    if git_policy["fail_if_unavailable"]:
        if not git_provenance.get("repository_root"):
            raise RuntimeError("Git repository root could not be captured")
        if not git_provenance.get("commit"):
            raise RuntimeError("Git HEAD commit could not be captured")
        if not git_provenance.get("status_sha256"):
            raise RuntimeError("Git status identity could not be captured")
        if not git_provenance.get("patch_sha256"):
            raise RuntimeError("Git patch identity could not be captured")

    untracked_files = git_provenance.get("untracked_files") or []
    if git_policy["fail_on_untracked_files"] and untracked_files:
        raise RuntimeError(
            "Release-quality training does not allow untracked files: "
            f"{untracked_files}"
        )

    if git_policy["require_clean_tree"] and git_provenance.get("is_dirty"):
        raise RuntimeError(
            "Release-quality training requires a clean Git working tree. "
            f"Status:\n{git_provenance.get('status_porcelain', '')}"
        )
