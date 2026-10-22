import hashlib

import pytest

from vehicle_insurance import utils


def _clean_provenance() -> dict:
    return {
        "repository_root": "D:/repo",
        "branch": "project/updations",
        "commit": "a" * 40,
        "is_dirty": False,
        "status_porcelain": "",
        "untracked_files": [],
        "status_sha256": "b" * 64,
        "patch_sha256": "c" * 64,
        "model_spec_sha256": "d" * 64,
        "schema_sha256": "e" * 64,
    }


def _strict_policy() -> dict:
    return {
        "require_clean_tree": True,
        "fail_if_unavailable": True,
        "fail_on_untracked_files": True,
    }


def test_capture_git_provenance_records_status_and_patch_identity(
    monkeypatch,
    tmp_path,
):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "model_spec.yaml").write_text("model", encoding="utf-8")
    (tmp_path / "configs" / "schema.yaml").write_text("schema", encoding="utf-8")
    (tmp_path / "uv.lock").write_text("lock", encoding="utf-8")

    def fake_git(repo_root, *arguments):
        assert repo_root == tmp_path
        if arguments == (
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ):
            return b" M src/example.py\n?? src/new.py\n"
        if arguments == ("diff", "--no-ext-diff", "--binary", "HEAD"):
            return b"binary-safe patch"
        if arguments == ("branch", "--show-current"):
            return b"project/updations\n"
        if arguments == ("rev-parse", "HEAD"):
            return b"a" * 40
        raise AssertionError(arguments)

    monkeypatch.setattr(utils, "_run_git", fake_git)

    provenance = utils.capture_git_provenance(tmp_path)

    assert provenance["is_dirty"] is True
    assert provenance["untracked_files"] == ["src/new.py"]
    assert (
        provenance["status_sha256"]
        == hashlib.sha256(b" M src/example.py\n?? src/new.py\n").hexdigest()
    )
    assert (
        provenance["patch_sha256"] == hashlib.sha256(b"binary-safe patch").hexdigest()
    )
    assert provenance["model_spec_sha256"] == hashlib.sha256(b"model").hexdigest()


def test_validate_git_provenance_accepts_clean_snapshot():
    utils.validate_git_provenance(_clean_provenance(), _strict_policy())


def test_validate_git_provenance_rejects_dirty_tree():
    provenance = _clean_provenance()
    provenance["is_dirty"] = True
    provenance["status_porcelain"] = " M src/example.py"

    with pytest.raises(RuntimeError, match="clean Git working tree"):
        utils.validate_git_provenance(provenance, _strict_policy())


def test_validate_git_provenance_rejects_untracked_files():
    provenance = _clean_provenance()
    provenance["untracked_files"] = ["src/new.py"]

    with pytest.raises(RuntimeError, match="untracked files"):
        utils.validate_git_provenance(provenance, _strict_policy())


def test_validate_git_provenance_rejects_unavailable_commit():
    provenance = _clean_provenance()
    provenance["commit"] = ""

    with pytest.raises(RuntimeError, match="HEAD commit"):
        utils.validate_git_provenance(provenance, _strict_policy())


def test_validate_git_provenance_rejects_invalid_policy():
    policy = _strict_policy()
    policy["require_clean_tree"] = "yes"

    with pytest.raises(ValueError, match="must be boolean"):
        utils.validate_git_provenance(_clean_provenance(), policy)
