# Git lineage and release preflight

The training pipeline records the source state used to create each candidate.
The commit SHA is retained for compatibility, but it is no longer treated as a
complete source identity by itself.

## Release policy

The active policy is defined in `configs/model_spec.yaml`:

```yaml
evaluation_policy:
  lineage:
    git:
      require_clean_tree: true
      fail_if_unavailable: true
      fail_on_untracked_files: true
```

Release-quality training must therefore run from a repository with committed
source and configuration files. Pipeline-generated artifacts, logs, and local
tests remain ignored according to `.gitignore` and do not become release source
changes.

## Preflight sequence

Before starting ingestion or model fitting, the pipeline captures:

- repository root;
- current branch or detached-head state;
- `HEAD` commit;
- complete non-ignored porcelain status;
- dirty-tree status;
- SHA-256 identity of the status output;
- SHA-256 identity of `git diff HEAD --binary`;
- hashes of `configs/model_spec.yaml`, `configs/schema.yaml`, and `uv.lock`.

The same immutable snapshot is passed to model training and written into the
candidate's `model_trainer/metrics.json` lineage and MLflow lineage artifact.

The compatibility fields remain available:

```yaml
git_branch: "<branch>"
git_commit: "<commit>"
```

The complete identity is stored under `lineage.git_provenance`:

```yaml
git_provenance:
  repository_root: "<repository-root>"
  branch: "<branch>"
  commit: "<commit>"
  is_dirty: false
  status_porcelain: ""
  untracked_files: []
  status_sha256: "<sha256>"
  patch_sha256: "<sha256>"
  model_spec_sha256: "<sha256>"
  schema_sha256: "<sha256>"
  dependency_lock_sha256: "<sha256>"
```

## Operator workflow

Use this sequence before a fresh release-quality candidate run:

```cmd
git status --short
git log -1 --oneline
uv run python -m vehicle_insurance.pipeline.training_pipeline
```

If the status command reports source/configuration changes or untracked source
files, commit them or move them outside the repository before running training.
The pipeline fails before ingestion when the strict Git policy is violated.

## Existing artifacts

> Regarding the existing model: its artifact was created before Git provenance
> existed, so its exact source state cannot be reconstructed. Keep it as
> historical evidence, but do not use it as a release-qualified model or
> retrofit unknown Git information. The next fresh run will train the current
> model again under a clean, traceable commit.

Artifacts created before `lineage.git_provenance` was introduced therefore
remain historical evidence only. Replay or release evaluation must treat them
as legacy/non-release artifacts rather than attempting to infer a missing
source patch or retrofit unknown Git information.

This policy is independent of Champion/Challenger behavior. The current phase
continues to use candidate-only evaluation, with Champion comparison and
promotion disabled.

Source-data identity is recorded alongside Git identity. New candidate lineage
must include the immutable MongoDB `source_manifest` produced by ingestion;
pre-manifest artifacts remain historical evidence and are not retrofitted.
See [MongoDB source provenance](mongodb_provenance.md) for the manifest and
retry contract.
