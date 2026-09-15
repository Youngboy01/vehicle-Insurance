# Existing-candidate evaluation replay

## Purpose

The replay utility reruns the canonical validation-based release evaluator
against an existing local `ModelTrainerArtifact`. It is used after a candidate's
lineage has been corrected or additional evidence, such as source-frozen parity,
has been recorded.

Replay is not retraining. It does not ingest MongoDB data, refit preprocessing,
fit a LightGBM model, download a model from MLflow, register a model, or mutate a
registry alias. It reads the candidate's recorded
`model_trainer/metrics.json`, verifies that the local `model_trainer/model.pkl`
exists, reconstructs the typed trainer artifact, and delegates the decision to
`components/modelEvaluation.py`.

## Command Prompt usage

Run from the repository root with the project's `uv` environment. Use the
candidate directory created by the fresh training run:

```cmd
set "CANDIDATE_ARTIFACT_DIR=artifact\YOUR_RUN_DIRECTORY"
uv run python -m vehicle_insurance.verification.replay_model_evaluation --candidate-artifact-dir "%CANDIDATE_ARTIFACT_DIR%"
```

Replace `YOUR_RUN_DIRECTORY` with the actual directory created under
`artifact\` by training. Do not copy a private historical artifact path from
another machine or run.

The default output is:

```text
%CANDIDATE_ARTIFACT_DIR%\model_evaluation\replay_report.json
```

The existing `model_evaluation\report.json` is not overwritten. To choose a
different report path:

```cmd
uv run python -m vehicle_insurance.verification.replay_model_evaluation --candidate-artifact-dir "%CANDIDATE_ARTIFACT_DIR%" --report-file-path "%CANDIDATE_ARTIFACT_DIR%\model_evaluation\replay_report_YYYY-MM-DD.json"
```

The command loads `.env` without overriding variables already supplied by the
execution environment. Under the current
`evaluation_policy.champion_challenger_workflow.enabled: false` setting, replay
does not query Champion and does not require MLflow credentials for a comparison.
Credentials become relevant only if the future Champion/Challenger workflow is
explicitly enabled.

## Decision and exit codes

The JSON summary printed to the console contains the candidate run/model IDs,
decision, promotion status, report path, and reason codes.

- Exit `0`: evaluation completed and all configured gates passed.
- Exit `2`: evaluation completed, but one or more gates rejected the candidate.
- Exit `1`: replay execution failed because the artifact, configuration, or
  MLflow evaluation dependency was unavailable or invalid.

Use `--fail-on-rejection` when an orchestration job should treat a policy
rejection as an execution failure and return exit `1` instead of exit `2`.

## Evidence recorded in the replay report

The report records an `evaluation_execution` object containing:

- `mode: replay`;
- the candidate artifact directory;
- the source `metrics.json` and trained-model paths;
- the SHA-256 hash of the source metrics file;
- the prior evaluation report path and hash when that report exists; and
- the UTC replay timestamp.

The evaluator then applies the locked policy in the normal order:

1. candidate identity and recorded validation metrics;
2. repeated-CV stability evidence;
3. contract, model-spec, dataset, dependency, feature-order, and fit-scope
   lineage;
4. source-frozen split/prediction parity evidence, including the persisted
   completion state, recording mode, candidate/attempt identity, certificate
   path and SHA-256 digest, and the referenced certificate's own identity;
5. Champion comparison is skipped under the current candidate-only policy and
   remains reserved for the future workflow; and
6. final-test metrics copied for reporting only.

The current `promotion.enabled: false` setting remains authoritative. A passing
replay therefore reports `eligible_but_promotion_disabled` and performs no
registration or alias change. If the future comparison workflow is enabled and
the registered Champion alias points to a legacy run without the required
validation evidence, the evaluator fails closed with an explicit reason code
rather than silently treating that run as a valid comparison baseline.

Artifacts created before `lineage.git_provenance` was introduced do not contain
the exact source-state evidence required for a release-quality decision. Replay
must treat such artifacts as legacy/non-release evidence; it must not invent or
retrofit a Git patch identity that was not captured during training.

> Regarding the existing model: its artifact was created before Git provenance
> existed, so its exact source state cannot be reconstructed. Keep it as
> historical evidence, but do not use it as a release-qualified model or
> retrofit unknown Git information. The next fresh run will train the current
> model again under a clean, traceable commit.

For source-frozen parity, replay always reloads the durable lineage from
`model_trainer/metrics.json`; an in-memory `verified` flag is not sufficient.
It resolves the candidate-relative certificate path from the current artifact
root, checks the certificate digest and exact candidate/certification-attempt
identity, and requires `recording_state: "complete"`. The configured
`required_recording_mode` is enforced: `local_only` evidence is rejected when
the release policy requires `local_and_mlflow`, including when the remote
artifact was logged but its tags were only partially recorded.

Replay output is treated as a new immutable record. The destination is
validated before the candidate is loaded and must not exist already; this
protects the model, metrics, source snapshots/manifests, parity certificate,
original decision report, and any same-file symlink or hard-link alias. The
default `model_evaluation/replay_report.json` therefore cannot be reused for a
second invocation; choose a new filename for each replay. Report serialization
uses a unique exclusively-created temporary file and atomic replacement, so a
pre-existing `<destination>.tmp` entry is never opened or truncated.

## When to run replay

Run replay after applying reviewed lineage/parity evidence to an existing
candidate, after changing evaluator code, or after changing the versioned
evaluation policy. Fresh training already runs source-frozen parity as a formal
stage before final evaluation. Do not use replay as a substitute for a fresh
training run when the data, preprocessing, model, split protocol, or model
specification changes.
