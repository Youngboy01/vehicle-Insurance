# Source-frozen parity certification

This utility proves that the production pipeline reproduces the frozen
`experimentation_v2.ipynb` workflow for the reference benchmark snapshot.

The normal training pipeline runs this certification as a formal stage before
final evaluation. The standalone CLI can also compare a completed candidate
artifact with the historical final MLflow run
`dc28826afdcc4e5cbce068db62439094` and verifies:

- the source snapshot/content hash;
- train, validation, and test ID sets and ordering;
- target ordering;
- canonical raw feature order;
- fitted transformed feature order and transformed values;
- final test prediction arrays within the configured tolerance;
- recomputed final metrics against the reference model, recorded candidate
  metrics, and the frozen notebook values.

The historical notebook did not persist a split-ID manifest. Therefore, the
utility reconstructs the split using the exact two `train_test_split` calls in
notebook cell 24 and records that fact in the report. It does not claim that a
missing manifest was loaded.

The reference model is loaded from the `reference_final_model_uri` configured
in `configs/model_spec.yaml`. For the current DagsHub MLflow run this is a
LoggedModel URI (`models:/m-3aacf46fc018452ab9aab96033eb7e56`), not a run
artifact path. The verifier checks that the downloaded model ID and source run
ID match the locked model lineage.

The verifier compares transformed feature order semantically. It ignores only
the sklearn transformer display prefix (`categorical__`, `numerical__`, or
`binary__`) because the production contract uses
`verbose_feature_names_out=False` while the historical notebook used the
default prefix. It still requires the same underlying feature sequence and
independently compares transformed values and predictions.

## Audit an existing candidate without changing lineage

Run from the repository root in Command Prompt. Set the candidate directory to
the actual run directory created by training:

```cmd
set "CANDIDATE_ARTIFACT_DIR=artifact\YOUR_RUN_DIRECTORY"
uv run python -m vehicle_insurance.verification.source_frozen_parity --candidate-artifact-dir "%CANDIDATE_ARTIFACT_DIR%"
```

Replace `YOUR_RUN_DIRECTORY` with the real directory under `artifact\`; do not
use a private historical path from another run.

The default source CSV is:

```text
%CANDIDATE_ARTIFACT_DIR%\data_ingestion\feature_store\data.csv
```

The adjacent source manifest is required for new release-quality candidates:

```text
%CANDIDATE_ARTIFACT_DIR%\data_ingestion\feature_store\source_manifest.json
```

The verifier validates that manifest against the CSV and candidate lineage,
including the snapshot ID, MongoDB namespace, explicit projection, indexed
ordering, and consistent read boundary. Pre-manifest artifacts remain
historical evidence and are not retrofitted.

The report is written to:

```text
%CANDIDATE_ARTIFACT_DIR%\source_frozen_parity\parity_report.json
```

The first run is read-only with respect to the candidate lineage. Its
`status: "verified"` means only that the numerical parity checks passed. A
lineage-bound run deliberately persists `status: "pending"` until the complete
recording transaction finishes.

## Apply successful evidence manually

The normal training pipeline applies verified evidence automatically and logs
the final certificate to the same candidate MLflow run. For a standalone
audit, inspect the report first and then rerun with explicit lineage recording:

```cmd
uv run python -m vehicle_insurance.verification.source_frozen_parity --candidate-artifact-dir "%CANDIDATE_ARTIFACT_DIR%" --apply-lineage --log-mlflow-evidence
```

`--apply-lineage` records a fresh certification attempt in the candidate
artifact. It first writes pending `model_trainer\metrics.json` evidence and a
pending certificate, then completes the requested local and remote steps. The
final metrics entry is `status: "verified"` only after the
certificate is durably written and its SHA-256 digest is committed. The
certificate reference is candidate-relative
(`source_frozen_parity/parity_report.json`), so it remains resolvable when the
approved artifact directory is relocated. `--log-mlflow-evidence` additionally
logs a pending remote copy, finalizes it only after every tag succeeds, and
records the MLflow artifact/tag completion flags for the final local commit.
No model registry alias is changed.

Recording modes are explicit: `audit_only` (no lineage mutation), `local_only`,
and `local_and_mlflow`. The release policy in `configs/model_spec.yaml`
requires `local_and_mlflow`; a local-only audit can be useful for diagnosis but
cannot satisfy release readiness.

The formal pipeline supplies both options automatically. Any local write,
remote log/tag, final certificate write, or process interruption leaves
`pending`/`failed` persisted evidence (or an unreadable artifact), and replay
rejects it. A newer failed attempt also supersedes an older certificate unless
the older, identity-matching certificate remains the explicitly referenced
one.

## Replay evaluation after certification

After a successful certificate has been applied to the candidate lineage, replay
the release evaluator against the existing artifact without retraining:

```cmd
uv run python -m vehicle_insurance.verification.replay_model_evaluation --candidate-artifact-dir "%CANDIDATE_ARTIFACT_DIR%"
```

The replay reads `model_trainer\metrics.json`, preserves the earlier
`model_evaluation\report.json`, and writes `model_evaluation\replay_report.json`.
It reruns validation, stability, lineage, and parity gates. Under the current
candidate-only policy, Champion comparison is skipped; it becomes active only
when the future workflow is explicitly enabled in `configs/model_spec.yaml`.
Final-test metrics remain reporting-only and `promotion.enabled: false` means no
MLflow registration or alias mutation occurs.

## Failure handling

The utility writes structured stage errors to the report whenever possible and
also writes traceable messages through the project's rotating logger and
`VehicleInsuranceException`. A non-zero process exit code means certification
or requested evidence recording did not complete successfully. Do not mark the
candidate verified manually after a failed check.
