# Source-frozen parity certification

This is a one-time certification utility for proving that the production
pipeline reproduces the frozen `experimentation_v2.ipynb` workflow for the
reference benchmark snapshot.

It is deliberately not part of the recurring training pipeline. The utility
compares a completed candidate artifact with the historical final MLflow run
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

## First run: verify without changing lineage

Run from the repository root in Command Prompt:

```cmd
cd /d D:\projVehicleInsurance\vehicle-Insurance
uv run python -m vehicle_insurance.verification.source_frozen_parity --candidate-artifact-dir artifact\08_31_2026_01_05_59
```

The default source CSV is:

```text
artifact\08_31_2026_01_05_59\data_ingestion\feature_store\data.csv
```

The report is written to:

```text
artifact\08_31_2026_01_05_59\source_frozen_parity\parity_report.json
```

The first run is read-only with respect to the candidate lineage. It must
return `status: "verified"` before evidence is applied.

## Apply successful evidence

After inspecting the report, rerun with explicit lineage recording:

```cmd
cd /d D:\projVehicleInsurance\vehicle-Insurance
uv run python -m vehicle_insurance.verification.source_frozen_parity --candidate-artifact-dir artifact\08_31_2026_01_05_59 --apply-lineage --log-mlflow-evidence
```

`--apply-lineage` updates only the candidate artifact's
`model_trainer\metrics.json` lineage. `--log-mlflow-evidence` additionally
logs `lineage/source_frozen_parity.json` and verification tags to that
candidate's existing MLflow run. No model registry alias is changed.

## Replay evaluation after certification

After a successful certificate has been applied to the candidate lineage, replay
the release evaluator against the existing artifact without retraining:

```cmd
cd /d D:\projVehicleInsurance\vehicle-Insurance
uv run python -m vehicle_insurance.verification.replay_model_evaluation --candidate-artifact-dir artifact\08_31_2026_01_05_59
```

The replay reads `model_trainer\metrics.json`, preserves the earlier
`model_evaluation\report.json`, and writes `model_evaluation\replay_report.json`.
It reruns validation, stability, lineage, parity, and read-only Champion gates.
Final-test metrics remain reporting-only and `promotion.enabled: false` means no
MLflow registration or alias mutation occurs.

## Failure handling

The utility writes structured stage errors to the report whenever possible and
also writes traceable messages through the project's rotating logger and
`VehicleInsuranceException`. A non-zero process exit code means certification
or requested evidence recording did not complete successfully. Do not mark the
candidate verified manually after a failed check.
