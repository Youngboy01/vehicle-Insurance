# Vehicle Insurance Cross-Sell Data Contract

**Contract version:** 1.0.0
**Status:** Phase 1 benchmark contract; no company partner is currently onboarded
**Model specification:** `configs/model_spec.yaml` version 3
**Last reviewed:** 2026-09-05

## 1. Purpose and scope

This contract defines the data accepted by the vehicle-insurance cross-sell response-propensity platform.

The platform ranks existing customers by their estimated likelihood of responding positively to a vehicle-insurance cross-sell offer. It supports marketing-campaign prioritization only. It is not an underwriting, pricing, claims, fraud, eligibility, approval, cancellation, or automatic adverse-decision system.

The contract applies separately to:

- **Training data:** historical, labeled customer-campaign records used to fit or evaluate a model.
- **Scoring data:** current customer records used for live or batch ranking. Scoring data must not contain the training target.

Phase 1 accepts exactly ten canonical predictive features. The identifier `id` is required for customer identity and deterministic ranking, but is not one of the ten predictive features. An equivalent company feature may be used only after semantic mapping approval under Section 8.

This document is the human-readable source of truth for the production data boundary. Code must enforce this contract rather than infer behavior from arbitrary input columns.

## 2. Canonical schemas

### 2.1 Training schema

Each training record must contain exactly these model-bound fields:

```text
id, Gender, Age, Driving_License, Region_Code,
Previously_Insured, Vehicle_Age, Vehicle_Damage,
Annual_Premium, Policy_Sales_Channel, Vintage, Response
```

`Response` is mandatory for training and is the binary target:

- `1`: the customer responded positively to the vehicle-insurance offer.
- `0`: the customer did not respond positively.

Operational provenance metadata is not part of the training record columns. It must be supplied in the separate batch envelope or lineage manifest defined in Section 7. Metadata must not silently enter schema validation or the feature matrix.

### 2.2 Scoring schema

Each scoring record must contain exactly these model-bound fields:

```text
id, Gender, Age, Driving_License, Region_Code,
Previously_Insured, Vehicle_Age, Vehicle_Damage,
Annual_Premium, Policy_Sales_Channel, Vintage
```

`Response` is forbidden during live scoring. Known post-campaign or outcome fields are also forbidden. The service must reject incomplete, target-bearing, unsupported, or ambiguous records before preprocessing.

### 2.3 Training data versus scoring data

Training data is historical labeled data. It contains the customer identifier, the ten canonical features, and the observed `Response` outcome. It is used for preprocessing fitting, model training, validation, and the one-time final test evaluation defined by the model specification.

Scoring data is unlabeled data available before a campaign decision. It contains the customer identifier and the same ten canonical features, but never contains `Response`. It is passed as raw canonical features to the immutable, self-contained preprocessing-plus-model serving artifact and is used to produce scores and campaign rankings. No model or preprocessor fitting occurs during scoring, and no detached local preprocessor is required by the API.

The same feature definitions, units, category meanings, and timing requirements apply to both data types. The only model-bound field that differs is `Response`.

### 2.4 Model-bound data and operational metadata

The contract separates data into two layers:

```text
Model-bound data:
  The approved training or scoring record columns. These are subject to the
  training/scoring schema and are the only fields that can reach preprocessing
  or model inference.

Operational metadata:
  Batch-level or lineage information retained for provenance, freshness,
  hashing, audit, and reproducibility. This information is never passed to
  schema validation, preprocessing, or the model.
```

The required transport pattern is a wrapper/envelope with separate `metadata` and `records` sections, or an equivalent separate lineage manifest. Operational metadata must not be mixed into model-bound record columns.

Example envelope:

```json
{
  "metadata": {
    "schema_version": "1.0.0",
    "snapshot_id": "runtime-generated",
    "data_as_of": null,
    "extracted_at": "runtime-generated",
    "source_system": "MongoDB Atlas",
    "database": "INSURANCE",
    "collection": "INSURANCE_DATA",
    "row_count": 381109,
    "content_hash": "runtime-generated"
  },
  "records": [
    {
      "id": 1,
      "Gender": "Male",
      "Age": 44,
      "Driving_License": 1,
      "Region_Code": 28.0,
      "Previously_Insured": 0,
      "Vehicle_Age": "> 2 Years",
      "Vehicle_Damage": "Yes",
      "Annual_Premium": 40454.0,
      "Policy_Sales_Channel": 26.0,
      "Vintage": 217,
      "Response": 1
    }
  ]
}
```

For the current MongoDB benchmark source, the collection records are the model-bound data. Ingestion generates the extraction metadata and stores it in the artifact/report or a separate lineage manifest. If a file or collection contains metadata columns, ingestion must explicitly separate them before strict schema validation; otherwise the input is rejected as containing unexpected columns.

## 2A. The ten canonical predictive features

These are the complete and ordered predictive inputs for the frozen model:

```text
1. Gender
2. Age
3. Driving_License
4. Region_Code
5. Previously_Insured
6. Vehicle_Age
7. Vehicle_Damage
8. Annual_Premium
9. Policy_Sales_Channel
10. Vintage
```

They are grouped as follows:

| Group | Canonical features | Treatment |
|---|---|---|
| Categorical | `Gender`, `Vehicle_Age`, `Vehicle_Damage`, `Region_Code`, `Policy_Sales_Channel` | One-hot encoded with unknown-category handling |
| Numeric | `Age`, `Annual_Premium`, `Vintage` | Passed through without scaling |
| Binary | `Driving_License`, `Previously_Insured` | Passed through as 0/1 |

The model feature matrix must contain these ten features only. The canonical raw input order is used for deterministic input construction; the transformed feature order and model input order come from the fitted pipeline. `id`, `Response`, provenance fields, timestamps, campaign outcomes, and any other columns must be excluded from the predictive feature matrix.

## 3. Field types, categories, ranges, and validation rules

The following rules apply after parsing the input into a tabular representation and before any preprocessing or model call. Field names are case-sensitive and must match the canonical spelling exactly.

| Field | Type | Required | Allowed values / validation | Model role |
|---|---|---:|---|---|
| `id` | Integer | Yes | Positive; unique and non-null within each input dataset/batch | Identifier and deterministic tie-break only; never a feature |
| `Gender` | String | Yes | Exactly `Male` or `Female` | Categorical |
| `Age` | Integer | Yes | Finite and initially `18–100`, inclusive | Numeric passthrough |
| `Driving_License` | Integer | Yes | Exactly `0` or `1` | Binary passthrough |
| `Region_Code` | Numeric category | Yes | Finite and non-negative; preserve the source category representation | Categorical one-hot input |
| `Previously_Insured` | Integer | Yes | Exactly `0` or `1` | Binary passthrough |
| `Vehicle_Age` | String | Yes | Exactly `< 1 Year`, `1-2 Year`, or `> 2 Years` | Categorical |
| `Vehicle_Damage` | String | Yes | Exactly `Yes` or `No` | Categorical |
| `Annual_Premium` | Number | Yes | Finite and strictly greater than `0` | Numeric passthrough; no scaling |
| `Policy_Sales_Channel` | Numeric category | Yes | Finite and strictly greater than `0`; preserve the source category representation | Categorical one-hot input |
| `Vintage` | Integer | Yes | Finite and greater than or equal to `0` | Numeric passthrough |
| `Response` | Integer | Training only | Exactly `0` or `1`; forbidden for scoring | Training target |

Validation rules:

1. Field names are case-sensitive and the required model-bound columns must be present.
2. Integer fields must be represented as integers; values such as `1.5` must not be silently rounded.
3. Numeric fields must be finite; `NaN`, positive infinity, and negative infinity are invalid.
4. Categorical values are case-sensitive and must use the documented spelling.
5. `Region_Code` and `Policy_Sales_Channel` are categorical even though their wire representation is numeric. They must not be treated as continuous measurements.
6. Training datasets must contain both target classes for supervised evaluation; scoring data must not contain a target column.
7. IDs must be present, non-null, positive, and unique within the submitted dataset. Duplicate IDs make deterministic campaign selection ambiguous and are rejected.
8. Values must not be silently clipped, rounded, recoded, or coerced into validity.

### 3.1 Range classification

The contract distinguishes fixed Phase 1 rules from observed evidence and future partner configuration:

| Field/rule | Current Phase 1 classification | Meaning |
|---|---|---|
| `Age` 18–100 | General safety range | Initial project guardrail; not a universal fact about every company dataset and configurable during onboarding with evidence |
| `Annual_Premium > 0` | Fixed structural rule | A premium cannot be zero or negative under the current feature definition; partner-specific upper bounds may be added |
| `Vintage >= 0` | Fixed structural rule | A tenure/vintage duration cannot be negative; the partner may define an upper bound during onboarding |
| `Region_Code >= 0` and finite | Fixed structural rule plus partner domain | The value must be a valid finite non-negative category; the accepted code set is source/partner-specific |
| `Policy_Sales_Channel > 0` and finite | Fixed structural rule plus partner domain | The value must be a valid finite positive category; the accepted code set is source/partner-specific |
| Current dataset min/max values | Benchmark-observed evidence | Measured from the exact MongoDB snapshot used for the run; not automatically a universal production limit |

The current MongoDB benchmark's observed numeric ranges must be calculated from the extracted snapshot and recorded in the ingestion lineage report. The contract deliberately does not invent those values or treat the repository reference CSV as proof of MongoDB contents.

Partner-specific business ranges, category sets, units, and freshness rules are configurable only after onboarding evidence and approval. A partner configuration must record the effective limits, source, owner, approval date, and configuration version. Any change that alters model compatibility requires a model/contract compatibility review.

## 4. Missing, null, NaN, invalid, and extra-column behavior

The following are invalid in both training and scoring records:

- Missing required fields
- Null, empty-string, or unsupported categorical values
- `NaN`, positive infinity, or negative infinity
- Non-integral values for integer fields
- Values outside the documented domain
- Duplicate or missing `id` values within the submitted dataset or scoring batch
- Unexpected columns in the model-bound payload

Missing-value policy:

- Missing required fields are rejected.
- Explicit `null`, empty strings, whitespace-only strings, and recognized missing-value tokens are rejected.
- Missing values are not imputed by default.
- The service must not guess values from other fields or use a fallback customer profile.

Non-finite and invalid-value policy:

- `NaN`, positive infinity, and negative infinity are rejected for every numeric field.
- Invalid categories, invalid binary values, non-integral integer values, and out-of-range values are rejected.
- The system must return the field name, received value where safe, expected type/domain, and corrective action.

Extra-column and target-leakage policy:

- Training input records must contain exactly the declared training columns. Operational metadata belongs in the separate envelope or lineage manifest.
- Scoring input records must contain exactly the declared scoring columns. Operational metadata belongs in the separate envelope or lineage manifest.
- Unexpected model-bound columns are rejected; they must not be silently ignored or passed through.
- `Response` is always rejected in scoring input, even if its value is `0`.
- Known post-campaign, follow-up, conversion, claim, approval, or outcome fields are rejected in scoring input.

The explicit forbidden-field deny-list for live scoring is:

```yaml
forbidden_scoring_columns:
  - Response
  - conversion
  - converted
  - offer_accepted
  - offer_accepted_date
  - contacted
  - campaign_outcome
  - claim_status
  - policy_issued_after_offer
```

The list is a named defense-in-depth control, not a replacement for the scoring allow-list. The allow-list remains the primary boundary: every column not explicitly approved for scoring is rejected. The deny-list ensures that known leakage fields receive a specific target/outcome error and cannot be accidentally approved during future implementation changes.

For training, `Response` is permitted only as the declared target. All other fields in the deny-list are forbidden because they contain post-campaign or downstream outcome information and must not be used as predictive features. The list may be extended when a new outcome, follow-up, claim, approval, or post-offer field is identified.

Unknown-category policy:

- The frozen encoder uses `handle_unknown="ignore"` for unseen categorical levels during transformation.
- For the current benchmark scoring path, a structurally valid but previously unseen `Region_Code` or `Policy_Sales_Channel` is accepted and encoded as all-zero across that feature's fitted one-hot columns. This preserves the frozen pipeline's behavior.
- For partner production data, a structurally valid but unapproved category code is rejected until its domain, meaning, and mapping are approved during onboarding.
- Malformed, missing, non-finite, incorrectly typed, negative, or semantically invalid category values are rejected before transformation in both benchmark and partner modes.

The machine-readable policy is:

```yaml
unknown_category_policy:
  benchmark_scoring:
    valid_unseen_category: accept_and_encode_all_zero
    malformed_or_invalid_category: reject
  partner_production:
    valid_unseen_category: reject_until_domain_approval
    malformed_or_invalid_category: reject
```

Missing values must be corrected by the data provider or handled by an explicitly approved preprocessing policy. The default policy is rejection; median/mode imputation is not permitted implicitly.

Validation must occur before transformation and model inference. Errors should identify the record where possible, list the invalid or missing fields, and provide an actionable correction message without exposing secrets or internal stack traces.

## 5. Feature and preprocessing contract

The contract distinguishes three different forms of feature order:

1. **Canonical raw input order:** the documented order of the ten raw predictive fields for schemas, examples, and deterministic DataFrame construction.
2. **Transformed feature order:** the order of columns emitted by the fitted preprocessing pipeline, including encoded categorical columns and passthrough columns.
3. **Model input column order:** the exact transformed order accepted by the fitted model.

Production guarantees are based on required feature names, correct meanings and types, an explicit feature allowlist, the approved preprocessing configuration, and the fitted pipeline's transformed feature order. The canonical raw input order must be documented, but the implementation must not assume that it is the transformed or model input order.

Categorical one-hot inputs:

- `Gender`
- `Vehicle_Age`
- `Vehicle_Damage`
- `Region_Code`
- `Policy_Sales_Channel`

Preprocessing rules:

- Use one-hot encoding with `handle_unknown="ignore"`.
- Do not drop the first category.
- Pass through `Age`, `Annual_Premium`, and `Vintage` without scaling.
- Pass through `Driving_License` and `Previously_Insured`.
- Drop all columns outside the explicit feature allowlist.
- Obtain and preserve transformed feature names and order from the fitted pipeline, for example through the fitted transformer's `get_feature_names_out()` output.
- Pass the transformed matrix to the model in the exact order produced by that fitted pipeline.
- Do not manually recreate transformed column names, category order, or model input order in serving code.
- The frozen benchmark pipeline has 220 transformed features.

`id` and `Response` must never reach the model feature matrix. The raw model-bound feature set must contain the ten canonical features only, with the documented raw input order used for deterministic input construction. The transformed feature matrix and model input order must come from the fitted pipeline. The production artifact should contain preprocessing and model inference as one self-contained pipeline.

Evaluation and model-acceptance rules are defined in the `evaluation_policy` section of `configs/model_spec.yaml`. This data contract defines input validity and ranking behavior; it does not replace the model specification's metric policy. Frozen final-test metrics are reference evidence only and must not be reused for candidate promotion.

## 6. Training-versus-scoring rules

| Rule | Training | Scoring |
|---|---|---|
| Ten canonical features | Required | Required |
| `id` | Required identifier | Required identifier |
| `Response` | Required target | Forbidden |
| Target leakage/post-campaign fields | Forbidden | Forbidden |
| Extra columns | Reject unless explicitly treated as metadata | Reject |
| Feature mapping | Must be documented | Must be approved before use |
| Model fit | Allowed | Not performed |
| Preprocessor fit | Training data only | Transform with frozen artifact |

## 7. Data quality and provenance requirements

Every training snapshot and partner scoring batch must carry, where applicable, the following operational metadata in a separate envelope or lineage manifest:

- `schema_version`
- `snapshot_id`
- Source system and collection/table name
- `data_as_of`
- Extraction timestamp
- Row count
- Column list and canonical mapping version
- Content hash or other immutable snapshot identifier
- Missingness and duplicate summary
- Target prevalence for labeled training data
- Split identifier and random seed for benchmark training data

The current project has no company-specific production data source. The active dataset is the benchmark `data.csv` uploaded to the configured MongoDB Atlas cluster. The application accesses database `INSURANCE` and collection `INSURANCE_DATA` through the configured MongoDB connection.

The MongoDB copy is the active ingestion source for the training pipeline. The repository copy at `notebooks/data.csv` is the local benchmark/reference copy and must not be assumed to be byte-for-byte identical to the MongoDB collection unless a comparison is performed.

For every ingestion run, the pipeline captures the actual MongoDB extraction timestamp, row count, schema version, and an immutable source manifest. The manifest records a MongoDB snapshot ID, snapshot-session read boundary, exact filter and projection, indexed ordering, database/collection namespace, ID range, and canonical/file SHA-256 hashes. The manifest is separate from model-bound columns and is propagated through ingestion, metrics, evaluation, MLflow lineage, and future release metadata. See [MongoDB source provenance](mongodb_provenance.md).

Partner-specific freshness, completeness, and onboarding validation rules remain inactive because no company is currently onboarded. They must be configured separately when a company source is introduced.

## 8. Equivalent-feature mapping

No company is currently onboarded. The active Phase 1 status is:

```yaml
feature_mapping:
  status: "not_active"
  approved_mappings: []
  note: "Phase 1 currently accepts the ten canonical features only."
```

A company may provide an equivalent feature instead of a canonical field only with written approval. Each approved mapping must document:

```yaml
- source_field: "customer_age_years"
  canonical_field: "Age"
  semantic_definition: "Customer age in completed years"
  source_type: "integer"
  unit: "years"
  valid_range: "18-100"
  available_as_of: "before campaign outreach"
  transformation: "none"
  category_mapping: null
  data_owner: "Company data team"
  approved_by: "Project/data-contract owner"
  approval_date: "YYYY-MM-DD"
  mapping_version: "1.0"
  validation_fixture: "<approved-validation-fixture-path>"
  leakage_review: "confirmed no post-campaign information"
```

Approval must confirm the exact meaning, units, valid domain, timing, category definitions, deterministic transformation, data owner, approver, approval date, mapping version, validation fixture, and absence of post-campaign information. The mapping must match the canonical feature's meaning, unit, timing, categories, range, and missing-value behavior. Similar names alone are insufficient. If a required canonical feature cannot be safely mapped, scoring must be rejected with a message naming the missing field and required mapping information.

## 9. Batch ranking and capacity contract

The selection contract is:

```yaml
selection:
  score_field: "response_propensity_score"
  capacity_type: "top_k_percentage"
  default_percentage: 10
  k_calculation: "floor"
  ranking_order:
    - score: "descending"
    - id: "ascending"
  selected_count: "exactly K"
```

For `N` valid unique customers and percentage capacity `p`:

```text
K = floor(N × p / 100)
```

The algorithm validates records and unique IDs, generates one score per customer, sorts by score descending, resolves exact score ties by ascending `id`, and selects exactly the first `K` customers.

Example: for `N = 10,000` and `p = 10%`, `K = floor(10,000 × 10 / 100) = 1,000`.

Required edge-case behavior:

- Empty batch: reject with a clear validation message.
- Duplicate IDs: reject the batch.
- Invalid or out-of-range percentage: reject the request.
- `K = 0`: select zero customers and return a warning. Use fixed-count capacity `1` when at least one customer must be selected.
- Fixed-count capacity greater than `N`: select all valid customers and return a warning. The effective selected count is `min(requested_k, N)`.
- Missing or invalid scores: reject according to the batch failure policy; never rank missing scores as zero.
- Extra recommendations beyond `K`: exclude them from the selected audience; deferred future functionality.

`top_k_percentage=10` is the configurable Phase 1 baseline, not a universal company requirement. An unselected customer is lower campaign priority; it is not rejected, denied, or ineligible.

Standard capacity warnings:

```text
Warning: requested fixed capacity exceeds the available population. All valid customers were selected.
```

```text
Warning: the requested percentage is too small for this population and calculated K is 0. No customers were selected. Use a fixed-count capacity if at least one customer must be selected.
```

## 10. Score terminology

The model output is an estimated positive-class response-propensity score used for ordering. Until calibration is formally evaluated, it must not be represented as a guaranteed or perfectly calibrated probability.

The canonical terminology is:

```yaml
score_semantics:
  canonical_name: "response_propensity_score"
  api_compatibility_name: "probability"
  meaning: "Estimated positive-class response-propensity score used for ranking"
  calibrated_probability: false
  calibration_status: "not_evaluated"
  campaign_capacity_rule: "descending_score_top_k"
  operating_threshold: "optional_diagnostic_only"
```

The current API may retain the compatibility field name `probability`, but user-facing documentation should prefer `response_propensity_score` and must explain the uncalibrated score semantics.

Do not describe the score as a guaranteed probability, chance of purchase, certain response, approval, eligibility, or policy decision. If literal probability terminology is adopted later, calibration must be evaluated and versioned using validation data, including reliability curves, Brier score, expected calibration error, calibration slope/intercept, and segment calibration results.

## 11. Rejection behavior

Validation failures must return a stable error code, an actionable human-readable message, and the affected field names where applicable. Responses must not expose stack traces, credentials, filesystem paths, or internal MLflow details.

Standard rejection examples:

```text
Scoring rejected: missing required canonical feature(s): Vehicle_Damage.
Provide the field or submit an approved semantic mapping before scoring.
```

```text
Scoring rejected: Response is a training-only target and must not be included in live scoring data.
Remove Response and resubmit the ten required customer features.
```

```text
Scoring rejected: unsupported columns detected: offer_accepted_date.
Remove post-campaign/outcome fields; only the approved scoring schema is accepted.
```

The corresponding structured response should follow this shape:

```json
{
  "error_code": "SCORING_SCHEMA_INVALID",
  "message": "Scoring rejected: missing required field(s): Vehicle_Damage.",
  "fields": ["Vehicle_Damage"]
}
```

Standard error codes include:

| Error code | Use |
|---|---|
| `SCORING_SCHEMA_INVALID` | Missing, unexpected, or malformed scoring fields |
| `SCORING_VALUE_INVALID` | Invalid category, binary value, type, range, or non-finite number |
| `SCORING_TARGET_FORBIDDEN` | `Response` supplied during scoring |
| `SCORING_DUPLICATE_ID` | Duplicate or missing customer IDs in a batch |
| `SCORING_EMPTY_BATCH` | No scoring records supplied |
| `FEATURE_MAPPING_REQUIRED` | Required company field has no approved canonical mapping |
| `TRAINING_SCHEMA_INVALID` | Invalid labeled training data |
| `PROVENANCE_METADATA_INVALID` | Required snapshot or provenance metadata is missing or invalid |

Serving operations use the same envelope for operational failures. `MODEL_NOT_READY` means the process is live but no load attempt has established a model; `MODEL_LOAD_FAILED` means startup could not validate the configured candidate; `MODEL_RELOAD_FAILED` means a controlled reload could not validate it; `DEPENDENCY_UNAVAILABLE` means a temporary MLflow/model-store dependency failed; `MODEL_INFERENCE_FAILED` means the loaded model could not score the request; and `INTERNAL_ERROR` means an unexpected application failure. Operational failures use an empty `fields` list and never expose stack traces or customer data.

Additional actionable messages should cover invalid categories, binary values, numeric ranges, nulls, NaN/infinity, duplicate IDs, empty batches, and unmapped company fields. The message must identify what failed and what the provider must correct.

## 12. Versioning and change control

Use semantic contract versions in `MAJOR.MINOR.PATCH` form.

### Major version

Increment the major version for breaking changes, including removing a required feature, renaming a field, changing field meaning or units, changing category definitions, changing missing-value behavior, changing the target definition, changing preprocessing behavior, or changing ranking/capacity behavior.

Example: `1.0.0` to `2.0.0`.

### Minor version

Increment the minor version for backward-compatible additions, including optional provenance metadata, optional diagnostics, a new approved equivalent mapping, or documentation additions that do not change existing behavior.

Example: `1.0.0` to `1.1.0`.

### Patch version

Increment the patch version for non-behavioral corrections such as documentation, formatting, or error-message improvements that do not change validation behavior.

Example: `1.1.0` to `1.1.1`.

Before approving any contract change:

1. Describe the proposed change.
2. Classify it as major, minor, or patch.
3. Check model, preprocessing, API, and mapping compatibility.
4. Update schemas, validation code, fixtures, tests, and examples.
5. Record the approver and approval date.
6. Re-run source-versus-frozen parity and integration tests where applicable.
7. Preserve the prior contract version for audit and rollback.

Every released model or scoring batch must record the contract version together with the model specification version, model URI/model ID, Git commit, dependency lock hash, and dataset snapshot/hash.

Example release lineage:

```yaml
lineage:
  contract_version: "1.0.0"
  model_spec_version: "3"
  model_uri: "models:/m-3aacf46fc018452ab9aab96033eb7e56"
  git_commit: "<commit>"
  git_provenance:
    repository_root: "<repository-root>"
    branch: "<branch>"
    commit: "<commit>"
    is_dirty: false
    status_sha256: "<sha256>"
    patch_sha256: "<sha256>"
    model_spec_sha256: "<sha256>"
    schema_sha256: "<sha256>"
  dependency_lock_hash: "<sha256>"
  dataset_snapshot_id: "<snapshot>"
  dataset_content_hash: "<sha256>"
  source_manifest:
    manifest_version: "1.0.0"
    snapshot_id: "mongodb-sha256:<manifest-hash>"
    immutable: true
    source:
      database: "INSURANCE"
      collection: "INSURANCE_DATA"
      namespace: "INSURANCE.INSURANCE_DATA"
    read_boundary:
      mode: "snapshot_session"
      read_concern: "snapshot"
```
