# 🚗 Vehicle Insurance Cross-Sell Prediction — MLOps Pipeline

An end-to-end machine learning system that predicts whether a health insurance customer will respond positively to a **vehicle insurance cross-sell** offer. The project goes beyond a notebook model: it is a modular pipeline with schema validation, experiment tracking, explicit model lineage, and a served REST API — all containerized with Docker.

> **Status:** MongoDB ingestion, contract validation, transformation, notebook-aligned training, MLflow lineage, source-frozen parity certification, and validation-based evaluation are implemented. Production approval, promotion, deployment, and monitoring remain in progress (see [Roadmap](#-roadmap)).

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Problem Definition and Scope](#-problem-definition-and-scope)
- [Model Card](#-model-card)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Pipeline Stages](#-pipeline-stages)
- [API Reference](#-api-reference)
- [Getting Started](#-getting-started)
- [Environment Variables](#-environment-variables)
- [Running the Training Pipeline](#-running-the-training-pipeline)
- [Replaying Existing-candidate Evaluation](#-replaying-existing-candidate-evaluation)
- [Git Lineage and Release Preflight](#-git-lineage-and-release-preflight)
- [Running the API](#-running-the-api)
- [Docker](#-docker)
- [Model Registry Workflow](#-model-registry-workflow)
- [Data Schema](#-data-schema)
- [Roadmap](#-roadmap)

---

## 🔎 Overview

The dataset used is the classic **health-to-vehicle-insurance cross-sell** dataset: given a customer's demographics, vehicle history, and policy details, predict whether they will respond positively to a vehicle-insurance offer (`Response` = 0/1). The target is highly imbalanced, which shapes the use of class-weighted LightGBM and validation-based top-K evaluation.

Rather than training a model once and calling it done, this project implements the pieces you'd expect in a real ML system:

- **Schema-validated data ingestion** from MongoDB
- **A reusable, versioned preprocessing pipeline** (fit once, applied consistently at train and inference time)
- **Experiment tracking, source-frozen parity certification, and model lineage** with promotion disabled pending owner approval and operational gates
- **A served model** behind a REST API, with retraining and hot-swap controls still subject to evaluation/promotion alignment
- **A containerized runtime** for reproducible deployment

---

## 🎯 Problem Definition and Scope

### Phase 1 project definition

This project is a **top-K vehicle-insurance cross-sell decision-support platform**. Given customer information available before a marketing campaign, it estimates each customer's response propensity for a vehicle-insurance cross-sell offer, ranks customers by that score, and supports selecting the highest-scoring customers for outreach at a company-selected capacity.

The decision unit is **one customer-campaign opportunity**. The intended user is a marketing or campaign-operations team deciding which customers to prioritize for outreach.

The API provides single-customer scoring through `/predict` and deterministic batch ranking/top-K selection through `/rank`. Selection-quality evaluation remains an offline model-evaluation concern. Budget-based selection, outreach-cost assumptions, conversion-value assumptions, and ROI calculations are later extensions.

### Target meaning

- `Response = 1`: the customer responded positively to the vehicle-insurance offer.
- `Response = 0`: the customer did not respond positively to the vehicle-insurance offer.

The model predicts **response propensity**, not claims, underwriting risk, pricing, eligibility, fraud, or policy approval. It must not be used for claim decisions, premium calculation, underwriting, fraud detection, eligibility decisions, policy cancellation, or automatic denial/adverse decisions.

The current dataset supports propensity modeling only. It does not contain the treatment/control information required to claim that contacting a customer caused an incremental response. Uplift modeling, A/B testing, and causal campaign measurement are future extensions requiring additional campaign data.

See the detailed [Model Card](docs/model_card.md) for the model's intended use, evaluation evidence, limitations, and operational risks.

---

## 📄 Model Card

The [model card](docs/model_card.md) documents the current LightGBM response-propensity model, including its target definition, data assumptions, evaluation results, intended use, prohibited uses, and planned improvements.

The model card is a living document and must be updated whenever the model, data, evaluation protocol, or serving contract changes.

---

## 🏗 Architecture

```
                    ┌─────────────────┐
                    │  MongoDB Atlas   │  (raw data source)
                    └────────┬─────────┘
                             │
                             ▼
   ┌─────────────────────────────────────────────────────┐
   │                  TRAINING PIPELINE                   │
   │                                                       │
   │  1. Data Ingestion   → immutable MongoDB snapshot + split │
   │  2. Data Validation  → Pandera schema checks           │
   │  3. Data Transform.  → fit on train, transform all splits │
   │  4. Model Training   → frozen LightGBM + MLflow lineage    │
   │  5. Source-frozen parity → split/prediction certification      │
   │  6. Model Evaluation → validation policy + final decision      │
   └───────────────────────────┬───────────────────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   MLflow Tracking       │  (tracked on DagsHub)
                    │   Candidate lineage     │
                    └────────────┬─────────────┘
                                 │ immutable candidate serving configured
                                 ▼
                    ┌───────────────────────┐
                    │      FastAPI Service    │
                    │  /predict  /rank         │
                    │  /health   /reload       │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌───────────────────────┐
                    │  Jinja2 + JS Dashboard  │
                    └───────────────────────┘
```

---

## 🛠 Tech Stack

| Layer                  | Technology |
|-------------------------|------------|
| Language / packaging    | Python 3.13, `uv`, `setuptools` |
| Data storage             | MongoDB Atlas |
| Data validation          | Pandera |
| Data processing          | pandas, NumPy, scikit-learn (`ColumnTransformer`, `Pipeline`, `OneHotEncoder`) |
| Model                    | LightGBM (`LGBMClassifier`) |
| Experiment tracking / registry | MLflow (hosted on DagsHub) |
| API                       | FastAPI, Uvicorn, Pydantic |
| Frontend                  | Jinja2 templates, vanilla JS/CSS |
| Containerization          | Docker (multi-stage build) |
| Logging / errors          | Custom rotating-file logger, centralized exception wrapper |

---

## 📁 Project Structure

```
vehicle-Insurance/
├── app.py                          # FastAPI application (serving layer)
├── main.py                         # Entry-point placeholder
├── Dockerfile                      # Multi-stage build (uv → slim runtime)
├── pyproject.toml / uv.lock        # Dependency management
├── configs/
│   ├── schema.yaml                 # Data contract and preprocessing rules
│   └── model_spec.yaml             # Frozen model, split, metric, and lineage policy
├── notebooks/
│   ├── notebook.ipynb              # Exploration
│   ├── experimentation.ipynb       # Earlier model experimentation
│   └── experimentation_v2.ipynb    # Frozen benchmark/model-reference workflow
├── static/style.css                # Dashboard styling
├── templates/index.html            # Dashboard UI
└── src/vehicle_insurance/
    ├── components/
    │   ├── dataIngestion.py        # MongoDB → feature store → train/validation/test split
    │   ├── dataValidation.py       # Pandera schema validation
    │   ├── dataTransformation.py   # ColumnTransformer pipeline
    │   ├── modelTrainer.py         # LightGBM training + MLflow logging
    │   └── modelEvaluation.py      # Validation-based evaluation policy
    ├── configurations/
    │   └── mongodb_connection.py   # MongoDB client (singleton)
    ├── data_access/
    │   └── Insurance_data.py       # MongoDB → DataFrame
    ├── entities/
    │   ├── config_entity.py        # Per-stage config dataclasses
    │   └── artifact_entity.py      # Per-stage output dataclasses
    ├── pipeline/
    │   └── training_pipeline.py    # End-to-end stage orchestration
    ├── constants/__init__.py       # Central constants
    ├── exceptions/handle_exception.py
    ├── logger/logger_config.py
    ├── verification/
    │   ├── source_frozen_parity.py # Notebook-to-production certification
    │   └── replay_model_evaluation.py # Existing-candidate evaluation replay
    └── utils/__init__.py           # pickle/yaml/numpy IO helpers, MLflow client
```

---

## ⚙️ Pipeline Stages

The training pipeline runs six stages in sequence, each producing a typed **artifact** consumed by the next stage. In addition, the evaluation policy can be replayed against an existing candidate artifact after lineage evidence is updated, without retraining.

### 1. Data Ingestion
Runs a controlled MongoDB extraction in a snapshot session using the explicit training projection and an ascending indexed `id` order. Before the first run, create/check that index in Atlas or Compass; no JavaScript is required. The pipeline writes an immutable per-run `feature_store/data.csv` beside `feature_store/source_manifest.json`; the manifest records the snapshot ID, MongoDB namespace, filter, projection, read boundary, extraction timestamps, row count, ID range, and content hash. If both files already exist they are hash-verified and reused; a partial or altered pair aborts rather than overwriting evidence. A retryable read failure aborts the extraction, while a deliberately new pipeline run creates a new snapshot ID. A verified snapshot is then split using the frozen stratified 80/10/10 protocol with `random_state=42`. The resulting `train.csv`, `validation.csv`, and `test.csv` files are disjoint and cover the source IDs exactly once. See [MongoDB source provenance](docs/mongodb_provenance.md).

### 2. Data Validation
Validates the training, validation, and test files against the versioned data contract using Pandera-backed checks. It checks required columns, target rules, dtypes, allowed categories, numeric constraints, null/NaN behavior, forbidden fields, and ID integrity, then writes a JSON validation report. If validation fails, the pipeline halts before transformation or training.

### 3. Data Transformation
Builds a `ColumnTransformer` driven by `configs/schema.yaml`:
- one-hot encoding with `handle_unknown="ignore"` for `Gender`, `Vehicle_Age`, `Vehicle_Damage`, `Region_Code`, and `Policy_Sales_Channel`;
- passthrough for `Age`, `Annual_Premium`, `Vintage`, `Driving_License`, and `Previously_Insured`;
- no numeric scaling.

The transformer is fit on training data only, applied to validation and test data without refitting, and persisted as an intermediate training artifact (`preprocessor.pkl`). Serving does not download or apply that detached artifact: the final MLflow model contains its own fitted preprocessor. The verified benchmark produces `219` train-only transformed features; the final train-plus-validation refit produces `220` features because a valid category is observed during final fitting.

### 4. Model Training
Loads the frozen LightGBM parameters and `scale_pos_weight` from `configs/model_spec.yaml`. The trainer first fits a train-only candidate for validation and repeated stability analysis, then clones and refits the preprocessing/model pipeline on train plus validation. The untouched test split is evaluated once through that final pipeline. Metrics, feature order, model URI, data hash, Git/dependency versions, and MLflow lineage are persisted locally and logged to MLflow.

### 5. Source-frozen Parity
Certifies the candidate's split IDs, feature order, transformed values, and predictions against the frozen source evidence. This stage runs automatically before evaluation and records verified evidence in the candidate lineage.

### 6. Model Evaluation
The evaluation component compares candidate validation metrics using primary top-K measures (`precision_at_10pct` and `lift_at_10pct`), supporting metrics, repeated-CV stability, lineage, and source-frozen parity. Provisional numeric thresholds are versioned in `configs/model_spec.yaml` and remain pending owner approval. Under the current candidate-only policy, Champion comparison is disabled and promotion remains disabled. Final-test metrics are reporting-only. See [Existing-candidate evaluation replay](docs/model_evaluation_replay.md) to regenerate evaluation evidence from a completed candidate without retraining.

---

## 🔌 API Reference

Base URL: `http://localhost:8080`

| Method | Endpoint    | Description |
|--------|-------------|-------------|
| `GET`  | `/`         | Serves the prediction dashboard UI |
| `GET`  | `/health`   | Backward-compatible aggregate health/readiness response |
| `GET`  | `/health/live` | Process liveness; does not depend on MLflow or model state |
| `GET`  | `/health/ready` | Model readiness and immutable model/evaluation metadata; returns `503` until a validated model is loaded |
| `POST` | `/predict`  | Runs inference on a single customer record |
| `POST` | `/rank`     | Scores a batch and returns the highest-scoring top-K campaign customers |
| `POST` | `/reload`   | Hot-swaps the in-memory model with the configured immutable MLflow candidate; it requires the `X-Reload-Token` header |

Training is intentionally not exposed through the public FastAPI service. A future CI/CD workflow or dedicated training job will invoke `python -m vehicle_insurance.pipeline.training_pipeline` outside the web process, publish an evaluated candidate, and use the controlled release/reload path when approved.

**Example `/predict` request:**
```json
{
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
  "Vintage": 5
}
```

**Example `/predict` response:**
```json
{
  "response_propensity_score": 0.8421,
  "probability": 0.8421,
  "prediction": null,
  "operating_threshold": null,
  "model_version": "m-e339c348e27d400d835b8e8bdf8fdbe1",
  "api_timestamp": "2026-07-08T10:15:00+00:00"
}
```

The canonical output is the double-valued `response_propensity_score`, which is used to rank customers. The compatibility field `probability` carries the same uncalibrated score and must not be interpreted as a guaranteed probability. `prediction` is `null` by default because the current configuration does not enable an operating threshold; if a threshold is explicitly enabled, it is a diagnostic binary result and not the campaign-selection rule. Campaign capacity is controlled by ranking and selecting the highest-scoring top-K customers. All API, model-load, and reload timestamps are timezone-aware ISO-8601 UTC values with an explicit `+00:00` offset.

Readiness metadata includes the immutable model ID, model URI, MLflow source run ID, model-spec/contract versions, Git commit, dataset snapshot ID, validation metrics, and report-only test metrics. Primary selection metrics are Precision@10% and Lift@10%; missing legacy metrics are reported as unavailable and do not cause a `KeyError` during model loading.

The returned score is intended for marketing prioritization. It is not a claim probability, underwriting score, price recommendation, eligibility decision, or causal estimate of whether contacting the customer will create an incremental response.

Before either endpoint calls the model, the API applies the shared scoring contract in `configs/schema.yaml`: required identifiers and fields, domain/range checks, finite numeric values, allowed categories, forbidden outcome fields, exact columns, and batch ID uniqueness. Customer IDs must be whole numbers from `1` through `9,007,199,254,740,991`, the largest integer range represented exactly by both the Python API and browser JavaScript; larger or malformed values are rejected before scoring.

All API failures use the stable JSON envelope `{"error_code": "...", "message": "...", "fields": []}`. Malformed scoring requests use `SCORING_SCHEMA_INVALID`, invalid domain values use `SCORING_VALUE_INVALID`, an unavailable serving model uses `MODEL_NOT_READY`, a failed candidate load uses `MODEL_LOAD_FAILED` or `MODEL_RELOAD_FAILED`, model execution uses `MODEL_INFERENCE_FAILED`, an unavailable MLflow/model-store dependency uses `DEPENDENCY_UNAVAILABLE`, and unexpected application failures use `INTERNAL_ERROR`. Messages are safe for clients and never contain stack traces, credentials, or customer attributes.

**Example `/rank` request:**
```json
{
  "records": [
    {
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
      "Vintage": 5
    }
  ],
  "top_k_percentage": 10
}
```

`/rank` accepts either `top_k_percentage` or `top_k_count` (not both). If omitted, it uses the versioned `selection.top_k_percentage` from `configs/model_spec.yaml`, applies floor conversion, and breaks score ties by ascending `id`. The optional operating threshold is never used to choose the campaign audience.

The dashboard is intentionally task-focused rather than a model-card viewer. It shows a small readiness indicator, explains what a response-propensity score means, and provides two workflows: estimate one customer's score or prioritize a campaign audience. The single-customer workflow describes the score as a relative campaign-prioritization signal, not a guaranteed probability or business decision. The batch workflow explains capacity-based selection, deterministic tie-breaking, and the meaning of unselected customers. Its JSON input is kept inside an advanced expandable section and can be populated with sample records, so the main page is not dominated by raw implementation data. Results show scored count, selected count, capacity, cutoff score, and the selected customer ranking. API validation failures are rendered with their error code, safe message, and affected fields rather than as `[object Object]` or an opaque generic toast. Technical model lineage and evaluation metadata remain available through readiness/API documentation rather than competing with the primary user workflow.

### Dashboard input guide

For single-customer scoring, use the values from the customer's existing customer or policy record. The dashboard includes an expandable explanation beside the form; the same meanings apply to every record in a batch:

| Field | What to enter |
|---|---|
| `id` | The unique customer identifier from the source system, restricted to `1–9,007,199,254,740,991`; it is used for traceability and deterministic tie-breaking, not as a model feature. |
| `Gender` | The recorded value `Male` or `Female`. |
| `Age` | Current customer age in years, from 18 through 100. |
| `Driving_License` | `1` when the customer has a driving license, otherwise `0`. |
| `Region_Code` | The numeric geographic region code from the source system; do not substitute a postal code without an approved mapping. |
| `Previously_Insured` | `1` when the customer already has vehicle insurance, otherwise `0`. |
| `Vehicle_Age` | One of the recorded vehicle-age brackets: `< 1 Year`, `1-2 Year`, or `> 2 Years`. |
| `Vehicle_Damage` | `Yes` or `No` according to the recorded vehicle-damage status. |
| `Annual_Premium` | The current positive annual premium amount in rupees. |
| `Policy_Sales_Channel` | The numeric sales-channel code from the source system, not the channel name. |
| `Vintage` | The number of days since the customer relationship or policy record began. |

Do not send `Response`, conversion outcomes, campaign results, or any field created after the campaign. Those values would leak the outcome into the scoring request and are rejected by the data contract.

Interactive Swagger docs are available at `/docs` once the server is running.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10–3.13
- [`uv`](https://github.com/astral-sh/uv)
- A MongoDB Atlas cluster (or local MongoDB instance) with your data loaded
- An MLflow tracking server (this project was tracked via [DagsHub](https://dagshub.com/))

### Installation
```bash
git clone https://github.com/Youngboy01/vehicle-Insurance.git
cd vehicle-Insurance

# using uv (reads pyproject.toml / uv.lock)
uv sync
```

---

## 🔐 Environment Variables

Create a `.env` file in the project root:

```env
MONGODB_URL=<your MongoDB Atlas connection string>
MLFLOW_TRACKING_URI=<your MLflow tracking server URI>
MLFLOW_TRACKING_USERNAME=<if using DagsHub-hosted MLflow>
MLFLOW_TRACKING_PASSWORD=<if using DagsHub-hosted MLflow>
RELOAD_TOKEN=<any secret string — protects the /reload endpoint>
```

---

## 🏃 Running the Training Pipeline

```cmd
uv run python -m vehicle_insurance.pipeline.training_pipeline
```

Parity recording is transactional: a pending certificate and pending local
lineage are written before the optional MLflow artifact/tags, and only a
certificate whose digest and candidate/attempt identity are complete is
persisted as `verified`. Failure or interruption leaves pending/failed evidence
that replay rejects. The release policy requires the `local_and_mlflow` mode;
local-only audits do not qualify a release.

Replay reports are fresh records: destinations are checked before artifact
loading and cannot overwrite model, metrics, snapshots/manifests, parity
certificates, the original decision report, or link aliases. The default
`model_evaluation/replay_report.json` must be given a new filename for each
replay. Report writes use unique exclusive temporary files; fresh replay
reports use atomic no-replace publication, while explicitly replaceable
ordinary reports retain atomic replacement. The dashboard validates raw batch
ID tokens before `JSON.parse`, so fractional or exponent forms cannot round
into a different safe integer.

Run this command from the repository root. The public module entry point executes ingestion, validation, transformation, model training, source-frozen parity, and final evaluation in sequence. It saves raw data, splits, transformed arrays, preprocessor, trained model, metrics, parity certificate, evaluation report, and lineage under a unique `artifact\<run-directory>` directory. Promotion remains disabled until the evaluation policy, source-frozen parity, segment review, and owner approval gates are complete. Stage-level benchmark helpers, if used during local development, are not part of the public release workflow.

### Replaying Existing-candidate Evaluation

When a completed candidate receives corrected or newly certified lineage, replay
the evaluation without rerunning ingestion, transformation, or training:

```cmd
set "CANDIDATE_ARTIFACT_DIR=artifact\YOUR_RUN_DIRECTORY"
uv run python -m vehicle_insurance.verification.replay_model_evaluation --candidate-artifact-dir "%CANDIDATE_ARTIFACT_DIR%"
```

Replace `YOUR_RUN_DIRECTORY` with the actual run directory created by training.
The replay writes `%CANDIDATE_ARTIFACT_DIR%\model_evaluation\replay_report.json`
and preserves the earlier `report.json`. It reads the recorded trainer metrics,
reruns the validation/stability/lineage/parity gates, skips Champion comparison
under the current candidate-only policy, keeps test metrics reporting-only, and
never mutates MLflow aliases. Exit code `0`
means the candidate passed; exit code `2` means evaluation completed but a
policy gate rejected the candidate; exit code `1` means replay execution failed.

See [Git Lineage and Release Preflight](docs/git_lineage.md) for the clean-tree
requirements and provenance recorded before a fresh candidate run.

> Regarding the existing model: its artifact was created before Git provenance
> existed, so its exact source state cannot be reconstructed. Keep it as
> historical evidence, but do not use it as a release-qualified model or
> retrofit unknown Git information. The next fresh run will train the current
> model again under a clean, traceable commit.

---

## 🖥 Running the API

```bash
uvicorn app:app --host 0.0.0.0 --port 8080 --reload
```

Then visit:
- `http://localhost:8080` — dashboard UI
- `http://localhost:8080/docs` — Swagger UI

During local development, the dashboard derives a cache-busting query value from the current `static/style.css` modification time. When the stylesheet changes, the rendered link changes automatically, so the browser fetches the new CSS without requiring `Ctrl+F5` or manual version editing.

> Current serving contract: the API loads the immutable candidate model ID, URI, and run ID recorded in `configs/model_spec.yaml` at startup and on `/reload`. It loads one self-contained sklearn preprocessing-plus-LightGBM pipeline, validates raw inputs, and calls its public `predict_proba()` interface. A failed startup load leaves the process live but not ready; a failed reload leaves the previously served engine unchanged. Promotion remains disabled; no registry alias is consulted.

---

## 🐳 Docker

The project ships with a multi-stage Dockerfile: a builder stage that uses `uv` (with layer caching) to resolve dependencies, and a slim `python:3.13-slim-bookworm` runtime stage with only `libgomp1` (required by LightGBM) added.

```bash
docker build -t vehicle-insurance .
docker run -p 8080:8080 --env-file .env vehicle-insurance
```

---

## 🏆 Model Registry Workflow

The current trainer logs a new candidate model and complete lineage to MLflow. The evaluation component consumes that candidate directly and applies validation top-K metrics, stability, source-frozen parity, immutable MongoDB snapshot provenance, and explicit acceptance thresholds. Promotion is currently disabled in `configs/model_spec.yaml`; the API serves only the explicitly configured immutable candidate and retraining/evaluation do not automatically change that configuration. Existing candidates can be replay-evaluated with the documented utility, but artifacts created before the source-manifest contract are historical evidence rather than release-qualified models.

---

## 🗂 Data Schema

Configured in `configs/schema.yaml` and enforced by Pandera in the validation stage:

| Field | Type | Notes |
|---|---|---|
| `id` | int | dropped before training |
| `Gender` | str | `Male` / `Female` |
| `Age` | int | 18–100 general safety range |
| `Driving_License` | int | 0 / 1 |
| `Region_Code` | float | ≥ 0 |
| `Previously_Insured` | int | 0 / 1 |
| `Vehicle_Age` | str | `< 1 Year`, `1-2 Year`, `> 2 Years` |
| `Vehicle_Damage` | str | `Yes` / `No` |
| `Annual_Premium` | float | > 0 |
| `Policy_Sales_Channel` | float | > 0 |
| `Vintage` | int | ≥ 0 |
| `Response` | int | marketing-response target: `0` = no positive response, `1` = positive response |

---

## 🗺 Roadmap

- [ ] CI pipeline (GitHub Actions): lint, test, build Docker image on push
- [ ] CD pipeline: automated deployment on merge to `main`
- [ ] Cloud deployment (e.g., AWS/GCP/Azure container hosting)
- [ ] Scheduled/automated retraining trigger
- [ ] Unit and integration tests for pipeline components
- [ ] Model/data drift monitoring

---
