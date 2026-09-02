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

This project is a **top-K vehicle-insurance cross-sell decision-support platform**. Given customer information available before a marketing campaign, it estimates each customer's likelihood of responding positively to a vehicle-insurance cross-sell offer, ranks customers by that response propensity, and supports selecting the highest-scoring customers for outreach at a company-selected capacity.

The decision unit is **one customer-campaign opportunity**. The intended user is a marketing or campaign-operations team deciding which customers to prioritize for outreach.

The current API performs single-customer inference. Batch ranking, top-K selection, and selection-quality evaluation are Phase 1 extensions built on top of the response score; they are not yet claimed to be implemented by the current `/predict` endpoint. Budget-based selection, outreach-cost assumptions, conversion-value assumptions, and ROI calculations are later extensions.

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
   │  1. Data Ingestion   → export + stratified 80/10/10 split │
   │  2. Data Validation  → Pandera schema checks           │
   │  3. Data Transform.  → fit on train, transform all splits │
   │  4. Model Training   → frozen LightGBM + MLflow lineage    │
   │  5. Model Evaluation → validation policy + replay evidence │
   └───────────────────────────┬───────────────────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   MLflow Tracking       │  (tracked on DagsHub)
                    │   Candidate lineage     │
                    └────────────┬─────────────┘
                                 │ registry serving integration pending
                                 ▼
                    ┌───────────────────────┐
                    │      FastAPI Service    │
                    │  /predict  /train        │
                    │  /health   /reload        │
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

The training pipeline runs five stages in sequence, each producing a typed **artifact** consumed by the next stage. In addition, the evaluation policy can be replayed against an existing candidate artifact after lineage evidence is updated, without retraining.

### 1. Data Ingestion
Pulls the raw `INSURANCE_DATA` collection from MongoDB Atlas into a timestamped feature-store CSV. The extracted DataFrame is sorted locally by ascending `id` and split using the frozen stratified 80/10/10 protocol with `random_state=42`. The resulting `train.csv`, `validation.csv`, and `test.csv` files are disjoint and cover the source IDs exactly once.

### 2. Data Validation
Validates the training, validation, and test files against the versioned data contract using Pandera-backed checks. It checks required columns, target rules, dtypes, allowed categories, numeric constraints, null/NaN behavior, forbidden fields, and ID integrity, then writes a JSON validation report. If validation fails, the pipeline halts before transformation or training.

### 3. Data Transformation
Builds a `ColumnTransformer` driven by `configs/schema.yaml`:
- one-hot encoding with `handle_unknown="ignore"` for `Gender`, `Vehicle_Age`, `Vehicle_Damage`, `Region_Code`, and `Policy_Sales_Channel`;
- passthrough for `Age`, `Annual_Premium`, `Vintage`, `Driving_License`, and `Previously_Insured`;
- no numeric scaling.

The transformer is fit on training data only, applied to validation and test data without refitting, and persisted as `preprocessor.pkl`. The verified benchmark produces `219` train-only transformed features; the final train-plus-validation refit produces `220` features because a valid category is observed during final fitting.

### 4. Model Training
Loads the frozen LightGBM parameters and `scale_pos_weight` from `configs/model_spec.yaml`. The trainer first fits a train-only candidate for validation and repeated stability analysis, then clones and refits the preprocessing/model pipeline on train plus validation. The untouched test split is evaluated once through that final pipeline. Metrics, feature order, model URI, data hash, Git/dependency versions, and MLflow lineage are persisted locally and logged to MLflow.

### 5. Model Evaluation
The evaluation component compares candidate validation metrics using primary top-K measures (`precision_at_10pct` and `lift_at_10pct`), supporting metrics, repeated-CV stability, lineage/source-frozen parity, and Champion validation evidence when available. Provisional numeric thresholds are versioned in `configs/model_spec.yaml` and remain pending owner approval. Final-test metrics are reporting-only. Registry alias mutation remains disabled while `promotion.enabled` is `false`. See [Existing-candidate evaluation replay](docs/model_evaluation_replay.md) to regenerate evaluation evidence from a completed candidate without retraining.

---

## 🔌 API Reference

Base URL: `http://localhost:8080`

| Method | Endpoint    | Description |
|--------|-------------|-------------|
| `GET`  | `/`         | Serves the prediction dashboard UI |
| `GET`  | `/health`   | Returns service health, loaded model metadata, and load time |
| `POST` | `/predict`  | Runs inference on a single customer record |
| `POST` | `/train`    | Kicks off the training pipeline as a background task (locked — rejects if training is already running) |
| `POST` | `/reload`   | Hot-swaps the in-memory model with the latest registered MLflow champion, no restart required (requires `X-Reload-Token` header) |

**Example `/predict` request:**
```json
{
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
  "prediction": 1,
  "response_propensity_score": 0.8421,
  "probability": 0.8421,
  "model_version": "reference-or-registered-version",
  "api_timestamp": "2026-07-08T10:15:00"
}
```

In the response, `prediction = 1` means that the model classifies the customer as likely to respond positively to the cross-sell offer; `prediction = 0` means that the model classifies the customer as unlikely to respond positively. The canonical semantic field is `response_propensity_score`. The compatibility field `probability` may remain temporarily, but it carries the same uncalibrated response-propensity score and must not be interpreted as a guaranteed probability.

The returned score is intended for marketing prioritization. It is not a claim probability, underwriting score, price recommendation, eligibility decision, or causal estimate of whether contacting the customer will create an incremental response.

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
cd /d D:\projVehicleInsurance\vehicle-Insurance
uv run python -m vehicle_insurance.pipeline.training_pipeline
```

The public module entry point executes ingestion, validation, transformation, model training, and evaluation in sequence. It returns an evaluation artifact and saves raw data, splits, transformed arrays, preprocessor, trained model, metrics, evaluation report, and lineage under a timestamped `artifact/` directory. Run from the repository root so relative paths resolve under the D: repository. Promotion remains disabled until the evaluation policy, source-frozen parity, segment review, and owner approval gates are complete. The private `tests\run_real_training.py` helper remains available for stage-level benchmark runs when evaluation is not required.

### Replaying Existing-candidate Evaluation

When a completed candidate receives corrected or newly certified lineage, replay
the evaluation without rerunning ingestion, transformation, or training:

```cmd
cd /d D:\projVehicleInsurance\vehicle-Insurance
uv run python -m vehicle_insurance.verification.replay_model_evaluation --candidate-artifact-dir artifact\08_31_2026_01_05_59
```

The replay writes `model_evaluation\replay_report.json` inside the candidate
artifact and preserves the earlier `report.json`. It reads the recorded trainer
metrics, reruns the validation/stability/lineage/parity/Champion gates, keeps
test metrics reporting-only, and never mutates MLflow aliases. Exit code `0`
means the candidate passed; exit code `2` means evaluation completed but a
policy gate rejected the candidate; exit code `1` means replay execution failed.

---

## 🖥 Running the API

```bash
uvicorn app:app --host 0.0.0.0 --port 8080 --reload
```

Then visit:
- `http://localhost:8080` — dashboard UI
- `http://localhost:8080/docs` — Swagger UI

> Note: the API loads the current registered **champion** model from MLflow at startup, so a model must already be registered before the API can serve predictions. The current trainer candidate is not registered while promotion is disabled.

---

## 🐳 Docker

The project ships with a multi-stage Dockerfile: a builder stage that uses `uv` (with layer caching) to resolve dependencies, and a slim `python:3.13-slim-bookworm` runtime stage with only `libgomp1` (required by LightGBM) added.

```bash
docker build -t vehicle-insurance .
docker run -p 8080:8080 --env-file .env vehicle-insurance
```

---

## 🏆 Model Registry Workflow

The current trainer logs a new candidate model and complete lineage to MLflow. The evaluation component consumes that candidate directly and applies validation top-K metrics, stability, source-frozen parity, and explicit acceptance thresholds. Promotion is currently disabled in `configs/model_spec.yaml`; no claim should be made that retraining or evaluation automatically updates the served champion. Existing candidates can be replay-evaluated with the documented utility.

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
