# 🚗 Vehicle Insurance Cross-Sell Prediction — MLOps Pipeline

An end-to-end machine learning system that predicts whether a health insurance customer will respond positively to a **vehicle insurance cross-sell** offer. The project goes beyond a notebook model: it is a modular pipeline with schema validation, experiment tracking, a champion/challenger promotion workflow, and a served REST API — all containerized with Docker.

> **Status:** Core pipeline, model registry workflow, and API are fully functional. CI/CD and cloud deployment are in progress (see [Roadmap](#-roadmap)).

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
- [Running the API](#-running-the-api)
- [Docker](#-docker)
- [Model Registry Workflow](#-model-registry-workflow)
- [Data Schema](#-data-schema)
- [Roadmap](#-roadmap)

---

## 🔎 Overview

The dataset used is the classic **health-to-vehicle-insurance cross-sell** dataset: given a customer's demographics, vehicle history, and policy details, predict whether they will respond positively to a vehicle-insurance offer (`Response` = 0/1). The target is highly imbalanced, which shapes several design decisions below, including class-weighted LightGBM and recall-oriented baseline evaluation.

Rather than training a model once and calling it done, this project implements the pieces you'd expect in a real ML system:

- **Schema-validated data ingestion** from MongoDB
- **A reusable, versioned preprocessing pipeline** (fit once, applied consistently at train and inference time)
- **Experiment tracking and a model registry** with a champion/challenger promotion rule
- **A served model** behind a REST API that can retrain and hot-swap itself
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
   │  1. Data Ingestion   → export + train/test split      │
   │  2. Data Validation  → Pandera schema checks           │
   │  3. Data Transform.  → scale/encode, save preprocessor │
   │  4. Model Training   → LightGBM + MLflow logging        │
   │  5. Model Evaluation → champion vs. challenger recall    │
   └───────────────────────────┬───────────────────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   MLflow Model Registry │  (tracked on DagsHub)
                    │   Champion / Challenger  │
                    └────────────┬─────────────┘
                                 │ loads champion model + preprocessor
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
| Data processing          | pandas, NumPy, scikit-learn (`ColumnTransformer`, `StandardScaler`, `RobustScaler`, `OneHotEncoder`) |
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
│   └── schema.yaml                 # Feature groups used by the transformer
├── notebooks/
│   ├── notebook.ipynb              # Exploration
│   └── experimentation.ipynb       # Model experimentation / MLflow runs
├── static/style.css                # Dashboard styling
├── templates/index.html            # Dashboard UI
└── src/vehicle_insurance/
    ├── components/
    │   ├── dataIngestion.py        # MongoDB → feature store → train/test split
    │   ├── dataValidation.py       # Pandera schema validation
    │   ├── dataTransformation.py   # ColumnTransformer pipeline
    │   ├── modelTrainer.py         # LightGBM training + MLflow logging
    │   └── modelEvaluation.py      # Champion/challenger comparison
    ├── configurations/
    │   └── mongodb_connection.py   # MongoDB client (singleton)
    ├── data_access/
    │   └── Insurance_data.py       # MongoDB → DataFrame
    ├── entities/
    │   ├── config_entity.py        # Per-stage config dataclasses
    │   └── artifact_entity.py      # Per-stage output dataclasses
    ├── pipeline/
    │   └── training_pipeline.py    # Orchestrates all 5 stages
    ├── constants/__init__.py       # Central constants
    ├── exceptions/handle_exception.py
    ├── logger/logger_config.py
    └── utils/__init__.py           # pickle/yaml/numpy IO helpers, MLflow client
```

---

## ⚙️ Pipeline Stages

The training pipeline (`TrainingPipeline.run_pipeline()`) runs five stages in sequence, each producing a typed **artifact** consumed by the next stage:

### 1. Data Ingestion
Pulls the raw collection from MongoDB into a "feature store" CSV, then splits it into train/test sets (`DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO = 0.25`).

### 2. Data Validation
Validates both train and test sets against a **Pandera `DataFrameSchema`** — checking dtypes, allowed categories (`Gender`, `Vehicle_Age`, `Vehicle_Damage`), and numeric ranges. Uses `lazy=True` to collect *all* schema violations at once rather than failing on the first one, and writes a JSON validation report. If validation fails, the pipeline halts before wasting compute on transformation/training.

### 3. Data Transformation
Builds a `ColumnTransformer` (wrapped in a `Pipeline`) driven by `configs/schema.yaml`:
- `StandardScaler` on standard numeric features (`Age`, `Vintage`)
- `RobustScaler` on outlier-prone features (`Annual_Premium`)
- `OneHotEncoder` (drop-first, unknown-safe) on categoricals (`Gender`, `Vehicle_Age`, `Vehicle_Damage`)

The transformer is **fit once on train data only**, applied to both train/test, and persisted as `preprocessor.pkl` so training and inference always use identical logic.

### 4. Model Training
Loads best hyperparameters from a prior experimentation run, trains a `LGBMClassifier` with `is_unbalance=True` to counter class imbalance, computes accuracy/precision/recall/F1/ROC-AUC, and **rejects the model if test recall falls below a configured threshold** (`MODEL_TRAINER_EXPECTED_SCORE = 0.75`). All params/metrics/artifacts are logged to MLflow, and the model is registered under a `Challenger` alias.

### 5. Model Evaluation
Compares the new `Challenger` model's recall against the current `Champion` in the MLflow Model Registry:
- No champion exists yet → challenger is promoted by default.
- Challenger beats champion by more than `MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE` → challenger is promoted to `Champion`.
- Otherwise → champion is retained and the challenger is discarded.

This is the same **champion/challenger** pattern used in real-world ML platforms to guard against silently shipping a worse model.

---

## 🔌 API Reference

Base URL: `http://localhost:8080`

| Method | Endpoint    | Description |
|--------|-------------|-------------|
| `GET`  | `/`         | Serves the prediction dashboard UI |
| `GET`  | `/health`   | Returns current champion model version, offline test recall, and load time |
| `POST` | `/predict`  | Runs inference on a single customer record |
| `POST` | `/train`    | Kicks off the training pipeline as a background task (locked — rejects if training is already running) |
| `POST` | `/reload`   | Hot-swaps the in-memory model with the latest MLflow champion, no restart required (requires `X-Reload-Token` header) |

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
  "probability": 0.8421,
  "model_version": "3",
  "api_timestamp": "2026-07-08T10:15:00"
}
```

In the response, `prediction = 1` means that the model classifies the customer as likely to respond positively to the cross-sell offer; `prediction = 0` means that the model classifies the customer as unlikely to respond positively. The `probability` field is the model's estimated response propensity score for the positive class. Until formal calibration is evaluated, it should not be interpreted as a guaranteed or perfectly calibrated probability.

The returned score is intended for marketing prioritization. It is not a claim probability, underwriting score, price recommendation, eligibility decision, or causal estimate of whether contacting the customer will create an incremental response.

Interactive Swagger docs are available at `/docs` once the server is running.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10–3.13
- [`uv`](https://github.com/astral-sh/uv) (recommended) or pip
- A MongoDB Atlas cluster (or local MongoDB instance) with your data loaded
- An MLflow tracking server (this project was tracked via [DagsHub](https://dagshub.com/))

### Installation
```bash
git clone https://github.com/Youngboy01/vehicle-Insurance.git
cd vehicle-Insurance

# using uv (recommended — reads pyproject.toml / uv.lock)
uv sync

# or using pip
pip install -e .
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

```bash
python -c "from vehicle_insurance.pipeline.training_pipeline import TrainingPipeline; TrainingPipeline().run_pipeline()"
```

This runs all five stages end-to-end and, if the trained model passes the recall threshold and beats the current champion, promotes it in the MLflow Model Registry. Artifacts (raw data, splits, transformed arrays, preprocessor, trained model, reports) are saved under a timestamped `artifact/` directory. The current pipeline trains and evaluates the response-propensity model; top-K batch selection and selection-quality evaluation are separate Phase 1 capabilities to be added later. Budget-based selection is deferred beyond the current implementation.

---

## 🖥 Running the API

```bash
uvicorn app:app --host 0.0.0.0 --port 8080 --reload
```

Then visit:
- `http://localhost:8080` — dashboard UI
- `http://localhost:8080/docs` — Swagger UI

> Note: the API loads the current **champion** model from MLflow at startup, so a model must already be registered before the API can serve predictions (i.e., run the training pipeline at least once first).

---

## 🐳 Docker

The project ships with a multi-stage Dockerfile: a builder stage that uses `uv` (with layer caching) to resolve dependencies, and a slim `python:3.13-slim-bookworm` runtime stage with only `libgomp1` (required by LightGBM) added.

```bash
docker build -t vehicle-insurance .
docker run -p 8080:8080 --env-file .env vehicle-insurance
```

---

## 🏆 Model Registry Workflow

```
 New model trained
        │
        ▼
 Registered as "Challenger" in MLflow
        │
        ▼
 Compare test recall vs. current "Champion"
        │
   ┌────┴─────┐
   │           │
 Better     Not better
   │           │
   ▼           ▼
Promote to   Champion
 Champion    retained
```

This means retraining (via the `/train` endpoint or a scheduled job) is **safe to run repeatedly** — the system will only ever update the served model if the new one is measurably better, and the change can be picked up live via `/reload` without redeploying the service.

---

## 🗂 Data Schema

Configured in `configs/schema.yaml` and enforced by Pandera in the validation stage:

| Field | Type | Notes |
|---|---|---|
| `id` | int | dropped before training |
| `Gender` | str | `Male` / `Female` |
| `Age` | int | > 0 |
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
