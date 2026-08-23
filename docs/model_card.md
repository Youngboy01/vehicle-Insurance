# Model Card: Vehicle Insurance Cross-Sell Response Propensity

## 1. Model summary

| Item | Current value |
|---|---|
| Model name | `Vehicle_Insurance_LGBM` |
| Model type | LightGBM binary classifier |
| Current reviewed champion | MLflow model version 7 at the time of review |
| Task | Binary response-propensity classification |
| Positive class | `Response = 1` |
| Decision unit | One customer-campaign opportunity |
| Intended user | Marketing or campaign-operations team |
| Serving interface | FastAPI `/predict` endpoint |
| Review status | Phase 1 research/portfolio system; not production-approved for autonomous decisions |
| Last reviewed | 2026-08-20 |

This model estimates whether a customer is likely to respond positively to a vehicle-insurance cross-sell offer. It is a marketing-prioritization model, not a claims, pricing, underwriting, fraud, or eligibility model.

## 2. Project definition

The broader Phase 1 system is a top-K vehicle-insurance cross-sell decision-support platform. Given customer information available before a marketing campaign, the system should estimate response propensity, rank customers, and support selecting the highest-scoring customers at a company-selected campaign capacity.

The current deployed API performs single-customer inference. Batch ranking, top-K selection, and selection-quality evaluation are planned Phase 1 capabilities built on top of the response score; they should not be described as already implemented by the current `/predict` endpoint. Budget-based selection, outreach-cost assumptions, conversion-value assumptions, and ROI calculations are later extensions.

## 3. Intended use

### Appropriate uses

The model may be used to:

- prioritize customers for vehicle-insurance marketing outreach;
- produce a response-propensity ranking for campaign planning;
- compare top-K campaign capacities using offline lift and precision/recall metrics;
- support a human campaign-operations workflow;
- monitor whether model scores remain useful over time.

The model output should be combined with the company-selected campaign capacity, business rules, customer preferences, and applicable legal or organizational policies. Financial assumptions are outside the current Phase 1 scope.

### Prohibited uses

The model must not be used as the sole basis for:

- approving or rejecting insurance claims;
- setting or recommending insurance premiums;
- underwriting or risk acceptance;
- determining insurance eligibility;
- fraud detection or investigation;
- cancelling or restricting a policy;
- automatically denying a service or applying another adverse action;
- predicting future vehicle-insurance claims;
- making a legal, financial, or eligibility decision about a customer.

The model is intended for marketing prioritization only. It should not be used to infer that a customer deserves or does not deserve insurance coverage.

## 4. Target definition

The target column is `Response`:

- `Response = 1`: the customer responded positively to the vehicle-insurance offer.
- `Response = 0`: the customer did not respond positively to the vehicle-insurance offer.

The model output is currently exposed as:

- `prediction = 1`: the classifier predicts a positive response at its current decision rule;
- `prediction = 0`: the classifier predicts no positive response at its current decision rule;
- `probability`: an estimated positive-class response propensity score;
- `model_version`: the MLflow champion version used for inference;
- `api_timestamp`: the time at which the API produced the response.

The `probability` field should not be interpreted as a guaranteed or perfectly calibrated probability until calibration has been evaluated and documented.

## 5. Prediction timing and decision unit

The intended scoring moment is immediately before a marketing campaign or outreach decision. The model should use only information available at that time, such as customer demographics, vehicle information, policy information, and historical status fields.

The decision unit is one customer-campaign opportunity. In the current benchmark data, one row is treated as one customer record. If future business data contains multiple policies or multiple campaign events for the same customer, the data model and split strategy must be revisited.

The target response is observed after the offer. It must never be used as an input feature at scoring time.

## 6. Data

### Data source

The current benchmark data is stored in `notebooks/data.csv` and is loaded into MongoDB for pipeline ingestion. The original dataset source, license, acquisition date, and content hash still need to be formally recorded as part of the data-provenance tasks.

The benchmark should not be presented as proof of production performance. It is a historical/static dataset used to demonstrate the modeling and MLOps workflow.

### Dataset profile observed during review

- Rows: 381,109
- Columns: 12, including `id` and `Response`
- Positive responses: 46,710
- Negative responses: 334,399
- Positive rate: approximately 12.26%
- Negative-to-positive ratio: approximately 7.16:1
- Nulls in the inspected CSV: none observed

### Features

| Feature | Role | Notes |
|---|---|---|
| `id` | Identifier | Dropped before training; uniqueness should be validated |
| `Gender` | Feature | Categorical value |
| `Age` | Feature | Numeric customer age |
| `Driving_License` | Feature | Binary indicator |
| `Region_Code` | Feature | Encoded region; numeric-versus-categorical treatment needs validation |
| `Previously_Insured` | Feature | Binary insurance-history indicator |
| `Vehicle_Age` | Feature | Categorical vehicle-age group |
| `Vehicle_Damage` | Feature | Categorical damage indicator |
| `Annual_Premium` | Feature | Numeric premium value |
| `Policy_Sales_Channel` | Feature | Encoded channel; numeric-versus-categorical treatment needs validation |
| `Vintage` | Feature | Customer tenure-like value in the benchmark data |
| `Response` | Target | Positive response label; never an inference input |

## 7. Training and preprocessing

The current pipeline contains these stages:

1. Ingest the MongoDB collection into a timestamped feature-store CSV.
2. Split the data into train and test sets.
3. Validate both sets with a Pandera schema.
4. Fit preprocessing on training data only.
5. Apply the fitted preprocessing to training and test data.
6. Train a LightGBM classifier with imbalance handling.
7. Log metrics and artifacts to MLflow.
8. Register the model as a challenger.
9. Compare challenger recall against the current champion.

The current preprocessing uses:

- `StandardScaler` for `Age` and `Vintage`;
- `RobustScaler` for `Annual_Premium`;
- one-hot encoding for `Gender`, `Vehicle_Age`, and `Vehicle_Damage`;
- passthrough behavior for the remaining configured features.

The current training model uses `LGBMClassifier` with `is_unbalance=True` and hyperparameters loaded from a previous experimentation artifact. Removing this hidden notebook-generated dependency is a required reproducibility improvement.

## 8. Evaluation evidence

The latest stored pipeline run reviewed was `artifact/04_13_2026_01_44_02`.

| Metric | Test | Train |
|---|---:|---:|
| Accuracy | 0.7174 | 0.7334 |
| Precision | 0.2894 | 0.3094 |
| Recall | 0.8991 | 0.9534 |
| F1 score | 0.4379 | 0.4672 |
| ROC-AUC | 0.8539 | 0.8894 |

These metrics describe the current stored run, not a final production guarantee.

### Evaluation limitations

The notebook and pipeline currently use different split protocols:

- notebook: 20% stratified test split;
- pipeline: 25% split without stratification.

The notebook evaluates candidates on the same test set before choosing the final candidate. That means the test set has influenced model selection and is not a completely untouched final estimate.

The notebook also applies SMOTE to the full training portion before cross-validation. A leakage-safe experiment should apply resampling within each training fold.

The current model is compared mainly using recall, precision, F1, accuracy, and ROC-AUC. PR-AUC, lift, gains, precision@K, recall@K, calibration, threshold economics, confidence intervals, and segment-level metrics are not yet part of the complete acceptance gate.

## 9. Operating point and campaign decisions

The current serving path returns a single-customer class prediction and score. It does not yet rank a batch or select a top-K campaign list.

The current campaign operating point has not yet been validated using:

- precision at a fixed top-K population;
- recall at a fixed top-K population;
- lift and cumulative gains at a fixed top-K population.

For the current Phase 1, the intended rule is to rank customers by response propensity and select the highest-scoring K customers, where K may be a company-selected percentage or customer count. The initial evaluation should compare top 1%, 5%, 10%, and 20%, with top 10% as the initial illustrative operating point.

Until this work is completed, the model should be treated as an experimental scoring component rather than a production campaign decision-maker.

Budget-based selection, outreach-cost modeling, conversion-value modeling, expected profit, and ROI are intentionally deferred. They may be added later as a configurable business-optimization layer.

## 10. Explainability and fairness

Global and local explainability for the frozen weighted LightGBM model has been completed with TreeSHAP in `notebooks/explainability_v1.ipynb`. The results provide model reason codes for analysis and future UI presentation. They explain the model's learned associations; they do not establish causal campaign impact.

Segment-level performance and fairness analysis are also not yet complete. At minimum, evaluate precision, recall, selection rate, false-positive rate, and calibration across relevant segments such as:

- gender;
- age bands;
- vehicle age;
- vehicle damage;
- previous insurance status;
- region;
- sales channel.

Small segment sizes and benchmark-data limitations must be reported alongside segment metrics. Fairness analysis should inform safeguards, not be reduced to a single overall score.

## 11. Limitations and failure modes

Important limitations include:

1. The benchmark data may not represent current customers, current campaigns, or a real deployment population.
2. The current dataset does not establish a campaign treatment/control design.
3. The model estimates association with historical response and does not establish causation.
4. The probability output has not been formally calibrated.
5. The current evaluation process uses inconsistent split designs and has test-set selection concerns.
6. `Region_Code` and `Policy_Sales_Channel` may be categorical codes rather than continuous measurements.
7. Data drift, novel categories, changed campaign behavior, and changed customer populations may reduce performance.
8. The API currently scores one record at a time and does not yet implement complete campaign-budget selection.
9. The current training process depends on an ignored local experiment pickle.
10. Offline model metrics do not prove incremental business value.

If input data is incomplete, invalid, out of distribution, or from a materially different population, the system should fail safely or route the case for review rather than silently presenting a confident score.

## 12. Monitoring plan

The production monitoring plan should include:

- missing-value and schema violation rates;
- novel-category rates;
- feature distribution drift;
- prediction-rate and score-distribution drift;
- positive-prediction-rate drift;
- latency and error rates;
- delayed-label precision, recall, PR-AUC, and calibration;
- performance by important segment;
- model version and threshold attached to every scored request.

Alerts should have defined thresholds, an owner, and a response procedure. Retraining should not automatically replace the champion without passing the documented evaluation and operational gates.

## 13. Privacy and security considerations

Customer-level data must be handled as sensitive business data even when the current benchmark is public. Do not log raw customer attributes by default. Use access controls for MongoDB, MLflow, model artifacts, and operational endpoints.

The model output should not be exposed as a basis for adverse customer decisions. Marketing use should follow applicable privacy, consent, communication, and anti-discrimination requirements.

## 14. Deployment status

The project includes a Docker image, FastAPI service, MLflow model registry integration, and a basic dashboard. However, deployment readiness is incomplete because CI/CD, secure operational access, robust tests, monitoring, rollback, and reproducible clean-environment training are still pending.

The model card should be updated before a model is promoted to a materially different dataset, evaluation protocol, serving contract, or business use.

## 15. Improvement roadmap

### Required before claiming a credible model

- Freeze train/validation/final-test evaluation design.
- Add honest baselines and cross-validation.
- Select the top-K operating point using validation data and top-K selection-quality metrics.
- Add PR-AUC, lift, gains, precision@K, recall@K, calibration, and confidence intervals.
- Validate data provenance, duplicates, split overlap, and target prevalence.

### Required before production-oriented deployment

- Remove the ignored experiment-pickle dependency.
- Make the inference artifact self-contained.
- Add tests, CI, input constraints, authentication, rate limiting, and readiness checks.
- Move training into a separate durable job or workflow.
- Add monitoring, rollback, and model-version lineage.

### Future extension with additional data

When treatment/control campaign data becomes available:

- measure incremental response;
- run controlled campaign experiments;
- compare model targeting with existing business targeting;
- build uplift or treatment-effect models;
- update this model card with causal-evaluation evidence.

## 16. Approval status

**Current status:** Suitable as a learning, portfolio, and Phase 1 research system. Not approved for autonomous claims, pricing, underwriting, eligibility, or other adverse decisions. Not yet proven to create incremental campaign value.

**Owner:** Project owner to be specified.

**Review cadence:** Update after any change to the dataset, target definition, split protocol, features, model, threshold, serving contract, or intended use.

## 17. SHAP explainability

The frozen weighted LightGBM model was explained in `notebooks/explainability_v1.ipynb` using TreeSHAP. The notebook loads the complete preprocessing-plus-LightGBM pipeline and does not retrain, retune, or modify the frozen model.

### Lineage and protocol

- Source frozen model run ID: `dc28826afdcc4e5cbce068db62439094`.
- Resolved MLflow model URI used for loading: `models:/m-3aacf46fc018452ab9aab96033eb7e56`.
- SHAP explanation run ID: `9a3d12ac156c48dda5a0a1e6a0b1d45f`.
- Explanation sample: 5,000 customers.
- Sample response prevalence: 12.2600%; full-data prevalence: 12.2563%.
- Transformed feature count: 220.
- Maximum SHAP additivity error: 0.00000000.
- Outreach boundary analyzed: configurable top 10%, equal to 38,110 of 381,109 customers in this run.

The protocol includes transformed-feature global importance and beeswarm plots, aggregation of one-hot SHAP values back to raw business features, numeric-feature dependence plots, local waterfall and grouped-contribution plots for customers around the outreach boundary, and MLflow logging of the plots, tables, metadata, and validation checks.

### Global findings and interpretation

Grouped mean absolute SHAP contribution was:

| Raw feature | Mean absolute SHAP contribution | Interpretation |
|---|---:|---|
| `Previously_Insured` | 2.350207 | Dominant ranking signal. Value `0` generally raises the response score and value `1` lowers it. |
| `Vehicle_Damage` | 0.968988 | `Yes` generally raises the score and `No` lowers it. |
| `Policy_Sales_Channel` | 0.416180 | Channel levels have materially different learned response patterns. |
| `Region_Code` | 0.329538 | Region levels contribute segment-specific response patterns. |
| `Age` | 0.326053 | Moderate nonlinear effect, generally positive in the mid-30s to early-40s and increasingly negative at older ages. |
| `Annual_Premium` | 0.068790 | Modest, dispersed, non-monotonic contribution. |
| `Vehicle_Age` | 0.054724 | Small contribution relative to the leading features. |
| `Vintage` | 0.052964 | Low and noisy contribution. |
| `Gender` | 0.050113 | Low global contribution, but still requires segment-level fairness review. |
| `Driving_License` | 0.003009 | Nearly negligible contribution in the frozen model. |

The strongest conclusion is that the model's ranking behavior is concentrated in `Previously_Insured`, `Vehicle_Damage`, age, and selected channel/region levels. This is consistent with the earlier ablation and permutation-importance experiments and explains why changing model families did not create a substantial performance improvement. The current data contains limited independent signal beyond these variables.

`Policy_Sales_Channel` and `Region_Code` must continue to be treated as categorical codes. A single encoded category is interpreted relative to the one-hot reference category; individual category effects should not be described as universally good or bad. Their grouped importance indicates that the broader channel and region patterns matter.

The Age dependence plot is inverted-U shaped rather than monotonic. Annual Premium and Vintage are weaker and non-monotonic, so the model does not apply a simple rule such as “higher premium means higher response.”

### Boundary-customer findings

The last selected customer at rank 38,110 had a score of approximately `0.7920018434`. The first unselected customer at rank 38,111 had a score of approximately `0.7920017617`. The difference is approximately `0.0000000817`, which is operationally indistinguishable at the capacity boundary.

The local explanations showed that boundary customers were mainly elevated by `Previously_Insured = 0`, vehicle-damage status, age, and selected channel or region levels. Small negative contributions from Annual Premium, Vintage, Gender, or Vehicle Age determined the precise ordering among near-tied customers.

The broad top-10% group is therefore useful for outreach prioritization, but the exact identity of the final few customers is fragile. A stable sort makes the result reproducible; it does not mean that the last selected customer is meaningfully more promising than the first unselected customer.

### Limitations and operational use

SHAP values are in the model's raw output space, not percentage points. They explain how features move an individual score relative to the model's baseline. They are not causal explanations, treatment-effect estimates, or evidence that changing a customer feature would change the customer's response.

Before production-oriented use:

- verify that `Previously_Insured`, Vehicle Damage, Age, Region Code, and Sales Channel are available at score-generation time and are not derived from the response;
- present aggregated raw-feature reason codes rather than exposing one-hot implementation details;
- define a deterministic tie policy and consider a review buffer around the top-10% cutoff;
- perform selection-rate, performance, and calibration checks across gender, age, region, and channel segments;
- calibrate the output only if the product presents it as a probability rather than as a ranking score;
- monitor feature drift, score drift, and delayed campaign outcomes after deployment.

The SHAP results do not justify changing the frozen model or rerunning the untouched test evaluation. They document the behavior of the approved model. Future causal improvement requires campaign treatment/control data and potentially an uplift or treatment-effect model.
