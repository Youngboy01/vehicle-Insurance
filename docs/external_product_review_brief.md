# Vehicle-insurance campaign prioritization: external review brief

Status: 22 October 2026. Code baseline: `project/issue_resolutions` at `f15df82` before this brief was added. This is a factual orientation and a request for independent judgment, not a claim that the current system is production-ready.

## The decision we want help making

We want to turn a technically functional insurance cross-sell benchmark into something a real user would choose to use. Please challenge the product premise and the amount of engineering already built. No insurer is onboarded, and no customer-discovery or measured business-impact evidence is recorded in this repository. The intended user so far is a campaign-operations or marketing team, but that is a hypothesis rather than an observed requirement.

The current narrow decision is: before a vehicle-insurance campaign, rank eligible customer records by predicted positive response to an offer, then let a human choose the highest-scoring customers within a campaign capacity. The system does **not** choose outreach channels, check consent, execute campaigns, or measure incremental response. A lower-ranked person is not denied insurance or judged ineligible.

We would particularly value an answer to this question: **what is the smallest credible product and evidence path that would make this useful to a real campaign team, rather than merely demonstrating a model and infrastructure?** If the answer is to narrow, redirect, or delete substantial parts of the current design, please say so.

## What exists today

| Area | Current implementation | Boundary to keep in mind |
|---|---|---|
| Data source | The Kaggle insurance cross-sell benchmark was loaded into MongoDB Atlas; training reads a snapshot-session extraction, writes a CSV and source manifest, then makes stratified 80/10/10 splits. | There is no partner production feed. The repository copy at `notebooks/data.csv` is not proven byte-identical to MongoDB or to a particular Kaggle download. |
| Data contract | Ten fixed predictive fields, a customer ID and a training-only `Response` label; Pandera-based checks and strict scoring validation. | No external-company schema mapping is active. An arbitrary customer CSV will not work merely because upload support is added. |
| Model | Class-weighted LightGBM in a self-contained scikit-learn preprocessing/model pipeline. Categorical fields are one-hot encoded; five numeric/binary fields pass through. | One benchmark model, not a portable pretrained model for another insurer. Scores are uncalibrated ranking signals, not reliable purchase probabilities. |
| Training | Train-only candidate for validation and repeated stability checks; final pipeline refit on train plus validation; untouched test split evaluated once. Git, dependency and MongoDB-source lineage are recorded. | This is an offline, manually invoked pipeline, not an automated retraining service. A successful process exit is not release approval. |
| Evaluation | Validation top-K metrics, supporting discrimination metrics, stability, provenance and source-frozen parity feed provisional candidate gates. Champion/Challenger comparison and promotion are disabled. | Thresholds are not approved business acceptance criteria. Source-frozen parity proves fidelity to this historical benchmark workflow, not fitness for new data or users. |
| Serving | FastAPI `/predict` and `/rank`, readiness/liveness checks, and token-guarded `/reload`; a small same-origin dashboard. The API loads an explicitly pinned MLflow model ID/URI. Ranking accepts a percentage or count, sorts by score with ID tie-breaking, and caps a batch at 10,000 records. | No campaign management, consent workflow, customer-data onboarding, multi-tenant service, public deployment or fleet-wide model rollout. Readiness means a structurally loadable model, not business approval. |
| Operations | Dockerfile and `.dockerignore` exist; local offline tests are tracked and run from a clean checkout. | No GitHub Actions workflow, verified production deployment, rollback exercise, live monitoring, or end-to-end service test against a real model store in the default suite. |

The API pins run `0a5a3eb264464111ad86075bd811531d` / model `m-e339c348e27d400d835b8e8bdf8fdbe1` in `configs/model_spec.yaml`. It is a candidate explicitly configured for serving, **not** a registered/promoted production Champion. `/reload` reloads that configured identity within one process; it does not promote a candidate or update other processes.

### Data and evaluation evidence

- User-identified public source: [Kaggle — Health Insurance Cross Sell Prediction](https://www.kaggle.com/datasets/anmolkumar/health-insurance-cross-sell-prediction/data). Its data card describes `Response` as customer **interest** in vehicle insurance and describes the features and source context. It does not establish actual purchases, outreach treatment/control, or causal campaign lift. The Kaggle listing currently displays a GPL 2 license; upstream origin, our acquisition/version record, redistribution rights, and suitability for any commercial use still need explicit review. Do not infer those rights from this brief.
- The documented benchmark has 381,109 labeled rows and 46,710 positive responses (about 12.26%). The active training source is the MongoDB copy, not the repository CSV. The source manifest records the exact extraction boundary and hashes for a given run; it does not prove that a future insurer's fields mean the same thing.
- The model card reports validation Precision@10% `0.396484`, Lift@10% `3.234938`, PR-AUC `0.370007`; final-test Precision@10% `0.396221`, Lift@10% `3.232797`, PR-AUC `0.373937`. These are random-split benchmark results, not temporal or external-cohort evidence. The final test is reporting-only; selecting product policies or new models from it would invalidate its untouched status.
- The data contract lists `Gender`, `Age`, `Driving_License`, `Region_Code`, `Previously_Insured`, `Vehicle_Age`, `Vehicle_Damage`, `Annual_Premium`, `Policy_Sales_Channel`, and `Vintage` as the ten predictive fields. `id` is a traceability/tie-break field, not a model feature. For an actual organization, field semantics, availability before outreach, missingness, permissible use and category mappings all require owner review.

### Evidence of software reproducibility—and its limit

The published `new/tests-updates` and `project/issue_resolutions` branches both reached `f15df82`. A clean Windows clone at that commit, with no `.env` and a fresh `uv sync --locked` environment, reported **108 passed, 1 skipped**. The skip was a symbolic-link test requiring a Windows privilege (`WinError 1314`). The default suite uses synthetic/local fixtures and does not run live MongoDB extraction, MLflow training, campaign delivery, or a production-like deployment. Earlier verification also passed the committed baseline when the tracked benchmark CSV was temporarily hidden; the default tests do not need that CSV. No latency SLO follows from these test results.

## Questions and risks we do not want to hide

1. **Product discovery:** Who has this decision, how often, with what capacity, constraints, budget, and existing workflow? What improvement would make them adopt a tool? No interviews, pilot, willingness-to-pay evidence, or observed campaign process are documented here.
2. **Label and decision validity:** Kaggle `Response` is interest, not necessarily purchase or incremental effect of contact. We have no treatment/control or delayed operational labels. A random stratified split may overstate performance for future campaigns; timing, repeated customers, contact history and selection bias need review before a live claim.
3. **Company-data portability:** No partner is onboarded, and the current model is tied to fixed benchmark semantics. An external CSV can be mapped only with explicit definitions, units, categories, as-of timing, missingness and leakage review. A training UI or free-form feature addition does not solve that prerequisite.
4. **Campaign policy and safety:** Consent, suppression lists, contact frequency, channel availability, protected attributes (including `Gender`), and human approval are outside the current software. Segment analysis is incomplete. The legal/organizational basis for using benchmark or partner data must be established before deployment.
5. **Release truth:** The service can load a pinned candidate that is structurally valid even though automatic promotion is disabled; readiness is not an acceptance certificate. The thresholds are provisional. We need a deliberate approved-model selection and recovery procedure before using output operationally.
6. **Known correctness/documentation gaps:** The trainer currently derives top-K from a configurable percentage but emits metric names fixed as `*_at_10pct`; a non-10 setting could mislabel evidence. Serving checks two `predict_proba` columns and value bounds but not that the number of returned rows equals the number scored. Public docs have historical/stale passages and still need precise Kaggle attribution and current data/license statements. These are targeted fixes, not reasons to build a new platform.
7. **Operational boundary:** The Dockerfile exists but has not been verified as a secure deployed image. `.dockerignore` excludes `.env` and some generated assets, but should be reviewed for private notes and other unintended build-context files. No CI or production monitoring exists. `/reload` is process-local; external exposure would need TLS, network access control, rate limiting, and a tested rollback path.
8. **Potential overengineering:** MongoDB snapshot lineage, Git provenance, parity certificates, replay evaluation and numerous destination guards were built for reproducibility and auditability. Which of these are essential for the first real user, and which impose maintenance cost without improving a decision? In particular, parity to one historical notebook is not a general-purpose quality gate for future partner models.

These are review hypotheses or visible current gaps, not a claim that every item needs immediate code. Please distinguish a demonstrable bug, a missing product prerequisite, an operational requirement conditional on deployment, and an optional refinement.

## Decisions requested from the reviewer

Please be independent of our current roadmap. We would like:

1. A sharply defined first user and job-to-be-done, or a reason to reject this product direction; compare at least two feasible product shapes (for example, a narrow campaign prioritization workflow versus an offline evaluation/planning tool) and the data access each requires.
2. A first useful workflow from data arrival to human action and feedback. State what the tool would do, what a human must approve, and what it must explicitly **not** promise. Identify the minimum dataset/contract and whether the existing frozen model is relevant outside this benchmark.
3. An evidence plan for usefulness: baseline comparison, out-of-time or external validation, selection quality at actual capacities, segment review, delayed outcome measurement, and whether a randomized outreach/control design is necessary for any incremental-value claim. Propose concrete go/no-go criteria, not just more model metrics.
4. A ruthless engineering plan: what to keep, simplify, postpone or remove in the current ingestion/training/evaluation/serving architecture; the few code defects or design choices that most threaten a pilot; and the minimum justified MLOps/deployment controls. Please resist adding generic AutoML, multi-tenancy, monitoring stacks or retraining automation until a user and data contract justify them.
5. A short phased roadmap with dependencies, approximate effort and explicit decision points. Separate work possible with only this public benchmark from work that requires a real partner, lawful data access or a deployment target. Tell us what additional evidence would change your recommendation.

We welcome corrections to any assumption or metric interpretation in this brief. Please cite the source file or experiment behind a technical finding and identify where a claim cannot be verified from the repository.

## Suggested reading route in the repository

Start here, then inspect implementation only where it bears on your recommendation:

1. `README.md` — current public positioning and API; some passages predate the latest clean-checkout run.
2. `docs/model_card.md` and `docs/data_contract.md` — intended use, metrics, target meaning, fixed schema, mapping policy and limitations. There is no separate finished data card yet.
3. `configs/model_spec.yaml` and `configs/schema.yaml` — actual pinned candidate, provisional gates and input contract.
4. `app.py`, `templates/index.html` — real user-facing scoring/ranking flow, readiness and reload behavior.
5. `src/vehicle_insurance/pipeline/training_pipeline.py` and `src/vehicle_insurance/components/` — stage orchestration, fit/evaluation sequence and data boundaries.
6. `docs/mongodb_provenance.md`, `docs/source_frozen_parity.md`, `docs/git_lineage.md`, `docs/model_evaluation_replay.md` — why the lineage/parity/replay machinery exists; decide whether it remains proportionate.
7. `tests/` and `Dockerfile` / `.dockerignore` — offline behavior checks and deployment assumptions.

The author's local-only historical logs/audits are long and sometimes stale; they are not needed to start this review and are not necessarily present in a Git clone. If shared separately, treat them as chronology, not authority over current code/configuration. Do not use `.env`, credentials, raw customer records or private artifacts in a review response.
