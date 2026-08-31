# Early Sepsis Prediction in ICU Patients

Course project for **Artificial Intelligence in Medicine (01VRUOV)**, Politecnico di Torino.

**Task.** Given a patient's structured clinical data from the first **N = 6 hours**
after ICU admission, predict whether they will meet **Sepsis-3 criteria** within
the following **M = 24 hours**. Binary classification (sepsis vs. no sepsis).

**Approach.** Develop on **MIMIC-IV**; externally validate on **eICU**
(cross-institutional) and **MIMIC-III** (temporal). Compare six optimized models
spanning classic ML and deep learning, with a comprehensive evaluation suite,
SHAP interpretability, and a demographic bias audit. Method adapts and extends
Nguyen & Mittal (2026).

**Deliverables.** Code + **presentation slides only** (no written report). The
graded artifact is a 15–20 min exam presentation built on the professor's
template (`slides/Presentation template.pptx`): Context → Objective → Data →
Workflow → Methods → Results → Discussion → Conclusion, with an Appendix for
details and anticipated Q&A.

## ⚠️ Compliance — read before touching data

Per the PhysioNet Data Use Agreement, **MIMIC/eICU patient-level data and any
derivatives must never be sent to, processed by, or retained by third-party APIs
or LLM services**, and must not be committed to version control. All extraction,
processing, and modeling happen in a controlled environment (Kaggle Notebook or
Colab, database via PhysioNet's official BigQuery channel). The `.gitignore`
enforces the no-commit rule — do not weaken it.

## Repository scope

This public repository contains only project deliverables and the material
needed to reproduce them: source code, SQL concepts, notebooks, safe aggregate
results/figures, and presentation slides. Local planning notes, personal
certificates, credentials, research papers, and the private reference library
are intentionally excluded from version control.

## Repository layout

```
requirements.txt               Python dependencies
sql/                           Sepsis-3 SQL concept (adapt from MIT-LCP/mimic-code)
src/
  config.py                    Locked parameters (N, M, seeds, feature/model lists)
  cohort.py                    Stage 1: cohort extraction + N/M windowing + labels
  features.py                  Stage 2: static + hourly time-series matrices
  splits.py                    Stage 3: patient-grouped split + SMOTE
  models/                      Stage 4: classic ML (classic.py) + LSTM (deep.py)
  evaluate.py                  Stage 5: metrics, ROC/PR/calibration, stat tests
  external.py                  Stage 6: eICU + MIMIC-III validation
  interpret.py                 Stage 7: SHAP + subgroup/bias audit
notebooks/                     Kaggle/Colab notebooks (one per stage)
data/{raw,interim,processed,external}/   Local data — gitignored
results/{figures,tables,models}/         Outputs (model binaries gitignored)
slides/                        Presentation deck + professor's template (.pptx)
```

## Models (brief §6)

| Model | Input | Family |
|---|---|---|
| Logistic Regression | static | classic ML |
| SVM | static | classic ML |
| Random Forest | static | classic ML (ensemble) |
| XGBoost | static | classic ML (ensemble) |
| MLP | static | deep learning |
| **LSTM** | hourly time-series | deep learning (sequence) |

## Setup

**Local development (uv).** `pyproject.toml` + `uv.lock` are the source of truth:

```bash
uv sync                       # create .venv and install all deps (incl. dev tools)
uv run python -c "import torch; print(torch.__version__)"   # sanity check
uv run jupyter lab            # launch notebooks
```

**Kaggle / Colab.** These use pip; install the mirrored top-level deps:

```bash
pip install -r requirements.txt
```

Most dependencies are pre-installed on Kaggle/Colab. Configure PhysioNet BigQuery
access with your own credentialed account (do not commit credentials).

## Pipeline status

**Stage 0 is complete; Stage 1 is ready to run.** The seven task decisions are locked
in `src/config.py` (including N=6h and M=24h), and the validated MIT-LCP
`sepsis3.sql` plus its `suspicion_of_infection.sql` dependency are present in
`sql/` and have been reviewed. The parameterized MIMIC-IV cohort/CONSORT SQL,
Python orchestration, leakage checks, synthetic boundary tests, and a clean
execution notebook are implemented.

The remaining Stage-1 step is to run `notebooks/01_cohort.ipynb` in the user's
credentialed BigQuery environment and review the protected audit sample plus the
aggregate CONSORT counts. No real cohort extract, feature matrix, trained model,
evaluation result, or completed presentation exists yet.
