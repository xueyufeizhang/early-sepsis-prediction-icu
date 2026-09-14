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
results/figures, and presentation slides. Local study notes (`docs/`), planning
notes, personal certificates, credentials, research papers, and the private
reference library are intentionally excluded from version control.

## Repository layout

```
requirements.txt               Python dependencies
sql/                           Sepsis-3 SQL concept (adapt from MIT-LCP/mimic-code)
src/
  config.py                    Locked parameters (N, M, seeds, feature/model lists)
  cohort.py                    Stage 1: cohort extraction + N/M windowing + labels
  features.py                  Stage 2: static + hourly time-series matrices
  splits.py                    Stage 3: patient-grouped split + SMOTE
  models/                      Stage 4: static orchestration (classic.py), Torch MLP (mlp.py), LSTM (lstm.py)
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

Stages 0–2 are complete. The leakage-safe MIMIC-IV cohort and both Stage-2
representations have been generated in the controlled environment: a static
patient matrix for tabular models and a six-hour value/mask tensor for the LSTM.
No patient-level artifact is stored in this repository.

Stage 3 has passed its controlled run: 28,820 patients were divided into a
23,056-row development set and a sealed 5,764-row internal test set, with five
frozen patient-grouped development folds. Stage 4 trains LR, XGBoost, RF, SVM and
PyTorch MLP through one shared workflow. SVM uses grouped probability
calibration; MLP uses grouped early stopping. The hourly LSTM is in progress.
Internal-test prediction and F1 threshold selection belong to Stage 5.

Run `notebooks/04_models.ipynb` beside the protected Stage-2/3 artifacts. Start
with `profile="smoke"`; use `"screening"` for LR/XGBoost imbalance comparisons or
`"tuning"` for model optimization. RF is available through
`train_random_forest` in `src/models/classic.py`; it has no notebook section yet.

Static models share `build_static_resampling_pipeline` in `src/splits.py`.
Imputation and numeric scaling are fitted on original training-fold rows;
SMOTENC runs next, followed by categorical one-hot encoding. This scaling is
required for SMOTENC even with tree models. The old Stage-3-only preprocessing
helpers have been removed; its notebook now uses the same preprocessing steps.

The code follows the notebook stages directly. `train_static_model` contains the
candidate/fold loop and result summaries. Extra synthetic-sample audits and
unused `run_*_stage4` wrappers have been removed.

For recovery, pass a private `checkpoint_dir`. A small manifest checks data,
splits and model settings. Use one experiment per directory, and choose a new
directory after changing code or dependencies. The notebook uses fresh suffixes
for this version; old checkpoints are not compatible. Existing Stage-2 features
and Stage-3 splits can be reused. Keep model, OOF and checkpoint files inside
the controlled environment.

Synthetic checks (no clinical data):

```bash
uv run python -m unittest discover -s tests
```
