# Engineering Review — Telecom Customer Churn Prediction System

## Scope

This review covers the repository Python source (`app.py`), the Jupyter notebook, `README.md`, dependency declarations, serialized artifacts, and repository hygiene. Changes were made only under this repository. No GitHub push was performed.

## Repository snapshot and constraints

- Repository: `Abdulrahman181/Telecom-Customer-Churn-Prediction-System`
- Local path: `/home/ubuntu/github_priority/Telecom-Customer-Churn-Prediction-System`
- The IBM Telco CSV is **not present** in the checkout.
- The audit environment does not have the complete heavy ML runtime installed (`scikit-learn`/`imbalanced-learn`, CatBoost, LightGBM, XGBoost, Optuna, SHAP, and Streamlit were not available for a full inference/training run).
- Consequently, this review does **not** claim a completed retraining run, metric reproduction, artifact compatibility, or end-to-end Streamlit execution.
- Existing `artifacts/*.pkl` files were preserved rather than replaced with invented results. They are legacy outputs whose training data snapshot and dependency versions are not recorded in the repository.

## Findings and remediation

### Critical/high priority findings fixed in source

1. **Non-portable notebook input path and missing-data behavior**
   - Original notebook hard-coded `/content/churn_prediction.csv`, which only works in a particular hosted notebook environment.
   - It now reads `CHURN_DATA_PATH`, defaults to `data/churn_prediction.csv`, and raises a clear `FileNotFoundError` explaining how to provide the dataset. The dataset is not fabricated or bundled.

2. **Notebook contained executable IPython install/magic syntax**
   - `!pip install ...` and `%matplotlib inline` made ordinary AST/static validation fail and coupled execution to an interactive environment.
   - Package installation is now documented through `requirements.txt`; the notebook contains standard Python only.

3. **Evaluation leakage in threshold selection**
   - The original notebook optimized the F2 threshold directly on `X_test/y_test`, then reported that same test performance.
   - The repaired notebook creates stratified train/validation/test partitions, selects the threshold on validation predictions, and reports the selected threshold on the untouched test split. The business curve is also based on validation data.

4. **Pre-split imputation leakage**
   - The original notebook filled missing `TotalCharges` using the full dataset median before splitting.
   - Cleaning now coerces the column but leaves missing values for the training-only `SimpleImputer` inside the pipeline.

5. **Fragile deployment paths and weak artifact checks**
   - `app.py` loaded artifacts relative to the process working directory and assumed a threshold was present.
   - It now resolves artifacts relative to `app.py`, checks artifact existence and threshold range, validates all required input columns, coerces numeric fields, and gives actionable errors. Prediction also verifies that class `1` is available rather than assuming probability-column order.

6. **Incomplete dependency declaration**
   - The notebook imported `imbalanced-learn`, Optuna, SHAP, plotting libraries, and IPython without declaring them.
   - `requirements.txt` now covers notebook, training, explainability, and runtime imports with minimum versions.

### Additional hygiene and maintainability changes

- Added `.gitignore` for Python caches, virtual environments, local data, and generated files.
- Added `tests/test_static_validation.py`, a dependency-free test that parses `app.py`, parses every notebook code cell, rejects shell/magic commands, and verifies stale notebook outputs are absent.
- Cleared all notebook output cells and execution counts so checked-in results cannot be mistaken for a fresh run.
- Updated the README with setup instructions, the `CHURN_DATA_PATH` contract, the validation/test split policy, and explicit caveats around historical benchmark claims and legacy artifacts.
- Kept the existing image files and serialized artifacts untouched because regenerating them without the real dataset would be invalid.

## Data leakage and reproducibility assessment

- Raw-row feature engineering is deterministic and does not use the target. Numeric missing-value imputation is now fitted inside the model pipeline.
- The split is stratified and uses `random_state=42`; cross-validation also uses a shuffled, seeded `StratifiedKFold`.
- Threshold selection is separated from final test evaluation in the notebook source.
- Feature selection remains an exploratory analysis block and is not silently applied to the final model pipeline; this avoids claiming that an unused selector controls production inference.
- The project still needs a dataset version/hash and a fully locked environment for byte-for-byte artifact reproducibility. Those cannot be invented during an audit.

## Validation performed

Successful checks in this environment:

- `python3 tests/test_static_validation.py`
- `ruff check app.py tests/test_static_validation.py`
- Ruff syntax/undefined-name checks on a sanitized concatenation of all notebook code cells (`--select E9,F821,F822,F823`)
- Python AST parsing for all Python files and every notebook code cell
- Notebook JSON parsing; 51 code cells; 0 non-empty output cells
- `git diff --check`

Not performed, by design/constraint:

- Full notebook execution or model retraining: dataset absent and heavy dependencies unavailable.
- Loading/predicting with the checked-in pickle artifacts: required sklearn/imbalanced-learn runtime is unavailable, and silently installing/retraining was outside the evidence available for this review.
- Streamlit UI smoke test: Streamlit is not installed in the audit environment.

## Changed files

- `app.py`
- `Telecom Customer Churn Prediction System.ipynb.ipynb`
- `README.md`
- `requirements.txt`
- `.gitignore`
- `tests/test_static_validation.py`
- `ENGINEERING_REVIEW.md`

## Next step for an operator

Install `requirements.txt`, obtain the IBM Telco CSV independently, set `CHURN_DATA_PATH`, run the notebook from the repository root, inspect validation and untouched-test metrics, and only then decide whether to replace the legacy artifacts. Record the dataset hash and resolved dependency versions with that run.
