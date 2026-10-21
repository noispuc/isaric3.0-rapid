# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment & Installation

The package is installed in editable mode in the `rapid` conda environment:

```bash
conda activate rapid
# already installed as: pip install -e .
```

Run tests (no test suite exists yet — create files under `tests/`):

```bash
conda run -n rapid pytest tests/ -v
conda run -n rapid pytest tests/test_predictive.py -v   # single file
conda run -n rapid pytest tests/ -k "class_weight" -v   # by keyword
```

Import check:

```bash
conda run -n rapid python -c "import isaric; print(isaric.__version__)"
```

## Architecture

### Abstract base + factory pattern

`src/isaric/rapid.py` defines `RAPID` (ABC). All model classes inherit from it. The entry point is always `RAPID.create(data, model="...", **params)`, which dispatches via an internal registry to the concrete subclass, calls `subclass.create(...)`, and returns a ready-to-fit instance.

State machine enforced by `_check_state`: `created → fitted → summarized → reported`. Calling `fit()` before `create()` or `summary()` before `fit()` raises `ValueError`.

### Subclass structure

Each subclass lives in `src/isaric/modeling/<module>.py` and follows this layout:

1. **Module-level factory functions** (`create_<model>_model(data, dependent_var, independent_vars, ...)`) — configure the sklearn/statsmodels model object and split X/y. These are where hyperparameter defaults live (including the current `class_weight='balanced'` issue).
2. **`create(cls, data, model, **params)`** — calls the factory, builds the instance, sets `self._model`, `self.X`, `self.y`.
3. **`_train_model(self, grid_search, param_grid, selection_metric)`** — called by the inherited `fit()`. Must end with `return self._model.fit(self.X, self.y)`.
4. **`_calculate_metrics(self)`** — evaluates on `self.X` (training data) at threshold 0.5. Known issue.
5. **`_build_result_df(self)`** — builds `self.result_df` (columns `Feature`, `Coefficient`).

Modules:
- `modeling/predictive.py` — Lasso, Ridge, ElasticNet, SVM, LogisticL2 (sklearn `LogisticRegression` / `SVC`)
- `modeling/treebased.py` — DecisionTree, RandomForest, XGBoost, LightGBM, CatBoost
- `modeling/regression.py` — LogisticRegression, GLM (statsmodels)
- `modeling/survival.py` — SurvivalCox, KaplanMeier (lifelines)
- `modeling/clustering.py` — LCA (stepmix), KMeans

### Evaluation & validation

- `modelevaluation/metrics.py` — `compute_classification_metrics`, `select_classification_threshold`
- `modelevaluation/bootstrap.py` — `bootstrap_metrics`, `bootstrap_coefficients`, `confidence_interval` (added Oct 2025)
- `modelevaluation/calibration.py` — `calibration_curve`
- `validation/netprofit.py` — `decision_curve_analysis`

## Current Work: branch `revarch_predictive`

**Scope:** only modify `modeling/predictive.py`, `modeling/treebased.py`, `modelevaluation/`, and `rapid.py` when strictly necessary. Do not touch `modeling/regression.py`, `modeling/survival.py`, `modeling/clustering.py`.

**Objective:** fix the open issues in `RAPID_issues.md` so that the dengue paper (Delpino et al.) can be reproduced using only `RAPID.create → fit → summary`, without accessing private attributes or implementing workarounds outside the package.

**Reference metrics (test set, logistic elastic-net):**

| Metric | Paper |
|---|---|
| AUC | 0.866 (0.860–0.871) |
| Calibration slope | 1.006 |
| Sensitivity | 0.359 |
| Specificity | 0.975 |
| PPV | 0.316 |
| Threshold | 0.155 |

**Rules:**
- No data leakage: preprocessing transformers must be fit on train only.
- No hidden sklearn defaults: any hyperparameter that changes model behavior (e.g. `class_weight`) must be explicit in `create()`.
- One commit per issue: `fix(<scope>): <description> (issue N)`.
- pytest for every fix before committing.

**Open issues (see `RAPID_issues.md` for detail):**

| # | Description | Status |
|---|---|---|
| 3 | `class_weight='balanced'` hardcoded in all sklearn classifiers | open |
| 4 | `_calculate_metrics` evaluates on training data at threshold 0.5 | open |
| 5 | Undersampling/SMOTE, logistic recalibration not in `fit()` pipeline; bootstrap of test metrics not implemented (bootstrap of coefficients added Oct 2025) | partially open |
| 6 | `Preprocess`/`onehot_encode`/`standardize` transform the whole dataframe without fit/transform separation | open |
| 7 | `select_classification_threshold` is O(n²); `summary(table_format="short")` breaks on predictive `result_df`; `import isaric` fails without catboost | **FIXED** (64750bb, 6a27e10, af7bef3) |
