# RAPID v0.2.0 — Latent Class Analysis (LCA) Technical Validation & Before/After Evidence

**Package:** ISARIC RAPID v0.2.0 (`isaric.modeling.clustering.LCA`)  
**Feature Branch:** `feat/LCA-fixes`  
**Verified Pre-LCA Baseline Commit:** `22b1a6d` (`22b1a6d28b44ee0676ab2a99cb3b06e0aa8af2c0` — `fix: LCA default interations constant identation.`)  
**Scope:** Requirements `LCA-01` through `LCA-08` and `GEN-03 (LCA)`

---

## 1. Executive Summary & Separation of Concerns

This document consolidates the reproducible before/after validation, numerical verification, automated test coverage, and known limitations for the Latent Class Analysis (`LCA`) module in RAPID v0.2.0.

### 1.1 Distinction Between Functional Changes and Statistical Estimation
All changes on `feat/LCA-fixes` were designed to improve **diagnostic reporting, visualization clarity, exploratory inspection across fitted class solutions, and formal decision persistence** without altering the underlying StepMix Expectation-Maximization (`EM`) estimation or likelihood ratio test mathematics:

- **Statistical Estimation (Unchanged):**
  - Core StepMix EM model fitting (`measurement="bernoulli"`), log-likelihood (`LL`), parameter counts (`n_par`), degrees of freedom (`dof`), Information Criteria (`AIC`, `BIC`, `CAIC`, `SABIC`), absolute `entropy`, `relative_entropy`, posterior classification diagnostics (`ave_pp_min`, `occ_k_min`, `min_class_perc`), Nested Likelihood Ratio Test statistics (`lrt_statistic`, `lrt_dof_diff`, `lrt_p_value`), Bootstrap LRT (`blrt_p_value`), class mixing proportions (`weights_`), and item-response probabilities (`k x p` class profiles) are **numerically identical** to the pre-LCA baseline commit `22b1a6d` within `0.000000e+00` (`tolerance = 1e-6`).
- **Functionality, Diagnostics & Workflow Integration (Updated):**
  - **LCA-01:** Integrated Nested LRT (`lrt_statistic`, `lrt_dof_diff`, `lrt_p_value`) and BLRT (`blrt_p_value`) columns directly into `grid_results` indexed by `n_clusters`, removing raw LRT dictionary dumps from `model_info` and preventing duplicate Nested LRT tables in PDF reports.
  - **LCA-02:** Corrected `convergence_rate` in `grid_results` so it is `None` when `bootstrap=False`, populated from `bootstrap_stats["stats"]["convergence_rate"]` when `bootstrap=True`, and omitted from `summary()` and PDF `report()` tables when all values are missing or `1.0`.
  - **LCA-03:** Enabled automatic rendering of `_default_plots` (including `radar_plot`) when `model.summary()` is called without an explicit `plots=` argument, while preserving explicit plot suppression (`summary(plots=[])`) in subsequent `report()` calls.
  - **LCA-04:** Replaced the single-axis `bootstrap_metrics_ci` plot with two separate figures: `bootstrap_metrics_ic` (`AIC`, `BIC`, `CAIC`, `SABIC` with 95% CI bands) and a two-panel `bootstrap_metrics_entropy` (`entropy` and `relative_entropy` on independent Y-axes).
  - **LCA-05:** Extended `ci_bootstrap` and bootstrap plots to include `k=1` for `AIC`, `BIC`, `CAIC`, `SABIC`, and `entropy` (`relative_entropy` remains undefined at `k=1` and is explicitly annotated).
  - **GEN-03 (LCA):** Implemented `LCA._summary_model_specific()` with 6 structured diagnostic sections (Model Metadata, Binary Indicator Frequency Table, Class Enumeration Results, Convergence Diagnostics, Reference/Exploratory Model Results with per-class `Prevalence`, `Est_N`, `AvePP`, `OCC`, and Outcome Crosstab) supporting `table_format="full"` and `"short"`.
  - **LCA-06 & LCA-07:** Added `model.summary(k=N)` to explore any previously fitted `k` solution in `model_objects` without retraining or mutating the reference model (`self.fitted_model`), and added validation rejecting ambiguous `summary(compare=[...], k=N)` calls.
  - **LCA-08:** Integrated formal model selection (`model.decide(selected=True, k=N, reason="...")`) with `model.report(format=["csv", "png", "pdf"])` and `model.save()` via `LCA._resolve_report_solution()` and `LCA._render_report_plot()`, plus automatic invalidation of stale decisions on re-fit.

### 1.2 Data Governance & Synthetic Benchmark Disclosure
- **No Patient-Level Clinical Data Executed or Committed:** All automated tests and before/after comparisons reported in this document were executed on the deterministic synthetic dataset (`generate_synthetic_lca_data(n_samples=150, seed=42)`).
- **Original Colab Notebook Preserved Unmodified:** The original exploratory notebook (`RAPIDv0_2_0_LCA_Clustering_sampleSINAN.ipynb`), which referenced an external SINAN clinical sample, was **not** executed, modified, or committed. A portable, synthetic-first validation notebook (`notebooks/validation/lca_development_local.ipynb`) is provided for local and CI verification.

---

## 2. Reproducible Benchmark Specification

### 2.1 Synthetic Dataset & Model Configuration

| Parameter | Value / Specification |
| :--- | :--- |
| **Generator Function** | `generate_synthetic_lca_data(n_samples=150, seed=42)` in `tests/lca/synthetic_data.py` |
| **Sample Size (`N`)** | `150` synthetic observations (`0` missing values) |
| **Binary Symptom Indicators (`p = 13`)** | `LCA_SYMPTOM_VARS`: `Fever`, `Myalgia`, `Headache`, `Rash`, `Vomiting`, `Nausea`, `Back pain`, `Conjunctivitis`, `Arthritis`, `Arthralgia`, `Petechiae`, `Tourniquet test`, `Retro-orbital pain` |
| **Simulated Ground-Truth Structure** | 3 latent profiles with mixing proportions `[0.50, 0.30, 0.20]` |
| **Binary Outcome Variable** | `severe` (`0/1`, simulated outcome probabilities `0.05`, `0.20`, `0.75` across latent classes) |
| **Random Seed** | `42` (`seed=42` for data generation; `random_seed=42` in `RAPID.create()`) |
| **Reference `n_clusters` at `create()`** | `n_clusters=4` |
| **Tested Class Enumeration Grid (`k`)** | `param_grid={"n_clusters": [1, 2, 3, 4]}` (plus non-consecutive `k = [1, 2, 4]` in unit tests) |
| **Training Scenarios Compared** | Scenario A: `bootstrap=False`<br>Scenario B: `bootstrap=True, n_iterations=6` |
| **Numerical Tolerance** | `1e-6` (`10^-6`) for floating-point metrics, weights, and probabilities; exact match for boolean flags |

### 2.2 Process-Isolated Before/After Comparison Methodology

The script `scripts/lca/run_before_after_comparison.py` compares the genuine pre-LCA baseline against the current working tree without mixing package versions in the same Python process:

1. Exports commit `22b1a6d28b44ee0676ab2a99cb3b06e0aa8af2c0` via `git archive` into an isolated temporary directory created with `tempfile.TemporaryDirectory(prefix="rapid_pre_lca_")`.
2. Launches **Subprocess 1 (`Before`)** with `sys.path[0]` pointing to `<baseline_checkout>/src` and asserts `isaric.__file__` resolves inside the baseline checkout.
3. Launches **Subprocess 2 (`After`)** with `sys.path[0]` pointing to the repository's `src/` directory and asserts `isaric.__file__` resolves inside the current working tree.
4. Compares all 77 metrics across `k = 1, 2, 3, 4` and writes the machine-readable audit table (`03_numerical_comparison.csv`).

---

## 3. Numerical Validation Results (`22b1a6d` vs `feat/LCA-fixes`)

Across all **77 evaluated metrics** for `k = 1, 2, 3, 4`:
- **68 `Pass`**: Exact numerical agreement within `1e-6` (observed difference `0.000000e+00`).
- **4 `Intentional Change`**: `convergence_rate` under `bootstrap=False` changed from `1.000000` (derived from single-fit `converged_`) to `None` per `LCA-02`.
- **5 `New Functionality`**: `k=1` LRT/BLRT columns (`None`) added to `grid_results` per `LCA-01`, and `k=1` rows added to `ci_bootstrap` (`[1, 2, 3, 4]` vs `[2, 3, 4]`) per `LCA-05`.
- **0 `Unexpected Regression`**.

### 3.1 Quantitative Comparison by `k` (`bootstrap=True, n_iterations=6, random_seed=42`)

| `k` | Metric | Before (`22b1a6d`) | After (`feat/LCA-fixes`) | Abs. Difference | Tolerance | Status |
| :-: | :--- | :---: | :---: | :---: | :---: | :---: |
| **1** | `LL` | `-1074.978215` | `-1074.978215` | `0.000000e+00` | `1e-06` | **Pass** |
| **1** | `AIC` / `BIC` | `2175.956431` / `2215.094690` | `2175.956431` / `2215.094690` | `0.000000e+00` | `1e-06` | **Pass** |
| **1** | `CAIC` / `SABIC` | `2228.094690` / `2239.090437` | `2228.094690` / `2239.090437` | `0.000000e+00` | `1e-06` | **Pass** |
| **1** | `entropy` / `relative_entropy` | `0.000000` / `None` | `0.000000` / `None` | `0.000000e+00` | `1e-06` | **Pass** |
| **1** | `ave_pp_min` / `occ_k_min` | `1.000000` / `None` | `1.000000` / `None` | `0.000000e+00` | `1e-06` | **Pass** |
| **1** | `convergence_rate` (`bootstrap=False`) | `1.000000` | `None` | `1.0 -> None` | By Design | **Intentional Change (`LCA-02`)** |
| **1** | `convergence_rate` (`bootstrap=True`) | `1.000000` | `1.000000` | `0.000000e+00` | `1e-06` | **Pass (`LCA-02`)** |
| **1** | `weights_` & `1x13` Item Probabilities | `[1.0]`, mean `0.402564` | `[1.0]`, mean `0.402564` | `0.000000e+00` | `1e-06` | **Pass** |
| **2** | `LL` | `-993.678813` | `-993.678813` | `0.000000e+00` | `1e-06` | **Pass** |
| **2** | `AIC` / `BIC` | `2041.357627` / `2122.644780` | `2041.357627` / `2122.644780` | `0.000000e+00` | `1e-06` | **Pass** |
| **2** | `CAIC` / `SABIC` | `2149.644780` / `2172.482101` | `2149.644780` / `2172.482101` | `0.000000e+00` | `1e-06` | **Pass** |
| **2** | `entropy` / `relative_entropy` | `23.382242` / `0.775110` | `23.382242` / `0.775110` | `0.000000e+00` | `1e-06` | **Pass** |
| **2** | `ave_pp_min` / `occ_k_min` | `0.940267` / `14.721326` | `0.940267` / `14.721326` | `0.000000e+00` | `1e-06` | **Pass** |
| **2** | `lrt_statistic` / `lrt_dof_diff` / `lrt_p_value` | `162.598804` / `14` / `2.13e-27` *(separate table)* | `162.598804` / `14.0` / `2.13e-27` *(in `grid_results`)* | `0.000000e+00` | `1e-06` | **Pass (`LCA-01`)** |
| **2** | `blrt_p_value` | `0.666667` *(in `model_info`)* | `0.666667` *(in `grid_results`)* | `3.333334e-11` | `1e-06` | **Pass (`LCA-01`)** |
| **2** | `weights_` & `2x13` Item Probabilities | `[0.475625, 0.524375]`, mean `0.408823` | `[0.475625, 0.524375]`, mean `0.408823` | `0.000000e+00` | `1e-06` | **Pass** |
| **3** | `LL` | `-958.649763` | `-958.649763` | `0.000000e+00` | `1e-06` | **Pass** |
| **3** | `AIC` / `BIC` | `1999.299525` / `2122.735572` | `1999.299525` / `2122.735572` | `0.000000e+00` | `1e-06` | **Pass** |
| **3** | `CAIC` / `SABIC` | `2163.735572` / `2198.414466` | `2163.735572` / `2198.414466` | `0.000000e+00` | `1e-06` | **Pass** |
| **3** | `entropy` / `relative_entropy` | `27.211433` / `0.834874` | `27.211433` / `0.834874` | `0.000000e+00` | `1e-06` | **Pass** |
| **3** | `ave_pp_min` / `occ_k_min` | `0.927999` / `12.575456` | `0.927999` / `12.575456` | `0.000000e+00` | `1e-06` | **Pass** |
| **3** | `lrt_statistic` / `lrt_dof_diff` / `lrt_p_value` | `70.058102` / `14` / `1.88e-09` *(separate table)* | `70.058102` / `14.0` / `1.88e-09` *(in `grid_results`)* | `0.000000e+00` | `1e-06` | **Pass (`LCA-01`)** |
| **3** | `blrt_p_value` | `0.666667` *(in `model_info`)* | `0.666667` *(in `grid_results`)* | `3.333334e-11` | `1e-06` | **Pass (`LCA-01`)** |
| **3** | `weights_` & `3x13` Item Probabilities | `[0.506151, 0.375599, 0.118250]`, mean `0.494485` | `[0.506151, 0.375599, 0.118250]`, mean `0.494485` | `0.000000e+00` | `1e-06` | **Pass** |
| **4** | `LL` | `-949.340247` | `-949.340247` | `0.000000e+00` | `1e-06` | **Pass** |
| **4** | `AIC` / `BIC` | `2008.680494` / `2174.265436` | `2008.680494` / `2174.265436` | `0.000000e+00` | `1e-06` | **Pass** |
| **4** | `CAIC` / `SABIC` | `2229.265436` / `2275.785904` | `2229.265436` / `2275.785904` | `0.000000e+00` | `1e-06` | **Pass** |
| **4** | `entropy` / `relative_entropy` | `40.653662` / `0.804497` | `40.653662` / `0.804497` | `0.000000e+00` | `1e-06` | **Pass** |
| **4** | `ave_pp_min` / `occ_k_min` | `0.854072` / `11.816340` | `0.854072` / `11.816340` | `0.000000e+00` | `1e-06` | **Pass** |
| **4** | `lrt_statistic` / `lrt_dof_diff` / `lrt_p_value` | `18.619031` / `14` / `0.180023` *(separate table)* | `18.619031` / `14.0` / `0.180023` *(in `grid_results`)* | `0.000000e+00` | `1e-06` | **Pass (`LCA-01`)** |
| **4** | `blrt_p_value` | `0.500000` *(in `model_info`)* | `0.500000` *(in `grid_results`)* | `0.000000e+00` | `1e-06` | **Pass (`LCA-01`)** |
| **4** | `weights_` & `4x13` Item Probabilities | `[0.115541, 0.153225, 0.435503, 0.295732]`, mean `0.476582` | `[0.115541, 0.153225, 0.435503, 0.295732]`, mean `0.476582` | `0.000000e+00` | `1e-06` | **Pass** |
| **1..4** | `ci_bootstrap` included `k` values | `[2, 3, 4]` | `[1, 2, 3, 4]` | Added `k=1` CIs | By Design | **New Functionality (`LCA-05`)** |

---

## 4. Functional Before/After Comparison Matrix

| Capability | Before (`22b1a6d`) | After (`feat/LCA-fixes`) | Requirement |
| :--- | :--- | :--- | :---: |
| **`grid_results` LRT & BLRT Integration** | 16 columns; `lrt_nested` and `lrt_blrt` stored as raw dictionary lists in `model_info` | 20 columns; `lrt_statistic`, `lrt_dof_diff`, `lrt_p_value`, and `blrt_p_value` aligned by `n_clusters` (`None` for lowest `k`, consecutive sorted `(k_prev, k_curr)` pairs thereafter) | **LCA-01** |
| **`convergence_rate` Semantics** | Populated as `1.0` from single-run `converged_` even when `bootstrap=False` | `None` when `bootstrap=False`; sourced from `bootstrap_stats` when `bootstrap=True`; omitted from `summary()` and PDF `report()` when all values are `None` or `1.0` | **LCA-02** |
| **Automatic Plots in `summary()`** | No plots rendered unless `plots=[...]` was passed; `radar_plot` absent from `_default_plots` | Automatically displays `_default_plots` (`information_criteria`, `ll_curve`, `entropy_curve`, `relative_entropy_curve`, `radar_plot`, plus `bootstrap_metrics_ic`, `bootstrap_metrics_entropy`, `selection_rate` when `bootstrap=True`); `summary(plots=[])` suppresses plots in both `summary()` and subsequent `report()` | **LCA-03** |
| **Bootstrap CI Visualizations** | Single `bootstrap_metrics_ci` figure combined IC (`~2000–2280`) and Entropy (`0–40`) on one Y-axis | Separated into `bootstrap_metrics_ic` (`AIC`, `BIC`, `CAIC`, `SABIC`) and two-panel `bootstrap_metrics_entropy` (`entropy` and `relative_entropy` on independent Y-axes) | **LCA-04** |
| **`k=1` in Bootstrap CI** | `k=1` omitted from `ci_bootstrap` and bootstrap plots | `k=1` included in `ci_bootstrap` and plotted for `AIC`, `BIC`, `CAIC`, `SABIC`, and `entropy` | **LCA-05** |
| **LCA-Specific `summary()` Diagnostics** | Generic `RAPID.summary()` output without indicator frequencies, convergence details, or AvePP/OCC tables | 6 structured LCA sections (`MODEL INFORMATION`, `BINARY INDICATOR FREQUENCY TABLE`, `GRID SEARCH RESULTS`, `CONVERGENCE DIAGNOSTICS`, `REFERENCE / EXPLORATORY MODEL RESULTS`, `OUTCOME BY CLASS`) with `table_format="full" \| "short"` | **GEN-03** |
| **Exploratory Inspection (`summary(k=N)`)** | Unsupported (`TypeError`) | `model.summary(k=N)` inspects any fitted `k` from `model_objects[k]` (metrics, prevalence, AvePP, OCC, profiles, outcome crosstab, and plots) without retraining or mutating `self.fitted_model` | **LCA-06 & LCA-07** |
| **Formal Selection (`decide(selected=True, k=N)`) in `report()` & `save()`** | `decide(k=3)` only recorded metadata; `report()` and `save()` still exported `Reference k=4` tables, plots, and 4-class `model.pkl` | `report(format=["csv", "png", "pdf"])` and `save()` export the `Selected k=3` profiles, class diagnostics, outcome crosstab, solution-specific plots (`radar_plot`, `profile_heatmap`, `cluster_distribution`, `crosstab`), and `model.pkl` (`n_components == 3`), while preserving full `grid_results` (`k=1..4`) and recording `reference_k=4, selected_k=3` | **LCA-08** |

---

## 5. Automated Test Suite & Reproducibility Commands

### 5.1 Test Execution Summary

| Test Module | Passed | XFail | Failed | Coverage Focus |
| :--- | :---: | :---: | :---: | :--- |
| `tests/lca/test_lca_smoke.py` | **27** | 0 | 0 | `RAPID.create(model="lca")`, binary indicator validation, `fit(bootstrap=False)`, `grid_results`, diagnostics (`ave_pp`, `occ_ratios`), outcome crosstabs, plot rendering |
| `tests/lca/test_lca_bootstrap.py` | **16** | 0 | 0 | `LCA-01` (LRT/BLRT in `grid_results`), `LCA-02` (`convergence_rate`), `LCA-04` (split IC & entropy plots), `LCA-05` (`k=1` in bootstrap CIs & plots), non-consecutive `k=[1, 2, 4]`, CI exception warning resilience, re-fit plot map cleanup |
| `tests/lca/test_lca_summary_report_pending.py` | **36** | 0 | 0 | `LCA-03` (default plots & `plots=[]` persistence), `GEN-03` (LCA summary sections & `table_format`), `LCA-06` (`summary(k=N)`), `LCA-07` (multi-`k` exploration, 1x2 outcome chi-square guard, `compare` + `k` validation), `LCA-08` (`decide(k=N)` integration with `report()` and `save()`, re-fit invalidation) |
| `tests/lca/test_lca_notebook_integration.py` | **3** | 0 | 0 | Validation notebook structure, absence of hardcoded paths, and end-to-end execution of `notebooks/validation/lca_development_local.ipynb` |
| **Total** | **82** | **0** | **0** | **100% Pass Rate (`82 PASSED, 0 XFAIL, 0 FAILED`)** |

### 5.2 Commands to Reproduce Validation & Comparison

Run from the repository root:

```bash
# 1. Run the complete automated LCA test suite (82 tests)
pytest tests/lca -ra

# 2. Run the fast unit/integration tests excluding the nbclient kernel test (81 tests)
pytest tests/lca -k "not nbclient" -ra

# 3. Reproduce the process-isolated Before (22b1a6d) vs After (current working tree) comparison
python scripts/lca/run_before_after_comparison.py --force

# 4. Generate synthetic baseline artifacts in artifacts/lca_baseline/synthetic/
python scripts/lca/generate_baseline_artifacts.py
```

---

## 6. Known Limitations & Out-of-Scope Items

1. **GEN-02 — PDF Wide-Table Column Overflow (General Report Layout):**
   - In `RAPID._generate_pdf_report()`, tables with many columns (such as the 14-column `Parameter Table` for 13 binary indicators or the 15-column `Class Enumeration Results` table) divide `(PAGE_W - 2*inch)` equally across columns without cell wrapping or landscape orientation switching. This affects wide tables across `RAPID` and is tracked separately under `GEN-02`.
2. **`bootstrap_stepmix()` Exception Denominator:**
   - In `bootstrap_stepmix()`, if a bootstrap replication raises a runtime exception during fitting (rather than returning `converged_ = False`), the exception is caught and skipped via `continue`, which reduces the `n_valid` denominator used when computing `convergence_rate`. Per methodological review instructions, the statistical estimation algorithm was not altered automatically; this behavior is documented and covered by `test_lca_bootstrap_non_consecutive_k_and_convergence_failures` in `tests/lca/test_lca_bootstrap.py`.
3. **Archive Deserialization Helper (`RAPID.load()`):**
   - `RAPID.save()` persists a `.rapid` ZIP archive containing `metadata.json` and `model.pkl` (serializing the selected `k` estimator when `decide(selected=True, k=N)` was called, or the reference estimator otherwise). While `.rapid` archives can be inspected and deserialized directly using Python's standard `zipfile`, `json`, and `pickle` modules, a `RAPID.load()` classmethod is not yet part of the `RAPID` base class contract.
