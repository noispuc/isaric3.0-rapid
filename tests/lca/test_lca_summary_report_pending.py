"""
Scenario C — Summary, selection of k, decide(), report(), and pending future requirements.

Verifies current behavior of:
- model.summary()
- model.summary(plots=[...])
- model.decide(selected=True, k=3, reason="...") (metadata recording)
- model.report(format=["csv", "png", "pdf"]) in an isolated output directory

Prepares xfail/pending tests for future LCA requirements:
- model.summary(k=3)
- Comparison of solutions with different k (e.g., summary(k=[2, 3, 4]))
- model.decide(selected=True, k=3, reason="...") activating k=3 as the reference/selected model
- model.report() after choosing k=3 reflecting the 3-class solution in exported results
"""

from pathlib import Path
import plotly.graph_objs as go
import pytest

import isaric as isa


# ============================================================================
# CURRENT BEHAVIOR TESTS (SCENARIO C)
# ============================================================================

def test_current_summary_without_args(model_no_bootstrap, monkeypatch, capsys):
    """
    Test A — Default summary without bootstrap:
    Verifies automatic display of metadata, binary indicator frequency table,
    class enumeration results, convergence diagnostics, reference model results,
    and default plots (including reference radar_plot).
    """
    shown = []
    monkeypatch.setattr(go.Figure, "show", lambda self, *a, **kw: shown.append(self))

    model_no_bootstrap.summary()
    out = capsys.readouterr().out

    # A. Model Metadata
    assert "MODEL INFORMATION" in out
    assert "stepmix" in out
    assert "Latent Class Analysis" in out
    assert "missing_data_strategy" in out

    # B. Binary Indicator Frequency Table
    assert "BINARY INDICATOR FREQUENCY TABLE" in out
    assert "Indicator" in out and "N_Valid" in out and "N_Missing" in out
    assert "Positive_n" in out and "Positive_%" in out

    # C. Class Enumeration Results
    assert "GRID SEARCH RESULTS (CLASS ENUMERATION)" in out
    assert "lrt_statistic" in out and "lrt_p_value" in out

    # D. Convergence Diagnostics
    assert "CONVERGENCE DIAGNOSTICS" in out
    assert "Enumeration converged" in out

    # E. Reference Model Results
    assert "REFERENCE MODEL RESULTS (Reference k = 4 — Not Automatically Selected)" in out
    assert "Reference Model Fit Metrics (k=4)" in out
    assert "Reference Model Class Prevalence & Diagnostics (k=4)" in out
    assert "Item-Response Probabilities (Class Profiles, Reference k=4)" in out
    assert "Class_1" in out and "Class_4" in out

    # Default plots automatically rendered
    expected_defaults = [
        "information_criteria",
        "ll_curve",
        "entropy_curve",
        "relative_entropy_curve",
        "radar_plot",
    ]
    assert model_no_bootstrap._displayed_plots == expected_defaults
    assert len(shown) == len(expected_defaults)


def test_summary_plots_empty_list_disables_plots(model_no_bootstrap, monkeypatch, capsys):
    """
    Test B — No plots:
    Verifies model.summary(plots=[]) displays all textual/tabular sections
    but produces zero plot-rendering calls and sets _displayed_plots = [].
    """
    shown = []
    monkeypatch.setattr(go.Figure, "show", lambda self, *a, **kw: shown.append(self))

    model_no_bootstrap.summary(plots=[])
    out = capsys.readouterr().out

    assert len(shown) == 0
    assert model_no_bootstrap._displayed_plots == []
    assert "MODEL INFORMATION" in out
    assert "BINARY INDICATOR FREQUENCY TABLE" in out
    assert "REFERENCE MODEL RESULTS" in out


def test_current_summary_with_explicit_plots(model_no_bootstrap, monkeypatch):
    """
    Test C & D — Single and multiple explicit plot selection in requested order.
    """
    shown = []
    monkeypatch.setattr(go.Figure, "show", lambda self, *a, **kw: shown.append(self))

    # Single explicit plot
    model_no_bootstrap.summary(plots=["information_criteria"])
    assert len(shown) == 1
    assert model_no_bootstrap._displayed_plots == ["information_criteria"]
    assert "Information Criteria" in shown[0].layout.title.text

    # Multiple explicit plots in custom order
    shown.clear()
    requested_plots = [
        "radar_plot",
        "information_criteria",
    ]
    model_no_bootstrap.summary(plots=requested_plots)

    assert len(shown) == 2
    assert model_no_bootstrap._displayed_plots == requested_plots
    assert "Radar (Reference k=4)" in shown[0].layout.title.text
    assert "Information Criteria" in shown[1].layout.title.text


def test_summary_default_plots_with_bootstrap(model_bootstrap, monkeypatch, capsys):
    """
    Test D (Bootstrap) — Default summary after fit(bootstrap=True) includes
    bootstrap_metrics_ic, bootstrap_metrics_entropy, and selection_rate,
    and performs zero bootstrap recalculations during summary().
    """
    from isaric.modelevaluation import gridsearch as gs_mod

    def _forbid_recalc(*args, **kwargs):
        raise AssertionError("Bootstrap must not be recalculated during summary()")

    monkeypatch.setattr(gs_mod, "bootstrap_stepmix", _forbid_recalc)
    monkeypatch.setattr(gs_mod, "bootstrap_ci_stepmix", _forbid_recalc)
    monkeypatch.setattr(gs_mod, "bootstrap_lrt_stepmix", _forbid_recalc)

    shown = []
    monkeypatch.setattr(go.Figure, "show", lambda self, *a, **kw: shown.append(self))

    model_bootstrap.summary()
    out = capsys.readouterr().out

    expected_boot_defaults = [
        "information_criteria",
        "ll_curve",
        "entropy_curve",
        "relative_entropy_curve",
        "radar_plot",
        "bootstrap_metrics_ic",
        "bootstrap_metrics_entropy",
        "selection_rate",
    ]
    assert model_bootstrap._displayed_plots == expected_boot_defaults
    assert len(shown) == len(expected_boot_defaults)
    assert "blrt_p_value" in out


def test_summary_reference_model_corresponds_to_actual_ref_k(
    synthetic_df, symptom_vars, monkeypatch, capsys
):
    """
    Test E — Reference model:
    Verifies that summary() and radar_plot reflect the actual reference k
    (e.g. k=2 when n_clusters=2 in create(), even if k=3 has lower AIC/BIC),
    never silently auto-selecting another k.
    """
    shown = []
    monkeypatch.setattr(go.Figure, "show", lambda self, *a, **kw: shown.append(self))

    model = isa.RAPID.create(
        data=synthetic_df,
        model="lca",
        n_clusters=2,
        measurement_vars=symptom_vars,
        random_seed=42,
    )
    model.fit(bootstrap=False, param_grid={"n_clusters": [1, 2, 3, 4]})
    model.summary(plots=["radar_plot"])
    out = capsys.readouterr().out

    assert model.fitted_model.n_components == 2
    assert "REFERENCE MODEL RESULTS (Reference k = 2 — Not Automatically Selected)" in out
    assert "Reference Model Fit Metrics (k=2)" in out
    assert "Class_1" in out and "Class_2" in out
    assert "Class_3" not in out
    assert len(shown) == 1
    assert "Radar (Reference k=2)" in shown[0].layout.title.text
    assert len(shown[0].data) == 2


def test_binary_indicator_frequency_table_with_missing_and_non_binary(
    model_no_bootstrap
):
    """
    Test F — Binary indicator frequency table:
    Verifies exact counts, denominators (N_Valid, never treating NaN as 0),
    percentages, missing-value counts, and non-binary variable handling.
    """
    import numpy as np
    import pandas as pd

    df_custom = pd.DataFrame(
        {
            "SympA": [1.0, 1.0, 0.0, 0.0, np.nan],   # 4 valid, 1 missing, 2 pos -> 50.0%
            "SympB": [1.0, 0.0, 1.0, np.nan, np.nan], # 3 valid, 2 missing, 2 pos -> 66.67%
            "SympC": [0.0, 0.0, 0.0, 0.0, 0.0],       # 5 valid, 0 missing, 0 pos -> 0.0%
        }
    )
    freq_df = model_no_bootstrap._build_indicator_frequency_df(
        data=df_custom, measurement_vars=["SympA", "SympB", "SympC"]
    ).set_index("Indicator")

    assert freq_df.loc["SympA", "N_Valid"] == 4
    assert freq_df.loc["SympA", "N_Missing"] == 1
    assert freq_df.loc["SympA", "Positive_n"] == 2
    assert np.isclose(freq_df.loc["SympA", "Positive_%"], 50.0, atol=1e-4)

    assert freq_df.loc["SympB", "N_Valid"] == 3
    assert freq_df.loc["SympB", "N_Missing"] == 2
    assert freq_df.loc["SympB", "Positive_n"] == 2
    assert np.isclose(freq_df.loc["SympB", "Positive_%"], 66.67, atol=1e-2)

    assert freq_df.loc["SympC", "N_Valid"] == 5
    assert freq_df.loc["SympC", "N_Missing"] == 0
    assert freq_df.loc["SympC", "Positive_n"] == 0
    assert np.isclose(freq_df.loc["SympC", "Positive_%"], 0.0, atol=1e-4)

    # Non-binary column handling: omits misleading binary prevalence
    df_nonbin = pd.DataFrame({"SympA": [0, 1, 2, 1]})
    freq_nonbin = model_no_bootstrap._build_indicator_frequency_df(
        data=df_nonbin, measurement_vars=["SympA"]
    )
    assert np.isnan(freq_nonbin.loc[0, "Positive_n"])
    assert np.isnan(freq_nonbin.loc[0, "Positive_%"])
    assert "Non-binary" in freq_nonbin.loc[0, "Note"]


def test_summary_metadata_matches_explicit_and_default_parameters(
    synthetic_df, symptom_vars, capsys
):
    """
    Test G — Metadata:
    Verifies displayed parameters match actual values used during fitting
    for both default and explicitly configured parameters.
    """
    # 1. Default parameters
    m_def = isa.RAPID.create(
        data=synthetic_df,
        model="lca",
        measurement_vars=symptom_vars,
    )
    m_def.fit(param_grid={"n_clusters": [1, 2]})
    params_def = m_def.model_info["parameters"]
    assert params_def["n_init"] == 5
    assert params_def["max_iter"] == 500
    assert m_def.model_info["random_seed"] == 42
    assert params_def["n_clusters_requested"] == 3
    assert params_def["n_clusters_tested"] == [1, 2]
    assert params_def["n_clusters_reference"] == 1
    assert params_def["selection_metric"] == "BIC"
    assert params_def["bootstrap"] is False
    assert params_def["n_iterations"] is None

    # 2. Explicit custom parameters
    m_exp = isa.RAPID.create(
        data=synthetic_df,
        model="lca",
        n_clusters=2,
        n_init=3,
        max_iter=250,
        random_seed=99,
        measurement_vars=symptom_vars,
    )
    m_exp.fit(
        bootstrap=True,
        n_iterations=4,
        selection_metric="AIC",
        param_grid={"n_clusters": [1, 2]},
    )
    params_exp = m_exp.model_info["parameters"]
    assert params_exp["n_init"] == 3
    assert params_exp["max_iter"] == 250
    assert m_exp.model_info["random_seed"] == 99
    assert params_exp["selection_metric"] == "AIC"
    assert params_exp["bootstrap"] is True
    assert params_exp["n_iterations"] == 4

    m_exp.summary(plots=[])
    out = capsys.readouterr().out
    assert "n_init=3" in out
    assert "max_iter=250" in out
    assert "selection_metric=AIC" in out
    assert "n_iterations=4" in out


def test_summary_repeated_calls_and_refit_hygiene(
    synthetic_df, symptom_vars, monkeypatch
):
    """
    Test H — Repeated calls and re-fit:
    Executes summary() -> summary(plots=[]) -> summary(plots=["information_criteria"]) -> summary()
    and fit(bootstrap=True) -> fit(bootstrap=False), verifying no plot contamination.
    """
    shown = []
    monkeypatch.setattr(go.Figure, "show", lambda self, *a, **kw: shown.append(self))

    model = isa.RAPID.create(
        data=synthetic_df,
        model="lca",
        n_clusters=3,
        measurement_vars=symptom_vars,
        random_seed=42,
    )
    model.fit(bootstrap=False, param_grid={"n_clusters": [1, 2, 3]})

    # 1. Default summary() -> 5 plots
    model.summary()
    assert len(shown) == 5
    assert len(model._displayed_plots) == 5

    # 2. summary(plots=[]) -> 0 plots
    shown.clear()
    model.summary(plots=[])
    assert len(shown) == 0
    assert model._displayed_plots == []

    # 3. summary(plots=["information_criteria"]) -> 1 plot
    shown.clear()
    model.summary(plots=["information_criteria"])
    assert len(shown) == 1
    assert model._displayed_plots == ["information_criteria"]

    # 4. Default summary() again -> 5 plots
    shown.clear()
    model.summary()
    assert len(shown) == 5
    assert len(model._displayed_plots) == 5

    # 5. Re-fit with bootstrap=True -> 8 default plots, then re-fit with bootstrap=False -> 5 default plots
    shown.clear()
    model.fit(bootstrap=True, n_iterations=5, param_grid={"n_clusters": [1, 2, 3]})
    model.summary()
    assert len(shown) == 8
    assert "bootstrap_metrics_ic" in model._displayed_plots

    shown.clear()
    model.fit(bootstrap=False, param_grid={"n_clusters": [1, 2, 3]})
    model.summary()
    assert len(shown) == 5
    assert "bootstrap_metrics_ic" not in model._displayed_plots
    assert "bootstrap_metrics_entropy" not in model._displayed_plots


def test_non_lca_models_summary_behavior_unchanged(
    synthetic_df, symptom_vars, monkeypatch, capsys
):
    """
    Test I — Non-LCA regression:
    Verify that non-LCA model families (e.g., KMeans, LogisticRegression) do not
    trigger LCA summary hooks or auto-render plots when summary() is called without plots=.
    """
    shown = []
    monkeypatch.setattr(go.Figure, "show", lambda self, *a, **kw: shown.append(self))

    # KMeans
    km = isa.RAPID.create(
        data=synthetic_df,
        model="kmeans",
        predictors=symptom_vars[:4],
        n_clusters=3,
        random_seed=42,
    )
    km.fit()
    km.summary()
    km_out = capsys.readouterr().out
    assert len(shown) == 0
    assert "MODEL INFORMATION" in km_out
    assert "BINARY INDICATOR FREQUENCY TABLE" not in km_out

    # Logistic Regression
    lr = isa.RAPID.create(
        data=synthetic_df,
        model="logistic",
        dependent_var="severe",
        independent_vars=["Fever", "Myalgia", "Headache"],
    )
    lr.fit()
    lr.summary()
    lr_out = capsys.readouterr().out
    assert len(shown) == 0
    assert "MODEL INFORMATION" in lr_out
    assert "BINARY INDICATOR FREQUENCY TABLE" not in lr_out

    # Passing k=... to non-LCA models raises a clear ValueError without breaking summary()
    with pytest.raises(ValueError, match="only supported for LCA"):
        km.summary(k=2)
    with pytest.raises(ValueError, match="only supported for LCA"):
        lr.summary(k=2)


def test_current_report_exports_csv_png_pdf_in_isolated_dir(
    model_no_bootstrap, tmp_path: Path, monkeypatch
):
    """
    Verify current behavior of model.report(format=['csv', 'png', 'pdf'])
    inside an isolated temporary directory (avoiding source directory pollution),
    and verify PDF story consistency for LCA-01 and LCA-02.
    """
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Table

    captured_stories = []
    orig_build = SimpleDocTemplate.build

    def _spy_build(self, flowables, *args, **kwargs):
        captured_stories.append(list(flowables))
        return orig_build(self, flowables, *args, **kwargs)

    monkeypatch.setattr(SimpleDocTemplate, "build", _spy_build)
    monkeypatch.chdir(tmp_path)
    model_no_bootstrap.summary(plots=["information_criteria", "profile_heatmap"])
    model_no_bootstrap.report(format=["csv", "png", "pdf"])

    csv_files = list(tmp_path.glob("*.csv"))
    png_files = list(tmp_path.glob("*.png"))
    pdf_files = list(tmp_path.glob("*.pdf"))

    assert len(csv_files) >= 2  # _results.csv, _grid_results.csv, _ave_pp.csv
    assert len(png_files) == 2
    assert len(pdf_files) == 1
    assert pdf_files[0].stat().st_size > 1000

    # Verify PDF story consistency (LCA-01 & LCA-02)
    assert len(captured_stories) == 1
    story = captured_stories[0]
    paragraphs = [f.getPlainText() for f in story if isinstance(f, Paragraph)]

    assert "Class Enumeration Results" in paragraphs
    # Nested LRT must not be duplicated as a separate table when integrated in grid_results
    assert "Nested LRT (Chi-square)" not in paragraphs

    # Locate the Table immediately following 'Class Enumeration Results'
    enum_idx = next(
        i for i, f in enumerate(story)
        if isinstance(f, Paragraph) and f.getPlainText() == "Class Enumeration Results"
    )
    enum_table = next(f for f in story[enum_idx + 1 :] if isinstance(f, Table))
    table_header = enum_table._cellvalues[0]

    assert "lrt_statistic" in table_header
    assert "lrt_dof_diff" in table_header
    assert "lrt_p_value" in table_header
    assert "convergence_rate" not in table_header


def test_report_preserves_explicit_empty_plots_and_legacy_no_summary(
    synthetic_df, symptom_vars, tmp_path: Path, monkeypatch
):
    """
    Prompt 3.1 (Fix 1):
    - summary(plots=[]) followed by report(format=["png", "pdf"]) must export 0 figures.
    - report(format=["png", "pdf"]) called without a preceding summary() must preserve
      legacy behavior and export all available plots in plots_map.
    """
    from reportlab.platypus import Image, Paragraph, SimpleDocTemplate

    captured_stories = []
    orig_build = SimpleDocTemplate.build

    def _spy_build(self, flowables, *args, **kwargs):
        captured_stories.append(list(flowables))
        return orig_build(self, flowables, *args, **kwargs)

    monkeypatch.setattr(SimpleDocTemplate, "build", _spy_build)

    # Case 1: summary(plots=[]) -> report(format=["png", "pdf"]) exports NO figures
    dir_empty = tmp_path / "empty_plots"
    dir_empty.mkdir()
    monkeypatch.chdir(dir_empty)

    m_empty = isa.RAPID.create(
        data=synthetic_df,
        model="lca",
        n_clusters=3,
        measurement_vars=symptom_vars,
        random_seed=42,
    )
    m_empty.fit(bootstrap=False, param_grid={"n_clusters": [1, 2, 3]})
    assert m_empty._displayed_plots is None

    m_empty.summary(plots=[])
    assert m_empty._displayed_plots == []

    m_empty.report(format=["png", "pdf"])
    assert len(list(dir_empty.glob("*.png"))) == 0
    assert len(list(dir_empty.glob("*.pdf"))) == 1

    story_empty = captured_stories[-1]
    paragraphs_empty = [f.getPlainText() for f in story_empty if isinstance(f, Paragraph)]
    images_empty = [f for f in story_empty if isinstance(f, Image)]
    assert "Figures" not in paragraphs_empty
    assert len(images_empty) == 0

    # Case 2: report(format=["png", "pdf"]) without calling summary() exports all plots_map figures
    dir_legacy = tmp_path / "legacy_no_summary"
    dir_legacy.mkdir()
    monkeypatch.chdir(dir_legacy)

    m_legacy = isa.RAPID.create(
        data=synthetic_df,
        model="lca",
        n_clusters=3,
        measurement_vars=symptom_vars,
        random_seed=42,
    )
    m_legacy.fit(bootstrap=False, param_grid={"n_clusters": [1, 2, 3]})
    assert m_legacy._displayed_plots is None

    m_legacy.report(format=["png", "pdf"])
    png_legacy = list(dir_legacy.glob("*.png"))
    assert len(png_legacy) == len(m_legacy.plots_map)
    assert len(list(dir_legacy.glob("*.pdf"))) == 1

    story_legacy = captured_stories[-1]
    paragraphs_legacy = [f.getPlainText() for f in story_legacy if isinstance(f, Paragraph)]
    images_legacy = [f for f in story_legacy if isinstance(f, Image)]
    assert "Figures" in paragraphs_legacy
    assert len(images_legacy) == len(m_legacy.plots_map)


def test_lca_missing_data_metadata_and_stepmix_nan_behavior(
    model_no_bootstrap, synthetic_df, symptom_vars
):
    """
    Prompt 3.1 (Fix 2):
    Verify missing_data_strategy accurately states that no automatic complete-case
    filtering is performed by RAPID LCA (caller must supply complete cases because
    StepMix measurement='bernoulli' rejects NaN).
    """
    import numpy as np

    strategy = model_no_bootstrap.model_info["missing_data_strategy"]
    assert "none" in strategy.lower()
    assert "complete-case preprocessing required from caller" in strategy
    assert "measurement='bernoulli' rejects NaN" in strategy
    assert model_no_bootstrap.model_info["N_missing_values"] == 0

    # Verify actual runtime behavior when NaN is present:
    # create() does not drop NaN rows, and StepMix fit() rejects NaN
    df_nan = synthetic_df.copy()
    df_nan.loc[df_nan.index[0], symptom_vars[0]] = np.nan

    m_nan = isa.RAPID.create(
        data=df_nan,
        model="lca",
        n_clusters=2,
        measurement_vars=symptom_vars,
        random_seed=42,
    )
    assert len(m_nan.X) == len(df_nan)
    assert int(m_nan.X.isna().sum().sum()) == 1

    with pytest.raises(ValueError, match="NaN"):
        m_nan.fit(bootstrap=False, param_grid={"n_clusters": [1, 2]})


def test_lca_summary_table_format_short_vs_full(model_no_bootstrap, capsys):
    """
    Prompt 3.1 (Fix 3):
    Verify table_format='short' produces a concise summary preserving essential
    metadata, indicator frequencies, class enumeration, and reference model results,
    while omitting verbose diagnostic columns shown in table_format='full'.
    """
    model_no_bootstrap.summary(table_format="full", plots=[])
    out_full = capsys.readouterr().out

    model_no_bootstrap.summary(table_format="short", plots=[])
    out_short = capsys.readouterr().out

    # Both formats preserve all essential sections and core metrics
    for section in [
        "MODEL INFORMATION",
        "BINARY INDICATOR FREQUENCY TABLE",
        "GRID SEARCH RESULTS (CLASS ENUMERATION)",
        "CONVERGENCE DIAGNOSTICS",
        "REFERENCE MODEL RESULTS",
        "missing_data_strategy",
        "AIC",
        "BIC",
        "SABIC",
        "lrt_p_value",
        "Class_1",
    ]:
        assert section in out_full
        assert section in out_short

    # Full format includes verbose diagnostic fields omitted from short format
    for full_only_field in [
        "N_missing_values",
        "occ_k_min",
        "min_mm_f",
        "lrt_dof_diff",
        "CAIC",
        "AvePP",
        "OCC",
    ]:
        assert full_only_field in out_full
        assert full_only_field not in out_short

    assert len(out_short) < len(out_full)


def test_current_decide_records_metadata(synthetic_df, symptom_vars):
    """Verify current behavior of model.decide(selected=True, k=3, reason=...)."""
    model = isa.RAPID.create(
        data=synthetic_df,
        model="lca",
        n_clusters=4,
        measurement_vars=symptom_vars,
        random_seed=42,
    )
    model.fit(bootstrap=False, param_grid={"n_clusters": [1, 2, 3, 4]})

    ret = model.decide(
        selected=True,
        k=3,
        reason="BIC elbow and clinically interpretable 3-class profiles.",
    )
    assert ret is model
    assert model._decision is not None
    assert model._decision["selected"] is True
    assert model._decision["k"] == 3
    assert "BIC elbow" in model._decision["reason"]


def test_known_out_of_scope_report_string_model_id_bug(model_no_bootstrap):
    """
    Document known out-of-scope type validation issue (BUG-02):
    Calling model.report(models="lca-20261103-124719") with a string instead
    of a list of RAPID objects currently raises AttributeError when iterating
    over characters of the string.
    """
    with pytest.raises((AttributeError, TypeError, ValueError)):
        model_no_bootstrap.report(models="lca-20261103-124719")


# ============================================================================
# LCA-06 & LCA-07: EXPLORATION AND COMPARISON OF CLASS SOLUTIONS
# ============================================================================

def test_lca06_summary_with_specific_k(model_no_bootstrap, capsys):
    """
    Test A — Summary for specific k (LCA-06):
    Train k=[1,2,3,4], call summary(k=3, plots=[]), and verify:
    - Three class profiles (Class_1, Class_2, Class_3; no Class_4).
    - Fit metrics correspond to k=3 from grid_results.
    - Prevalence and diagnostics (AvePP, OCC) correspond to k=3.
    - Full enumeration table (k=1..4) remains present.
    - Reference metadata remains unchanged (reference_k=4).
    """
    import numpy as np

    model_no_bootstrap.summary(k=3, plots=[])
    out = capsys.readouterr().out

    # Header and metadata
    assert "EXPLORATORY MODEL RESULTS (k=3 — Reference k=4)" in out
    assert "exploratory_k" in out
    assert "reference_k" in out
    assert model_no_bootstrap.model_info["reference_k"] == 4
    assert "exploratory_k" not in model_no_bootstrap.model_info

    # Full enumeration table remains present (all k=1, 2, 3, 4)
    assert "GRID SEARCH RESULTS (CLASS ENUMERATION)" in out
    grid_by_k = model_no_bootstrap.grid_results.set_index("n_clusters")
    for k_val in [1, 2, 3, 4]:
        assert f"{grid_by_k.loc[k_val, 'AIC']:.4f}" in out

    # Exploratory k=3 sections
    assert "--- Exploratory Model Fit Metrics (k=3) ---" in out
    assert "--- Exploratory Model Class Prevalence & Diagnostics (k=3) ---" in out
    assert "--- Item-Response Probabilities (Class Profiles, Exploratory k=3) ---" in out
    assert "Class_1" in out and "Class_2" in out and "Class_3" in out
    assert "Class_4" not in out

    # Verify numerical accuracy of resolved k=3 solution
    sol_k3 = model_no_bootstrap._resolve_k_solution(3)
    assert sol_k3["k"] == 3
    assert sol_k3["ref_k"] == 4
    assert sol_k3["is_exploratory"] is True
    assert len(sol_k3["profiles_df"]) == 3
    assert np.isclose(sol_k3["grid_row"]["AIC"], grid_by_k.loc[3, "AIC"], rtol=1e-12)
    assert np.isclose(sol_k3["grid_row"]["BIC"], grid_by_k.loc[3, "BIC"], rtol=1e-12)

    prev_df = model_no_bootstrap._build_class_prevalence_df(sol_k3, is_short=False)
    assert len(prev_df) == 3
    assert np.allclose(prev_df["Prevalence"].values, np.round(model_no_bootstrap.model_objects[3].weights_, 4))
    ave_pp_k3 = model_no_bootstrap.DiagClass["ave_pp"].loc[3, ["Class_1", "Class_2", "Class_3"]].values
    assert np.allclose(prev_df["AvePP"].values, np.round(ave_pp_k3.astype(float), 4))


def test_lca07_sequential_exploration_and_comparison_workflow(
    model_no_bootstrap, monkeypatch, capsys
):
    """
    Test B — Sequential exploration and comparison of k solutions (LCA-07):
    Execute summary(k=2) -> summary(k=4) -> summary(k=3) -> summary()
    without refitting, confirming each call displays the requested solution
    and the final summary() returns to the original reference model (k=4).
    """
    shown = []
    monkeypatch.setattr(go.Figure, "show", lambda self, *a, **kw: shown.append(self))

    # 1. Explore k=2
    model_no_bootstrap.summary(k=2, plots=["radar_plot"])
    out_k2 = capsys.readouterr().out
    assert "EXPLORATORY MODEL RESULTS (k=2 — Reference k=4)" in out_k2
    assert "Class_1" in out_k2 and "Class_2" in out_k2
    assert "Class_3" not in out_k2
    assert len(shown[-1].data) == 2
    assert "Exploratory k=2" in shown[-1].layout.title.text

    # 2. Explore k=4
    model_no_bootstrap.summary(k=4, plots=["radar_plot"])
    out_k4 = capsys.readouterr().out
    assert "EXPLORATORY MODEL RESULTS (k=4 — Reference k=4)" in out_k4
    assert "Class_4" in out_k4
    assert len(shown[-1].data) == 4
    assert "Exploratory k=4" in shown[-1].layout.title.text

    # 3. Explore k=3
    model_no_bootstrap.summary(k=3, plots=["radar_plot"])
    out_k3 = capsys.readouterr().out
    assert "EXPLORATORY MODEL RESULTS (k=3 — Reference k=4)" in out_k3
    assert "Class_3" in out_k3 and "Class_4" not in out_k3
    assert len(shown[-1].data) == 3
    assert "Exploratory k=3" in shown[-1].layout.title.text

    # 4. Return to default reference summary()
    model_no_bootstrap.summary(plots=["radar_plot"])
    out_ref = capsys.readouterr().out
    assert "REFERENCE MODEL RESULTS (Reference k = 4 — Not Automatically Selected)" in out_ref
    assert "Class_4" in out_ref
    assert len(shown[-1].data) == 4
    assert "Reference k=4" in shown[-1].layout.title.text


def test_lca06_state_preservation_even_on_plot_exception(
    synthetic_df, symptom_vars, monkeypatch, capsys
):
    """
    Test C — State preservation (including outcome and custom class_names):
    Verify fitted_model, result_df, grid_results, metrics, model_info, _decision,
    n_clusters, model_objects, and outcome_crosstab_df remain completely untouched
    before and after exploratory summaries, even when a plot raises an exception.
    """
    import copy
    import pandas as pd

    model = isa.RAPID.create(
        data=synthetic_df,
        model="lca",
        n_clusters=4,
        measurement_vars=symptom_vars,
        outcome="severe",
        random_seed=42,
    )
    model.fit(bootstrap=False, param_grid={"n_clusters": [1, 2, 3, 4]})
    model.class_names = ["Ref_A", "Ref_B", "Ref_C", "Ref_D"]

    fitted_id_before = id(model.fitted_model)
    model_objs_ids_before = {k: id(v) for k, v in model.model_objects.items()}
    result_df_before = model.result_df.copy(deep=True)
    grid_before = model.grid_results.copy(deep=True)
    metrics_before = copy.deepcopy(model.metrics)
    info_before = copy.deepcopy(model.model_info)
    decision_before = copy.deepcopy(model._decision)
    outcome_ct_before = model.outcome_crosstab_df.copy(deep=True)

    # Explore k=2 and k=3 with outcome present and custom class_names on k=4
    model.summary(k=3, plots=[])
    out_k3 = capsys.readouterr().out
    # Custom class_names for k=4 must NOT be applied to k=3
    assert "Ref_A" not in out_k3
    assert "Class_1" in out_k3 and "Class_3" in out_k3 and "Class_4" not in out_k3
    assert "OUTCOME BY CLASS — severe (Exploratory k = 3)" in out_k3

    # Simulate an exception during plotting in summary(k=2)
    def _boom(*args, **kwargs):
        raise RuntimeError("Simulated plotting failure")

    monkeypatch.setattr(model, "_radar_plot", _boom)
    model.plots_map["radar_plot"] = model._radar_plot

    with pytest.raises(RuntimeError, match="Simulated plotting failure"):
        model.summary(k=2, plots=["radar_plot"])

    # Verify scientific state was never mutated
    assert id(model.fitted_model) == fitted_id_before
    assert model.fitted_model.n_components == 4
    assert model.n_clusters == 4
    assert {k: id(v) for k, v in model.model_objects.items()} == model_objs_ids_before
    pd.testing.assert_frame_equal(model.result_df, result_df_before)
    pd.testing.assert_frame_equal(model.grid_results, grid_before)
    pd.testing.assert_frame_equal(model.outcome_crosstab_df, outcome_ct_before)
    assert model.metrics == metrics_before
    assert model.model_info == info_before
    assert model._decision == decision_before


def test_lca06_plot_correctness_plotly_and_matplotlib(model_no_bootstrap):
    """
    Test D — Plot correctness across k values and backends (Plotly & Matplotlib):
    - Radar with k=3 displays 3 classes.
    - Heatmap with k=2 displays 2 classes.
    - Cluster distribution corresponds to requested k.
    - Enumeration plots retain all fitted k=[1, 2, 3, 4].
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # 1. Radar plot for k=3 (Plotly & Matplotlib)
    radar_pl = model_no_bootstrap._radar_plot(backend="plotly", k=3)
    assert len(radar_pl.data) == 3
    assert [t.name for t in radar_pl.data] == ["Class_1", "Class_2", "Class_3"]
    assert "Exploratory k=3" in radar_pl.layout.title.text

    radar_mpl = model_no_bootstrap._radar_plot(backend="matplotlib", k=3)
    ax_radar = radar_mpl.axes[0]
    assert "Exploratory k=3" in ax_radar.get_title()
    _, legend_labels = ax_radar.get_legend_handles_labels()
    assert legend_labels == ["Class_1", "Class_2", "Class_3"]
    plt.close(radar_mpl)

    # 2. Profile heatmap for k=2 (Plotly & Matplotlib)
    hm_pl = model_no_bootstrap._profile_heatmap(backend="plotly", k=2)
    assert np.asarray(hm_pl.data[0].z).shape == (2, len(model_no_bootstrap.measurement_vars))
    assert "Exploratory k=2" in hm_pl.layout.title.text

    hm_mpl = model_no_bootstrap._profile_heatmap(backend="matplotlib", k=2)
    ax_hm = hm_mpl.axes[0]
    assert len(ax_hm.get_yticks()) == 2
    assert "Exploratory k=2" in ax_hm.get_title()
    plt.close(hm_mpl)

    # 3. Cluster distribution for k=3 and k=2 (Plotly & Matplotlib)
    dist_k3_pl = model_no_bootstrap._cluster_distribution(backend="plotly", k=3)
    assert list(dist_k3_pl.data[0].x) == ["Class_1", "Class_2", "Class_3"]
    assert sum(dist_k3_pl.data[0].y) == len(model_no_bootstrap.X)
    assert "Exploratory k=3" in dist_k3_pl.layout.title.text

    dist_k2_mpl = model_no_bootstrap._cluster_distribution(backend="matplotlib", k=2)
    ax_dist = dist_k2_mpl.axes[0]
    assert len(ax_dist.patches) == 2
    assert "Exploratory k=2" in ax_dist.get_title()
    plt.close(dist_k2_mpl)

    # 4. Enumeration plots retain all fitted k=[1, 2, 3, 4] even when k=3 is passed
    ic_pl = model_no_bootstrap._information_criteria(backend="plotly", k=3)
    assert list(ic_pl.data[0].x) == [1, 2, 3, 4]
    ll_pl = model_no_bootstrap._ll_curve(backend="plotly", k=3)
    assert list(ll_pl.data[0].x) == [1, 2, 3, 4]
    ent_pl = model_no_bootstrap._entropy_curve(backend="plotly", k=3)
    assert list(ent_pl.data[0].x) == [1, 2, 3, 4]
    rel_pl = model_no_bootstrap._relative_entropy_curve(backend="plotly", k=3)
    assert list(rel_pl.data[0].x) == [1, 2, 3, 4]


def test_lca06_invalid_k_and_unfitted_validation(
    model_no_bootstrap, synthetic_df, symptom_vars
):
    """
    Test E — Invalid k and unfitted summary(k=...) validation:
    Verify clear ValueErrors for k=99, k=0, k not fitted, string k, boolean k,
    list k, and summary(k=...) before fit().
    """
    # k out of range (k=99, k=0, k=-1)
    with pytest.raises(ValueError, match=r"k=99 is not available.*\[1, 2, 3, 4\]"):
        model_no_bootstrap.summary(k=99, plots=[])

    with pytest.raises(ValueError, match=r"k=0 is not available.*\[1, 2, 3, 4\]"):
        model_no_bootstrap.summary(k=0, plots=[])

    # Non-integer / boolean / list k
    for bad_k in ["3", True, False, 2.5, [2, 3, 4]]:
        with pytest.raises(ValueError, match="must be an integer"):
            model_no_bootstrap.summary(k=bad_k, plots=[])

    # k not fitted in non-consecutive enumeration [1, 2, 4]
    m_gap = isa.RAPID.create(
        data=synthetic_df,
        model="lca",
        n_clusters=4,
        measurement_vars=symptom_vars,
        random_seed=42,
    )
    m_gap.fit(bootstrap=False, param_grid={"n_clusters": [1, 2, 4]})
    with pytest.raises(ValueError, match=r"k=3 is not available.*\[1, 2, 4\]"):
        m_gap.summary(k=3, plots=[])

    # Calling summary(k=2) before fit()
    m_unfitted = isa.RAPID.create(
        data=synthetic_df,
        model="lca",
        n_clusters=3,
        measurement_vars=symptom_vars,
        random_seed=42,
    )
    with pytest.raises(ValueError, match="state 'created'"):
        m_unfitted.summary(k=2)


def test_lca06_k1_one_class_solution_summary_and_plots(
    model_no_bootstrap, monkeypatch, capsys
):
    """
    Test F — k=1 exploration:
    Verify class-profile, prevalence, and metric presentation for k=1 without
    inventing undefined relative_entropy or LRT values.
    """
    shown = []
    monkeypatch.setattr(go.Figure, "show", lambda self, *a, **kw: shown.append(self))

    model_no_bootstrap.summary(
        k=1,
        plots=["radar_plot", "profile_heatmap", "cluster_distribution"],
    )
    out = capsys.readouterr().out

    assert "EXPLORATORY MODEL RESULTS (k=1 — Reference k=4)" in out
    assert "--- Exploratory Model Fit Metrics (k=1) ---" in out
    assert "--- Exploratory Model Class Prevalence & Diagnostics (k=1) ---" in out
    assert "Class_1" in out
    assert "Class_2" not in out

    # Verify fit metrics table for k=1 does not include undefined relative_entropy
    sol_k1 = model_no_bootstrap._resolve_k_solution(1)
    fit_k1_df = model_no_bootstrap._build_solution_fit_metrics_df(sol_k1, is_short=False)
    assert "relative_entropy" not in fit_k1_df.columns
    assert fit_k1_df.loc[0, "n_clusters"] == 1

    # Verify plots for k=1 rendered 1 class
    assert len(shown) == 3
    assert len(shown[0].data) == 1  # radar_plot with 1 class trace


def test_lca06_bootstrap_exploratory_summary_without_recalculation(
    model_bootstrap, monkeypatch, capsys
):
    """
    Test G — Bootstrap exploratory summary:
    After fit(bootstrap=True), confirm summary(k=3) displays k=3 tables and profiles
    while preserving overall bootstrap enumeration results and performing zero
    bootstrap recalculations.
    """
    from isaric.modelevaluation import gridsearch as gs_mod

    def _forbid_recalc(*args, **kwargs):
        raise AssertionError("Bootstrap must not be recalculated during summary(k=3)")

    monkeypatch.setattr(gs_mod, "bootstrap_stepmix", _forbid_recalc)
    monkeypatch.setattr(gs_mod, "bootstrap_ci_stepmix", _forbid_recalc)
    monkeypatch.setattr(gs_mod, "bootstrap_lrt_stepmix", _forbid_recalc)

    shown = []
    monkeypatch.setattr(go.Figure, "show", lambda self, *a, **kw: shown.append(self))

    model_bootstrap.summary(k=3)
    out = capsys.readouterr().out

    assert "EXPLORATORY MODEL RESULTS (k=3 — Reference k=4)" in out
    assert "blrt_p_value" in out
    assert "Class_1" in out and "Class_3" in out and "Class_4" not in out
    assert len(shown) == 8
    # Radar plot in default plots reflects Exploratory k=3 (3 traces)
    radar_fig = next(f for f in shown if "Radar" in f.layout.title.text)
    assert "Exploratory k=3" in radar_fig.layout.title.text
    assert len(radar_fig.data) == 3


def test_lca06_exploratory_table_format_full_vs_short(model_no_bootstrap, capsys):
    """
    Test H — Exploratory summary table formats:
    Verify summary(k=3, table_format='full') and summary(k=3, table_format='short')
    preserve the distinction between full and short presentations.
    """
    model_no_bootstrap.summary(k=3, table_format="full", plots=[])
    out_full = capsys.readouterr().out

    model_no_bootstrap.summary(k=3, table_format="short", plots=[])
    out_short = capsys.readouterr().out

    for common in [
        "EXPLORATORY MODEL RESULTS (k=3 — Reference k=4)",
        "exploratory_k",
        "AIC",
        "BIC",
        "SABIC",
        "Class_1",
        "Class_3",
    ]:
        assert common in out_full
        assert common in out_short

    for full_only in ["CAIC", "AvePP", "OCC", "lrt_dof_diff", "N_missing_values"]:
        assert full_only in out_full
        assert full_only not in out_short

    assert len(out_short) < len(out_full)


def test_lca06_report_isolation_after_exploratory_summary(
    model_no_bootstrap, tmp_path: Path, monkeypatch
):
    """
    Test J — Report isolation:
    Calling summary(k=3) must not silently switch the model exported by report().
    report() must continue exporting the reference k=4 model until formal selection
    (LCA-08) is implemented.
    """
    import pandas as pd

    monkeypatch.chdir(tmp_path)
    model_no_bootstrap.summary(k=3, plots=["radar_plot"])
    model_no_bootstrap.report(format=["csv", "png", "pdf"])

    results_csv = next(
        p for p in tmp_path.glob("*_results.csv")
        if not p.name.endswith("_grid_results.csv")
    )
    df_exported = pd.read_csv(results_csv)
    assert len(df_exported) == 4
    assert model_no_bootstrap.fitted_model.n_components == 4
    assert len(list(tmp_path.glob("*.png"))) == 1
    assert len(list(tmp_path.glob("*.pdf"))) == 1


def test_summary_rejects_simultaneous_compare_and_k(
    model_no_bootstrap, synthetic_df, symptom_vars, capsys
):
    """
    Prompt 4.1 (Item 1):
    summary(compare=[...], k=N) must not silently ignore k.
    Verify simultaneous use of compare and k raises a clear ValueError, while
    independent use of compare=[...] and k=N continues to work.
    """
    other_lca = isa.RAPID.create(
        data=synthetic_df,
        model="lca",
        n_clusters=3,
        measurement_vars=symptom_vars,
        random_seed=99,
    )
    other_lca.fit(bootstrap=False, param_grid={"n_clusters": [1, 2, 3]})

    with pytest.raises(ValueError, match="Cannot specify both 'compare' and 'k'"):
        model_no_bootstrap.summary(compare=[other_lca], k=3)

    # Independent compare=[...] still works
    model_no_bootstrap.summary(compare=[other_lca])
    out_cmp = capsys.readouterr().out
    assert "RAPID — MODEL COMPARISON" in out_cmp

    # Independent k=3 still works
    model_no_bootstrap.summary(k=3, plots=[])
    out_k3 = capsys.readouterr().out
    assert "EXPLORATORY MODEL RESULTS (k=3 — Reference k=4)" in out_k3


def test_lca_outcome_computation_categorical_continuous_k1_and_insufficient_obs(
    synthetic_df, symptom_vars, capsys
):
    """
    Prompt 4.1 (Item 2):
    Verify _compute_outcome_for_solution() and outcome summary sections across:
    1. Categorical outcome (chi-square) for reference and exploratory k.
    2. Continuous outcome (>10 unique numeric values -> ANOVA) for reference and exploratory k.
    3. One-class solution (k=1 -> 1-row crosstab, no statistical test, no error).
    4. Insufficient observations / degenerate outcome (single category or <=1 obs per group).
    """
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(42)
    df_out = synthetic_df.copy()
    df_out["cont_score"] = rng.normal(loc=50.0, scale=12.5, size=len(df_out))
    df_out["const_outcome"] = 1

    # 1. Categorical outcome ("severe")
    m_cat = isa.RAPID.create(
        data=df_out,
        model="lca",
        n_clusters=4,
        measurement_vars=symptom_vars,
        outcome="severe",
        random_seed=42,
    )
    m_cat.fit(bootstrap=False, param_grid={"n_clusters": [1, 2, 3, 4]})
    assert m_cat.outcome_test_result is not None
    assert m_cat.outcome_test_result["test"] == "chi-square"
    assert np.isfinite(m_cat.outcome_test_result["statistic"])
    assert 0.0 <= m_cat.outcome_test_result["p_value"] <= 1.0

    ct_k3, test_k3 = m_cat._compute_outcome_for_solution(m_cat.model_objects[3])
    assert len(ct_k3) == 3
    assert np.allclose(ct_k3.sum(axis=1).values, 1.0, atol=1e-3)
    assert test_k3 is not None and test_k3["test"] == "chi-square"
    assert test_k3["dof"] == 2

    # 3. One-class solution (k=1) & 1x2 contingency table with categorical outcome (0/1)
    ct_k1, test_k1 = m_cat._compute_outcome_for_solution(m_cat.model_objects[1])
    assert ct_k1.shape == (1, 2)
    assert test_k1 is None

    m_cat.summary(k=1, plots=[])
    out_k1 = capsys.readouterr().out
    assert "OUTCOME BY CLASS — severe (Exploratory k = 1)" in out_k1
    assert "Statistical test: not applicable (single class or insufficient observations)" in out_k1
    assert "chi-square: statistic=" not in out_k1.split("OUTCOME BY CLASS")[1]

    # Also verify a 1x2 contingency table when a k>=2 model assigns all observations to 1 modal class
    class _SingleModalClassStub:
        n_components = 2

        @staticmethod
        def predict(X):
            return np.zeros(len(X), dtype=int)

    ct_1x2_stub, test_1x2_stub = m_cat._compute_outcome_for_solution(_SingleModalClassStub())
    assert ct_1x2_stub.shape == (1, 2)
    assert test_1x2_stub is None

    # 2. Continuous outcome ("cont_score")
    m_cont = isa.RAPID.create(
        data=df_out,
        model="lca",
        n_clusters=4,
        measurement_vars=symptom_vars,
        outcome="cont_score",
        random_seed=42,
    )
    m_cont.fit(bootstrap=False, param_grid={"n_clusters": [1, 2, 3, 4]})
    assert m_cont.outcome_test_result is not None
    assert m_cont.outcome_test_result["test"] == "ANOVA"
    assert np.isfinite(m_cont.outcome_test_result["statistic"])
    assert 0.0 <= m_cont.outcome_test_result["p_value"] <= 1.0

    ref_anova_before = dict(m_cont.outcome_test_result)
    m_cont.summary(k=2, plots=[])
    out_cont_k2 = capsys.readouterr().out
    assert "OUTCOME BY CLASS — cont_score (Exploratory k = 2)" in out_cont_k2
    assert "ANOVA:" in out_cont_k2
    assert m_cont.outcome_test_result == ref_anova_before

    # 4a. Degenerate categorical outcome (only 1 unique category -> insufficient columns for chi-square)
    m_const = isa.RAPID.create(
        data=df_out,
        model="lca",
        n_clusters=3,
        measurement_vars=symptom_vars,
        outcome="const_outcome",
        random_seed=42,
    )
    m_const.fit(bootstrap=False, param_grid={"n_clusters": [1, 2, 3]})
    assert m_const.outcome_crosstab_df is not None
    assert m_const.outcome_test_result is None

    # 4b. Insufficient continuous observations (all 12 non-NaN continuous values in class 0, 0 in class 1)
    m_cont_sparse = isa.RAPID.create(
        data=df_out,
        model="lca",
        n_clusters=2,
        measurement_vars=symptom_vars,
        outcome="cont_score",
        random_seed=42,
    )
    m_cont_sparse.fit(bootstrap=False, param_grid={"n_clusters": [1, 2]})
    pred_k2 = m_cont_sparse.model_objects[2].predict(m_cont_sparse.X)
    sparse_series = pd.Series(np.nan, index=df_out.index)
    idx_c0 = np.where(pred_k2 == 0)[0][:12]
    sparse_series.iloc[idx_c0] = np.linspace(10.0, 120.0, len(idx_c0))
    m_cont_sparse.outcome_data = sparse_series

    ct_sparse, test_sparse = m_cont_sparse._compute_outcome_for_solution(
        m_cont_sparse.model_objects[2]
    )
    assert ct_sparse is not None
    assert test_sparse is None


def test_lca_sequential_exploration_preserves_decision_data_metrics_and_report(
    synthetic_df, symptom_vars, tmp_path: Path, monkeypatch
):
    """
    Prompt 4.1 (Item 3):
    Confirm that sequential exploratory summaries do not alter the reference model,
    stored decisions, input data X, metrics, model_info, or future report outputs.
    """
    import copy
    import pandas as pd

    model = isa.RAPID.create(
        data=synthetic_df,
        model="lca",
        n_clusters=4,
        measurement_vars=symptom_vars,
        outcome="severe",
        random_seed=42,
    )
    model.fit(bootstrap=False, param_grid={"n_clusters": [1, 2, 3, 4]})
    model.decide(selected=True, k=3, reason="Recorded decision before exploratory summaries.")

    # Snapshot all state before sequential exploration
    fitted_model_before = model.fitted_model
    x_before = model.X.copy(deep=True)
    result_df_before = model.result_df.copy(deep=True)
    grid_before = model.grid_results.copy(deep=True)
    metrics_before = copy.deepcopy(model.metrics)
    model_info_before = copy.deepcopy(model.model_info)
    decision_before = copy.deepcopy(model._decision)

    # Export report before exploratory summaries
    dir_before = tmp_path / "report_before"
    dir_before.mkdir()
    monkeypatch.chdir(dir_before)
    model.summary(plots=["information_criteria", "radar_plot"])
    model.report(format=["csv", "png", "pdf"])

    # Run sequential exploratory summaries across k=1, k=2, k=3
    model.summary(k=1, plots=[])
    model.summary(k=2, plots=["profile_heatmap"])
    model.summary(k=3, table_format="short", plots=["radar_plot"])
    model.summary(plots=["information_criteria", "radar_plot"])

    # Verify reference model, decision, data, and metrics are unchanged
    assert model.fitted_model is fitted_model_before
    assert model.fitted_model.n_components == 4
    pd.testing.assert_frame_equal(model.X, x_before)
    pd.testing.assert_frame_equal(model.result_df, result_df_before)
    pd.testing.assert_frame_equal(model.grid_results, grid_before)
    assert model.metrics == metrics_before
    assert model.model_info == model_info_before
    assert model._decision == decision_before

    # Export report after exploratory summaries and compare exported CSV contents
    dir_after = tmp_path / "report_after"
    dir_after.mkdir()
    monkeypatch.chdir(dir_after)
    model.report(format=["csv", "png", "pdf"])

    for suffix in ["_results.csv", "_grid_results.csv", "_ave_pp.csv"]:
        file_b = next(
            p for p in dir_before.glob(f"*{suffix}")
            if suffix != "_results.csv" or not p.name.endswith("_grid_results.csv")
        )
        file_a = next(
            p for p in dir_after.glob(f"*{suffix}")
            if suffix != "_results.csv" or not p.name.endswith("_grid_results.csv")
        )
        csv_b = pd.read_csv(file_b).drop(columns=["exported_at"], errors="ignore")
        csv_a = pd.read_csv(file_a).drop(columns=["exported_at"], errors="ignore")
        pd.testing.assert_frame_equal(csv_b, csv_a)
    assert len(list(dir_after.glob("*.png"))) == 2
    assert len(list(dir_after.glob("*.pdf"))) == 1


def test_other_model_families_summary_regression(
    synthetic_df, symptom_vars, tmp_path: Path, monkeypatch, capsys
):
    """
    Prompt 4.1 (Item 4) & Prompt 5 (Test O):
    Verify RAPID.summary() and RAPID.report() across all non-LCA model families:
    1. Clustering: KMeans (model='kmeans')
    2. Inferential Regression: LogisticRegression (model='logistic') and GLM (model='glm')
    3. Predictive ML: Lasso (model='lasso')
    4. Tree-based ML: RandomForest (model='random_forest')
    5. Survival Analysis: Survival (model='survival')
    """
    import numpy as np
    import pandas as pd

    shown = []
    monkeypatch.setattr(go.Figure, "show", lambda self, *a, **kw: shown.append(self))

    rng = np.random.default_rng(42)
    df_multi = synthetic_df.copy()
    df_multi["AVAL"] = rng.uniform(1.0, 30.0, size=len(df_multi))
    df_multi["event"] = df_multi["severe"].astype(int)
    preds = ["Fever", "Myalgia", "Headache", "Rash"]

    models_to_test = [
        ("kmeans", isa.RAPID.create(data=df_multi, model="kmeans", predictors=preds, n_clusters=3, random_seed=42)),
        ("logistic", isa.RAPID.create(data=df_multi, model="logistic", dependent_var="severe", independent_vars=preds)),
        ("glm", isa.RAPID.create(data=df_multi, model="glm", dependent_var="AVAL", independent_vars=preds, family="gaussian")),
        ("lasso", isa.RAPID.create(data=df_multi, model="lasso", dependent_var="severe", independent_vars=preds, random_seed=42)),
        ("random_forest", isa.RAPID.create(data=df_multi, model="random_forest", dependent_var="severe", independent_vars=preds, n_estimators=10, random_seed=42)),
        ("survival", isa.RAPID.create(data=df_multi, model="survival", duration_var="AVAL", event_var="event", independent_vars=preds)),
    ]

    for name, mdl in models_to_test:
        # Calling summary() before fit() must raise state error
        with pytest.raises(ValueError, match="state 'created'"):
            mdl.summary()

        mdl.fit()
        mdl.summary(table_format="full")
        out_full = capsys.readouterr().out
        assert "MODEL INFORMATION" in out_full
        assert "RESULTS" in out_full
        assert "BINARY INDICATOR FREQUENCY TABLE" not in out_full

        mdl.summary(table_format="short")
        out_short = capsys.readouterr().out
        assert "RESULTS" in out_short

        # Parameter k is rejected for non-LCA families
        with pytest.raises(ValueError, match="only supported for LCA"):
            mdl.summary(k=2)

        # Verify report(format=['csv', 'pdf']) preserves legacy contract for non-LCA models
        fam_dir = tmp_path / f"report_{name}"
        fam_dir.mkdir()
        monkeypatch.chdir(fam_dir)
        mdl.summary(plots=[])
        mdl.report(format=["csv", "pdf"])
        capsys.readouterr()

        res_csv = next(fam_dir.glob("*_results.csv"))
        df_res = pd.read_csv(res_csv)
        assert list(df_res.columns[:3]) == ["model_id", "model_type", "exported_at"]
        assert "reference_k" not in df_res.columns
        assert "selected_k" not in df_res.columns
        assert len(list(fam_dir.glob("*.pdf"))) == 1

    assert len(shown) == 0


# ============================================================================
# LCA-08: FORMAL MODEL SELECTION, REPORT INTEGRATION & PERSISTENCE TESTS
# ============================================================================

def test_pending_decide_k_updates_active_solution(synthetic_df, symptom_vars):
    """
    LCA-08 (formerly XFAIL 1) — Formal selection via decide(selected=True, k=3):
    Verifies that decide(selected=True, k=3) records the formal selection and
    resolves k=3 as the active report solution (_resolve_report_solution) while
    preserving the original fitted reference model (k=4) and all enumerated
    solutions in model_objects.
    """
    import numpy as np

    model = isa.RAPID.create(
        data=synthetic_df,
        model="lca",
        n_clusters=4,
        measurement_vars=symptom_vars,
        random_seed=42,
    )
    model.fit(bootstrap=False, param_grid={"n_clusters": [1, 2, 3, 4]})
    ref_fitted_before = model.fitted_model

    model.decide(
        selected=True,
        k=np.int64(3),
        reason="Selected k=3 after reviewing enumeration.",
        notes="Verified with clinical team.",
    )

    # 1. Decision metadata accurately recorded
    assert model._decision is not None
    assert model._decision["selected"] is True
    assert model._decision["k"] == 3
    assert isinstance(model._decision["k"], int)
    assert model._decision["reference_k"] == 4
    assert model._decision["reason"] == "Selected k=3 after reviewing enumeration."
    assert model._decision["notes"] == "Verified with clinical team."
    assert "timestamp" in model._decision

    # 2. Original reference model and all enumerated solutions are preserved
    assert model.fitted_model is ref_fitted_before
    assert model.fitted_model.n_components == 4
    assert len(model.result_df) == 4
    assert sorted(model.model_objects.keys()) == [1, 2, 3, 4]

    # 3. Resolved report solution uses the formally selected k=3
    rep_sol = model._resolve_report_solution()
    assert rep_sol["k"] == 3
    assert rep_sol["ref_k"] == 4
    assert rep_sol["selected_k"] == 3
    assert rep_sol["is_selected"] is True
    assert rep_sol["solution_role"] == "selected"
    assert rep_sol["decision_status"] == "selected_k"
    assert rep_sol["model"] is model.model_objects[3]
    assert len(rep_sol["profiles_df"]) == 3
    assert len(rep_sol["prevalence_df"]) == 3
    grid_k3 = model.grid_results.set_index("n_clusters").loc[3]
    assert np.isclose(rep_sol["metrics"]["aic"], grid_k3["AIC"], rtol=1e-12)
    assert np.isclose(rep_sol["metrics"]["bic"], grid_k3["BIC"], rtol=1e-12)


def test_pending_report_after_decide_k_exports_chosen_k_profiles(
    synthetic_df, symptom_vars, tmp_path: Path, monkeypatch
):
    """
    LCA-08 (formerly XFAIL 2) — CSV export after decide(selected=True, k=3):
    Verifies that calling model.report(format=['csv']) after decide(selected=True, k=3)
    exports the 3-class profile table, full grid_results, and k=3 diagnostics/outcomes.
    """
    import numpy as np
    import pandas as pd

    monkeypatch.chdir(tmp_path)
    model = isa.RAPID.create(
        data=synthetic_df,
        model="lca",
        n_clusters=4,
        measurement_vars=symptom_vars,
        outcome="severe",
        random_seed=42,
    )
    model.fit(bootstrap=False, param_grid={"n_clusters": [1, 2, 3, 4]})
    model.decide(selected=True, k=3, reason="Selected k=3 for final report.")
    model.report(format=["csv"])

    results_csv = next(
        p for p in tmp_path.glob("*_results.csv")
        if not p.name.endswith("_grid_results.csv")
    )
    df_exported = pd.read_csv(results_csv)
    assert len(df_exported) == 3
    assert list(df_exported["Class"]) == ["Class_1", "Class_2", "Class_3"]
    assert (df_exported["reference_k"] == 4).all()
    assert (df_exported["selected_k"] == 3).all()
    assert (df_exported["decision_status"] == "selected_k").all()

    # Verify exported profiles match k=3 fitted model exactly
    expected_k3_profiles = model._resolve_k_solution(3, mode="selected")["profiles_df"]
    for col in symptom_vars:
        assert np.allclose(df_exported[col].values, expected_k3_profiles[col].values, rtol=1e-12)

    # Grid results CSV retains all 4 enumerated k solutions
    grid_csv = next(tmp_path.glob("*_grid_results.csv"))
    df_grid = pd.read_csv(grid_csv)
    assert list(df_grid["n_clusters"]) == [1, 2, 3, 4]

    # Outcome crosstab CSV has 3 rows for selected k=3
    outcome_csv = next(tmp_path.glob("*_outcome_crosstab.csv"))
    df_outcome = pd.read_csv(outcome_csv)
    assert len(df_outcome) == 3
    assert list(df_outcome["class"]) == ["Class_1", "Class_2", "Class_3"]

    # Class diagnostics CSV has 3 rows for reported_k=3, while ave_pp CSV retains full enumeration
    diag_csv = next(tmp_path.glob("*_class_diagnostics.csv"))
    df_diag = pd.read_csv(diag_csv)
    assert len(df_diag) == 3
    assert (df_diag["reported_k"] == 3).all()

    ave_pp_csv = next(tmp_path.glob("*_ave_pp.csv"))
    df_ave_pp = pd.read_csv(ave_pp_csv)
    assert len(df_ave_pp) == 4


def test_lca08_png_and_plot_consistency_for_selected_k(
    synthetic_df, symptom_vars, tmp_path: Path, monkeypatch
):
    """
    Test C — PNG & plot consistency for selected k=3 (Plotly and Matplotlib):
    Verifies radar_plot, profile_heatmap, cluster_distribution, and crosstab reflect
    Selected k=3 while enumeration plots retain all fitted k=[1, 2, 3, 4].
    """
    import matplotlib.pyplot as plt
    import numpy as np

    model = isa.RAPID.create(
        data=synthetic_df,
        model="lca",
        n_clusters=4,
        measurement_vars=symptom_vars,
        outcome="severe",
        random_seed=42,
    )
    model.fit(bootstrap=False, param_grid={"n_clusters": [1, 2, 3, 4]})
    model.decide(selected=True, k=3, reason="Selected k=3")

    # Plotly & Matplotlib via _render_report_plot
    radar_pl = model._render_report_plot("radar_plot", backend="plotly")
    assert len(radar_pl.data) == 3
    assert "Selected k=3" in radar_pl.layout.title.text

    radar_mpl = model._render_report_plot("radar_plot", backend="matplotlib")
    assert "Selected k=3" in radar_mpl.axes[0].get_title()
    _, labels = radar_mpl.axes[0].get_legend_handles_labels()
    assert labels == ["Class_1", "Class_2", "Class_3"]
    plt.close(radar_mpl)

    hm_pl = model._render_report_plot("profile_heatmap", backend="plotly")
    assert np.asarray(hm_pl.data[0].z).shape == (3, len(symptom_vars))
    assert "Selected k=3" in hm_pl.layout.title.text

    hm_mpl = model._render_report_plot("profile_heatmap", backend="matplotlib")
    assert len(hm_mpl.axes[0].get_yticks()) == 3
    assert "Selected k=3" in hm_mpl.axes[0].get_title()
    plt.close(hm_mpl)

    dist_pl = model._render_report_plot("cluster_distribution", backend="plotly")
    assert list(dist_pl.data[0].x) == ["Class_1", "Class_2", "Class_3"]
    assert "Selected k=3" in dist_pl.layout.title.text

    ct_pl = model._render_report_plot("crosstab", backend="plotly")
    assert np.asarray(ct_pl.data[0].z).shape[0] == 3
    assert "Selected k=3" in ct_pl.layout.title.text

    ic_pl = model._render_report_plot("information_criteria", backend="plotly")
    assert list(ic_pl.data[0].x) == [1, 2, 3, 4]

    # Export PNGs via report(format=["png"])
    monkeypatch.chdir(tmp_path)
    model.summary(plots=["radar_plot", "profile_heatmap"])
    model.report(format=["png"])
    png_files = sorted(p.name for p in tmp_path.glob("*.png"))
    assert len(png_files) == 2
    assert any("radar_plot" in f for f in png_files)
    assert any("profile_heatmap" in f for f in png_files)


def test_lca08_pdf_consistency_before_and_after_decide(
    synthetic_df, symptom_vars, tmp_path: Path, monkeypatch
):
    """
    Test D & E — PDF report consistency before and after decide(selected=True, k=3):
    Verifies programmatic PDF story contents when undecided (Reference k=4) vs
    after selecting k=3 (Selected k=3).
    """
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Table

    captured_stories = []
    orig_build = SimpleDocTemplate.build

    def _spy_build(self, flowables, *args, **kwargs):
        captured_stories.append(list(flowables))
        return orig_build(self, flowables, *args, **kwargs)

    monkeypatch.setattr(SimpleDocTemplate, "build", _spy_build)
    monkeypatch.chdir(tmp_path)

    model = isa.RAPID.create(
        data=synthetic_df,
        model="lca",
        n_clusters=4,
        measurement_vars=symptom_vars,
        outcome="severe",
        random_seed=42,
    )
    model.fit(bootstrap=False, param_grid={"n_clusters": [1, 2, 3, 4]})

    # 1. PDF before decide() -> Reference k=4
    model.summary(plots=["radar_plot"])
    model.report(format=["pdf"])
    story_before = captured_stories[-1]
    paras_before = [f.getPlainText() for f in story_before if isinstance(f, Paragraph)]
    assert "Fit Metrics (Reference k=4)" in paras_before
    assert "Parameter Table — Item-Response Probabilities (Reference k=4)" in paras_before
    assert "Selected: None (No formal decision recorded)" in paras_before

    # Locate Parameter Table in story_before -> header + 4 class rows = 5 rows
    param_idx_before = next(
        i for i, f in enumerate(story_before)
        if isinstance(f, Paragraph) and "Parameter Table" in f.getPlainText()
    )
    param_tbl_before = next(f for f in story_before[param_idx_before + 1:] if isinstance(f, Table))
    assert len(param_tbl_before._cellvalues) == 5  # 1 header + 4 classes

    # 2. PDF after decide(selected=True, k=3) -> Selected k=3
    model.decide(
        selected=True,
        k=3,
        reason="Best balance between BIC and clinical interpretability",
    )
    model.report(format=["pdf"])
    story_after = captured_stories[-1]
    paras_after = [f.getPlainText() for f in story_after if isinstance(f, Paragraph)]

    assert "Fit Metrics (Selected k=3)" in paras_after
    assert "Parameter Table — Item-Response Probabilities (Selected k=3)" in paras_after
    assert "Class Sizes & Classification Diagnostics (Selected k=3)" in paras_after
    assert "Outcome by Class — severe (Selected k=3)" in paras_after
    assert "Class Enumeration Results" in paras_after
    assert "Selected: True" in paras_after
    assert "Reference k: 4" in paras_after
    assert "Selected k: 3" in paras_after
    assert "Reason: Best balance between BIC and clinical interpretability" in paras_after

    # Verify Fit Metrics table in PDF contains k=3 AIC (1999.2995), not k=4 AIC (2008.6805)
    fit_idx_after = next(
        i for i, f in enumerate(story_after)
        if isinstance(f, Paragraph) and f.getPlainText() == "Fit Metrics (Selected k=3)"
    )
    fit_tbl_after = next(f for f in story_after[fit_idx_after + 1:] if isinstance(f, Table))
    fit_kv = dict(fit_tbl_after._cellvalues[1:])
    assert fit_kv["aic"] == "1999.2995"
    assert fit_kv["bic"] == "2122.7356"

    # Verify Parameter Table in PDF has 1 header + 3 class rows = 4 rows
    param_idx_after = next(
        i for i, f in enumerate(story_after)
        if isinstance(f, Paragraph) and "Parameter Table" in f.getPlainText()
    )
    param_tbl_after = next(f for f in story_after[param_idx_after + 1:] if isinstance(f, Table))
    assert len(param_tbl_after._cellvalues) == 4  # 1 header + 3 classes
    assert [r[0] for r in param_tbl_after._cellvalues[1:]] == ["Class_1", "Class_2", "Class_3"]

    # Verify Class Enumeration table still has 1 header + 4 rows = 5 rows
    enum_idx_after = next(
        i for i, f in enumerate(story_after)
        if isinstance(f, Paragraph) and f.getPlainText() == "Class Enumeration Results"
    )
    enum_tbl_after = next(f for f in story_after[enum_idx_after + 1:] if isinstance(f, Table))
    assert len(enum_tbl_after._cellvalues) == 5


def test_lca08_repeated_decisions_rejection_and_k_none_fallback(
    synthetic_df, symptom_vars, tmp_path: Path, monkeypatch
):
    """
    Test G & Section 4 — Repeated decisions, selected=False rejection, and k=None:
    1. decide(selected=True, k=2) -> report exports 2 classes.
    2. decide(selected=True, k=3) -> report exports 3 classes (overwrites k=2 cleanly).
    3. decide(selected=False, k=3) -> report falls back to Reference k=4 (decision_status='rejected').
    4. decide(selected=True, reason='...') without k -> report uses Reference k=4 (decision_status='selected_reference').
    """
    import pandas as pd

    model = isa.RAPID.create(
        data=synthetic_df,
        model="lca",
        n_clusters=4,
        measurement_vars=symptom_vars,
        random_seed=42,
    )
    model.fit(bootstrap=False, param_grid={"n_clusters": [1, 2, 3, 4]})

    def _export_and_read(subdir_name: str) -> pd.DataFrame:
        d = tmp_path / subdir_name
        d.mkdir()
        monkeypatch.chdir(d)
        model.report(format=["csv"])
        res_csv = next(
            p for p in d.glob("*_results.csv")
            if not p.name.endswith("_grid_results.csv")
        )
        return pd.read_csv(res_csv)

    # 1. Decide k=2
    model.decide(selected=True, k=2, reason="Initial k=2")
    df_k2 = _export_and_read("step1_k2")
    assert len(df_k2) == 2
    assert (df_k2["selected_k"] == 2).all()

    # 2. Revise decision to k=3
    model.decide(selected=True, k=3, reason="Revised to k=3")
    df_k3 = _export_and_read("step2_k3")
    assert len(df_k3) == 3
    assert (df_k3["selected_k"] == 3).all()
    assert model._decision["reason"] == "Revised to k=3"

    # 3. Reject k=3 (selected=False) -> falls back to reference k=4
    model.decide(selected=False, k=3, reason="Rejected k=3 after clinical review")
    df_rej = _export_and_read("step3_rejected")
    assert len(df_rej) == 4
    assert df_rej["selected_k"].isna().all()
    assert (df_rej["decision_status"] == "rejected").all()

    # 4. Decide selected=True without k -> falls back to reference k=4 without inventing selected_k
    model.decide(selected=True, reason="Accepted reference model")
    df_ref = _export_and_read("step4_no_k")
    assert len(df_ref) == 4
    assert df_ref["selected_k"].isna().all()
    assert (df_ref["decision_status"] == "selected_reference").all()


def test_lca08_invalid_decide_parameters_and_refit_invalidation(
    synthetic_df, symptom_vars
):
    """
    Test H & I — Validation of invalid decide() arguments and stale decision
    invalidation on re-fit (both when new range excludes and includes old k).
    """
    m_unfitted = isa.RAPID.create(
        data=synthetic_df,
        model="lca",
        n_clusters=3,
        measurement_vars=symptom_vars,
    )
    with pytest.raises(ValueError, match="state 'created'"):
        m_unfitted.decide(selected=True, k=2)

    model = isa.RAPID.create(
        data=synthetic_df,
        model="lca",
        n_clusters=4,
        measurement_vars=symptom_vars,
        random_seed=42,
    )
    model.fit(bootstrap=False, param_grid={"n_clusters": [1, 2, 4]})

    # Unavailable k values (k=99, k=0, k=-1, unfitted k=3)
    for bad_val in [99, 0, -1, 3]:
        with pytest.raises(ValueError, match=r"not available.*\[1, 2, 4\]"):
            model.decide(selected=True, k=bad_val)

    # Invalid types for k (bool, str, float, list)
    for bad_type in [True, False, "2", 2.5, [1, 2]]:
        with pytest.raises(ValueError, match="must be an integer"):
            model.decide(selected=True, k=bad_type)

    # Invalid type for selected
    with pytest.raises(ValueError, match="Parameter 'selected' must be a boolean"):
        model.decide(selected="yes", k=2)

    # Re-fit invalidation Case 1: old selected k=4 is not in new param_grid [1, 2]
    model.decide(selected=True, k=4, reason="Selected k=4")
    assert model._decision is not None
    model.fit(bootstrap=False, param_grid={"n_clusters": [1, 2]})
    assert model._decision is None
    assert model._resolve_report_solution()["decision_status"] == "undecided"
    assert model._resolve_report_solution()["k"] == 1

    # Re-fit invalidation Case 2: old selected k=2 IS in new param_grid [1, 2, 3], still invalidated
    model.decide(selected=True, k=2, reason="Selected k=2")
    assert model._decision is not None
    model.fit(bootstrap=False, param_grid={"n_clusters": [1, 2, 3]})
    assert model._decision is None
    assert model._resolve_report_solution()["decision_status"] == "undecided"


def test_lca08_explicit_plot_selection_k1_and_outcome_in_report(
    synthetic_df, symptom_vars, tmp_path: Path, monkeypatch
):
    """
    Test J, K & L — Explicit empty plots with decide(), k=1 formal selection,
    and continuous/categorical outcome reporting for selected k.
    """
    import numpy as np
    import pandas as pd
    from reportlab.platypus import Image, Paragraph, SimpleDocTemplate

    captured_stories = []
    orig_build = SimpleDocTemplate.build

    def _spy_build(self, flowables, *args, **kwargs):
        captured_stories.append(list(flowables))
        return orig_build(self, flowables, *args, **kwargs)

    monkeypatch.setattr(SimpleDocTemplate, "build", _spy_build)

    rng = np.random.default_rng(42)
    df_out = synthetic_df.copy()
    df_out["cont_score"] = rng.normal(loc=50.0, scale=10.0, size=len(df_out))

    model = isa.RAPID.create(
        data=df_out,
        model="lca",
        n_clusters=4,
        measurement_vars=symptom_vars,
        outcome="cont_score",
        random_seed=42,
    )
    model.fit(bootstrap=False, param_grid={"n_clusters": [1, 2, 3, 4]})

    # Test J: summary(plots=[]) -> decide(selected=True, k=3) -> report(format=["png", "pdf"]) exports 0 figures
    dir_no_plots = tmp_path / "no_plots_k3"
    dir_no_plots.mkdir()
    monkeypatch.chdir(dir_no_plots)
    model.summary(plots=[])
    model.decide(selected=True, k=3, reason="k=3 with continuous outcome and no plots")
    model.report(format=["csv", "png", "pdf"])

    assert len(list(dir_no_plots.glob("*.png"))) == 0
    story_no_plots = captured_stories[-1]
    assert len([f for f in story_no_plots if isinstance(f, Image)]) == 0
    paras_no_plots = [f.getPlainText() for f in story_no_plots if isinstance(f, Paragraph)]
    assert "Figures" not in paras_no_plots
    assert "Outcome by Class — cont_score (Selected k=3)" in paras_no_plots
    assert any("ANOVA:" in p for p in paras_no_plots)

    # Test L: Formal selection of k=1
    dir_k1 = tmp_path / "selected_k1"
    dir_k1.mkdir()
    monkeypatch.chdir(dir_k1)
    model.summary(plots=["radar_plot"])
    model.decide(selected=True, k=1, reason="Single class selected")
    model.report(format=["csv", "png", "pdf"])

    res_k1 = pd.read_csv(
        next(p for p in dir_k1.glob("*_results.csv") if not p.name.endswith("_grid_results.csv"))
    )
    assert len(res_k1) == 1
    assert (res_k1["selected_k"] == 1).all()
    assert len(list(dir_k1.glob("*.png"))) == 1

    story_k1 = captured_stories[-1]
    paras_k1 = [f.getPlainText() for f in story_k1 if isinstance(f, Paragraph)]
    assert "Fit Metrics (Selected k=1)" in paras_k1
    assert "Selected k: 1" in paras_k1
    assert any("not applicable" in p for p in paras_k1)


def test_lca08_multi_model_report_save_persistence_and_no_recalculation(
    model_bootstrap, synthetic_df, symptom_vars, tmp_path: Path, monkeypatch
):
    """
    Test M, N & Section 9 — Multi-model report isolation, save() persistence
    consistency, and zero statistical/bootstrap recalculation during decide() & report().
    """
    import json
    import pickle
    import zipfile
    import pandas as pd
    from isaric.modelevaluation import gridsearch as gs_mod
    from stepmix.stepmix import StepMix

    # Forbid any StepMix fitting or bootstrap recalculation during decide(), report(), and save()
    def _forbid_fit_or_boot(*args, **kwargs):
        raise AssertionError("No re-fitting or bootstrap recalculation is allowed during decide/report/save")

    model_b = isa.RAPID.create(
        data=synthetic_df,
        model="lca",
        n_clusters=4,
        measurement_vars=symptom_vars,
        random_seed=99,
    )
    model_b.fit(bootstrap=False, param_grid={"n_clusters": [1, 2, 3, 4]})

    monkeypatch.setattr(StepMix, "fit", _forbid_fit_or_boot)
    monkeypatch.setattr(gs_mod, "bootstrap_stepmix", _forbid_fit_or_boot)
    monkeypatch.setattr(gs_mod, "bootstrap_ci_stepmix", _forbid_fit_or_boot)
    monkeypatch.setattr(gs_mod, "bootstrap_lrt_stepmix", _forbid_fit_or_boot)

    grid_a_before = model_bootstrap.grid_results.copy(deep=True)
    ci_a_before = model_bootstrap.lrt_results["ci_bootstrap"].copy(deep=True)

    # Model A selects k=3; Model B selects k=2
    model_bootstrap.decide(selected=True, k=3, reason="Model A selected k=3")
    model_b.decide(selected=True, k=2, reason="Model B selected k=2")

    monkeypatch.chdir(tmp_path)
    model_bootstrap.report(models=[model_bootstrap, model_b], format=["csv", "pdf"])

    # Verify independent CSV exports for Model A (3 rows) and Model B (2 rows)
    res_a = pd.read_csv(
        next(
            p for p in tmp_path.glob(f"{model_bootstrap.model_id}*_results.csv")
            if not p.name.endswith("_grid_results.csv")
        )
    )
    res_b = pd.read_csv(
        next(
            p for p in tmp_path.glob(f"{model_b.model_id}*_results.csv")
            if not p.name.endswith("_grid_results.csv")
        )
    )
    assert len(res_a) == 3
    assert (res_a["selected_k"] == 3).all()
    assert len(res_b) == 2
    assert (res_b["selected_k"] == 2).all()
    assert len(list(tmp_path.glob("*.pdf"))) == 1

    # Verify save() persists the selected k=3 estimator and unambiguous metadata
    model_bootstrap.save()
    saved_zip = tmp_path / model_bootstrap.saved_filename
    assert saved_zip.exists()
    with zipfile.ZipFile(saved_zip, "r") as zf:
        meta = json.loads(zf.read("metadata.json").decode("utf-8"))
        saved_est = pickle.loads(zf.read("model.pkl"))

    assert meta["reference_k"] == 4
    assert meta["selected_k"] == 3
    assert meta["serialized_model_k"] == 3
    assert meta["serialized_model_role"] == "selected"
    assert meta["decision_status"] == "selected_k"
    assert saved_est.n_components == 3
    # Live reference estimator remains k=4
    assert model_bootstrap.fitted_model.n_components == 4

    # Verify numerical invariance of grid_results and ci_bootstrap
    pd.testing.assert_frame_equal(model_bootstrap.grid_results, grid_a_before)
    pd.testing.assert_frame_equal(model_bootstrap.lrt_results["ci_bootstrap"], ci_a_before)

    # Restore model_bootstrap._decision to None so session fixture stays clean
    model_bootstrap._decision = None
