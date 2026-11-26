"""
Scenario A — Fast smoke tests for LCA without bootstrap (k = [1, 2, 3, 4]).

Verifies:
- Local package import origin
- Synthetic binary data generator
- LCA object creation and pipeline state transition (created -> fitted)
- Independence of separate model instances
- grid_results availability, expected k=[1, 2, 3, 4], and information criteria
- Presence of fitted StepMix objects in model_objects for each k
- Execution of summary() and available non-bootstrap plots (Plotly & Matplotlib)
- Known issues LCA-01, LCA-02, and LCA-03 (baseline + xfail)
"""

from pathlib import Path
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import pytest

import isaric as isa
from isaric.modeling.clustering import LCA
from stepmix.stepmix import StepMix
from tests.lca.synthetic_data import generate_synthetic_lca_data


def test_local_package_import_origin(repo_root: Path):
    """Verify that `import isaric` loads from the local repository src/isaric."""
    expected_init = (repo_root / "src" / "isaric" / "__init__.py").resolve()
    actual_init = Path(isa.__file__).resolve()
    assert actual_init == expected_init
    assert isa.__version__ == "0.2.0"


def test_synthetic_data_generator_reproducibility(symptom_vars):
    """Verify synthetic binary symptom data generator is binary and deterministic."""
    df1 = generate_synthetic_lca_data(n_samples=120, seed=42, symptom_vars=symptom_vars)
    df2 = generate_synthetic_lca_data(n_samples=120, seed=42, symptom_vars=symptom_vars)

    assert df1.shape == (120, len(symptom_vars) + 1)
    pd.testing.assert_frame_equal(df1, df2)

    for col in symptom_vars:
        unique_vals = set(df1[col].unique())
        assert unique_vals.issubset({0, 1}), f"Column {col} is not binary: {unique_vals}"
        assert len(unique_vals) == 2, f"Column {col} has zero variance"


def test_lca_creation_and_initial_state(synthetic_df, symptom_vars):
    """Verify LCA object creation and initial 'created' state."""
    model = isa.RAPID.create(
        data=synthetic_df,
        model="lca",
        n_clusters=4,
        measurement_vars=symptom_vars,
        random_seed=42,
    )
    assert isinstance(model, LCA)
    assert model.model_type == "lca"
    assert model._state == "created"
    assert model.fitted_model is None
    assert model.result_df is None
    assert model.model_id.startswith("lca-")


def test_lca_fit_without_bootstrap_state_and_attributes(model_no_bootstrap, symptom_vars):
    """Verify fit(bootstrap=False) completes and transitions state to 'fitted'."""
    assert model_no_bootstrap._state == "fitted"
    assert model_no_bootstrap.fitted_model is not None
    assert isinstance(model_no_bootstrap.fitted_model, StepMix)
    assert model_no_bootstrap.result_df is not None
    assert isinstance(model_no_bootstrap.result_df, pd.DataFrame)
    assert list(model_no_bootstrap.result_df.columns) == symptom_vars
    assert len(model_no_bootstrap.result_df) == 4


def test_lca_grid_results_k_values_and_information_criteria(model_no_bootstrap):
    """Verify grid_results contains k=[1, 2, 3, 4] and valid information criteria."""
    grid = model_no_bootstrap.grid_results
    assert isinstance(grid, pd.DataFrame)
    assert list(grid["n_clusters"]) == [1, 2, 3, 4]

    expected_cols = {
        "n_clusters",
        "LL",
        "AIC",
        "BIC",
        "CAIC",
        "SABIC",
        "entropy",
        "relative_entropy",
        "ave_pp_min",
        "occ_k_min",
        "min_mm_f",
        "min_class_perc",
        "n_par",
        "dof",
        "converged",
    }
    assert expected_cols.issubset(set(grid.columns))

    # Information criteria must be finite numbers and mathematically consistent with LL
    n_samples = len(model_no_bootstrap.X)
    for _, row in grid.iterrows():
        k = int(row["n_clusters"])
        ll = float(row["LL"])
        npar = int(row["n_par"])

        assert np.isfinite(ll) and ll < 0
        assert np.isclose(row["AIC"], -2 * ll + 2 * npar, rtol=1e-5)
        assert np.isclose(row["BIC"], -2 * ll + npar * np.log(n_samples), rtol=1e-5)
        assert np.isclose(row["CAIC"], -2 * ll + npar * (np.log(n_samples) + 1), rtol=1e-5)
        assert np.isfinite(row["SABIC"])
        assert np.isfinite(row["entropy"]) and row["entropy"] >= 0.0

        if k == 1:
            assert np.isnan(row["relative_entropy"])
        else:
            assert 0.0 <= float(row["relative_entropy"]) <= 1.0

    # Explicit baseline numerical regression check (synthetic_df, n=150, seed=42)
    baseline_expected = {
        1: {"LL": -1074.978215, "AIC": 2175.956431, "BIC": 2215.094690, "CAIC": 2228.094690, "SABIC": 2239.090437},
        2: {"LL": -993.678813, "AIC": 2041.357627, "BIC": 2122.644780, "CAIC": 2149.644780, "SABIC": 2172.482101},
        3: {"LL": -958.649763, "AIC": 1999.299525, "BIC": 2122.735572, "CAIC": 2163.735572, "SABIC": 2198.414466},
        4: {"LL": -949.340247, "AIC": 2008.680494, "BIC": 2174.265436, "CAIC": 2229.265436, "SABIC": 2275.785904},
    }
    grid_by_k = grid.set_index("n_clusters")
    for k, exp_metrics in baseline_expected.items():
        for metric_name, exp_val in exp_metrics.items():
            assert np.isclose(
                grid_by_k.loc[k, metric_name], exp_val, rtol=1e-6, atol=1e-5
            ), f"Regression mismatch for k={k}, {metric_name}"


def test_lca_model_objects_present_for_all_k(model_no_bootstrap):
    """Verify fitted StepMix instances exist in model_objects for every k=[1, 2, 3, 4]."""
    assert hasattr(model_no_bootstrap, "model_objects")
    assert sorted(model_no_bootstrap.model_objects.keys()) == [1, 2, 3, 4]

    for k, stepmix_obj in model_no_bootstrap.model_objects.items():
        assert isinstance(stepmix_obj, StepMix)
        assert stepmix_obj.n_components == k
        assert hasattr(stepmix_obj, "converged_")
        probs = stepmix_obj.predict_proba(model_no_bootstrap.X)
        assert probs.shape == (len(model_no_bootstrap.X), k)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_lca_separate_model_instances_do_not_overwrite(synthetic_df, symptom_vars):
    """
    Verify Section 4 requirement: separate objects (model_no_bootstrap and
    model_bootstrap) remain independent and do not overwrite each other.
    """
    m_no_boot = isa.RAPID.create(
        data=synthetic_df,
        model="lca",
        n_clusters=3,
        measurement_vars=symptom_vars,
        random_seed=42,
    )
    m_no_boot.fit(bootstrap=False, param_grid={"n_clusters": [1, 2, 3]})
    grid_before = m_no_boot.grid_results.copy()

    m_boot = isa.RAPID.create(
        data=synthetic_df,
        model="lca",
        n_clusters=2,
        measurement_vars=symptom_vars,
        random_seed=42,
    )
    m_boot.fit(bootstrap=True, n_iterations=5, param_grid={"n_clusters": [1, 2]})

    assert m_no_boot is not m_boot
    assert list(m_no_boot.grid_results["n_clusters"]) == [1, 2, 3]
    assert list(m_boot.grid_results["n_clusters"]) == [1, 2]
    assert m_no_boot.lrt_results["blrt"] is None
    assert m_boot.lrt_results["blrt"] is not None
    pd.testing.assert_frame_equal(m_no_boot.grid_results, grid_before)


def test_lca_summary_execution_without_bootstrap(model_no_bootstrap, capsys):
    """Verify summary() executes and outputs MODEL INFORMATION, RESULTS, and GRID SEARCH RESULTS."""
    model_no_bootstrap.summary()
    captured = capsys.readouterr().out

    assert "MODEL INFORMATION" in captured
    assert "RESULTS" in captured
    assert "GRID SEARCH RESULTS" in captured


@pytest.mark.parametrize(
    "plot_name",
    [
        "profile_heatmap",
        "cluster_distribution",
        "radar_plot",
        "information_criteria",
        "ll_curve",
        "entropy_curve",
        "relative_entropy_curve",
    ],
)
@pytest.mark.parametrize("backend", ["plotly", "matplotlib"])
def test_lca_non_bootstrap_plots_execution(model_no_bootstrap, plot_name, backend):
    """Verify all non-bootstrap plots execute in both Plotly and Matplotlib backends."""
    assert plot_name in model_no_bootstrap.plots_map
    fig = model_no_bootstrap.plots_map[plot_name](backend=backend)
    assert fig is not None

    if backend == "plotly":
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1
    else:
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)


# ============================================================================
# LCA-01 AND LCA-02 VERIFICATION & REMAINING SCENARIO A XFAIL TARGETS
# ============================================================================

def test_baseline_lca01_lrt_currently_in_model_info(
    model_no_bootstrap, synthetic_df, symptom_vars, capsys
):
    """
    Verify LCA-01 contract:
    - model_info no longer contains detailed 'lrt_nested' or 'lrt_blrt' lists.
    - Internal structures (nested_models_lrt, lrt_results['nested']) are preserved.
    - summary() displays LRT columns in the enumeration table.
    - Non-consecutive k values do not invent comparison results.
    """
    assert "lrt_nested" not in model_no_bootstrap.model_info
    assert "lrt_blrt" not in model_no_bootstrap.model_info

    # Internal structure preserved
    nested_df = model_no_bootstrap.lrt_results.get("nested")
    assert isinstance(nested_df, pd.DataFrame)
    assert list(nested_df["n_clusters"]) == [2, 3, 4]

    # summary() displays LRT results in GRID SEARCH RESULTS
    model_no_bootstrap.summary()
    out = capsys.readouterr().out
    assert "lrt_statistic" in out
    assert "lrt_dof_diff" in out
    assert "lrt_p_value" in out

    # Non-consecutive k=[1, 2, 4]: k=4 has no k=3 predecessor, so LRT must be NaN
    m_gap = isa.RAPID.create(
        data=synthetic_df,
        model="lca",
        n_clusters=4,
        measurement_vars=symptom_vars,
        random_seed=42,
    )
    m_gap.fit(bootstrap=False, param_grid={"n_clusters": [1, 2, 4]})
    g_gap = m_gap.grid_results.set_index("n_clusters")
    assert list(g_gap.index) == [1, 2, 4]
    assert np.isnan(g_gap.loc[1, "lrt_p_value"])
    assert np.isfinite(g_gap.loc[2, "lrt_p_value"])
    assert np.isnan(g_gap.loc[4, "lrt_p_value"])
    assert np.isnan(g_gap.loc[4, "lrt_statistic"])


def test_xfail_lca01_lrt_in_grid_results(model_no_bootstrap):
    """
    Verify LCA-01: Nested LRT statistics are integrated into grid_results by n_clusters,
    k=1 has NaN, numerical values match nested_models_lrt, and no artificial BLRT is present.
    """
    grid = model_no_bootstrap.grid_results
    assert len(grid) == 4
    assert list(grid["n_clusters"]) == [1, 2, 3, 4]
    assert grid["n_clusters"].is_unique

    for col in ("lrt_statistic", "lrt_dof_diff", "lrt_p_value"):
        assert col in grid.columns

    # No artificial BLRT values when bootstrap=False
    if "blrt_p_value" in grid.columns:
        assert grid["blrt_p_value"].isna().all()

    grid_by_k = grid.set_index("n_clusters")
    assert np.isnan(grid_by_k.loc[1, "lrt_statistic"])
    assert np.isnan(grid_by_k.loc[1, "lrt_dof_diff"])
    assert np.isnan(grid_by_k.loc[1, "lrt_p_value"])

    nested_by_k = model_no_bootstrap.nested_models_lrt.set_index("n_clusters")
    for k in [2, 3, 4]:
        assert np.isclose(
            grid_by_k.loc[k, "lrt_statistic"], nested_by_k.loc[k, "lr_statistic"], rtol=1e-12
        )
        assert np.isclose(
            grid_by_k.loc[k, "lrt_dof_diff"], nested_by_k.loc[k, "dof_diff"], rtol=1e-12
        )
        assert np.isclose(
            grid_by_k.loc[k, "lrt_p_value"], nested_by_k.loc[k, "p_value"], rtol=1e-12
        )

    assert "lrt_nested" not in model_no_bootstrap.model_info


def test_baseline_lca02_convergence_rate_present_without_bootstrap(model_no_bootstrap):
    """
    Verify LCA-02 internal data integrity when bootstrap=False:
    - Boolean 'converged' remains available for every k.
    - 'convergence_rate' has no artificial 1.0/0.0 values when bootstrap=False.
    """
    grid = model_no_bootstrap.grid_results
    assert "converged" in grid.columns
    assert grid["converged"].notna().all()
    assert set(grid["converged"].unique()).issubset({True, False})

    if "convergence_rate" in grid.columns:
        assert grid["convergence_rate"].isna().all()


def test_xfail_lca02_no_convergence_rate_without_bootstrap(model_no_bootstrap, capsys):
    """
    Verify LCA-02 contract:
    When bootstrap=False, convergence_rate is either absent or all-NaN internally,
    and is NOT displayed in the user-facing summary() table.
    """
    grid = model_no_bootstrap.grid_results
    if "convergence_rate" in grid.columns:
        assert grid["convergence_rate"].isna().all()

    model_no_bootstrap.summary()
    out = capsys.readouterr().out
    assert "GRID SEARCH RESULTS" in out
    assert "converged" in out
    assert "convergence_rate" not in out


def test_xfail_lca03_summary_automatically_displays_default_plots(
    model_no_bootstrap, monkeypatch, capsys
):
    """
    Verify LCA-03 & GEN-03: calling summary() with no arguments automatically
    renders all default LCA plots (including radar_plot for the reference k)
    and outputs LCA-specific sections (Binary Indicator Frequency Table,
    Class Enumeration, Convergence Diagnostics, Reference Model Results).
    """
    shown_figures = []
    monkeypatch.setattr(go.Figure, "show", lambda self, *a, **kw: shown_figures.append(self))
    model_no_bootstrap.summary()
    out = capsys.readouterr().out

    assert len(shown_figures) == 5
    assert model_no_bootstrap._displayed_plots == [
        "information_criteria",
        "ll_curve",
        "entropy_curve",
        "relative_entropy_curve",
        "radar_plot",
    ]
    titles = [fig.layout.title.text for fig in shown_figures]
    assert "Information Criteria" in titles[0]
    assert "Log-Likelihood" in titles[1]
    assert "Entropy by Number of Classes" in titles[2]
    assert "Relative Entropy" in titles[3]
    assert "Radar (Reference k=4)" in titles[4]

    assert "MODEL INFORMATION" in out
    assert "BINARY INDICATOR FREQUENCY TABLE" in out
    assert "GRID SEARCH RESULTS (CLASS ENUMERATION)" in out
    assert "CONVERGENCE DIAGNOSTICS" in out
    assert "REFERENCE MODEL RESULTS (Reference k = 4" in out
