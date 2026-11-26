"""
Scenario B — Tests for LCA with bootstrap (k = [1, 2, 3, 4], n_iterations=6).

Verifies:
- Execution of fit(bootstrap=True, n_iterations=6, param_grid={"n_clusters": [1, 2, 3, 4]})
- Structure and content of grid_results
- Available bootstrap metrics and artifacts (lrt_results, bootstrap_results)
- Existence of k=1 in grid_results, bootstrap_stats, and selection_rate
- Availability and names of bootstrap plots in plots_map
- Plotly and Matplotlib execution of bootstrap_metrics_ci and selection_rate
- Known issues LCA-04 and LCA-05 (baseline + xfail)
"""

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import pytest


def test_lca_bootstrap_training_and_grid_results(model_bootstrap, capsys):
    """Verify training with bootstrap completes and populates grid_results (including LRT & BLRT) for k=[1, 2, 3, 4]."""
    assert model_bootstrap._state == "fitted"
    assert model_bootstrap._bootstrap is True
    assert model_bootstrap._n_iterations == 6

    grid = model_bootstrap.grid_results
    assert isinstance(grid, pd.DataFrame)
    assert list(grid["n_clusters"]) == [1, 2, 3, 4]

    # LCA-01 with bootstrap: Nested LRT and BLRT integrated into grid_results by n_clusters
    for col in ("lrt_statistic", "lrt_dof_diff", "lrt_p_value", "blrt_p_value"):
        assert col in grid.columns

    grid_by_k = grid.set_index("n_clusters")
    assert np.isnan(grid_by_k.loc[1, "lrt_p_value"])
    assert np.isnan(grid_by_k.loc[1, "blrt_p_value"])

    blrt_by_k = model_bootstrap.lrt_results["blrt"].set_index("n_clusters")
    for k in [2, 3, 4]:
        assert np.isclose(
            grid_by_k.loc[k, "blrt_p_value"], blrt_by_k.loc[k, "blrt_p_value"], rtol=1e-12
        )

    assert "lrt_nested" not in model_bootstrap.model_info
    assert "lrt_blrt" not in model_bootstrap.model_info

    # LCA-02 with bootstrap: convergence_rate comes from bootstrap_stats replications
    assert grid["convergence_rate"].notna().all()
    model_bootstrap.summary()
    out = capsys.readouterr().out
    assert "lrt_p_value" in out
    assert "blrt_p_value" in out
    if (grid["convergence_rate"].dropna() == 1.0).all():
        assert "convergence_rate" not in out


def test_lca_bootstrap_results_structure_and_metrics(model_bootstrap):
    """Verify lrt_results and bootstrap_results structures and available metrics."""
    assert hasattr(model_bootstrap, "lrt_results")
    lrt = model_bootstrap.lrt_results
    assert isinstance(lrt, dict)
    assert set(["nested", "blrt", "ci_bootstrap", "bootstrap_stats"]).issubset(lrt.keys())

    # BLRT (k vs k-1 for k=2, 3, 4)
    blrt_df = lrt["blrt"]
    assert isinstance(blrt_df, pd.DataFrame)
    assert list(blrt_df["n_clusters"]) == [2, 3, 4]
    assert "blrt_p_value" in blrt_df.columns
    for pval in blrt_df["blrt_p_value"]:
        assert 0.0 <= float(pval) <= 1.0

    # CI Bootstrap DataFrame
    ci_df = lrt["ci_bootstrap"]
    assert isinstance(ci_df, pd.DataFrame)
    assert {"metric", "theta_hat", "z0", "ci_low", "ci_high", "n_clusters"}.issubset(
        ci_df.columns
    )
    assert {"AIC", "BIC", "entropy", "relative_entropy"}.issubset(set(ci_df["metric"].unique()))

    # Bootstrap raw stats across repetitions
    stats_df = lrt["bootstrap_stats"]
    assert isinstance(stats_df, pd.DataFrame)
    assert sorted(stats_df["n_clusters"].unique().tolist()) == [1, 2, 3, 4]

    # Selection rate DataFrame
    sel_df = lrt["selection_rate"]
    assert isinstance(sel_df, pd.DataFrame)
    assert list(sel_df["n_clusters"]) == [1, 2, 3, 4]
    for metric_col in ["AIC", "BIC", "CAIC", "SABIC", "entropy"]:
        assert metric_col in sel_df.columns
        assert np.isclose(sel_df[metric_col].sum(), 100.0, atol=1e-3)

    # RAPID.bootstrap_results dictionary populated by _validate_bootstrap()
    boot_res = model_bootstrap.bootstrap_results
    assert isinstance(boot_res, dict)
    assert boot_res["ci_by_k"] is ci_df
    assert boot_res["selection_rate"] is sel_df
    assert boot_res["bootstrap_stats"] is stats_df


def test_lca_bootstrap_plots_registered_in_plots_map(model_bootstrap):
    """Verify bootstrap plots (bootstrap_metrics_ic, bootstrap_metrics_entropy, selection_rate) are registered after fit(bootstrap=True)."""
    expected_plots = {
        "profile_heatmap",
        "cluster_distribution",
        "radar_plot",
        "information_criteria",
        "ll_curve",
        "entropy_curve",
        "relative_entropy_curve",
        "bootstrap_metrics_ic",
        "bootstrap_metrics_entropy",
        "selection_rate",
    }
    assert expected_plots.issubset(set(model_bootstrap.plots_map.keys()))
    assert "bootstrap_metrics_ci" not in model_bootstrap.plots_map
    assert "bootstrap_metrics_ic" in model_bootstrap._default_plots
    assert "bootstrap_metrics_entropy" in model_bootstrap._default_plots
    assert "selection_rate" in model_bootstrap._default_plots

    # Calling summary with old plot name 'bootstrap_metrics_ci' raises explicit ValueError (no silent alias)
    with pytest.raises(ValueError, match="bootstrap_metrics_ic"):
        model_bootstrap.summary(plots=["bootstrap_metrics_ci"])


@pytest.mark.parametrize(
    "plot_name",
    ["bootstrap_metrics_ic", "bootstrap_metrics_entropy", "selection_rate"],
)
@pytest.mark.parametrize("backend", ["plotly", "matplotlib"])
def test_lca_bootstrap_plots_execution_both_backends(model_bootstrap, plot_name, backend):
    """Verify bootstrap_metrics_ic, bootstrap_metrics_entropy, and selection_rate execute in Plotly and Matplotlib."""
    fig = model_bootstrap.plots_map[plot_name](backend=backend)
    assert fig is not None

    if backend == "plotly":
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1
    else:
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)


# ============================================================================
# LCA-04 AND LCA-05 VERIFICATION (SEPARATED PLOTS & k=1 INCLUSION)
# ============================================================================

def test_baseline_lca04_incompatible_metric_scales_on_single_axis(
    model_bootstrap, monkeypatch
):
    """
    Verify LCA-04 contract:
    - bootstrap_metrics_ic displays ONLY AIC, BIC, CAIC, SABIC.
    - bootstrap_metrics_entropy displays ONLY entropy and relative_entropy in two panels.
    - Plotted values and CI bands match ci_bootstrap without recalculating bootstrap.
    """
    from isaric.modelevaluation import gridsearch as gs_mod

    # Ensure no bootstrap recalculation occurs during plot generation
    def _forbid_recalc(*args, **kwargs):
        raise AssertionError("Bootstrap must not be recalculated during plot generation")

    monkeypatch.setattr(gs_mod, "bootstrap_stepmix", _forbid_recalc)
    monkeypatch.setattr(gs_mod, "bootstrap_ci_stepmix", _forbid_recalc)

    ci_df = model_bootstrap.lrt_results["ci_bootstrap"]

    # 1. Information Criteria plot (Plotly & Matplotlib)
    fig_ic_pl = model_bootstrap.plots_map["bootstrap_metrics_ic"](backend="plotly")
    ic_names_pl = [t.name for t in fig_ic_pl.data if t.name]
    assert set(ic_names_pl) == {"AIC", "BIC", "CAIC", "SABIC"}
    assert fig_ic_pl.layout.xaxis.title.text == "Number of Classes (k)"
    assert "Information Criterion" in fig_ic_pl.layout.yaxis.title.text

    # Verify numerical equivalence of plotted BIC trace with ci_bootstrap
    bic_sub = ci_df[ci_df["metric"] == "BIC"].sort_values("n_clusters")
    bic_trace = next(t for t in fig_ic_pl.data if t.name == "BIC")
    assert list(bic_trace.x) == list(bic_sub["n_clusters"])
    assert np.allclose(bic_trace.y, bic_sub["theta_hat"].values, rtol=1e-12)

    fig_ic_mpl = model_bootstrap.plots_map["bootstrap_metrics_ic"](backend="matplotlib")
    assert len(fig_ic_mpl.axes) == 1
    ax_ic = fig_ic_mpl.axes[0]
    assert "Information Criterion" in ax_ic.get_ylabel()
    mpl_labels = [line.get_label() for line in ax_ic.get_lines() if not line.get_label().startswith("_")]
    assert set(mpl_labels) == {"AIC", "BIC", "CAIC", "SABIC"}
    plt.close(fig_ic_mpl)


def test_xfail_lca04_compatible_scales_in_bootstrap_plot(model_bootstrap):
    """
    Verify LCA-04 two-panel entropy visualization:
    - bootstrap_metrics_entropy separates 'entropy' and 'relative_entropy' onto
      two independent panels / Y-axes in both Plotly and Matplotlib.
    - Plotly and Matplotlib contain equivalent data points.
    """
    ci_df = model_bootstrap.lrt_results["ci_bootstrap"]

    # Plotly two-panel verification
    fig_ent_pl = model_bootstrap.plots_map["bootstrap_metrics_entropy"](backend="plotly")
    named_traces = {t.name: t for t in fig_ent_pl.data if t.name}
    assert set(named_traces.keys()) == {"entropy", "relative_entropy"}

    ent_yaxis = getattr(named_traces["entropy"], "yaxis", None) or "y"
    rel_yaxis = getattr(named_traces["relative_entropy"], "yaxis", None) or "y"
    assert ent_yaxis != rel_yaxis, "entropy and relative_entropy must use independent Y-axes"
    assert "Entropy" in fig_ent_pl.layout.yaxis.title.text
    assert "Relative Entropy" in fig_ent_pl.layout.yaxis2.title.text

    # Matplotlib two-panel verification
    fig_ent_mpl = model_bootstrap.plots_map["bootstrap_metrics_entropy"](backend="matplotlib")
    assert len(fig_ent_mpl.axes) == 2
    ax_ent, ax_rel = fig_ent_mpl.axes
    assert "Entropy" in ax_ent.get_ylabel()
    assert "Relative Entropy" in ax_rel.get_ylabel()

    # Verify Plotly and Matplotlib plot identical coordinates from ci_bootstrap
    ent_sub = ci_df[ci_df["metric"] == "entropy"].dropna(subset=["theta_hat"]).sort_values("n_clusters")
    rel_sub = ci_df[ci_df["metric"] == "relative_entropy"].dropna(subset=["theta_hat"]).sort_values("n_clusters")

    assert list(named_traces["entropy"].x) == [1, 2, 3, 4]
    assert list(named_traces["relative_entropy"].x) == [2, 3, 4]
    assert np.allclose(ax_ent.get_lines()[0].get_xdata(), ent_sub["n_clusters"].values)
    assert np.allclose(ax_ent.get_lines()[0].get_ydata(), ent_sub["theta_hat"].values, rtol=1e-12)
    assert np.allclose(ax_rel.get_lines()[0].get_xdata(), rel_sub["n_clusters"].values)
    assert np.allclose(ax_rel.get_lines()[0].get_ydata(), rel_sub["theta_hat"].values, rtol=1e-12)
    plt.close(fig_ent_mpl)


def test_baseline_lca05_k1_missing_from_ci_bootstrap(model_bootstrap):
    """
    Verify LCA-05 data structure and baseline numerical consistency:
    - k=1 is present in ci_bootstrap for AIC, BIC, CAIC, SABIC, and entropy.
    - relative_entropy is not defined at k=1 and remains absent/NaN for k=1.
    - Results for k=2, 3, 4 match the baseline without regression.
    """
    ci_df = model_bootstrap.lrt_results["ci_bootstrap"]
    stats_ks = sorted(model_bootstrap.lrt_results["bootstrap_stats"]["n_clusters"].unique().tolist())
    ci_ks = sorted(ci_df["n_clusters"].unique().tolist())

    assert stats_ks == [1, 2, 3, 4]
    assert ci_ks == [1, 2, 3, 4]

    k1_metrics = set(ci_df[ci_df["n_clusters"] == 1]["metric"].unique())
    assert k1_metrics == {"AIC", "BIC", "CAIC", "SABIC", "entropy"}
    assert "relative_entropy" not in k1_metrics

    # Baseline numerical check for k=2, 3, 4 in ci_bootstrap (synthetic_df, seed=42, n_iterations=6)
    bic_ci = ci_df[ci_df["metric"] == "BIC"].set_index("n_clusters")
    assert np.isclose(bic_ci.loc[1, "theta_hat"], 2215.094690, rtol=1e-6)
    assert np.isclose(bic_ci.loc[2, "theta_hat"], 2122.644780, rtol=1e-6)
    assert np.isclose(bic_ci.loc[3, "theta_hat"], 2122.735572, rtol=1e-6)
    assert np.isclose(bic_ci.loc[4, "theta_hat"], 2174.265436, rtol=1e-6)


def test_xfail_lca05_k1_present_in_bootstrap_ci_plot(
    model_bootstrap, synthetic_df, symptom_vars
):
    """
    Verify LCA-05 across scenarios:
    - k=[1, 2, 3, 4]: k=1 is in ci_bootstrap and plotted on x-axis of bootstrap_metrics_ic
      and bootstrap_metrics_entropy.
    - k=[1]: single-class enumeration works with bootstrap=False and bootstrap=True,
      populating k=1 in ci_bootstrap and rendering both bootstrap plots without artificial LRT/BLRT.
    """
    import isaric as isa

    ci_df = model_bootstrap.lrt_results["ci_bootstrap"]
    assert 1 in ci_df["n_clusters"].values

    fig_ic = model_bootstrap.plots_map["bootstrap_metrics_ic"](backend="plotly")
    for trace in fig_ic.data:
        if trace.name in {"AIC", "BIC", "CAIC", "SABIC"}:
            assert list(trace.x) == [1, 2, 3, 4]

    # Scenario k=[1] with bootstrap=False and bootstrap=True
    m_k1_noboot = isa.RAPID.create(
        data=synthetic_df,
        model="lca",
        n_clusters=1,
        measurement_vars=symptom_vars,
        random_seed=42,
    )
    m_k1_noboot.fit(bootstrap=False, param_grid={"n_clusters": [1]})
    assert list(m_k1_noboot.grid_results["n_clusters"]) == [1]
    assert np.isnan(m_k1_noboot.grid_results.loc[0, "lrt_p_value"])
    assert np.isnan(m_k1_noboot.grid_results.loc[0, "relative_entropy"])

    m_k1_boot = isa.RAPID.create(
        data=synthetic_df,
        model="lca",
        n_clusters=1,
        measurement_vars=symptom_vars,
        random_seed=42,
    )
    m_k1_boot.fit(bootstrap=True, n_iterations=5, param_grid={"n_clusters": [1]})
    assert list(m_k1_boot.grid_results["n_clusters"]) == [1]
    assert np.isnan(m_k1_boot.grid_results.loc[0, "lrt_p_value"])
    assert np.isnan(m_k1_boot.grid_results.loc[0, "blrt_p_value"])
    assert m_k1_boot.lrt_results["blrt"] is None
    assert list(m_k1_boot.lrt_results["ci_bootstrap"]["n_clusters"].unique()) == [1]

    fig_k1_ic = m_k1_boot.plots_map["bootstrap_metrics_ic"](backend="plotly")
    fig_k1_ent = m_k1_boot.plots_map["bootstrap_metrics_entropy"](backend="plotly")
    fig_k1_ent_mpl = m_k1_boot.plots_map["bootstrap_metrics_entropy"](backend="matplotlib")
    plt.close(fig_k1_ent_mpl)
    assert len(fig_k1_ic.data) >= 2
    assert len(fig_k1_ent.data) >= 2


def test_lca_bootstrap_non_consecutive_k_and_convergence_failures(
    synthetic_df, symptom_vars, monkeypatch, capsys
):
    """
    Verify:
    1. Non-consecutive k=[1, 2, 4] with bootstrap=True only computes Nested LRT and BLRT
       for consecutive (k-1, k) pairs (k=2), both in grid_results and in internal
       structures (lrt_results['nested'] and lrt_results['blrt']).
    2. When bootstrap replications have non-convergences (converged=False),
       convergence_rate < 1.0 is computed and displayed in summary().
    3. Documented behavior of bootstrap_stepmix when estimator_rep.fit raises an Exception:
       failed replications are skipped via 'except Exception: continue', reducing the
       denominator of converged.mean() from n_bootstrap to n_succeeded.
    """
    import isaric as isa
    from isaric.modelevaluation import gridsearch as gs_mod
    from stepmix.stepmix import StepMix

    # 1. Non-consecutive k=[1, 2, 4] with simulated partial convergence in bootstrap_stepmix
    orig_bootstrap_stepmix = gs_mod.bootstrap_stepmix

    def _mock_bootstrap_stepmix(fitted_estimator, X, Y=None, n_bootstrap=5, **kwargs):
        df_boot = orig_bootstrap_stepmix(
            fitted_estimator, X, Y=Y, n_bootstrap=n_bootstrap, **kwargs
        )
        # Simulate non-convergence on the first replication for k=2
        if fitted_estimator.n_components == 2 and len(df_boot) >= 2:
            df_boot = df_boot.copy()
            df_boot.loc[df_boot.index[0], "converged"] = False
        return df_boot

    monkeypatch.setattr(gs_mod, "bootstrap_stepmix", _mock_bootstrap_stepmix)

    m_gap = isa.RAPID.create(
        data=synthetic_df,
        model="lca",
        n_clusters=4,
        measurement_vars=symptom_vars,
        random_seed=42,
    )
    m_gap.fit(
        bootstrap=True,
        n_iterations=4,
        param_grid={"n_clusters": [1, 2, 4]},
    )

    # Check internal structures for non-consecutive k
    assert list(m_gap.lrt_results["nested"]["n_clusters"]) == [2]
    assert list(m_gap.lrt_results["blrt"]["n_clusters"]) == [2]

    # Check grid_results for non-consecutive k
    g_gap = m_gap.grid_results.set_index("n_clusters")
    assert list(g_gap.index) == [1, 2, 4]
    assert np.isnan(g_gap.loc[1, "lrt_p_value"])
    assert np.isnan(g_gap.loc[1, "blrt_p_value"])
    assert np.isfinite(g_gap.loc[2, "lrt_p_value"])
    assert np.isfinite(g_gap.loc[2, "blrt_p_value"])
    assert np.isnan(g_gap.loc[4, "lrt_p_value"])
    assert np.isnan(g_gap.loc[4, "blrt_p_value"])

    # Check convergence_rate < 1.0 for k=2 (3/4 = 0.75) and display in summary()
    assert np.isclose(g_gap.loc[2, "convergence_rate"], 0.75, atol=1e-6)
    assert np.isclose(g_gap.loc[1, "convergence_rate"], 1.0, atol=1e-6)
    m_gap.summary()
    out = capsys.readouterr().out
    assert "convergence_rate" in out

    # 3. Verify and document bootstrap_stepmix exception-skipping denominator behavior
    fitted_k2 = m_gap.model_objects[2]
    call_counter = {"n": 0}
    orig_fit = StepMix.fit

    def _flaky_fit(self, X, Y=None, **kwargs):
        call_counter["n"] += 1
        if call_counter["n"] in (1, 2):
            raise RuntimeError("Simulated EM numerical failure in bootstrap replication")
        return orig_fit(self, X, Y=Y, **kwargs)

    monkeypatch.setattr(StepMix, "fit", _flaky_fit)
    stats_with_exceptions = orig_bootstrap_stepmix(
        fitted_estimator=fitted_k2,
        X=m_gap.X.values,
        n_bootstrap=5,
        parametric=True,
        random_state=42,
    )
    # 2 exceptions out of 5 replications are discarded by 'except Exception: continue',
    # leaving 3 rows (all converged=True), so mean() is 3/3 = 1.0 instead of 3/5 = 0.6.
    assert len(stats_with_exceptions) == 3
    assert float(stats_with_exceptions["converged"].mean()) == 1.0


def test_lca_run_lrt_ci_exception_emits_warning_and_continues(
    synthetic_df, symptom_vars, monkeypatch
):
    """
    Verify that if bootstrap_ci_stepmix raises an exception for a specific k (e.g., k=2),
    LCA._run_lrt emits a UserWarning naming k and the error message, and continues
    computing CIs for the remaining k values (k=1, 3, 4).
    """
    import isaric as isa
    from isaric.modelevaluation import gridsearch as gs_mod

    orig_ci_fn = gs_mod.bootstrap_ci_stepmix

    def _failing_ci_for_k2(fitted_estimator, stats_bootstrap, X, **kwargs):
        if fitted_estimator.n_components == 2:
            raise RuntimeError("Simulated CI failure on 2-class model")
        return orig_ci_fn(fitted_estimator, stats_bootstrap, X, **kwargs)

    monkeypatch.setattr(gs_mod, "bootstrap_ci_stepmix", _failing_ci_for_k2)

    model = isa.RAPID.create(
        data=synthetic_df,
        model="lca",
        n_clusters=4,
        measurement_vars=symptom_vars,
        random_seed=42,
    )
    with pytest.warns(UserWarning, match=r"k=2.*Simulated CI failure on 2-class model"):
        model.fit(
            bootstrap=True,
            n_iterations=5,
            param_grid={"n_clusters": [1, 2, 3, 4]},
        )

    ci_df = model.lrt_results["ci_bootstrap"]
    assert isinstance(ci_df, pd.DataFrame)
    assert sorted(ci_df["n_clusters"].unique().tolist()) == [1, 3, 4]


def test_lca_refit_without_bootstrap_clears_bootstrap_plots(
    synthetic_df, symptom_vars
):
    """
    Verify that re-fitting an LCA model with bootstrap=False after a fit(bootstrap=True)
    removes bootstrap plots from both plots_map and _default_plots.
    """
    import isaric as isa

    model = isa.RAPID.create(
        data=synthetic_df,
        model="lca",
        n_clusters=3,
        measurement_vars=symptom_vars,
        random_seed=42,
    )

    model.fit(bootstrap=True, n_iterations=6, param_grid={"n_clusters": [1, 2, 3]})
    assert "bootstrap_metrics_ic" in model.plots_map
    assert "bootstrap_metrics_entropy" in model.plots_map
    assert "selection_rate" in model.plots_map
    assert "bootstrap_metrics_ic" in model._default_plots
    assert "bootstrap_metrics_entropy" in model._default_plots
    assert "selection_rate" in model._default_plots

    model.fit(bootstrap=False, param_grid={"n_clusters": [1, 2, 3]})
    assert "bootstrap_metrics_ic" not in model.plots_map
    assert "bootstrap_metrics_entropy" not in model.plots_map
    assert "selection_rate" not in model.plots_map
    assert "bootstrap_metrics_ic" not in model._default_plots
    assert "bootstrap_metrics_entropy" not in model._default_plots
    assert "selection_rate" not in model._default_plots

    # Standard LCA plots remain available
    expected_base = {
        "profile_heatmap",
        "cluster_distribution",
        "radar_plot",
        "information_criteria",
        "ll_curve",
        "entropy_curve",
        "relative_entropy_curve",
    }
    assert expected_base == set(model.plots_map.keys())
    assert model._default_plots == [
        "information_criteria",
        "ll_curve",
        "entropy_curve",
        "relative_entropy_curve",
    ]


