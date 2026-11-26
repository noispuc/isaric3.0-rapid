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

def test_current_summary_without_args(model_no_bootstrap, capsys):
    """Verify current behavior of model.summary() with default arguments."""
    model_no_bootstrap.summary()
    out = capsys.readouterr().out

    assert "MODEL INFORMATION" in out
    assert "RESULTS" in out
    assert "GRID SEARCH RESULTS" in out
    # Sets _displayed_plots to _default_plots
    assert model_no_bootstrap._displayed_plots == [
        "information_criteria",
        "ll_curve",
        "entropy_curve",
        "relative_entropy_curve",
    ]


def test_current_summary_with_explicit_plots(model_no_bootstrap, monkeypatch):
    """Verify current behavior of model.summary(plots=[...])."""
    shown = []
    monkeypatch.setattr(go.Figure, "show", lambda self, *a, **kw: shown.append(self))

    requested_plots = [
        "profile_heatmap",
        "cluster_distribution",
        "information_criteria",
    ]
    model_no_bootstrap.summary(plots=requested_plots)

    assert len(shown) == len(requested_plots)
    assert model_no_bootstrap._displayed_plots == requested_plots


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
# FUTURE / PENDING REQUIREMENTS (MARKED STRICT XFAIL UNTIL IMPLEMENTED)
# ============================================================================

@pytest.mark.xfail(
    strict=True,
    reason="Pending LCA requirement: model.summary(k=3) is not yet supported in summary() signature.",
)
def test_pending_summary_with_specific_k(model_no_bootstrap, capsys):
    """Future requirement: inspect a specific k solution via model.summary(k=3)."""
    model_no_bootstrap.summary(k=3)
    out = capsys.readouterr().out
    assert "RESULTS" in out and "Class_4" not in out


@pytest.mark.xfail(
    strict=True,
    reason="Pending LCA requirement: comparing multiple k solutions from enumeration via model.summary(k=[2, 3, 4]) is not yet implemented.",
)
def test_pending_summary_compare_different_k_solutions(model_no_bootstrap, capsys):
    """Future requirement: compare candidate k solutions directly from a single fitted LCA enumeration."""
    model_no_bootstrap.summary(k=[2, 3, 4])
    out = capsys.readouterr().out
    assert len(out.strip()) > 0


@pytest.mark.xfail(
    strict=True,
    reason="Pending LCA requirement: model.decide(selected=True, k=3, ...) currently only records metadata and does not update fitted_model/result_df from reference n_clusters=4 to k=3.",
)
def test_pending_decide_k_updates_active_solution(synthetic_df, symptom_vars):
    """
    Future requirement: calling model.decide(selected=True, k=3, reason=...)
    should switch the active solution (fitted_model, result_df, metrics) to k=3.
    """
    model = isa.RAPID.create(
        data=synthetic_df,
        model="lca",
        n_clusters=4,
        measurement_vars=symptom_vars,
        random_seed=42,
    )
    model.fit(bootstrap=False, param_grid={"n_clusters": [1, 2, 3, 4]})
    model.decide(selected=True, k=3, reason="Selected k=3 after reviewing enumeration.")

    assert model.fitted_model.n_components == 3
    assert len(model.result_df) == 3


@pytest.mark.xfail(
    strict=True,
    reason="Pending LCA requirement: model.report() after decide(k=3) still exports result_df for initial n_clusters=4 instead of chosen k=3.",
)
def test_pending_report_after_decide_k_exports_chosen_k_profiles(
    synthetic_df, symptom_vars, tmp_path: Path, monkeypatch
):
    """
    Future requirement: calling model.report() after model.decide(selected=True, k=3)
    should export the 3-class profile table in the report CSV/PDF.
    """
    import pandas as pd

    monkeypatch.chdir(tmp_path)
    model = isa.RAPID.create(
        data=synthetic_df,
        model="lca",
        n_clusters=4,
        measurement_vars=symptom_vars,
        random_seed=42,
    )
    model.fit(bootstrap=False, param_grid={"n_clusters": [1, 2, 3, 4]})
    model.decide(selected=True, k=3, reason="Selected k=3 for final report.")
    model.report(format=["csv"])

    results_csv = next(tmp_path.glob("*_results.csv"))
    df_exported = pd.read_csv(results_csv)
    assert len(df_exported) == 3
