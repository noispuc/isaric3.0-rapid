"""
Survival curve visualization for the RAPID methodology.

This module provides functions to generate survival curves (Step 6.3
of the RAPID methodology). Survival curves display the probability of
surviving past a certain time using the Kaplan-Meier estimator.

Backends:
- plotly: Interactive figures for notebook display (default).
- matplotlib: Static figures for report export (PNG, PDF).

Techniques:
- kaplan_meier_curve: Plot survival curve from lifelines KMF.
- compare_survival_curves: Compare survival between groups.
- baseline_survival_curve: Plot baseline survival from Cox model.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter


def kaplan_meier_curve(
    data: pd.DataFrame,
    duration_var: str,
    event_var: str,
    title: str = "Kaplan-Meier Survival Curve",
    xaxis_title: str = "Time",
    yaxis_title: str = "Survival Probability",
    color: str = '#2a9d8f',
    height: int = 500,
    width: int = 700,
    backend: str = "plotly"
):
    """
    Generate a Kaplan-Meier survival curve.

    Plots the survival function estimated from time-to-event data
    using the Kaplan-Meier product-limit formula.

    Args:
        data: Input DataFrame.
        duration_var: Time-to-event column.
        event_var: Event indicator column (1=event, 0=censored).
        title: Plot title.
        xaxis_title: X-axis label.
        yaxis_title: Y-axis label.
        color: Line color.
        height: Figure height in pixels.
        width: Figure width in pixels.
        backend: "plotly" or "matplotlib".

    Returns:
        Plotly Figure or Matplotlib Figure.

    Raises:
        ValueError: If required columns are not found or backend invalid.
    """
    for col in [duration_var, event_var]:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    kmf = KaplanMeierFitter()
    kmf.fit(
        data[duration_var],
        event_observed=data[event_var]
    )

    if backend == "plotly":
        return _kaplan_meier_plotly(
            kmf=kmf,
            title=title,
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            color=color,
            height=height,
            width=width
        )
    elif backend == "matplotlib":
        return _kaplan_meier_matplotlib(
            kmf=kmf,
            title=title,
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            color=color,
            height=height,
            width=width
        )
    else:
        raise ValueError(f"backend must be 'plotly' or 'matplotlib'. Received: {backend}")


def compare_survival_curves(
    data: pd.DataFrame,
    duration_var: str,
    event_var: str,
    group_col: str,
    group_values: Optional[List[str]] = None,
    title: str = "Survival Curves by Group",
    xaxis_title: str = "Time",
    yaxis_title: str = "Survival Probability",
    colors: Optional[List[str]] = None,
    height: int = 500,
    width: int = 700,
    backend: str = "plotly"
):
    """
    Compare survival curves between multiple groups.

    Plots Kaplan-Meier curves for each subgroup to compare survival.

    Args:
        data: Input DataFrame.
        duration_var: Time-to-event column.
        event_var: Event indicator column.
        group_col: Column defining groups to compare.
        group_values: List of group values to include (None = all).
        title: Plot title.
        xaxis_title: X-axis label.
        yaxis_title: Y-axis label.
        colors: List of line colors.
        height: Figure height in pixels.
        width: Figure width in pixels.
        backend: "plotly" or "matplotlib".

    Returns:
        Plotly Figure or Matplotlib Figure.

    Raises:
        ValueError: If required columns are not found or backend invalid.
    """
    for col in [duration_var, event_var, group_col]:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    if group_values is None:
        group_values = data[group_col].unique().tolist()

    if colors is None:
        colors = ['#2a9d8f', '#e76f51', '#264653', '#e9c46a', '#f4a261']

    # Fit KMF for each group
    kmf_dict = {}
    kmf = KaplanMeierFitter()

    for group in group_values:
        group_data = data[data[group_col] == group]
        if len(group_data) > 0:
            kmf.fit(
                group_data[duration_var],
                event_observed=group_data[event_var],
                label=str(group)
            )
            kmf_dict[str(group)] = kmf

    if backend == "plotly":
        return _compare_survival_plotly(
            kmf_dict=kmf_dict,
            title=title,
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            colors=colors,
            height=height,
            width=width
        )
    elif backend == "matplotlib":
        return _compare_survival_matplotlib(
            kmf_dict=kmf_dict,
            title=title,
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            colors=colors,
            height=height,
            width=width
        )
    else:
        raise ValueError(f"backend must be 'plotly' or 'matplotlib'. Received: {backend}")


def baseline_survival_curve(
    fitted_model,
    title: str = "Baseline Survival Curve (Cox Model)",
    xaxis_title: str = "Time",
    yaxis_title: str = "Survival Probability",
    color: str = '#2a9d8f',
    height: int = 500,
    width: int = 700,
    backend: str = "plotly"
):
    """
    Plot the baseline survival function from a fitted Cox model.

    Args:
        fitted_model: Fitted lifelines CoxPHFitter.
        title: Plot title.
        xaxis_title: X-axis label.
        yaxis_title: Y-axis label.
        color: Line color.
        height: Figure height in pixels.
        width: Figure width in pixels.
        backend: "plotly" or "matplotlib".

    Returns:
        Plotly Figure or Matplotlib Figure.

    Raises:
        ValueError: If model does not have baseline_survival_ or backend invalid.
    """
    if not hasattr(fitted_model, 'baseline_survival_'):
        raise ValueError(
            "Fitted model does not have baseline_survival_ attribute."
        )

    survival_func = fitted_model.baseline_survival_

    if backend == "plotly":
        return _baseline_survival_plotly(
            survival_func=survival_func,
            title=title,
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            color=color,
            height=height,
            width=width
        )
    elif backend == "matplotlib":
        return _baseline_survival_matplotlib(
            survival_func=survival_func,
            title=title,
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            color=color,
            height=height,
            width=width
        )
    else:
        raise ValueError(f"backend must be 'plotly' or 'matplotlib'. Received: {backend}")


# ============================================================================
# PLOTLY BACKEND
# ============================================================================

def _kaplan_meier_plotly(
    kmf: KaplanMeierFitter,
    title: str,
    xaxis_title: str,
    yaxis_title: str,
    color: str,
    height: int,
    width: int
) -> go.Figure:
    """Build Plotly Kaplan-Meier curve."""
    fig = go.Figure(data=go.Scatter(
        x=kmf.survival_function_.index,
        y=kmf.survival_function_.iloc[:, 0].values,
        mode='lines',
        line=dict(color=color, width=2),
        hovertemplate='Time: %{x:.1f}<br>Survival: %{y:.3f}<extra></extra>',
        name='Survival'
    ))

    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        yaxis=dict(range=[0, 1]),
        height=height,
        width=width,
        template='plotly_white'
    )

    return fig


def _compare_survival_plotly(
    kmf_dict: dict,
    title: str,
    xaxis_title: str,
    yaxis_title: str,
    colors: List[str],
    height: int,
    width: int
) -> go.Figure:
    """Build Plotly comparison survival curves."""
    fig = go.Figure()

    for i, (label, kmf) in enumerate(kmf_dict.items()):
        fig.add_trace(go.Scatter(
            x=kmf.survival_function_.index,
            y=kmf.survival_function_.iloc[:, 0].values,
            mode='lines',
            line=dict(color=colors[i % len(colors)], width=2),
            name=label,
            hovertemplate=f'Group: {label}<br>Time: %{{x:.1f}}<br>Survival: %{{y:.3f}}<extra></extra>'
        ))

    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        yaxis=dict(range=[0, 1]),
        height=height,
        width=width,
        template='plotly_white'
    )

    return fig


def _baseline_survival_plotly(
    survival_func: pd.DataFrame,
    title: str,
    xaxis_title: str,
    yaxis_title: str,
    color: str,
    height: int,
    width: int
) -> go.Figure:
    """Build Plotly baseline survival curve."""
    fig = go.Figure(data=go.Scatter(
        x=survival_func.index,
        y=survival_func.iloc[:, 0].values,
        mode='lines',
        line=dict(color=color, width=2),
        hovertemplate='Time: %{x:.1f}<br>Survival: %{y:.3f}<extra></extra>',
        name='Baseline Survival'
    ))

    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        yaxis=dict(range=[0, 1]),
        height=height,
        width=width,
        template='plotly_white'
    )

    return fig


# ============================================================================
# MATPLOTLIB BACKEND
# ============================================================================

def _kaplan_meier_matplotlib(
    kmf: KaplanMeierFitter,
    title: str,
    xaxis_title: str,
    yaxis_title: str,
    color: str,
    height: int,
    width: int
) -> plt.Figure:
    """Build Matplotlib Kaplan-Meier curve."""
    fig, ax = plt.subplots(figsize=(width/100, height/100))
    
    ax.plot(
        kmf.survival_function_.index,
        kmf.survival_function_.iloc[:, 0].values,
        color=color,
        linewidth=2,
        label='Survival'
    )
    
    ax.set_xlabel(xaxis_title)
    ax.set_ylabel(yaxis_title)
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    fig.tight_layout()
    return fig


def _compare_survival_matplotlib(
    kmf_dict: dict,
    title: str,
    xaxis_title: str,
    yaxis_title: str,
    colors: List[str],
    height: int,
    width: int
) -> plt.Figure:
    """Build Matplotlib comparison survival curves."""
    fig, ax = plt.subplots(figsize=(width/100, height/100))
    
    for i, (label, kmf) in enumerate(kmf_dict.items()):
        ax.plot(
            kmf.survival_function_.index,
            kmf.survival_function_.iloc[:, 0].values,
            color=colors[i % len(colors)],
            linewidth=2,
            label=label
        )
    
    ax.set_xlabel(xaxis_title)
    ax.set_ylabel(yaxis_title)
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    fig.tight_layout()
    return fig


def _baseline_survival_matplotlib(
    survival_func: pd.DataFrame,
    title: str,
    xaxis_title: str,
    yaxis_title: str,
    color: str,
    height: int,
    width: int
) -> plt.Figure:
    """Build Matplotlib baseline survival curve."""
    fig, ax = plt.subplots(figsize=(width/100, height/100))
    
    ax.plot(
        survival_func.index,
        survival_func.iloc[:, 0].values,
        color=color,
        linewidth=2,
        label='Baseline Survival'
    )
    
    ax.set_xlabel(xaxis_title)
    ax.set_ylabel(yaxis_title)
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    fig.tight_layout()
    return fig

# ============================================================================
# RESIDUAL PLOTS (COX MODEL DIAGNOSTICS)
# ============================================================================

def schoenfeld_residuals_plot(
    residuals: np.ndarray,
    times: np.ndarray,
    covariate_name: str,
    title: Optional[str] = None,
    height: int = 500,
    width: int = 700,
    backend: str = "plotly"
):
    """
    Generate a Schoenfeld residuals plot for a single covariate.

    Schoenfeld residuals are used to test the Proportional Hazards
    assumption. If the assumption holds, residuals should show no
    pattern over time (random scatter around zero).

    Args:
        residuals: Schoenfeld residuals for the covariate.
        times: Time points corresponding to residuals.
        covariate_name: Name of the covariate being tested.
        title: Plot title (default: auto-generated).
        height: Figure height in pixels.
        width: Figure width in pixels.
        backend: "plotly" or "matplotlib".

    Returns:
        Plotly Figure or Matplotlib Figure.
    """
    if title is None:
        title = f"Schoenfeld Residuals - {covariate_name}"

    if backend == "plotly":
        return _schoenfeld_residuals_plotly(
            residuals=residuals,
            times=times,
            covariate_name=covariate_name,
            title=title,
            height=height,
            width=width
        )
    elif backend == "matplotlib":
        return _schoenfeld_residuals_matplotlib(
            residuals=residuals,
            times=times,
            covariate_name=covariate_name,
            title=title,
            height=height,
            width=width
        )
    else:
        raise ValueError(f"backend must be 'plotly' or 'matplotlib'. Received: {backend}")


def martingale_residuals_plot(
    residuals: np.ndarray,
    covariate: np.ndarray,
    covariate_name: str,
    title: Optional[str] = None,
    add_smoother: bool = True,
    height: int = 500,
    width: int = 700,
    backend: str = "plotly"
):
    """
    Generate a Martingale residuals plot for a single covariate.

    Martingale residuals are used to assess the functional form of a
    covariate. A smoothed line helps identify non-linear patterns.

    Args:
        residuals: Martingale residuals.
        covariate: Covariate values.
        covariate_name: Name of the covariate.
        title: Plot title (default: auto-generated).
        add_smoother: Add a smoothed trend line.
        height: Figure height in pixels.
        width: Figure width in pixels.
        backend: "plotly" or "matplotlib".

    Returns:
        Plotly Figure or Matplotlib Figure.
    """
    if title is None:
        title = f"Martingale Residuals - {covariate_name}"

    if backend == "plotly":
        return _martingale_residuals_plotly(
            residuals=residuals,
            covariate=covariate,
            covariate_name=covariate_name,
            title=title,
            add_smoother=add_smoother,
            height=height,
            width=width
        )
    elif backend == "matplotlib":
        return _martingale_residuals_matplotlib(
            residuals=residuals,
            covariate=covariate,
            covariate_name=covariate_name,
            title=title,
            add_smoother=add_smoother,
            height=height,
            width=width
        )
    else:
        raise ValueError(f"backend must be 'plotly' or 'matplotlib'. Received: {backend}")


def deviance_residuals_plot(
    residuals: np.ndarray,
    covariate: np.ndarray,
    covariate_name: str,
    title: Optional[str] = None,
    height: int = 500,
    width: int = 700,
    backend: str = "plotly"
):
    """
    Generate a Deviance residuals plot for a single covariate.

    Deviance residuals are a transformation of Martingale residuals,
    more symmetric around zero. Used to identify outliers and
    influential observations.

    Args:
        residuals: Deviance residuals.
        covariate: Covariate values.
        covariate_name: Name of the covariate.
        title: Plot title (default: auto-generated).
        height: Figure height in pixels.
        width: Figure width in pixels.
        backend: "plotly" or "matplotlib".

    Returns:
        Plotly Figure or Matplotlib Figure.
    """
    if title is None:
        title = f"Deviance Residuals - {covariate_name}"

    if backend == "plotly":
        return _deviance_residuals_plotly(
            residuals=residuals,
            covariate=covariate,
            covariate_name=covariate_name,
            title=title,
            height=height,
            width=width
        )
    elif backend == "matplotlib":
        return _deviance_residuals_matplotlib(
            residuals=residuals,
            covariate=covariate,
            covariate_name=covariate_name,
            title=title,
            height=height,
            width=width
        )
    else:
        raise ValueError(f"backend must be 'plotly' or 'matplotlib'. Received: {backend}")

# ============================================================================
# TIME-DEPENDENT PLOTS (SURVIVAL)
# ============================================================================

def brier_score_plot(
    time_points: np.ndarray,
    brier_scores: np.ndarray,
    target_time: float,
    title: Optional[str] = None,
    height: int = 500,
    width: int = 700,
    backend: str = "plotly"
):
    """
    Generate a time-dependent Brier Score plot.

    Args:
        time_points: Array of time points.
        brier_scores: Array of Brier Score values.
        target_time: Target time point (for reference line).
        title: Plot title (default: auto-generated).
        height: Figure height in pixels.
        width: Figure width in pixels.
        backend: "plotly" or "matplotlib".

    Returns:
        Plotly Figure or Matplotlib Figure.
    """
    if title is None:
        title = f"Time-Dependent Brier Score (t={target_time})"

    if backend == "plotly":
        return _brier_score_plotly(time_points, brier_scores, target_time, title, height, width)
    elif backend == "matplotlib":
        return _brier_score_matplotlib(time_points, brier_scores, target_time, title, height, width)
    else:
        raise ValueError(f"backend must be 'plotly' or 'matplotlib'. Received: {backend}")


def roc_time_dependent_plot(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc: float,
    target_time: float,
    title: Optional[str] = None,
    height: int = 500,
    width: int = 700,
    backend: str = "plotly"
):
    """
    Generate a time-dependent ROC curve plot.

    Args:
        fpr: False positive rate array.
        tpr: True positive rate array.
        auc: AUC value.
        target_time: Target time point.
        title: Plot title (default: auto-generated).
        height: Figure height in pixels.
        width: Figure width in pixels.
        backend: "plotly" or "matplotlib".

    Returns:
        Plotly Figure or Matplotlib Figure.
    """
    if title is None:
        title = f"ROC at t={target_time} (AUC = {auc:.3f})"

    if backend == "plotly":
        return _roc_time_dependent_plotly(fpr, tpr, auc, target_time, title, height, width)
    elif backend == "matplotlib":
        return _roc_time_dependent_matplotlib(fpr, tpr, auc, target_time, title, height, width)
    else:
        raise ValueError(f"backend must be 'plotly' or 'matplotlib'. Received: {backend}")


def calibration_plot(
    predicted_survival: np.ndarray,
    observed_survival: np.ndarray,
    target_time: float,
    title: Optional[str] = None,
    height: int = 500,
    width: int = 700,
    backend: str = "plotly"
):
    """
    Generate a calibration plot for survival models.

    Compares predicted survival probabilities with observed
    Kaplan-Meier estimates at the target time.

    Args:
        predicted_survival: Array of predicted survival probabilities.
        observed_survival: Array of observed survival probabilities.
        target_time: Target time point.
        title: Plot title (default: auto-generated).
        height: Figure height in pixels.
        width: Figure width in pixels.
        backend: "plotly" or "matplotlib".

    Returns:
        Plotly Figure or Matplotlib Figure.
    """
    if title is None:
        title = f"Calibration at t={target_time}"

    if backend == "plotly":
        return _calibration_plotly(predicted_survival, observed_survival, target_time, title, height, width)
    elif backend == "matplotlib":
        return _calibration_matplotlib(predicted_survival, observed_survival, target_time, title, height, width)
    else:
        raise ValueError(f"backend must be 'plotly' or 'matplotlib'. Received: {backend}")

# ============================================================================
# PLOTLY BACKEND - RESIDUAL PLOTS
# ============================================================================

def _schoenfeld_residuals_plotly(
    residuals: np.ndarray,
    times: np.ndarray,
    covariate_name: str,
    title: str,
    height: int,
    width: int
) -> go.Figure:
    """Build Plotly Schoenfeld residuals plot."""
    fig = go.Figure()

    # Scatter points
    fig.add_trace(go.Scatter(
        x=times,
        y=residuals,
        mode='markers',
        marker=dict(color='#2a9d8f', size=8),
        name='Residuals',
        hovertemplate='Time: %{x:.2f}<br>Residual: %{y:.3f}<extra></extra>'
    ))

    # Reference line at 0
    fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.7)

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title=f"Schoenfeld Residual ({covariate_name})",
        height=height,
        width=width,
        template='plotly_white'
    )

    return fig


def _martingale_residuals_plotly(
    residuals: np.ndarray,
    covariate: np.ndarray,
    covariate_name: str,
    title: str,
    add_smoother: bool,
    height: int,
    width: int
) -> go.Figure:
    """Build Plotly Martingale residuals plot."""
    fig = go.Figure()

    # Scatter points
    fig.add_trace(go.Scatter(
        x=covariate,
        y=residuals,
        mode='markers',
        marker=dict(color='#2a9d8f', size=8, opacity=0.6),
        name='Residuals',
        hovertemplate=f'{covariate_name}: %{{x}}<br>Residual: %{{y:.3f}}<extra></extra>'
    ))

    # Smoother (LOWESS)
    if add_smoother and len(covariate) > 10:
        try:
            import statsmodels.api as sm
            lowess = sm.nonparametric.lowess(residuals, covariate, frac=0.3)
            fig.add_trace(go.Scatter(
                x=lowess[:, 0],
                y=lowess[:, 1],
                mode='lines',
                line=dict(color='#e76f51', width=2),
                name='Smoother'
            ))
        except Exception:
            pass

    # Reference line at 0
    fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.7)

    fig.update_layout(
        title=title,
        xaxis_title=covariate_name,
        yaxis_title="Martingale Residual",
        height=height,
        width=width,
        template='plotly_white'
    )

    return fig


def _deviance_residuals_plotly(
    residuals: np.ndarray,
    covariate: np.ndarray,
    covariate_name: str,
    title: str,
    height: int,
    width: int
) -> go.Figure:
    """Build Plotly Deviance residuals plot."""
    fig = go.Figure()

    # Scatter points
    fig.add_trace(go.Scatter(
        x=covariate,
        y=residuals,
        mode='markers',
        marker=dict(color='#2a9d8f', size=8, opacity=0.6),
        name='Residuals',
        hovertemplate=f'{covariate_name}: %{{x}}<br>Residual: %{{y:.3f}}<extra></extra>'
    ))

    # Reference line at 0
    fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.7)

    fig.update_layout(
        title=title,
        xaxis_title=covariate_name,
        yaxis_title="Deviance Residual",
        height=height,
        width=width,
        template='plotly_white'
    )

    return fig

# ============================================================================
# PLOTLY BACKEND - TIME-DEPENDENT PLOTS
# ============================================================================

def _brier_score_plotly(time_points, brier_scores, target_time, title, height, width):
    """Build Plotly Brier Score plot."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_points, y=brier_scores,
        mode='lines',
        line=dict(color='#2a9d8f', width=2),
        name='Brier Score',
        hovertemplate='Time: %{x:.1f}<br>Brier Score: %{y:.4f}<extra></extra>'
    ))
    fig.add_vline(x=target_time, line_dash="dash", line_color="red", opacity=0.7)
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Brier Score",
        height=height, width=width,
        template='plotly_white'
    )
    return fig


def _roc_time_dependent_plotly(fpr, tpr, auc, target_time, title, height, width):
    """Build Plotly time-dependent ROC plot."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        line=dict(color='#2a9d8f', width=2),
        name=f'AUC = {auc:.3f}',
        hovertemplate='FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        line=dict(color='gray', width=1, dash='dash'),
        name='Random',
        hoverinfo='skip'
    ))
    fig.update_layout(
        title=title,
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=height, width=width,
        template='plotly_white'
    )
    return fig


def _calibration_plotly(predicted_survival, observed_survival, target_time, title, height, width):
    """Build Plotly calibration plot for survival."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=predicted_survival, y=observed_survival,
        mode='markers',
        marker=dict(color='#2a9d8f', size=8, opacity=0.6),
        name='Patients',
        hovertemplate='Predicted: %{x:.3f}<br>Observed: %{y:.3f}<extra></extra>'
    ))
    # Diagonal reference
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        line=dict(color='gray', width=1, dash='dash'),
        name='Perfect calibration',
        hoverinfo='skip'
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Predicted Survival",
        yaxis_title="Observed Survival (KM)",
        height=height, width=width,
        template='plotly_white'
    )
    return fig


# ============================================================================
# MATPLOTLIB BACKEND - TIME-DEPENDENT PLOTS
# ============================================================================

def _brier_score_matplotlib(time_points, brier_scores, target_time, title, height, width):
    """Build Matplotlib Brier Score plot."""
    fig, ax = plt.subplots(figsize=(width/100, height/100))
    ax.plot(time_points, brier_scores, color='#2a9d8f', linewidth=2, label='Brier Score')
    ax.axvline(x=target_time, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.set_xlabel("Time")
    ax.set_ylabel("Brier Score")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    return fig


def _roc_time_dependent_matplotlib(fpr, tpr, auc, target_time, title, height, width):
    """Build Matplotlib time-dependent ROC plot."""
    fig, ax = plt.subplots(figsize=(width/100, height/100))
    ax.plot(fpr, tpr, color='#2a9d8f', linewidth=2, label=f'AUC = {auc:.3f}')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1, label='Random')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    return fig


def _calibration_matplotlib(predicted_survival, observed_survival, target_time, title, height, width):
    """Build Matplotlib calibration plot for survival."""
    fig, ax = plt.subplots(figsize=(width/100, height/100))
    ax.scatter(predicted_survival, observed_survival, color='#2a9d8f', s=40, alpha=0.6, label='Patients')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1, label='Perfect calibration')
    ax.set_xlabel("Predicted Survival")
    ax.set_ylabel("Observed Survival (KM)")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    return fig

# ============================================================================
# MATPLOTLIB BACKEND - RESIDUAL PLOTS
# ============================================================================

def _schoenfeld_residuals_matplotlib(
    residuals: np.ndarray,
    times: np.ndarray,
    covariate_name: str,
    title: str,
    height: int,
    width: int
) -> plt.Figure:
    """Build Matplotlib Schoenfeld residuals plot."""
    fig, ax = plt.subplots(figsize=(width/100, height/100))

    ax.scatter(times, residuals, color='#2a9d8f', s=40, alpha=0.7)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)

    ax.set_xlabel("Time")
    ax.set_ylabel(f"Schoenfeld Residual ({covariate_name})")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    return fig


def _martingale_residuals_matplotlib(
    residuals: np.ndarray,
    covariate: np.ndarray,
    covariate_name: str,
    title: str,
    add_smoother: bool,
    height: int,
    width: int
) -> plt.Figure:
    """Build Matplotlib Martingale residuals plot."""
    fig, ax = plt.subplots(figsize=(width/100, height/100))

    ax.scatter(covariate, residuals, color='#2a9d8f', s=40, alpha=0.6)

    # Smoother
    if add_smoother and len(covariate) > 10:
        try:
            import statsmodels.api as sm
            lowess = sm.nonparametric.lowess(residuals, covariate, frac=0.3)
            ax.plot(lowess[:, 0], lowess[:, 1], color='#e76f51', linewidth=2,
                    label='Smoother')
            ax.legend()
        except Exception:
            pass

    ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)

    ax.set_xlabel(covariate_name)
    ax.set_ylabel("Martingale Residual")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    return fig


def _deviance_residuals_matplotlib(
    residuals: np.ndarray,
    covariate: np.ndarray,
    covariate_name: str,
    title: str,
    height: int,
    width: int
) -> plt.Figure:
    """Build Matplotlib Deviance residuals plot."""
    fig, ax = plt.subplots(figsize=(width/100, height/100))

    ax.scatter(covariate, residuals, color='#2a9d8f', s=40, alpha=0.6)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)

    ax.set_xlabel(covariate_name)
    ax.set_ylabel("Deviance Residual")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    return fig