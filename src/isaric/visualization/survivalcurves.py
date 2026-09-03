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