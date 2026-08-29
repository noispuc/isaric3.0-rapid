"""
Survival curve visualization for the RAPID methodology.

This module provides functions to generate survival curves (Step 6.3
of the RAPID methodology). Survival curves display the probability of
surviving past a certain time using the Kaplan-Meier estimator.

Techniques:
- kaplan_meier_curve: Plot survival curve from lifelines KMF.
- compare_survival_curves: Compare survival between groups.
- baseline_survival_curve: Plot baseline survival from Cox model.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
import plotly.graph_objs as go
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
    width: int = 700
) -> go.Figure:
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

    Returns:
        Plotly Figure object.

    Raises:
        ValueError: If required columns are not found.
    """
    for col in [duration_var, event_var]:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    kmf = KaplanMeierFitter()
    kmf.fit(
        data[duration_var],
        event_observed=data[event_var]
    )

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
    width: int = 700
) -> go.Figure:
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

    Returns:
        Plotly Figure object.

    Raises:
        ValueError: If required columns are not found.
    """
    for col in [duration_var, event_var, group_col]:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    if group_values is None:
        group_values = data[group_col].unique().tolist()

    if colors is None:
        colors = ['#2a9d8f', '#e76f51', '#264653', '#e9c46a', '#f4a261']

    fig = go.Figure()
    kmf = KaplanMeierFitter()

    for i, group in enumerate(group_values):
        group_data = data[data[group_col] == group]

        if len(group_data) == 0:
            continue

        kmf.fit(
            group_data[duration_var],
            event_observed=group_data[event_var],
            label=str(group)
        )

        fig.add_trace(go.Scatter(
            x=kmf.survival_function_.index,
            y=kmf.survival_function_.iloc[:, 0].values,
            mode='lines',
            line=dict(color=colors[i % len(colors)], width=2),
            name=str(group),
            hovertemplate=f'Group: {group}<br>Time: %{{x:.1f}}<br>Survival: %{{y:.3f}}<extra></extra>'
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


def baseline_survival_curve(
    fitted_model,
    title: str = "Baseline Survival Curve (Cox Model)",
    xaxis_title: str = "Time",
    yaxis_title: str = "Survival Probability",
    color: str = '#2a9d8f',
    height: int = 500,
    width: int = 700
) -> go.Figure:
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

    Returns:
        Plotly Figure object.

    Raises:
        ValueError: If model does not have baseline_survival_.
    """
    if not hasattr(fitted_model, 'baseline_survival_'):
        raise ValueError(
            "Fitted model does not have baseline_survival_ attribute."
        )

    survival_func = fitted_model.baseline_survival_

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