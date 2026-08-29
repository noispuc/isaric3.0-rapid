"""
Line plot visualization for the RAPID methodology.

This module provides functions to generate line plots (Step 6.1 of
the RAPID methodology). Line plots display data that changes
continuously over a specified range or time, illustrating trends.

Techniques:
- time_series_plot: Line plot for trends over time.
- multi_line_plot: Multiple lines for comparison.
- line_with_ci: Line plot with confidence intervals.
"""

import pandas as pd
import numpy as np
from typing import List, Optional
import plotly.graph_objs as go


def time_series_plot(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str = "Time Series Plot",
    xaxis_title: Optional[str] = None,
    yaxis_title: Optional[str] = None,
    color: str = '#2a9d8f',
    height: int = 500,
    width: int = 700
) -> go.Figure:
    """
    Generate a time series line plot.

    Shows the trend of a single metric over time.

    Args:
        data: Input DataFrame.
        x_col: Column for time (x-axis).
        y_col: Column for values (y-axis).
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
    for col in [x_col, y_col]:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    fig = go.Figure(data=go.Scatter(
        x=data[x_col],
        y=data[y_col],
        mode='lines+markers',
        line=dict(color=color, width=2),
        marker=dict(size=6),
        hovertemplate='%{x}<br>%{y}<extra></extra>'
    ))

    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title or x_col,
        yaxis_title=yaxis_title or y_col,
        height=height,
        width=width,
        template='plotly_white'
    )

    return fig


def multi_line_plot(
    data: pd.DataFrame,
    x_col: str,
    y_cols: List[str],
    title: str = "Multi-Line Plot",
    xaxis_title: Optional[str] = None,
    yaxis_title: Optional[str] = None,
    colors: Optional[List[str]] = None,
    height: int = 500,
    width: int = 700
) -> go.Figure:
    """
    Generate a line plot with multiple lines for comparison.

    Plots multiple y columns against the same x column.

    Args:
        data: Input DataFrame.
        x_col: Column for x-axis (shared).
        y_cols: List of columns for y-axis (multiple lines).
        title: Plot title.
        xaxis_title: X-axis label.
        yaxis_title: Y-axis label.
        colors: List of line colors (optional).
        height: Figure height in pixels.
        width: Figure width in pixels.

    Returns:
        Plotly Figure object.

    Raises:
        ValueError: If required columns are not found.
    """
    if x_col not in data.columns:
        raise ValueError(f"Column '{x_col}' not found in DataFrame.")

    for col in y_cols:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    if colors is None:
        colors = ['#2a9d8f', '#e76f51', '#264653', '#e9c46a', '#f4a261']

    fig = go.Figure()

    for i, col in enumerate(y_cols):
        fig.add_trace(go.Scatter(
            x=data[x_col],
            y=data[col],
            mode='lines+markers',
            name=col,
            line=dict(color=colors[i % len(colors)], width=2),
            marker=dict(size=6),
            hovertemplate=f'{col}<br>%{{x}}<br>%{{y}}<extra></extra>'
        ))

    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title or x_col,
        yaxis_title=yaxis_title or 'Value',
        height=height,
        width=width,
        template='plotly_white',
        hovermode='x unified'
    )

    return fig


def line_with_ci(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    ci_lower_col: str,
    ci_upper_col: str,
    title: str = "Line Plot with Confidence Intervals",
    xaxis_title: Optional[str] = None,
    yaxis_title: Optional[str] = None,
    color: str = '#2a9d8f',
    height: int = 500,
    width: int = 700
) -> go.Figure:
    """
    Generate a line plot with confidence intervals.

    Shows the trend of a metric with shaded confidence bands.

    Args:
        data: Input DataFrame.
        x_col: Column for x-axis.
        y_col: Column for values (y-axis).
        ci_lower_col: Column for lower CI bound.
        ci_upper_col: Column for upper CI bound.
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
    for col in [x_col, y_col, ci_lower_col, ci_upper_col]:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    fig = go.Figure()

    # Confidence interval band
    fig.add_trace(go.Scatter(
        x=list(data[x_col]) + list(data[x_col])[::-1],
        y=list(data[ci_upper_col]) + list(data[ci_lower_col])[::-1],
        fill='toself',
        fillcolor='rgba(42, 157, 143, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=False,
        hoverinfo='skip'
    ))

    # Main line
    fig.add_trace(go.Scatter(
        x=data[x_col],
        y=data[y_col],
        mode='lines+markers',
        line=dict(color=color, width=2),
        marker=dict(size=6),
        hovertemplate='%{x}<br>%{y}<extra></extra>'
    ))

    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title or x_col,
        yaxis_title=yaxis_title or y_col,
        height=height,
        width=width,
        template='plotly_white'
    )

    return fig