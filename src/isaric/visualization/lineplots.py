"""
Line plot visualization for the RAPID methodology.

This module provides functions to generate line plots (Step 6.1 of
the RAPID methodology). Line plots display data that changes
continuously over a specified range or time, illustrating trends.

Backends:
- plotly: Interactive figures for notebook display (default).
- matplotlib: Static figures for report export (PNG, PDF).

Techniques:
- time_series_plot: Line plot for trends over time.
- multi_line_plot: Multiple lines for comparison.
- line_with_ci: Line plot with confidence intervals.
"""

import pandas as pd
import numpy as np
from typing import List, Optional
import plotly.graph_objs as go
import matplotlib.pyplot as plt


def time_series_plot(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str = "Time Series Plot",
    xaxis_title: Optional[str] = None,
    yaxis_title: Optional[str] = None,
    color: str = '#2a9d8f',
    height: int = 500,
    width: int = 700,
    backend: str = "plotly"
):
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
        backend: "plotly" or "matplotlib".

    Returns:
        Plotly Figure or Matplotlib Figure.

    Raises:
        ValueError: If required columns are not found or backend invalid.
    """
    for col in [x_col, y_col]:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    if backend == "plotly":
        return _time_series_plotly(
            data=data,
            x_col=x_col,
            y_col=y_col,
            title=title,
            xaxis_title=xaxis_title or x_col,
            yaxis_title=yaxis_title or y_col,
            color=color,
            height=height,
            width=width
        )
    elif backend == "matplotlib":
        return _time_series_matplotlib(
            data=data,
            x_col=x_col,
            y_col=y_col,
            title=title,
            xaxis_title=xaxis_title or x_col,
            yaxis_title=yaxis_title or y_col,
            color=color,
            height=height,
            width=width
        )
    else:
        raise ValueError(f"backend must be 'plotly' or 'matplotlib'. Received: {backend}")


def multi_line_plot(
    data: pd.DataFrame,
    x_col: str,
    y_cols: List[str],
    title: str = "Multi-Line Plot",
    xaxis_title: Optional[str] = None,
    yaxis_title: Optional[str] = None,
    colors: Optional[List[str]] = None,
    height: int = 500,
    width: int = 700,
    backend: str = "plotly"
):
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
        backend: "plotly" or "matplotlib".

    Returns:
        Plotly Figure or Matplotlib Figure.

    Raises:
        ValueError: If required columns are not found or backend invalid.
    """
    if x_col not in data.columns:
        raise ValueError(f"Column '{x_col}' not found in DataFrame.")

    for col in y_cols:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    if colors is None:
        colors = ['#2a9d8f', '#e76f51', '#264653', '#e9c46a', '#f4a261']

    if backend == "plotly":
        return _multi_line_plotly(
            data=data,
            x_col=x_col,
            y_cols=y_cols,
            title=title,
            xaxis_title=xaxis_title or x_col,
            yaxis_title=yaxis_title or 'Value',
            colors=colors,
            height=height,
            width=width
        )
    elif backend == "matplotlib":
        return _multi_line_matplotlib(
            data=data,
            x_col=x_col,
            y_cols=y_cols,
            title=title,
            xaxis_title=xaxis_title or x_col,
            yaxis_title=yaxis_title or 'Value',
            colors=colors,
            height=height,
            width=width
        )
    else:
        raise ValueError(f"backend must be 'plotly' or 'matplotlib'. Received: {backend}")


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
    width: int = 700,
    backend: str = "plotly"
):
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
        backend: "plotly" or "matplotlib".

    Returns:
        Plotly Figure or Matplotlib Figure.

    Raises:
        ValueError: If required columns are not found or backend invalid.
    """
    for col in [x_col, y_col, ci_lower_col, ci_upper_col]:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    if backend == "plotly":
        return _line_with_ci_plotly(
            data=data,
            x_col=x_col,
            y_col=y_col,
            ci_lower_col=ci_lower_col,
            ci_upper_col=ci_upper_col,
            title=title,
            xaxis_title=xaxis_title or x_col,
            yaxis_title=yaxis_title or y_col,
            color=color,
            height=height,
            width=width
        )
    elif backend == "matplotlib":
        return _line_with_ci_matplotlib(
            data=data,
            x_col=x_col,
            y_col=y_col,
            ci_lower_col=ci_lower_col,
            ci_upper_col=ci_upper_col,
            title=title,
            xaxis_title=xaxis_title or x_col,
            yaxis_title=yaxis_title or y_col,
            color=color,
            height=height,
            width=width
        )
    else:
        raise ValueError(f"backend must be 'plotly' or 'matplotlib'. Received: {backend}")


# ============================================================================
# PLOTLY BACKEND
# ============================================================================

def _time_series_plotly(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    xaxis_title: str,
    yaxis_title: str,
    color: str,
    height: int,
    width: int
) -> go.Figure:
    """Build Plotly time series plot."""
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
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        height=height,
        width=width,
        template='plotly_white'
    )

    return fig


def _multi_line_plotly(
    data: pd.DataFrame,
    x_col: str,
    y_cols: List[str],
    title: str,
    xaxis_title: str,
    yaxis_title: str,
    colors: List[str],
    height: int,
    width: int
) -> go.Figure:
    """Build Plotly multi-line plot."""
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
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        height=height,
        width=width,
        template='plotly_white',
        hovermode='x unified'
    )

    return fig


def _line_with_ci_plotly(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    ci_lower_col: str,
    ci_upper_col: str,
    title: str,
    xaxis_title: str,
    yaxis_title: str,
    color: str,
    height: int,
    width: int
) -> go.Figure:
    """Build Plotly line with confidence intervals."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=list(data[x_col]) + list(data[x_col])[::-1],
        y=list(data[ci_upper_col]) + list(data[ci_lower_col])[::-1],
        fill='toself',
        fillcolor='rgba(42, 157, 143, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=False,
        hoverinfo='skip'
    ))

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
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        height=height,
        width=width,
        template='plotly_white'
    )

    return fig


# ============================================================================
# MATPLOTLIB BACKEND
# ============================================================================

def _time_series_matplotlib(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    xaxis_title: str,
    yaxis_title: str,
    color: str,
    height: int,
    width: int
) -> plt.Figure:
    """Build Matplotlib time series plot."""
    fig, ax = plt.subplots(figsize=(width/100, height/100))
    
    ax.plot(data[x_col], data[y_col], color=color, linewidth=2, marker='o', markersize=6)
    
    ax.set_xlabel(xaxis_title)
    ax.set_ylabel(yaxis_title)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    fig.tight_layout()
    return fig


def _multi_line_matplotlib(
    data: pd.DataFrame,
    x_col: str,
    y_cols: List[str],
    title: str,
    xaxis_title: str,
    yaxis_title: str,
    colors: List[str],
    height: int,
    width: int
) -> plt.Figure:
    """Build Matplotlib multi-line plot."""
    fig, ax = plt.subplots(figsize=(width/100, height/100))
    
    for i, col in enumerate(y_cols):
        ax.plot(
            data[x_col],
            data[col],
            color=colors[i % len(colors)],
            linewidth=2,
            marker='o',
            markersize=6,
            label=col
        )
    
    ax.set_xlabel(xaxis_title)
    ax.set_ylabel(yaxis_title)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    fig.tight_layout()
    return fig


def _line_with_ci_matplotlib(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    ci_lower_col: str,
    ci_upper_col: str,
    title: str,
    xaxis_title: str,
    yaxis_title: str,
    color: str,
    height: int,
    width: int
) -> plt.Figure:
    """Build Matplotlib line with confidence intervals."""
    fig, ax = plt.subplots(figsize=(width/100, height/100))
    
    # Confidence interval band
    ax.fill_between(
        data[x_col],
        data[ci_lower_col],
        data[ci_upper_col],
        alpha=0.2,
        color=color
    )
    
    # Main line
    ax.plot(
        data[x_col],
        data[y_col],
        color=color,
        linewidth=2,
        marker='o',
        markersize=6
    )
    
    ax.set_xlabel(xaxis_title)
    ax.set_ylabel(yaxis_title)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    fig.tight_layout()
    return fig