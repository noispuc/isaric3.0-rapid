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

def binned_residuals_plot(
    bin_centers: np.ndarray,
    mean_residuals: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
    n_per_bin: Optional[np.ndarray] = None,
    title: str = "Binned Residual Plot",
    xaxis_title: str = "Predicted Probability",
    yaxis_title: str = "Mean Residual",
    color: str = '#2a9d8f',
    height: int = 500,
    width: int = 700,
    backend: str = "plotly"
):
    """
    Generate a binned residual plot for binary classification models.

    Each point represents the mean residual in a bin of predicted
    probabilities. The error bars correspond to ±2 standard errors.
    If the model is well-calibrated, most points should fall within
    the bounds.

    Args:
        bin_centers: Mean predicted probability per bin.
        mean_residuals: Mean residual per bin.
        lower_bound: Lower confidence bound (±2 SE).
        upper_bound: Upper confidence bound (±2 SE).
        n_per_bin: Number of observations per bin (optional).
        title: Plot title.
        xaxis_title: X-axis label.
        yaxis_title: Y-axis label.
        color: Marker color.
        height: Figure height in pixels.
        width: Figure width in pixels.
        backend: "plotly" or "matplotlib".

    Returns:
        Plotly Figure or Matplotlib Figure.

    Raises:
        ValueError: If backend invalid.
    """
    if backend == "plotly":
        return _binned_residuals_plotly(
            bin_centers, mean_residuals, lower_bound, upper_bound,
            n_per_bin, title, xaxis_title, yaxis_title, color, height, width
        )
    elif backend == "matplotlib":
        return _binned_residuals_matplotlib(
            bin_centers, mean_residuals, lower_bound, upper_bound,
            n_per_bin, title, xaxis_title, yaxis_title, color, height, width
        )
    else:
        raise ValueError(
            f"backend must be 'plotly' or 'matplotlib'. Received: {backend}"
        )

def information_criteria_plot(
    grid_results: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    x_col: str = "n_clusters",
    title: str = "Information Criteria by Number of Clusters",
    xaxis_title: Optional[str] = None,
    yaxis_title: str = "Value",
    height: int = 500,
    width: int = 800,
    backend: str = "plotly"
):
    """
    Generate a multi-line plot of information criteria across models.

    Useful for class enumeration in LCA and cluster selection in KMeans,
    showing how AIC, BIC, CAIC, SABIC (and entropy) vary with the number
    of components/clusters.

    Args:
        grid_results: DataFrame with columns for each metric.
        metrics: List of metric column names to plot.
            Default: ["AIC", "BIC", "CAIC", "SABIC"].
        x_col: Column for x-axis (default: "n_clusters").
        title: Plot title.
        xaxis_title: X-axis label (defaults to x_col).
        yaxis_title: Y-axis label.
        height: Figure height in pixels.
        width: Figure width in pixels.
        backend: "plotly" or "matplotlib".

    Returns:
        Plotly Figure or Matplotlib Figure.

    Raises:
        ValueError: If required columns are not found or backend invalid.
    """
    if x_col not in grid_results.columns:
        raise ValueError(
            f"Column '{x_col}' not found in grid_results. "
            f"Available: {grid_results.columns.tolist()}"
        )

    if metrics is None:
        metrics = ["AIC", "BIC", "CAIC", "SABIC"]

    # Filtra só métricas que existem
    metrics = [m for m in metrics if m in grid_results.columns]
    if not metrics:
        raise ValueError(
            f"None of the specified metrics found in grid_results. "
            f"Available: {grid_results.columns.tolist()}"
        )

    if backend == "plotly":
        return _information_criteria_plotly(
            grid_results, metrics, x_col, title,
            xaxis_title or x_col, yaxis_title, height, width
        )
    elif backend == "matplotlib":
        return _information_criteria_matplotlib(
            grid_results, metrics, x_col, title,
            xaxis_title or x_col, yaxis_title, height, width
        )
    else:
        raise ValueError(
            f"backend must be 'plotly' or 'matplotlib'. Received: {backend}"
        )

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


def _binned_residuals_plotly(
    bin_centers, mean_residuals, lower_bound, upper_bound,
    n_per_bin, title, xaxis_title, yaxis_title, color, height, width
) -> go.Figure:
    """Build Plotly binned residual plot."""
    import plotly.graph_objs as go

    fig = go.Figure()

    # Faixa de confiança (upper + lower, preenchida)
    fig.add_trace(go.Scatter(
        x=list(bin_centers) + list(bin_centers)[::-1],
        y=list(upper_bound) + list(lower_bound)[::-1],
        fill='toself',
        fillcolor='rgba(231, 111, 81, 0.15)',
        line=dict(color='rgba(0,0,0,0)'),
        hoverinfo='skip',
        showlegend=False,
        name='±2 SE'
    ))

    # Linha zero
    fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.6)

    # Pontos médios
    hovertext = None
    if n_per_bin is not None:
        hovertext = [f"n={int(n)}" for n in n_per_bin]

    fig.add_trace(go.Scatter(
        x=bin_centers,
        y=mean_residuals,
        mode='markers',
        marker=dict(color=color, size=10),
        name='Bins',
        text=hovertext,
        hovertemplate=(
            'Predicted: %{x:.3f}<br>'
            'Mean Residual: %{y:.4f}<br>'
            '%{text}<extra></extra>'
        )
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

def _information_criteria_plotly(
    grid_results, metrics, x_col, title, xaxis_title, yaxis_title, height, width
) -> go.Figure:
    """Build Plotly information criteria plot."""
    fig = go.Figure()

    colors = ['#2a9d8f', '#e76f51', '#264653', '#e9c46a', '#f4a261']
    markers = ['circle', 'square', 'diamond', 'triangle-up', 'cross']

    for i, metric in enumerate(metrics):
        fig.add_trace(go.Scatter(
            x=grid_results[x_col],
            y=grid_results[metric],
            mode='lines+markers',
            name=metric,
            line=dict(color=colors[i % len(colors)], width=2),
            marker=dict(
                size=10,
                symbol=markers[i % len(markers)]
            ),
            hovertemplate=(
                f'{metric}<br>'
                f'{x_col}: %{{x}}<br>'
                f'Value: %{{y:.2f}}<extra></extra>'
            )
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

def _binned_residuals_matplotlib(
    bin_centers, mean_residuals, lower_bound, upper_bound,
    n_per_bin, title, xaxis_title, yaxis_title, color, height, width
) -> plt.Figure:
    """Build Matplotlib binned residual plot."""
    fig, ax = plt.subplots(figsize=(width/100, height/100))

    # Faixa de confiança
    ax.fill_between(
        bin_centers,
        lower_bound,
        upper_bound,
        alpha=0.15,
        color='#e76f51',
        label='±2 SE'
    )

    # Linha zero
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.6)

    # Pontos
    ax.scatter(
        bin_centers, mean_residuals,
        color=color, s=60, zorder=5, label='Bins'
    )

    ax.set_xlabel(xaxis_title)
    ax.set_ylabel(yaxis_title)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    return fig

def _information_criteria_matplotlib(
    grid_results, metrics, x_col, title, xaxis_title, yaxis_title, height, width
) -> plt.Figure:
    """Build Matplotlib information criteria plot."""
    fig, ax = plt.subplots(figsize=(width/100, height/100))

    colors = ['#2a9d8f', '#e76f51', '#264653', '#e9c46a', '#f4a261']
    markers = ['o', 's', 'D', '^', 'x']

    for i, metric in enumerate(metrics):
        ax.plot(
            grid_results[x_col],
            grid_results[metric],
            marker=markers[i % len(markers)],
            color=colors[i % len(colors)],
            linewidth=2,
            markersize=8,
            label=metric
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
