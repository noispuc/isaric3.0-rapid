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

def bootstrap_ci_plot(
    ci_df: pd.DataFrame,
    metric: str,
    x_col: str = "n_clusters",
    theta_col: str = "theta_hat",
    lower_col: str = "ci_low",
    upper_col: str = "ci_high",
    title: Optional[str] = None,
    xaxis_title: Optional[str] = None,
    yaxis_title: Optional[str] = None,
    color: str = '#2a9d8f',
    height: int = 500,
    width: int = 700,
    backend: str = "plotly"
):
    """
    Plot a metric with its bootstrap confidence intervals.

    Args:
        ci_df: DataFrame with columns x_col, theta_col, lower_col, upper_col.
        metric: Metric name (for the title).
        x_col: Column for x-axis (default "n_clusters").
        theta_col: Column for point estimates.
        lower_col: Column for lower CI.
        upper_col: Column for upper CI.
        title: Plot title (default: f"{metric} with Bootstrap CI").
        xaxis_title: X-axis label.
        yaxis_title: Y-axis label.
        color: Line color.
        height: Figure height in pixels.
        width: Figure width in pixels.
        backend: "plotly" or "matplotlib".

    Returns:
        Plotly Figure or Matplotlib Figure.
    """
    if title is None:
        title = f"{metric} with Bootstrap CI"

    # Filtra o metric escolhido
    df = ci_df[ci_df['metric'] == metric].copy()
    if df.empty:
        raise ValueError(f"Metric '{metric}' not found in ci_df.")

    df = df.sort_values(x_col)

    if backend == "plotly":
        return _bootstrap_ci_plotly(
            df, title, xaxis_title or x_col, yaxis_title or metric,
            color, height, width, theta_col, lower_col, upper_col, x_col
        )
    elif backend == "matplotlib":
        return _bootstrap_ci_matplotlib(
            df, title, xaxis_title or x_col, yaxis_title or metric,
            color, height, width, theta_col, lower_col, upper_col, x_col
        )
    else:
        raise ValueError(f"backend must be 'plotly' or 'matplotlib'. Received: {backend}")

def selection_rate_plot(
    selection_rate_df: pd.DataFrame,
    x_col: str = "n_clusters",
    metrics: Optional[List[str]] = None,
    title: str = "Bootstrap Selection Rate",
    xaxis_title: Optional[str] = None,
    yaxis_title: str = "Selection Rate (%)",
    height: int = 500,
    width: int = 800,
    backend: str = "plotly"
):
    """
    Plot bootstrap selection rate per metric and k.

    Args:
        selection_rate_df: DataFrame from bootstrap_selection_rate.
        x_col: Column for x-axis (default "n_clusters").
        metrics: List of metric columns to plot.
            If None, uses all numeric columns except x_col.
        title: Plot title.
        xaxis_title: X-axis label.
        yaxis_title: Y-axis label.
        height: Figure height in pixels.
        width: Figure width in pixels.
        backend: "plotly" or "matplotlib".

    Returns:
        Plotly Figure or Matplotlib Figure.
    """
    if x_col not in selection_rate_df.columns:
        raise ValueError(f"Column '{x_col}' not found in DataFrame.")

    if metrics is None:
        metrics = [
            c for c in selection_rate_df.columns
            if c != x_col and pd.api.types.is_numeric_dtype(selection_rate_df[c])
        ]

    if not metrics:
        raise ValueError("No metric columns to plot.")

    if backend == "plotly":
        return _selection_rate_plotly(
            selection_rate_df, metrics, x_col, title,
            xaxis_title or x_col, yaxis_title, height, width
        )
    elif backend == "matplotlib":
        return _selection_rate_plot_matplotlib(
            selection_rate_df, metrics, x_col, title,
            xaxis_title or x_col, yaxis_title, height, width
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

def _bootstrap_ci_plotly(
    df, title, xaxis_title, yaxis_title, color, height, width,
    theta_col, lower_col, upper_col, x_col
) -> go.Figure:
    """Build Plotly bootstrap CI plot."""
    fig = go.Figure()

    # Faixa de CI
    fig.add_trace(go.Scatter(
        x=list(df[x_col]) + list(df[x_col])[::-1],
        y=list(df[upper_col]) + list(df[lower_col])[::-1],
        fill='toself',
        fillcolor='rgba(42, 157, 143, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=False,
        hoverinfo='skip'
    ))

    # Pontos observados
    fig.add_trace(go.Scatter(
        x=df[x_col],
        y=df[theta_col],
        mode='lines+markers',
        line=dict(color=color, width=2),
        marker=dict(size=10),
        name='Observed',
        hovertemplate='k: %{x}<br>Value: %{y:.4f}<extra></extra>'
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

def _selection_rate_plotly(
    df, metrics, x_col, title, xaxis_title, yaxis_title, height, width
) -> go.Figure:
    """Build Plotly selection rate plot."""
    fig = go.Figure()

    colors = ['#2a9d8f', '#e76f51', '#264653', '#e9c46a', '#f4a261', '#8a5a44']

    for i, metric in enumerate(metrics):
        fig.add_trace(go.Scatter(
            x=df[x_col],
            y=df[metric],
            mode='lines+markers',
            name=metric,
            line=dict(color=colors[i % len(colors)], width=2),
            marker=dict(size=8),
            hovertemplate=f'{metric}<br>k: %{{x}}<br>Rate: %{{y:.1f}}%<extra></extra>'
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

def grid_search_metric_plot(
    grid_results: pd.DataFrame,
    metrics: List[str],
    x_col: str = "n_clusters",
    convergence_col: Optional[str] = "converged",
    title: str = "Metric vs. Number of Clusters",
    xaxis_title: Optional[str] = None,
    yaxis_title: str = "Value",
    height: int = 500,
    width: int = 800,
    backend: str = "plotly"
):
    """
    Plot metric(s) from a grid search across the number of clusters.

    Optionally marks points by convergence status (o = converged,
    x = failed).

    Args:
        grid_results: DataFrame with grid search results.
        metrics: List of metric column names to plot.
        x_col: Column for x-axis (default: "n_clusters").
        convergence_col: Column indicating convergence. If None, no markers.
        title: Plot title.
        xaxis_title: X-axis label (defaults to x_col).
        yaxis_title: Y-axis label.
        height: Figure height in pixels.
        width: Figure width in pixels.
        backend: "plotly" or "matplotlib".

    Returns:
        Plotly Figure or Matplotlib Figure.

    Raises:
        ValueError: If required columns not found or backend invalid.
    """
    if x_col not in grid_results.columns:
        raise ValueError(
            f"Column '{x_col}' not found in grid_results. "
            f"Available: {grid_results.columns.tolist()}"
        )

    metrics = [m for m in metrics if m in grid_results.columns]
    if not metrics:
        raise ValueError(
            f"None of the specified metrics found in grid_results. "
            f"Available: {grid_results.columns.tolist()}"
        )

    if backend == "plotly":
        return _grid_search_metric_plotly(
            grid_results, metrics, x_col, convergence_col, title,
            xaxis_title or x_col, yaxis_title, height, width
        )
    elif backend == "matplotlib":
        return _grid_search_metric_plot_matplotlib(
            grid_results, metrics, x_col, convergence_col, title,
            xaxis_title or x_col, yaxis_title, height, width
        )
    else:
        raise ValueError(
            f"backend must be 'plotly' or 'matplotlib'. Received: {backend}"
        )


def _grid_search_metric_plotly(
    grid_results, metrics, x_col, convergence_col, title,
    xaxis_title, yaxis_title, height, width
) -> go.Figure:
    """Build Plotly grid search metric plot with convergence markers."""
    fig = go.Figure()

    colors = ['#2a9d8f', '#e76f51', '#264653', '#e9c46a', '#f4a261']

    for i, metric in enumerate(metrics):
        color = colors[i % len(colors)]

        # Linha principal
        fig.add_trace(go.Scatter(
            x=grid_results[x_col],
            y=grid_results[metric],
            mode='lines',
            name=metric,
            line=dict(color=color, width=2),
            showlegend=True,
            hovertemplate=f'{metric}<br>{x_col}: %{{x}}<br>Value: %{{y:.2f}}<extra></extra>'
        ))

        # Markers de convergência
        if convergence_col and convergence_col in grid_results.columns:
            # Converte para bool/str para filtrar
            conv_series = grid_results[convergence_col].astype(str).str.lower()

            # Converte 'true' para True
            conv_true_mask = conv_series.isin(['true', '1', 'yes'])
            conv_false_mask = ~conv_true_mask

            # Convergiu (círculo)
            if conv_true_mask.any():
                fig.add_trace(go.Scatter(
                    x=grid_results.loc[conv_true_mask, x_col],
                    y=grid_results.loc[conv_true_mask, metric],
                    mode='markers',
                    marker=dict(color=color, size=10, symbol='circle'),
                    name=f'{metric} (converged)',
                    showlegend=False,
                    hovertemplate=f'{metric} converged<br>{x_col}: %{{x}}<br>Value: %{{y:.2f}}<extra></extra>'
                ))

            # Não convergiu (x)
            if conv_false_mask.any():
                fig.add_trace(go.Scatter(
                    x=grid_results.loc[conv_false_mask, x_col],
                    y=grid_results.loc[conv_false_mask, metric],
                    mode='markers',
                    marker=dict(color=color, size=10, symbol='x'),
                    name=f'{metric} (failed)',
                    showlegend=False,
                    hovertemplate=f'{metric} failed<br>{x_col}: %{{x}}<br>Value: %{{y:.2f}}<extra></extra>'
                ))
        else:
            # Sem coluna de convergência: só markers
            fig.add_trace(go.Scatter(
                x=grid_results[x_col],
                y=grid_results[metric],
                mode='markers',
                marker=dict(color=color, size=8),
                showlegend=False,
                hoverinfo='skip'
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


def _grid_search_metric_plot_matplotlib(
    grid_results, metrics, x_col, convergence_col, title,
    xaxis_title, yaxis_title, height, width
) -> plt.Figure:
    """Build Matplotlib grid search metric plot with convergence markers."""
    fig, ax = plt.subplots(figsize=(width/100, height/100))

    colors = ['#2a9d8f', '#e76f51', '#264653', '#e9c46a', '#f4a261']

    for i, metric in enumerate(metrics):
        color = colors[i % len(colors)]

        # Linha
        ax.plot(
            grid_results[x_col],
            grid_results[metric],
            color=color,
            linewidth=2,
            label=metric
        )

        # Markers de convergência
        if convergence_col and convergence_col in grid_results.columns:
            conv_series = grid_results[convergence_col].astype(str).str.lower()
            conv_true_mask = conv_series.isin(['true', '1', 'yes'])
            conv_false_mask = ~conv_true_mask

            ax.scatter(
                grid_results.loc[conv_true_mask, x_col],
                grid_results.loc[conv_true_mask, metric],
                color=color, s=80, marker='o', zorder=5
            )
            ax.scatter(
                grid_results.loc[conv_false_mask, x_col],
                grid_results.loc[conv_false_mask, metric],
                color=color, s=80, marker='x', zorder=5
            )
        else:
            ax.scatter(
                grid_results[x_col],
                grid_results[metric],
                color=color, s=50, zorder=5
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

def _bootstrap_ci_matplotlib(
    df, title, xaxis_title, yaxis_title, color, height, width,
    theta_col, lower_col, upper_col, x_col
) -> plt.Figure:
    """Build Matplotlib bootstrap CI plot."""
    fig, ax = plt.subplots(figsize=(width/100, height/100))

    ax.fill_between(
        df[x_col], df[lower_col], df[upper_col],
        alpha=0.2, color=color, label='95% CI'
    )
    ax.plot(
        df[x_col], df[theta_col],
        'o-', color=color, linewidth=2, markersize=8, label='Observed'
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


def _selection_rate_plot_matplotlib(
    df, metrics, x_col, title, xaxis_title, yaxis_title, height, width
) -> plt.Figure:
    """Build Matplotlib selection rate plot."""
    fig, ax = plt.subplots(figsize=(width/100, height/100))

    colors = ['#2a9d8f', '#e76f51', '#264653', '#e9c46a', '#f4a261', '#8a5a44']

    for i, metric in enumerate(metrics):
        ax.plot(
            df[x_col], df[metric],
            marker='o', color=colors[i % len(colors)],
            linewidth=2, markersize=8, label=metric
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