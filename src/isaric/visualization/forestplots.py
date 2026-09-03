"""
Forest plot visualization for the RAPID methodology.

This module provides functions to generate forest plots (Step 6.5 of
the RAPID methodology). Forest plots display estimated effect sizes
and confidence intervals for multiple predictors.

Backends:
- plotly: Interactive figures for notebook display (default).
- matplotlib: Static figures for report export (PNG, PDF).

Techniques:
- odds_ratio_plot: Forest plot for Odds Ratios (logistic regression).
- hazard_ratio_plot: Forest plot for Hazard Ratios (Cox model).
- coefficient_plot: Forest plot for regression coefficients.
"""

import pandas as pd
import numpy as np
from typing import Optional
import plotly.graph_objs as go
import matplotlib.pyplot as plt


def odds_ratio_plot(
    df: pd.DataFrame,
    label_col: str = 'Variable',
    effect_col: str = 'OddsRatio',
    lower_col: str = 'LowerCI',
    upper_col: str = 'UpperCI',
    title: str = "Forest Plot - Odds Ratios",
    height: int = 600,
    width: int = 800,
    backend: str = "plotly"
):
    """
    Generate a forest plot for Odds Ratios.

    Displays Odds Ratios with 95% confidence intervals for each
    predictor. A vertical reference line at 1.0 indicates no effect.

    Args:
        df: DataFrame with effect sizes and confidence intervals.
        label_col: Column name for variable labels.
        effect_col: Column name for Odds Ratios.
        lower_col: Column name for lower CI bound.
        upper_col: Column name for upper CI bound.
        title: Plot title.
        height: Figure height in pixels.
        width: Figure width in pixels.
        backend: "plotly" or "matplotlib".

    Returns:
        Plotly Figure or Matplotlib Figure.

    Raises:
        ValueError: If required columns are not found or backend is invalid.
    """
    _validate_columns(df, [label_col, effect_col, lower_col, upper_col])
    
    if backend == "plotly":
        return _build_forest_plotly(
            df=df,
            label_col=label_col,
            effect_col=effect_col,
            lower_col=lower_col,
            upper_col=upper_col,
            title=title,
            xaxis_title="Odds Ratio",
            null_value=1.0,
            log_scale=True,
            height=height,
            width=width
        )
    elif backend == "matplotlib":
        return _build_forest_matplotlib(
            df=df,
            label_col=label_col,
            effect_col=effect_col,
            lower_col=lower_col,
            upper_col=upper_col,
            title=title,
            xaxis_title="Odds Ratio",
            null_value=1.0,
            log_scale=True,
            height=height,
            width=width
        )
    else:
        raise ValueError(f"backend must be 'plotly' or 'matplotlib'. Received: {backend}")


def hazard_ratio_plot(
    df: pd.DataFrame,
    label_col: str = 'Variable',
    effect_col: str = 'HazardRatio',
    lower_col: str = 'LowerCI',
    upper_col: str = 'UpperCI',
    title: str = "Forest Plot - Hazard Ratios",
    height: int = 600,
    width: int = 800,
    backend: str = "plotly"
):
    """
    Generate a forest plot for Hazard Ratios.

    Displays Hazard Ratios with 95% confidence intervals for each
    predictor. A vertical reference line at 1.0 indicates no effect.

    Args:
        df: DataFrame with effect sizes and confidence intervals.
        label_col: Column name for variable labels.
        effect_col: Column name for Hazard Ratios.
        lower_col: Column name for lower CI bound.
        upper_col: Column name for upper CI bound.
        title: Plot title.
        height: Figure height in pixels.
        width: Figure width in pixels.
        backend: "plotly" or "matplotlib".

    Returns:
        Plotly Figure or Matplotlib Figure.

    Raises:
        ValueError: If required columns are not found.
    """
    _validate_columns(df, [label_col, effect_col, lower_col, upper_col])

    if backend == "plotly":
        return _build_forest_plotly(
            df=df,
            label_col=label_col,
            effect_col=effect_col,
            lower_col=lower_col,
            upper_col=upper_col,
            title=title,
            xaxis_title="Hazard Ratio",
            null_value=1.0,
            log_scale=True,
            height=height,
            width=width
        )
    elif backend == "matplotlib":
        return _build_forest_matplotlib(
            df=df,
            label_col=label_col,
            effect_col=effect_col,
            lower_col=lower_col,
            upper_col=upper_col,
            title=title,
            xaxis_title="Hazard Ratio",
            null_value=1.0,
            log_scale=True,
            height=height,
            width=width
        )
    else:
        raise ValueError(f"backend must be 'plotly' or 'matplotlib'. Received: {backend}")


def coefficient_plot(
    df: pd.DataFrame,
    label_col: str = 'Variable',
    effect_col: str = 'Coefficient',
    lower_col: str = 'LowerCI',
    upper_col: str = 'UpperCI',
    title: str = "Forest Plot - Coefficients",
    height: int = 600,
    width: int = 800,
    backend: str = "plotly"
):
    """
    Generate a forest plot for regression coefficients.

    Displays coefficients with 95% confidence intervals for each
    predictor. A vertical reference line at 0.0 indicates no effect.

    Args:
        df: DataFrame with coefficients and confidence intervals.
        label_col: Column name for variable labels.
        effect_col: Column name for coefficients.
        lower_col: Column name for lower CI bound.
        upper_col: Column name for upper CI bound.
        title: Plot title.
        height: Figure height in pixels.
        width: Figure width in pixels.
        backend: "plotly" or "matplotlib".

    Returns:
        Plotly Figure or Matplotlib Figure.

    Raises:
        ValueError: If required columns are not found.
    """
    _validate_columns(df, [label_col, effect_col, lower_col, upper_col])

    if backend == "plotly":
        return _build_forest_plotly(
            df=df,
            label_col=label_col,
            effect_col=effect_col,
            lower_col=lower_col,
            upper_col=upper_col,
            title=title,
            xaxis_title="Coefficient",
            null_value=0.0,
            log_scale=False,
            height=height,
            width=width
        )
    elif backend == "matplotlib":
        return _build_forest_matplotlib(
            df=df,
            label_col=label_col,
            effect_col=effect_col,
            lower_col=lower_col,
            upper_col=upper_col,
            title=title,
            xaxis_title="Coefficient",
            null_value=0.0,
            log_scale=False,
            height=height,
            width=width
        )
    else:
        raise ValueError(f"backend must be 'plotly' or 'matplotlib'. Received: {backend}")


# ============================================================================
# PLOTLY BACKEND
# ============================================================================

def _build_forest_plotly(
    df: pd.DataFrame,
    label_col: str,
    effect_col: str,
    lower_col: str,
    upper_col: str,
    title: str,
    xaxis_title: str,
    null_value: float,
    log_scale: bool,
    height: int,
    width: int
) -> go.Figure:
    """
    Build a Plotly forest plot.
    """
    plot_df = df.copy()
    plot_df = plot_df.sort_values(by=effect_col, ascending=True)

    traces = []

    traces.append(go.Scatter(
        x=plot_df[effect_col],
        y=plot_df[label_col],
        mode='markers',
        name=xaxis_title,
        marker=dict(color='#2a9d8f', size=10),
        hovertemplate='%{y}<br>%{x:.3f}<extra></extra>'
    ))

    for _, row in plot_df.iterrows():
        traces.append(go.Scatter(
            x=[row[lower_col], row[upper_col]],
            y=[row[label_col], row[label_col]],
            mode='lines',
            showlegend=False,
            line=dict(color='#2a9d8f', width=2),
            hoverinfo='skip'
        ))

    xaxis_config = dict(title=xaxis_title)
    if log_scale:
        xaxis_config['type'] = 'log'

    layout = go.Layout(
        title=title,
        xaxis=xaxis_config,
        yaxis=dict(
            title='',
            automargin=True,
            tickmode='array',
            tickvals=plot_df[label_col].tolist(),
            ticktext=plot_df[label_col].tolist()
        ),
        shapes=[
            dict(
                type='line',
                x0=null_value,
                y0=-0.5,
                x1=null_value,
                y1=len(plot_df) - 0.5,
                line=dict(color='red', width=2, dash='dash')
            )
        ],
        margin=dict(l=200, r=50, t=80, b=50),
        height=height,
        width=width,
        template='plotly_white'
    )

    return go.Figure(data=traces, layout=layout)


# ============================================================================
# MATPLOTLIB BACKEND
# ============================================================================

def _build_forest_matplotlib(
    df: pd.DataFrame,
    label_col: str,
    effect_col: str,
    lower_col: str,
    upper_col: str,
    title: str,
    xaxis_title: str,
    null_value: float,
    log_scale: bool,
    height: int,
    width: int
) -> plt.Figure:
    """
    Build a Matplotlib forest plot.
    """
    plot_df = df.copy()
    plot_df = plot_df.sort_values(by=effect_col, ascending=True)
    
    fig, ax = plt.subplots(figsize=(width/100, height/100))
    
    y_positions = range(len(plot_df))
    
    # Plot confidence intervals
    for i, (_, row) in enumerate(plot_df.iterrows()):
        ax.plot(
            [row[lower_col], row[upper_col]],
            [i, i],
            color='#2a9d8f',
            linewidth=2
        )
    
    # Plot point estimates
    ax.scatter(
        plot_df[effect_col],
        y_positions,
        color='#2a9d8f',
        s=80,
        zorder=5
    )
    
    # Null effect line
    ax.axvline(x=null_value, color='red', linestyle='--', linewidth=2)
    
    # Labels
    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(plot_df[label_col].tolist())
    ax.set_xlabel(xaxis_title)
    ax.set_title(title)
    
    if log_scale:
        ax.set_xscale('log')
    
    ax.grid(axis='x', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    fig.tight_layout()
    return fig


def _validate_columns(
    df: pd.DataFrame,
    required_cols: list
) -> None:
    """
    Validate that required columns are present in DataFrame.
    """
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Required columns not found: {missing_cols}"
        )