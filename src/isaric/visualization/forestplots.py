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
    plot_df = plot_df.sort_values(by=effect_col, ascending=True).reset_index(drop=True)

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

    # Calcula range manualmente para evitar ticks esquisitos
    all_values = pd.concat([
        plot_df[effect_col],
        plot_df[lower_col],
        plot_df[upper_col]
    ]).replace([np.inf, -np.inf], np.nan).dropna()

    xaxis_config = dict(title=xaxis_title)

    if log_scale:
        xaxis_config['type'] = 'log'

        # Filtra só valores positivos (log não aceita <= 0)
        positive_values = all_values[all_values > 0]

        if len(positive_values) > 0:
            x_min = positive_values.min()
            x_max = positive_values.max()

            # Margem de 5% em escala log
            log_min = np.log10(x_min)
            log_max = np.log10(x_max)
            log_margin = (log_max - log_min) * 0.05

            xaxis_config['range'] = [
                log_min - log_margin,
                log_max + log_margin
            ]

            # Ticks "bonitos": potências de 10 e alguns intermediários
            # Ex: [0.1, 0.5, 1, 2, 5, 10, 50, 100]
            tick_candidates = [
                0.1, 0.2, 0.5,
                1, 2, 5,
                10, 20, 50,
                100, 200, 500,
                1000
            ]
            ticks = [
                t for t in tick_candidates
                if x_min * 0.9 <= t <= x_max * 1.1
            ]
            if ticks:
                xaxis_config['tickvals'] = ticks
                xaxis_config['ticktext'] = [
                    str(t) for t in ticks
                ]

    else:
        # Escala linear (coeficientes)
        if len(all_values) > 0:
            x_min = all_values.min()
            x_max = all_values.max()
            margin = (x_max - x_min) * 0.1
            xaxis_config['range'] = [x_min - margin, x_max + margin]

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
    df, label_col, effect_col, lower_col, upper_col,
    title, xaxis_title, null_value, log_scale, height, width
):
    plot_df = df.copy().sort_values(by=effect_col, ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(width/100, height/100))

    y_positions = range(len(plot_df))

    for i, (_, row) in enumerate(plot_df.iterrows()):
        ax.plot(
            [row[lower_col], row[upper_col]],
            [i, i],
            color='#2a9d8f',
            linewidth=2
        )

    ax.scatter(plot_df[effect_col], y_positions, color='#2a9d8f', s=80, zorder=5)
    ax.axvline(x=null_value, color='red', linestyle='--', linewidth=2)

    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(plot_df[label_col].tolist())
    ax.set_xlabel(xaxis_title)
    ax.set_title(title)

    if log_scale:
        ax.set_xscale('log')

        # Limits e ticks customizados
        all_values = pd.concat([
            plot_df[effect_col], plot_df[lower_col], plot_df[upper_col]
        ]).replace([np.inf, -np.inf], np.nan).dropna()
        positive = all_values[all_values > 0]
        if len(positive) > 0:
            ax.set_xlim(positive.min() * 0.9, positive.max() * 1.1)
            ticks = [t for t in [0.1, 0.5, 1, 2, 5, 10, 100]
                     if positive.min() * 0.9 <= t <= positive.max() * 1.1]
            if ticks:
                ax.set_xticks(ticks)
                ax.set_xticklabels([str(t) for t in ticks])

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